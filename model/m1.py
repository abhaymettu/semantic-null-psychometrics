"""M1: independent-source decomposition of a latent correlation matrix (design decision D1-D3).

Sigma0 = A K_sem A + G K_nuis G + Lambda Lambda' + I,  Sigma = D^-1/2 Sigma0 D^-1/2.
Likelihood: Wishart / normal-theory ML on the polychoric matrix at an effective sample size (D10).

M2: the spec's loading split, one general factor with loadings mu + U b + delta (D6), fitted on the
same likelihood; its readout keeps the semantic x residual cross term as its own matrix.
"""
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def loading_mask(n_items, structure, groups=None, n_factors=1):
    """0/1 mask (J, F) for Lambda. efa1: one column. efa: n_factors columns, lower-triangular
    (upper triangle zero, the standard exploratory identification). bifactor: general + one
    column per group. cfa: one column per group."""
    if structure == "efa1":
        return jnp.ones((n_items, 1))
    if structure == "efa":
        return jnp.tril(jnp.ones((n_items, n_factors)))
    if structure == "bifactor":
        labels = sorted(set(groups))
        cols = [jnp.ones(n_items)] + [jnp.array([g == l for g in groups], float) for l in labels]
        return jnp.stack(cols, 1)
    if structure == "cfa":
        labels = sorted(set(groups))
        return jnp.stack([jnp.array([g == l for g in groups], float) for l in labels], 1)
    raise ValueError(structure)


def components(a, Lambda, K_sem, g=None, K_nuis=None):
    """Raw (unstandardized) PSD components and the standardizing vector d = diag(Sigma0)."""
    S_sem = a[:, None] * K_sem * a[None, :]
    S_res = Lambda @ Lambda.T
    S_nuis = jnp.zeros_like(S_sem) if K_nuis is None else g[:, None] * K_nuis * g[None, :]
    d = jnp.diag(S_sem) + jnp.diag(S_nuis) + jnp.diag(S_res) + 1.0
    return S_sem, S_nuis, S_res, d


def wishart_loglik(Sigma, R, n):
    """Normal-theory ML log-likelihood up to terms constant in Sigma (no -log|R| - J), so values
    are comparable across parameter values of one model, not against a saturated model."""
    L = jnp.linalg.cholesky(Sigma)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    Sinv_R = jax.scipy.linalg.cho_solve((L, True), R)
    return -0.5 * n * (logdet + jnp.trace(Sinv_R))


def with_const(Lambda, b0):
    """Append the free constant-loading residual column (one shared loading b0): a uniform rank-1
    residual term b0^2 11' that competes with the uniform part c 11' of a level kernel."""
    return Lambda if b0 is None else jnp.concatenate([Lambda, b0 * jnp.ones((Lambda.shape[0], 1))], 1)


def m1_model(R, n_eff, K_sem, mask, K_nuis=None, prior_scale=1.0, sem_scale="item", const_factor=False):
    J, F = mask.shape
    # hierarchical (non-centred) item scales: a shared scale lets every a_k shrink to 0 together,
    # which matters because 42 free scales otherwise pick up noise as a positive semantic share
    # sem_scale="global": one scale for all items (A K A = alpha^2 K), the spec's single dial; it
    # cannot bend item-by-item to absorb residual misfit, so its random-kernel floor is lower
    sigma_a = numpyro.sample("sigma_a", dist.HalfNormal(prior_scale))
    if sem_scale == "global":
        a = numpyro.deterministic("a", sigma_a * jnp.ones(J))
    else:
        a = numpyro.deterministic("a", sigma_a * numpyro.sample("a_raw", dist.HalfNormal(1.0).expand([J]).to_event(1)))
    g = None
    if K_nuis is not None:
        sigma_g = numpyro.sample("sigma_g", dist.HalfNormal(prior_scale))
        g = numpyro.deterministic("g", sigma_g * numpyro.sample("g_raw", dist.HalfNormal(1.0).expand([J]).to_event(1)))
    lam = numpyro.sample("lam", dist.Normal(0.0, prior_scale).expand([J, F]).to_event(2))
    b0 = numpyro.sample("b0", dist.Normal(0.0, prior_scale)) if const_factor else None
    Lambda = with_const(lam * mask, b0)
    S_sem, S_nuis, S_res, d = components(a, Lambda, K_sem, g, K_nuis)
    s = 1.0 / jnp.sqrt(d)
    Sigma = (S_sem + S_nuis + S_res + jnp.eye(J)) * s[:, None] * s[None, :]
    numpyro.factor("lik", wishart_loglik(Sigma, R, n_eff))


def m2_model(R, n_eff, U, prior_scale=1.0):
    """Spec loading split, one factor: lambda = mu + U b + delta (raw scale, unit raw uniqueness)."""
    J, m = U.shape
    mu = numpyro.sample("mu", dist.Normal(0.0, prior_scale))
    b = numpyro.sample("b", dist.Normal(0.0, prior_scale).expand([m]).to_event(1))
    sd = numpyro.sample("sigma_delta", dist.HalfNormal(prior_scale))
    delta = numpyro.sample("delta", dist.Normal(0.0, 1.0).expand([J]).to_event(1)) * sd
    lam = mu + U @ b + delta
    d = lam**2 + 1.0
    s = 1.0 / jnp.sqrt(d)
    Sigma = (jnp.outer(lam, lam) + jnp.eye(J)) * s[:, None] * s[None, :]
    numpyro.factor("lik", wishart_loglik(Sigma, R, n_eff))


def estimands(Sigma_sem, Sigma_nuis, Sigma_res, uniq):
    """The estimands from standardized components (unit-diagonal total). Single definition used by
    the fit readout AND by the simulation code to compute truth."""
    J = Sigma_sem.shape[0]
    off = 1.0 - jnp.eye(J)
    Csem, Cnuis, Cres = Sigma_sem * off, Sigma_nuis * off, Sigma_res * off
    Ctot = Csem + Cnuis + Cres
    sem, nuis, res = jnp.diag(Sigma_sem), jnp.diag(Sigma_nuis), jnp.diag(Sigma_res)
    return dict(
        pi_offdiag=Csem.sum() / Ctot.sum(), nuis_offdiag=Cnuis.sum() / Ctot.sum(),
        sem_cov_share=Csem.sum(1) / Ctot.sum(1),
        pi_common=sem.sum() / (sem.sum() + nuis.sum() + res.sum()),
        pi_total=sem.mean(), sem_share=sem, nuis_share=nuis, res_share=res, uniq=uniq,
    )


def pi_shape(Sigma_sem, Sigma_nuis, Sigma_res, uniq, groups=None):
    """design decision D14 (PROPOSED, not yet in `estimands`): level-free PROJECTION COEFFICIENT of the
    S-aligned part on the non-uniform pair-covariance pattern (not a variance fraction: it can be
    negative, and above 1 even with PSD components under correct specification).
    s = sqrt(uniq) is the standardized image of a raw equal-loading direction; Q removes each component's
    projection on off(s s'), so the kernel level, b0 and any equal-raw-loading factor drop out, and the
    value is constant along the D13 ridge. That ridge needs one global semantic scale (A K A = alpha^2 K):
    with per-item scales the level term a_i a_j c is not equal-loading and the invariance does not hold.
    groups (D14b, PROPOSED): Q also removes off(s s' o B_g) for each group's within-block indicator B_g,
    i.e. every raw equal-within-group pattern. That is the scale-membership exact-exchange class (K0 in
    span{I, B_g, 11'} traded against equal bifactor loadings), on which this version is invariant.
    Shares sum to 1; sem_energy is a non-additive diagnostic; den_ratio = ||Q(C)||^2 / ||off(C)||^2 (report it;
    the floor must be fixed in advance). NaN when ||Q(C)||^2 <= max(1e-12, 100 eps) ||off(C)||^2
    (nothing outside the removed patterns to share)."""
    J = Sigma_sem.shape[0]
    assert groups is None or len(groups) == J, "groups needs one label per item"
    off = 1.0 - jnp.eye(J)
    s = jnp.sqrt(uniq)
    ss = jnp.outer(s, s) * off
    Us = [ss] + ([] if groups is None else
                 [ss * jnp.outer(m, m) for m in (jnp.array([x == l for x in groups], float) for l in sorted(set(groups)))])
    U = jnp.stack([u.ravel() for u in Us], 1)
    Up = jnp.linalg.pinv(U)  # (k, J^2); pinv: a singleton or single group makes the columns dependent

    def Q(M):  # never form the J^2 x J^2 projector (memory under vmap over draws)
        m = (M * off).ravel()
        return m - U @ (Up @ m)

    qs, qn, qr = Q(Sigma_sem), Q(Sigma_nuis), Q(Sigma_res)
    qc = qs + qn + qr
    den = jnp.sum(qc * qc)
    ratio = den / jnp.sum(((Sigma_sem + Sigma_nuis + Sigma_res) * off) ** 2)
    den = jnp.where(ratio <= jnp.maximum(1e-12, 100 * jnp.finfo(den.dtype).eps), jnp.nan, den)  # float32 roundoff ~1e-12
    return dict(sem=jnp.sum(qs * qc) / den, nuis=jnp.sum(qn * qc) / den, res=jnp.sum(qr * qc) / den,
                sem_energy=jnp.sum(qs * qs) / den, den_ratio=ratio)


def m1_readout(a, lam, K_sem, mask, g=None, K_nuis=None, b0=None):
    """Standardized per-item shares and component matrices for one parameter draw (numpy-able)."""
    Lambda = with_const(lam * mask, b0)
    S_sem, S_nuis, S_res, d = components(a, Lambda, K_sem, g, K_nuis)
    s = 1.0 / jnp.sqrt(d)
    std = lambda M: M * s[:, None] * s[None, :]
    out = estimands(std(S_sem), std(S_nuis), std(S_res), 1.0 / d)
    out.update(Lambda=Lambda * s[:, None], Sigma_sem=std(S_sem), Sigma_nuis=std(S_nuis), Sigma_res=std(S_res))
    return out


def m2_readout(mu, b, delta_raw, sigma_delta, U):
    lam = mu + U @ b + delta_raw * sigma_delta
    s = 1.0 / jnp.sqrt(lam**2 + 1.0)
    lam_sem = (mu + U @ b) * s           # semantic-predicted standardized loading (includes mu, see D6)
    lam_dev = (U @ b) * s                # the part that varies with semantics
    lam_res = delta_raw * sigma_delta * s
    return dict(
        loading=lam * s, loading_sem=lam_sem, loading_sem_dev=lam_dev, loading_res=lam_res,
        Sigma_semsem=jnp.outer(lam_sem, lam_sem), Sigma_cross=jnp.outer(lam_sem, lam_res) + jnp.outer(lam_res, lam_sem),
        Sigma_resres=jnp.outer(lam_res, lam_res),
        # variance-component ratio of the raw loading profile; ignores the sample cov of the two parts
        r2_sem=jnp.var(U @ b) / (jnp.var(U @ b) + jnp.var(delta_raw * sigma_delta)),
    )
