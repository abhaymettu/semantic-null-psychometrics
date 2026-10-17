"""Two-step polychoric correlations for ordinal items, vectorized over all item pairs.

Step 1: thresholds per item from marginal cumulative proportions.
Step 2: per pair, the latent correlation maximizing the bivariate-normal rectangle likelihood of
the observed contingency table, with thresholds held fixed. Pairwise-complete counts; responses
are coded 0..C-1 with -1 missing. Optional per-person weights give a fast multinomial bootstrap.
"""
import numpy as np
from scipy.special import ndtr, ndtri, owens_t

BIG = 10.0  # stands in for +/- infinity in thresholds; ndtr(10) = 1 - 7.6e-24


def bvn_cdf(h, k, rho):
    """P(X <= h, Y <= k) for standard bivariate normal with correlation rho (Owen 1956, via T)."""
    h, k, rho = np.broadcast_arrays(np.asarray(h, float), np.asarray(k, float), np.asarray(rho, float))
    h = np.where(h == 0, 1e-12, h)
    k = np.where(k == 0, 1e-12, k)
    s = np.sqrt(1.0 - rho * rho)
    ah = (k - rho * h) / (h * s)
    ak = (h - rho * k) / (k * s)
    beta = np.where((h * k < 0), 0.5, 0.0)
    return 0.5 * ndtr(h) + 0.5 * ndtr(k) - owens_t(h, ah) - owens_t(k, ak) - beta


def _onehot(Y, C):
    N, J = Y.shape
    O = np.zeros((N, J * C), dtype=np.float32)
    rows = np.repeat(np.arange(N), J)
    cols = (np.arange(J) * C)[None, :] + np.where(Y < 0, 0, Y)
    mask = (Y >= 0).ravel()
    O[rows[mask], cols.ravel()[mask]] = 1.0
    return O


def _thresholds(marg):
    """marg: (J, C) counts -> (J, C+1) thresholds with -BIG / +BIG ends."""
    p = marg / marg.sum(1, keepdims=True)
    cum = np.clip(np.cumsum(p, 1)[:, :-1], 1e-9, 1 - 1e-9)
    t = ndtri(cum)
    J = marg.shape[0]
    return np.concatenate([np.full((J, 1), -BIG), np.clip(t, -BIG, BIG), np.full((J, 1), BIG)], 1)


def _pair_loglik(rho, tj, tk, tab):
    """rho: (P,), tj/tk: (P, C+1), tab: (P, C, C) -> (P,) log-likelihood."""
    r = rho[:, None, None]
    F = bvn_cdf(tj[:, :, None], tk[:, None, :], r)  # (P, C+1, C+1)
    cell = F[:, 1:, 1:] - F[:, :-1, 1:] - F[:, 1:, :-1] + F[:, :-1, :-1]
    return (tab * np.log(np.clip(cell, 1e-12, None))).sum((1, 2))


def polychoric(Y, C=None, weights=None, iters=60):
    """Y: (N, J) int, 0..C-1, -1 missing. Returns dict(R, thresholds, n_pairs, C)."""
    Y = np.asarray(Y)
    C = int(Y.max()) + 1 if C is None else C
    N, J = Y.shape
    O = _onehot(Y, C)
    Ow = O if weights is None else O * np.asarray(weights, np.float32)[:, None]
    T = (Ow.T @ O).astype(np.float64).reshape(J, C, J, C).transpose(0, 2, 1, 3)  # (J, J, C, C)
    iu, ju = np.triu_indices(J, 1)
    tabs = T[iu, ju]
    # thresholds from each item's own marginal (all non-missing responses)
    marg = np.stack([T[j, j].diagonal() for j in range(J)])
    th = _thresholds(marg)
    tj, tk = th[iu], th[ju]
    # vectorized golden-section search on (-0.999, 0.999)
    lo, hi = np.full(len(iu), -0.999), np.full(len(iu), 0.999)
    g = (np.sqrt(5) - 1) / 2
    x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
    f1, f2 = _pair_loglik(x1, tj, tk, tabs), _pair_loglik(x2, tj, tk, tabs)
    for _ in range(iters):
        left = f1 > f2  # maximum lies in [lo, x2]
        hi = np.where(left, x2, hi)
        lo = np.where(left, lo, x1)
        x2n = np.where(left, x1, lo + g * (hi - lo))
        x1n = np.where(left, hi - g * (hi - lo), x2)
        f2n = np.where(left, f1, np.nan)
        f1n = np.where(left, np.nan, f2)
        need1, need2 = np.isnan(f1n), np.isnan(f2n)
        f1n[need1] = _pair_loglik(x1n[need1], tj[need1], tk[need1], tabs[need1])
        f2n[need2] = _pair_loglik(x2n[need2], tj[need2], tk[need2], tabs[need2])
        x1, x2, f1, f2 = x1n, x2n, f1n, f2n
    rho = (lo + hi) / 2
    # an item with < 2 observed categories carries no information on rho: NaN, never a fake 0.999
    n_obs_cat = (marg > 0).sum(1)
    degenerate = (n_obs_cat[iu] < 2) | (n_obs_cat[ju] < 2)
    rho = np.where(degenerate, np.nan, rho)
    at_bound = np.abs(np.abs(rho) - 0.999) < 1e-6
    R = np.eye(J)
    R[iu, ju] = R[ju, iu] = rho
    n_pairs = np.zeros((J, J))
    n_pairs[iu, ju] = n_pairs[ju, iu] = tabs.sum((1, 2))
    flags = dict(degenerate_items=[int(j) for j in np.flatnonzero(n_obs_cat < 2)],
                 at_bound_pairs=[(int(i), int(j)) for i, j in zip(iu[at_bound], ju[at_bound])])
    return dict(R=R, thresholds=th[:, 1:-1], n_pairs=n_pairs, C=C, flags=flags)


def nearest_correlation(R, eps=1e-6, iters=200):
    """Higham (2002) alternating projections; returns R unchanged if already PD."""
    if np.linalg.eigvalsh(R).min() > eps:
        return R, False
    Y, dS = R.copy(), np.zeros_like(R)
    for _ in range(iters):
        Rk = Y - dS
        w, V = np.linalg.eigh(Rk)
        X = (V * np.maximum(w, eps)) @ V.T
        dS = X - Rk
        Y = X.copy()
        np.fill_diagonal(Y, 1.0)
    return Y, True


def bootstrap_neff(Y, R, B=200, seed=0, C=None):
    """Multinomial bootstrap of the polychoric matrix. Returns (n_eff, deff, boot_sd).

    deff = median over pairs of Var_boot(r) / ((1 - r^2)^2 / N), so n_eff = N / deff is the sample
    size at which normal-theory (Wishart) variance matches the bootstrap variance.
    """
    rng = np.random.default_rng(seed)
    N, J = Y.shape
    iu, ju = np.triu_indices(J, 1)
    draws = np.empty((B, len(iu)))
    for b in range(B):
        w = rng.multinomial(N, np.full(N, 1.0 / N))
        draws[b] = polychoric(Y, C=C, weights=w)["R"][iu, ju]
    sd = draws.std(0, ddof=1)
    r = R[iu, ju]
    deff = float(np.median(sd**2 / ((1 - r**2) ** 2 / N)))
    deff = max(deff, 1.0)  # bootstrap noise must never make the likelihood more confident than N
    return N / deff, deff, sd
