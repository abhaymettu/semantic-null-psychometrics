"""Fit API: fit_decomposition for M1, fit_m2 for the spec loading split."""
import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
from jax import random
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import initialize_model
from scipy.optimize import minimize

from model.m1 import loading_mask, m1_model, m1_readout, m2_model, m2_readout

jax.config.update("jax_enable_x64", True)


def _map(model, kwargs, seed, n_starts=4):
    """Posterior mode on the UNCONSTRAINED space (log scale for a, g), L-BFGS, best of n_starts.

    This includes the log-Jacobian, so it is not the constrained MAP and never returns a = 0
    exactly; it is a preview / starting point. Reported intervals come from NUTS.
    """
    best = None
    for i in range(n_starts):
        info, potential_fn, postprocess_fn, _ = initialize_model(random.PRNGKey(seed + i), model, model_kwargs=kwargs)
        x0, unravel = jax.flatten_util.ravel_pytree(info.z)
        vg = jax.jit(jax.value_and_grad(lambda x: potential_fn(unravel(x))))
        f = lambda x: tuple(np.asarray(v, float) for v in vg(jnp.asarray(x)))
        r = minimize(f, np.asarray(x0), jac=True, method="L-BFGS-B", options=dict(maxiter=5000))
        fun = r.fun if np.isfinite(r.fun) else np.inf
        if best is None or fun < best[3]:
            best = (r, unravel, postprocess_fn, fun)
    r, unravel, post, _ = best
    sample = post(unravel(jnp.asarray(r.x)))
    return {k: np.asarray(v)[None] for k, v in sample.items()}, dict(converged=bool(r.success), neg_logpost=float(r.fun), message=str(r.message))


def _nuts(model, kwargs, seed, num_warmup, num_samples, num_chains):
    mcmc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, chain_method="sequential", progress_bar=False)
    mcmc.run(random.PRNGKey(seed), **kwargs, extra_fields=("diverging",))
    samples = mcmc.get_samples(group_by_chain=True)
    from numpyro.diagnostics import summary
    # diagnostics on sign-identified draws: a loading column (M1) or the whole profile (M2) can flip
    diag_s = dict(samples)
    if "lam" in diag_s:
        diag_s["lam"] = _sign_fix(np.asarray(diag_s["lam"]))
    if "b0" in diag_s:  # only b0^2 enters Sigma
        diag_s["b0"] = np.abs(np.asarray(diag_s["b0"]))
    if "mu" in diag_s:
        flip = np.where(np.asarray(diag_s["mu"]) < 0, -1, 1)
        for k in ("mu", "b", "delta"):
            v = np.asarray(diag_s[k])
            diag_s[k] = v * flip.reshape(flip.shape + (1,) * (v.ndim - 2))
    summ = summary(diag_s)
    diag = dict(rhat_max=float(max(np.nanmax(v["r_hat"]) for v in summ.values())),
                ess_min=float(min(np.nanmin(v["n_eff"]) for v in summ.values())),
                divergences=int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum()))
    flat = {k: np.asarray(v).reshape((-1,) + v.shape[2:]) for k, v in samples.items()}
    return flat, diag


def _sign_fix(Lambda):
    """Each column's sign is not identified. Orient each column of each draw so its largest-|.|
    entry is positive (stable for bifactor group columns whose sum can be near 0)."""
    idx = np.abs(Lambda).argmax(-2)[..., None, :]
    s = np.sign(np.take_along_axis(Lambda, idx, -2))
    s[s == 0] = 1
    return Lambda * s


def fit_decomposition(R, n_eff, K_sem, *, K_nuis=None, n_factors=1, structure="efa1", groups=None,
                      method="map", seed=0, num_warmup=500, num_samples=500, num_chains=2, prior_scale=1.0,
                      sem_scale="item", n_starts=4, const_factor=False):
    R, K_sem = np.asarray(R, float), np.asarray(K_sem, float)
    J = R.shape[0]
    if structure in ("bifactor", "cfa") and groups is None:
        raise ValueError(f"structure={structure!r} needs groups (one label per item)")
    if structure == "efa1" and n_factors != 1:
        raise ValueError("efa1 is one factor; use structure='bifactor' or 'cfa' with groups")
    mask = loading_mask(J, structure, groups, n_factors)
    kwargs = dict(R=jnp.asarray(R), n_eff=float(n_eff), K_sem=jnp.asarray(K_sem), mask=mask,
                  K_nuis=None if K_nuis is None else jnp.asarray(K_nuis, float), prior_scale=prior_scale,
                  sem_scale=sem_scale, const_factor=const_factor)
    if method == "map":
        draws, diag = _map(m1_model, kwargs, seed, n_starts=n_starts)
    else:
        draws, diag = _nuts(m1_model, kwargs, seed, num_warmup, num_samples, num_chains)
    g, b0 = draws.get("g"), draws.get("b0")
    ro = jax.vmap(lambda a, lam, gg, bb: m1_readout(a, lam, kwargs["K_sem"], mask, gg, kwargs["K_nuis"],
                                                    bb if const_factor else None))(
        jnp.asarray(draws["a"]), jnp.asarray(draws["lam"]),
        jnp.asarray(g) if g is not None else jnp.zeros_like(jnp.asarray(draws["a"])),
        jnp.asarray(b0) if b0 is not None else jnp.zeros(len(draws["a"])))
    out = {k: np.asarray(v) for k, v in ro.items()}
    out["Lambda"] = _sign_fix(out["Lambda"])
    for k in ("Sigma_sem", "Sigma_nuis", "Sigma_res"):
        out[k] = out[k].mean(0)
    out["diagnostics"] = diag
    out["draws"] = draws
    return out


def fit_m2(R, n_eff, U, *, method="map", seed=0, num_warmup=500, num_samples=500, num_chains=2, prior_scale=1.0):
    kwargs = dict(R=jnp.asarray(R, float), n_eff=float(n_eff), U=jnp.asarray(U, float), prior_scale=prior_scale)
    if method == "map":
        draws, diag = _map(m2_model, kwargs, seed)
    else:
        draws, diag = _nuts(m2_model, kwargs, seed, num_warmup, num_samples, num_chains)
    ro = jax.vmap(lambda mu, b, d, sd: m2_readout(mu, b, d, sd, kwargs["U"]))(
        jnp.asarray(draws["mu"]), jnp.asarray(draws["b"]), jnp.asarray(draws["delta"]), jnp.asarray(draws["sigma_delta"]))
    out = {k: np.asarray(v) for k, v in ro.items()}
    out["diagnostics"] = diag
    out["draws"] = draws
    return out
