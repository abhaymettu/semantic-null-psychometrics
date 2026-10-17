"""Level sweep and partial identification of the S-aligned share (design decision D11).

K_c = c 11' + (1 - c) K0 with K0 = shape_kernel(S_rep). The data inform the shape K0 but barely the
level c (a uniform rank-1 part the residual general factor can also carry). For each c on a grid:
M1 MAP fit, its log-likelihood gap to the saturated model, and the estimands. Admissible c = those
whose deviance is within 3.84 of the best c on the n_eff scale. That cut is a descriptive
likelihood-ratio band under a misspecified model; it is NOT a calibrated partial-identification
interval until its coverage is checked on known truth. Reported: the range of each
estimand over that band, the c = 0 (shape-only) value, and the value at c_obs, the level the chosen
encoder representation happens to have (conditional, not identified).
Level-preserving null: permute the items of K0 and rebuild K_c at the SAME c, so every
null kernel has exactly the observed off-diagonal multiset and level (recorded per draw).
--null-strata scale permutes within D/A/S only: does S explain anything beyond scale membership.
--const-factor adds a free constant-loading residual column (b0^2 11'). With the global scale the
c = 0 model then nests every c > 0 model (alpha^2 c 11' is absorbed by b0^2), so a c preference that
survives without it and vanishes with it is the residual asking for a uniform component, not
information about the kernel level.

Run: .venv/bin/python -m integration.level_sweep --embed-dir <wt>/embed [--model M] [--rep abtt-ref]
     [--sem-scale global] [--null 50] [--null-at 0,obs] [--null-strata none|scale] [--const-factor]
"""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from fit.api import fit_decomposition
from integration.dass42 import OUT, load_items, load_kernel
from model.kernels import level_kernel, nuisance_kernel, null_permutation, offdiag_mean, shape_kernel
from model.m1 import loading_mask, m1_readout, wishart_loglik
from scripts.load_data import matrix

GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]
KEYS = ("pi_offdiag", "pi_common", "pi_total", "nuis_offdiag")
NULL_KEYS = KEYS + ("deviance_reduction_vs_noS",)


def _gap(f, R, n_eff):
    Sig = f["Sigma_sem"] + f["Sigma_nuis"] + f["Sigma_res"] + np.diag(np.asarray(f["uniq"]).reshape(-1))
    return float(wishart_loglik(jnp.asarray(Sig), jnp.asarray(R), n_eff) - wishart_loglik(jnp.asarray(R), jnp.asarray(R), n_eff))


def ridge(f, K0, R, n_eff, kw):
    """With a global scale and the free constant column b0, alpha^2 K_c + b0^2 11'
    is the same matrix for every c in [0, c_max], c_max = b0^2 / (b0^2 + alpha0^2), where (alpha0, b0) is
    the c = 0 fit. The identified set for the S-aligned share is therefore the ridge from the c = 0 fit
    (uniform part all residual) to c_max (uniform part all semantic). Exact algebra, one fit."""
    d = f["draws"]
    alpha2, u = float(d["a"][0][0]) ** 2, float(d["b0"][0]) ** 2
    c_max = u / (u + alpha2)
    J = len(K0)
    mask = loading_mask(J, kw["structure"], kw["groups"], kw["n_factors"])
    Kn = None if kw["K_nuis"] is None else jnp.asarray(kw["K_nuis"])
    g = jnp.asarray(d["g"][0]) if "g" in d else None
    ro = {k: np.asarray(v) for k, v in m1_readout(jnp.full(J, np.sqrt(alpha2 + u)), jnp.asarray(d["lam"][0]),
                                                   jnp.asarray(level_kernel(K0, c_max)), mask, g, Kn).items()}
    end = {k: float(np.asarray(ro[k]).reshape(-1)[0]) for k in KEYS}
    start = {k: float(np.asarray(f[k]).reshape(-1)[0]) for k in KEYS}
    return dict(c_max=c_max, alpha0_sq=alpha2, b0_sq=u, at_c0=start, at_c_max=end, converged_c0=f["diagnostics"]["converged"],
                range={k: sorted([start[k], end[k]]) for k in KEYS},
                loglik_gap_c0=_gap(f, R, n_eff), loglik_gap_c_max_reconstructed=_gap(ro, R, n_eff),
                note="identified ridge under global scale + free equal-loading residual; the uniform part cannot "
                     "be split between the kernel level and any equal-loading non-semantic component (e.g. "
                     "acquiescence: all DASS-42 items keyed one way). MAP point: (alpha0, b0) carry the prior/Jacobian of "
                     "the unconstrained mode, so the ridge is exact for the fitted point, not the profile. NOT calibrated.")


def fit_at(R, n_eff, K, kw, seed=0, keep=False):
    f = fit_decomposition(R, n_eff, K, method="map", seed=seed, **kw)
    gap = _gap(f, R, n_eff)
    row = {k: float(np.asarray(f[k]).reshape(-1)[0]) for k in KEYS}
    row.update(loglik_gap=gap, converged=f["diagnostics"]["converged"])
    if "b0" in f["draws"]:
        row["b0"] = float(np.asarray(f["draws"]["b0"]).reshape(-1)[0])
    return (row, f) if keep else row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-dir", type=Path, required=True)
    ap.add_argument("--model", default="bge-large-en-v1.5")
    ap.add_argument("--rep", default="abtt-ref")
    ap.add_argument("--structure", default="bifactor", choices=["bifactor", "cfa", "efa1", "efa"])
    ap.add_argument("--n-factors", type=int, default=1, help="for structure=efa")
    ap.add_argument("--starts", type=int, default=4, help="MAP random starts per fit")
    ap.add_argument("--sem-scale", default="global")
    ap.add_argument("--variant", default="primary")
    ap.add_argument("--null", type=int, default=0)
    ap.add_argument("--null-at", default="0,obs")
    ap.add_argument("--null-strata", default="none", choices=["none", "scale"])
    ap.add_argument("--const-factor", action="store_true")
    ap.add_argument("--grid", default=None, help="comma list of c to replace the default grid (c_obs always added)")
    ap.add_argument("--baseline-json", type=Path, default=None,
                    help="reuse baseline_noS from a committed output with the same structure, scale, variant and starts "
                         "(K = 0 does not depend on the encoder) instead of refitting it")
    a = ap.parse_args()

    items = load_items()
    order = sorted(items, key=lambda r: ("DAS".index(r["scale"]), int(r["position"])))
    ids = [r["item_id"] for r in order]
    all_ids, R_all = matrix("dass42", a.variant, "polychoric")
    pos = [all_ids.index(i) for i in ids]
    R = R_all[np.ix_(pos, pos)]
    n_eff = json.loads((OUT / "dass42_neff.json").read_text())[a.variant]["n_eff"]
    S, F, _ = load_kernel(a.embed_dir, a.model, a.rep, ids)
    K0 = shape_kernel(S)
    m0, L = offdiag_mean(K0), offdiag_mean(S)
    c_obs = (L - m0) / (1 - m0)
    kw = dict(K_nuis=nuisance_kernel(F), structure=a.structure, groups=[r["scale"] for r in order],
              sem_scale=a.sem_scale, n_factors=a.n_factors, n_starts=a.starts, const_factor=a.const_factor)
    struct_tag = a.structure + (str(a.n_factors) if a.structure == "efa" else "") + ("+const" if a.const_factor else "")
    kfile = a.embed_dir / "out" / f"dass42__{a.model}__S_{a.rep}.npy"
    prov = dict(git_head=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                git_dirty=bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()),
                kernel_file=str(kfile), kernel_sha256=hashlib.sha256(kfile.read_bytes()).hexdigest(), starts=a.starts)
    grid = GRID if a.grid is None else [float(x) for x in a.grid.split(",")]
    t0 = time.time()

    # baseline without any S-aligned component (K = 0): the deviance reduction S buys at each c is
    # the alignment statistic that does not depend on how the shares are summed
    if a.baseline_json is None:
        base = fit_at(R, n_eff, np.zeros_like(K0), kw)
    else:
        src = json.loads(a.baseline_json.read_text())
        want = dict(structure=struct_tag, sem_scale=a.sem_scale, variant=a.variant, n_eff=n_eff)
        if any(src[k] != v for k, v in want.items()) or src["provenance"]["starts"] != a.starts or not src["baseline_noS"]["converged"]:
            raise SystemExit(f"refuse: {a.baseline_json} baseline is not the same converged K = 0 fit ({want}, starts={a.starts})")
        base = src["baseline_noS"]
        prov.update(baseline_file=str(a.baseline_json), baseline_sha256=hashlib.sha256(a.baseline_json.read_bytes()).hexdigest())
    cs = sorted(set(grid + [round(c_obs, 4)]))
    rows, ridge_out = [], None
    for c in cs:
        r, f = fit_at(R, n_eff, level_kernel(K0, c), kw, keep=True)
        if c == 0.0 and a.const_factor and a.sem_scale == "global":
            ridge_out = ridge(f, K0, R, n_eff, kw)
        r["c"], r["K_offdiag_mean"] = c, offdiag_mean(level_kernel(K0, c))
        r["deviance_reduction_vs_noS"] = 2 * (r["loglik_gap"] - base["loglik_gap"])
        rows.append(r)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    best = max(r["loglik_gap"] for r in rows)
    adm = [r for r in rows if 2 * (best - r["loglik_gap"]) <= 3.84]
    out = dict(
        provenance=prov, model=a.model, rep=a.rep, structure=struct_tag, sem_scale=a.sem_scale, variant=a.variant, n_eff=n_eff,
        S_rep_offdiag_mean=L, K0_offdiag_mean=m0, c_obs=c_obs, baseline_noS=base, grid=rows,
        admissible_c=[r["c"] for r in adm],
        lr_band_3p84={k: [min(r[k] for r in adm), max(r[k] for r in adm)] for k in KEYS},
        shape_only={k: next(r[k] for r in rows if r["c"] == 0.0) for k in KEYS},
        at_c_obs={k: next(r[k] for r in rows if r["c"] == round(c_obs, 4)) for k in KEYS},
        ridge=ridge_out,
        status="descriptive LR band (3.84) over kernel level under a misspecified model; MAP; NOT a calibrated "
               "partial-identification interval; NOT calibrated (sim recovery pending)",
    )
    strata = None if a.null_strata == "none" else ["DAS".index(r["scale"]) for r in order]
    if a.null:
        out["null"] = {}
        for tag in a.null_at.split(","):
            c = 0.0 if tag == "0" else round(c_obs, 4)
            obs = next(r for r in rows if r["c"] == c)
            vals, levels = [], []
            for b in range(a.null):
                p = null_permutation(len(K0), strata, rng=2000 + b)
                Kp = level_kernel(K0[np.ix_(p, p)], c)
                levels.append(offdiag_mean(Kp))
                x = fit_at(R, n_eff, Kp, kw, seed=b)
                x["deviance_reduction_vs_noS"] = 2 * (x["loglik_gap"] - base["loglik_gap"])
                vals.append(x)
            v = {k: np.array([x[k] for x in vals]) for k in NULL_KEYS}
            out["null"][f"c={c}"] = {
                k: dict(observed=obs[k], null_mean=float(v[k].mean()), null_q95=float(np.quantile(v[k], .95)),
                        p_upper=float((1 + (v[k] >= obs[k]).sum()) / (1 + a.null))) for k in NULL_KEYS}
            out["null"][f"c={c}"]["kind"] = ("item-label permutation of K0" + (" within D/A/S" if strata else "")
                                             + ", level rebuilt at same c (exact level and value multiset)")
            out["null"][f"c={c}"].update(
                n=a.null, seeds=f"null_permutation rng=2000+b, fit seed=b, b<{a.null}", strata=a.null_strata,
                observed_level=offdiag_mean(level_kernel(K0, c)), null_levels_min=min(levels), null_levels_max=max(levels),
                draws={k: [x[k] for x in vals] for k in NULL_KEYS + ("converged",)})
            print(tag, json.dumps(out["null"][f"c={c}"]["pi_offdiag"]), flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    null_tag = f"__null{a.null}-{a.null_strata}" if a.null else ""
    grid_tag = "" if a.grid is None else "__grid" + a.grid.replace(",", "_")
    path = OUT / f"dass42_level__{a.model}__{a.rep}__{struct_tag}__{a.sem_scale}{grid_tag}{null_tag}.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("c_obs", "admissible_c", "lr_band_3p84", "shape_only", "at_c_obs", "ridge")}, indent=1))


if __name__ == "__main__":
    main()
