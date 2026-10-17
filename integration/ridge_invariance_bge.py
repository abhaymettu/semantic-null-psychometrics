"""Population-algebra check of design decision D14 / D14b ridge invariance on the four calibration truths
(B1, B2, B4, N0) built with the real bge-large-en-v1.5 abtt-ref kernel. No fit, optimizer, sampler or MC.
The only RNG is np.random.default_rng(0) inside the frozen truth definition (loadings), as in
integration.calibrate.truth. Usage (from repo root): python -m integration.ridge_invariance_bge OUT.json
(or python integration/ridge_invariance_bge.py OUT.json with the repo root on PYTHONPATH)."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from integration.calibrate import CELLS, setup, truth
from model.kernels import level_kernel
from model.m1 import loading_mask, m1_readout, pi_shape

ROOT = Path(__file__).resolve().parents[1]


def build(cell, K0, Kn, groups, seed=0):
    """Same construction as calibrate.truth, but returns truth and both ridge ends as component dicts."""
    p = CELLS[cell]
    J = len(K0)
    rng = np.random.default_rng(seed)
    mask = np.asarray(loading_mask(J, "bifactor", groups))
    lam = mask * np.column_stack([rng.uniform(0.9, 1.3, J)] + [rng.uniform(0.3, 0.7, J)] * 3)
    cols = [lam]
    if p["m"]:
        cols.append(np.full((J, 1), p["m"]))
    if p["omit"]:
        o = np.zeros((J, 1)); o[rng.choice(J, 6, replace=False)] = 0.6
        cols.append(o)
    L = np.hstack(cols)
    g = np.full(J, 0.15)
    ro = lambda aa, KK, LL: {k: np.asarray(v) for k, v in m1_readout(
        jnp.asarray(aa), jnp.asarray(LL), jnp.asarray(KK), jnp.ones(LL.shape), jnp.asarray(g), jnp.asarray(Kn)).items()}
    t = ro(np.full(J, p["alpha"]), level_kernel(K0, p["c"]), L)
    u = p["alpha"] ** 2 * p["c"] + p["m"] ** 2
    Lg = np.hstack([lam] + ([cols[-1]] if p["omit"] else []))
    a_lo = np.sqrt(p["alpha"] ** 2 * (1 - p["c"]))
    lo = ro(np.full(J, a_lo), K0, np.hstack([Lg, np.full((J, 1), np.sqrt(u))]))
    s2 = p["alpha"] ** 2 * (1 - p["c"]) + u
    c_hi = u / s2 if s2 > 0 else 0.0
    hi = ro(np.full(J, np.sqrt(s2)), level_kernel(K0, c_hi) if s2 > 0 else K0, Lg)
    ends = dict(truth=dict(alpha=p["alpha"], c=p["c"], b0=p["m"]),
                lower=dict(alpha=float(a_lo), c=0.0, b0=float(np.sqrt(u))),
                upper=dict(alpha=float(np.sqrt(s2)), c=float(c_hi), b0=0.0))
    return t, lo, hi, ends


def sigma(r):
    return r["Sigma_sem"] + r["Sigma_nuis"] + r["Sigma_res"] + np.diag(r["uniq"])


def shape(r, groups):
    out = pi_shape(*(jnp.asarray(r[k]) for k in ("Sigma_sem", "Sigma_nuis", "Sigma_res", "uniq")), groups)
    return float(out["sem"])


def main():
    out_path = Path(sys.argv[1])
    K0, Kn, groups = setup()
    kfile = ROOT / "embed" / "out" / "dass42__bge-large-en-v1.5__S_abtt-ref.npy"
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    res = dict(git_head=head or os.environ.get("GIT_HEAD"), kernel_file=str(kfile.relative_to(ROOT)),
               kernel_sha256=hashlib.sha256(kfile.read_bytes()).hexdigest() if kfile.exists() else None,
               method="population algebra only; loadings from the truth builder's seed-0 RNG", cells={})
    for cell in CELLS:
        t, lo, hi, ends = build(cell, K0, Kn, groups)
        _, tr = truth(cell, K0, Kn, groups)  # repo function, cross-check of the rebuilt truth
        S = {"truth": sigma(t), "lower": sigma(lo), "upper": sigma(hi)}
        R = {"truth": t, "lower": lo, "upper": hi}
        row = dict(params=ends)
        for name, grp in (("pi_shape_D14", None), ("pi_shape_D14b_groups", groups)):
            v = {e: shape(R[e], grp) for e in R}
            row[name] = dict(values=v, max_abs_diff=max(abs(v[e] - v["truth"]) for e in ("lower", "upper")))
        row["pi_offdiag"] = dict(truth=float(t["pi_offdiag"]), lower=float(lo["pi_offdiag"]), upper=float(hi["pi_offdiag"]))
        row["max_abs_Sigma_diff_vs_truth"] = {e: float(np.abs(S[e] - S["truth"]).max()) for e in ("lower", "upper")}
        row["crosscheck_calibrate_truth"] = dict(
            pi_shape=tr["pi_shape"], pi_shape_groups=tr["pi_shape_groups"], ridge=tr["ridge"],
            matches=bool(abs(tr["pi_shape"] - row["pi_shape_D14"]["values"]["truth"]) < 1e-12
                         and np.allclose(tr["ridge"], [row["pi_offdiag"]["lower"], row["pi_offdiag"]["upper"]], atol=1e-12)))
        res["cells"][cell] = row
    out_path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
