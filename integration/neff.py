"""Effective sample size for the Wishart likelihood (design decision D10) and a cross-check of the
pairwise polychoric estimator in fit/ against the aggregate built by scripts/build_aggregates.py.

Rows come from the loader (scripts/build_aggregates.load_dass) and the git-ignored
cache ($SEMNULL_CACHE); only aggregate numbers are written.
Run: SEMNULL_CACHE=<data worktree>/data/cache .venv/bin/python -m integration.neff [B]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from fit.polychoric import bootstrap_neff, polychoric
from scripts.build_aggregates import load_dass
from scripts.load_data import matrix

OUT = Path(__file__).resolve().parent / "out"


def main(B=200):
    Y, _, variants, counts = load_dass()
    res = {"dataset": "dass42", "B": B, "seed": 0}
    for v in ("primary", "primary_engnat"):
        Yv = Y[variants[v]].astype(np.int8)
        t = time.time()
        R = polychoric(Yv, C=4)["R"]
        ids, R_data = matrix("dass42", v, "polychoric")
        n_eff, deff, sd = bootstrap_neff(Yv, R, B=B, seed=0, C=4)
        res[v] = dict(n=int(len(Yv)), n_eff=round(n_eff, 1), deff=round(deff, 4),
                      boot_sd_median=float(np.median(sd)), boot_sd_max=float(sd.max()),
                      max_abs_diff_vs_data_lane=float(np.abs(R - R_data).max()),
                      seconds=round(time.time() - t, 1))
        print(v, res[v], flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "dass42_neff.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
