"""Split-half stability of the M1 decomposition on DASS-42 (aggregates only out).

Run: SEMNULL_CACHE=<data worktree>/data/cache .venv/bin/python -m integration.split --embed-dir <worktree>/embed
     [--model M] [--rep abtt-ref] [--structure bifactor] [--sem-scale global] [--splits 5]
For each random half split of the primary rows: polychoric per half, M1 MAP per half (n_eff = half n / deff).
Reports |pi_A - pi_B| and the correlation of per-item semantic shares across halves.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from fit.api import fit_decomposition
from fit.polychoric import polychoric
from integration.dass42 import OUT, load_items, load_kernel
from model.kernels import nuisance_kernel
from scripts.build_aggregates import load_dass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-dir", type=Path, required=True)
    ap.add_argument("--model", default="bge-large-en-v1.5")
    ap.add_argument("--rep", default="abtt-ref")
    ap.add_argument("--structure", default="bifactor")
    ap.add_argument("--sem-scale", default="global")
    ap.add_argument("--splits", type=int, default=5)
    a = ap.parse_args()

    items = load_items()
    order = sorted(items, key=lambda r: ("DAS".index(r["scale"]), int(r["position"])))
    ids = [r["item_id"] for r in order]
    col = [int(r["data_column"][1:-1]) - 1 for r in order]  # load_dass columns are Q1A..Q42A
    Y, _, variants, _ = load_dass()
    Y = Y[variants["primary"]][:, col].astype(np.int8)
    deff = json.loads((OUT / "dass42_neff.json").read_text())["primary"]["deff"]
    S, F, _ = load_kernel(a.embed_dir, a.model, a.rep, ids)
    kw = dict(K_nuis=nuisance_kernel(F), structure=a.structure, groups=[r["scale"] for r in order],
              sem_scale=a.sem_scale, method="map")
    rows = []
    for s in range(a.splits):
        idx = np.random.default_rng(s).permutation(len(Y))
        halves = [idx[: len(Y) // 2], idx[len(Y) // 2:]]
        fits = [fit_decomposition(polychoric(Y[h], C=4)["R"], len(h) / deff, S, seed=s, **kw) for h in halves]
        pa, pb = (float(f["pi_common"][0]) for f in fits)
        sa, sb = (f["sem_share"][0] for f in fits)
        ra, rb = (f["res_share"][0] for f in fits)
        rows.append(dict(split=s, pi_A=pa, pi_B=pb, abs_diff=abs(pa - pb),
                         sem_share_corr=float(np.corrcoef(sa, sb)[0, 1]) if sa.std() > 0 and sb.std() > 0 else None,
                         res_share_corr=float(np.corrcoef(ra, rb)[0, 1])))
        print(rows[-1], flush=True)
    out = dict(model=a.model, rep=a.rep, structure=a.structure, sem_scale=a.sem_scale, deff=deff, splits=rows,
               max_abs_diff=max(r["abs_diff"] for r in rows),
               status="stability across halves of the same sample; not calibration and not replication in new data")
    (OUT / f"dass42_split__{a.model}__{a.rep}__{a.structure}__{a.sem_scale}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
