"""Command adapter for the simulation recovery harness (the harness is not included in this repository).

Run: .venv/bin/python -m integration.sim_adapter <in_dir> <out_dir>
in_dir: y.npy (n x K int8 0..C-1), S.npy (K x K), meta.json {C, m}. Writes out_dir/fit.json, L.npy.
Env: SEMNULL_SEM_SCALE (global|item, default global), SEMNULL_NUTS ("warmup,samples,chains",
default 300,300,2), SEMNULL_BOOT (bootstrap draws for n_eff, default 40).

pi reported = mean over items of the standardized S-aligned share of TOTAL latent variance, which
is the sim generator's pi (Sigma = pi A + (1 - pi) R with unit-diagonal A and R). pi_common (share
of common variance) is reported alongside.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

from fit.api import fit_decomposition
from fit.polychoric import bootstrap_neff, nearest_correlation, polychoric


def main(in_dir, out_dir):
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    y = np.load(in_dir / "y.npy")
    S = np.load(in_dir / "S.npy")
    meta = json.loads((in_dir / "meta.json").read_text())
    C, m = int(meta["C"]), int(meta["m"])
    sem_scale = os.environ.get("SEMNULL_SEM_SCALE", "global")
    w, s, c = (int(x) for x in os.environ.get("SEMNULL_NUTS", "300,300,2").split(","))
    B = int(os.environ.get("SEMNULL_BOOT", "40"))
    pc = polychoric(y, C=C)
    R, adjusted = nearest_correlation(np.nan_to_num(pc["R"], nan=0.0))
    n_eff, deff, _ = bootstrap_neff(y, R, B=B, seed=0, C=C) if B > 0 else (len(y), 1.0, None)
    structure = "efa1" if m == 1 else "efa"
    fit = fit_decomposition(R, n_eff, S, n_factors=m, structure=structure, method="nuts", seed=0,
                            num_warmup=w, num_samples=s, num_chains=c, sem_scale=sem_scale)
    pi = fit["sem_share"].mean(1)
    level = 0.9
    d = fit["diagnostics"]
    out = dict(kind="posterior", pi=float(pi.mean()), pi_lo=float(np.quantile(pi, (1 - level) / 2)),
               pi_hi=float(np.quantile(pi, 1 - (1 - level) / 2)), level=level,
               diag=dict(rhat_max=d["rhat_max"], ess_bulk_min=d["ess_min"], divergences=d["divergences"]),
               extra=dict(model="M1", sem_scale=sem_scale, structure=structure, n_eff=n_eff, deff=deff,
                          R_adjusted_to_psd=bool(adjusted), degenerate_items=pc["flags"]["degenerate_items"],
                          pi_common=float(fit["pi_common"].mean()), pi_offdiag=float(fit["pi_offdiag"].mean())))
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "L.npy", fit["Lambda"].mean(0))
    (out_dir / "fit.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main(*sys.argv[1:3])
