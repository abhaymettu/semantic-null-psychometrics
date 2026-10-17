"""M0 descriptive baseline (a replication, not a contribution) for all three kernels, no model fit:
off-diagonal r(R, S), QAP p (2000 permutations), first-eigenvector congruence, via `dass42.descriptive`
on the same R and abtt-ref kernels as the headline. Uses the DASS-42 aggregates rebuilt locally.
  .venv/bin/python -m integration.m0 -> integration/out/dass42_m0.json
"""
import hashlib
import json
import subprocess
from pathlib import Path

from integration.dass42 import OUT, ROOT, descriptive, load_items, load_kernel
from scripts.load_data import matrix

MODELS = ("bge-large-en-v1.5", "all-mpnet-base-v2", "qwen3-embedding-0.6b")

if __name__ == "__main__":
    order = sorted(load_items(), key=lambda r: ("DAS".index(r["scale"]), int(r["position"])))
    ids = [r["item_id"] for r in order]
    all_ids, R_all = matrix("dass42", "primary", "polychoric")
    pos = [all_ids.index(i) for i in ids]
    R = R_all[pos][:, pos]
    out = {"git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()}
    for m in MODELS:
        S, _, _ = load_kernel(Path(ROOT / "embed"), m, "abtt-ref", ids)
        f = ROOT / "embed" / "out" / f"dass42__{m}__S_abtt-ref.npy"
        out[m] = dict(descriptive(R, S), kernel_sha256=hashlib.sha256(f.read_bytes()).hexdigest())
    (OUT / "dass42_m0.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
