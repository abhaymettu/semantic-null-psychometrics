#!/bin/bash
# R3 item-heldout launcher (integration/r3_heldout.py), as frozen for the paper's run; run from the repo root.
# 3 jobs (one per encoder), at most 2 at a time, single-threaded. ONE wall budget of 5 h for the whole launcher, shared by the
# 2 workers (not 5 h per job): the runner stops drawing null permutations at the deadline and writes INCOMPLETE_TIMEOUT; a job
# still queued at the deadline writes INCOMPLETE_STARVED without fitting; GNU timeout (deadline + 10 min) is the hard stop, and a
# killed job's output keeps its last checkpoint (status STARTED or RUNNING). All of these are INCOMPLETE: no verdict, no rerun.
# The runner checkpoints its one output file after the observed fits and after every null draw, and refuses if it exists.
# Exits nonzero if any job exits nonzero. Kernels come from the repo's embed/ (frozen INPUTS).
# Usage: integration/run_r3_heldout.sh <freeze-hash>   (freeze-hash: full 40-hex hash of the last commit touching INPUTS)
set -euo pipefail
cd "$(dirname "$0")/.."
FREEZE=${1:?freeze hash required}
EMBED=embed
F7=${FREEZE:0:7}
PATHS=$(.venv/bin/python -c 'from integration.r3_heldout import INPUTS; print(" ".join(INPUTS))')
git ls-files --error-unmatch $PATHS > /dev/null || { echo "refuse: an input is untracked" >&2; exit 2; }
[[ -z "$(git ls-files --others --exclude-standard -- $PATHS)" ]] || { echo "refuse: untracked files under the inputs" >&2; exit 2; }
LAST=$(git log -1 --format=%H HEAD -- $PATHS)
[[ "$FREEZE" == "$LAST" ]] || { echo "refuse: $FREEZE is not the full hash of the last commit touching the inputs ($LAST)" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "refuse: tracked files dirty" >&2; exit 3; }
.venv/bin/python -m integration.r3_heldout check-inputs --embed-dir "$EMBED" || { echo "refuse: freeze input assertions failed" >&2; exit 5; }
LOG=integration/out/r3_heldout__$F7.log
for M in bge-large-en-v1.5 all-mpnet-base-v2 qwen3-embedding-0.6b; do
  [[ ! -e "integration/out/r3_heldout__${M}__abtt-ref.json" ]] || { echo "refuse: output exists for $M" >&2; exit 4; }
done
[[ ! -e "$LOG" ]] || { echo "refuse: $LOG exists (runs once)" >&2; exit 4; }
BUDGET=18000
DEADLINE=$(( $(date +%s) + BUDGET ))
ENVH=$(uv pip freeze --python .venv/bin/python | shasum -a 256 | cut -c1-12)
echo "$(date -u +%FT%TZ) start freeze=$FREEZE head=$(git rev-parse --short HEAD) uv_pip_freeze=$ENVH deadline_epoch=$DEADLINE" >> "$LOG"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 \
  XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
# sh -c args: $1 log, $2 freeze7, $3 embed dir, $4 deadline, $5 model (xargs appends it). Slowest encoder first.
RC=0
printf '%s\n' qwen3-embedding-0.6b bge-large-en-v1.5 all-mpnet-base-v2 | xargs -P 2 -n 1 sh -c '
  T=$(( $4 - $(date +%s) + 600 )); [ "$T" -ge 60 ] || T=60
  /usr/bin/time -p /opt/homebrew/bin/timeout -k 30 "$T" .venv/bin/python -m integration.r3_heldout run --embed-dir "$3" --model "$5" --deadline "$4" > "integration/out/r3_heldout__$5.$2.stdout" 2>&1
  rc=$?; echo "$(date -u +%FT%TZ) job=$5 rc=$rc" >> "$1"; [ "$rc" -eq 0 ]' _ "$LOG" "$F7" "$EMBED" "$DEADLINE" || RC=$?
echo "$(date -u +%FT%TZ) done xargs_rc=$RC" >> "$LOG"
exit "$RC"
