#!/bin/bash
# submit_deriv7_cond.sh -- draft submitter for the cond_deriv primary run.
#
# SAFETY: prints the qsub command and EXITS by default (dry run). Pass --go to
# actually submit. Submission is ask-gated (Sanaa approval) -- do not --go
# without it.
#
# Pool: FRC-* sweeps MINUS FRC-Re25k/sweep_dT_1p5em2 (past-wall, unlearnable per
# Prop 2; approved drop). 17 roots. Config identical to the control
# (deriv7_filtered_floor0.1) except --model cond_deriv, for a clean comparison.
set -euo pipefail

WORKTREE=/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning
cd "$WORKTREE/training"

# Build the root list relative to training/ (resolves at job runtime via the
# data symlink). NOTE: the data/ symlink must resolve identically on the submit
# host and the compute node (it points at the shared package-stable ensemble).
# Exclude the approved-drop tier.
ROOTS=()
for r in data/ensemble_N5_7lag/FRC-*/sweep_dT_*; do
  [[ "$r" == "data/ensemble_N5_7lag/FRC-Re25k/sweep_dT_1p5em2" ]] && continue
  [[ -e "$r/packed/inputs.npy" ]] || continue
  ROOTS+=("$r")
done
echo "[submit_deriv7_cond] ${#ROOTS[@]} sweep roots (expect 17):"
printf '   %s\n' "${ROOTS[@]}"
# hard guard: refuse to build the qsub on an unexpected pool (missing packed
# dir, unresolved symlink, or an accidental extra/dropped tier).
if [[ ${#ROOTS[@]} -ne 17 ]]; then
  echo "[submit_deriv7_cond] ERROR: expected 17 roots, got ${#ROOTS[@]}." >&2
  exit 2
fi

QSUB=(qsub -q ibgpu.q -l gpu=1 -N deriv7_cond -j y
  -m ea -M "${QG_NOTIFY_EMAIL:-sanaamz@mit.edu}"
  "$WORKTREE/scripts/sge/train_deriv_cond_job.sh"
  --model cond_deriv
  --sweep-roots "${ROOTS[@]}"
  --n-snapshots 7 --out-orders 3
  --epochs 300 --lr 5.0e-5 --batch-size 4 --compute-dtype float64
  --rel-floor 0.1
  --run-name deriv7_cond)

echo
echo "[submit_deriv7_cond] qsub command:"
printf '%q ' "${QSUB[@]}"; echo

if [[ "${1:-}" == "--go" ]]; then
  echo "[submit_deriv7_cond] SUBMITTING (--go given)..."
  "${QSUB[@]}"
else
  echo "[submit_deriv7_cond] DRY RUN (pass --go to submit)."
fi
