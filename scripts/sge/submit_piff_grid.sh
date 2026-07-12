#!/bin/bash
# submit_piff_grid.sh - the 6-run Pi_FF hyperparameter grid (ML SPEC 01 S3.2):
# lr in {1.0e-4, 3.0e-4, 1.0e-3} x weight_decay in {1.0e-5, 1.0e-4} on FPC-const
# s=4; one GPU job each (-q ibgpu.q -l gpu=1 ONLY — the single permitted pair).
#
# GATE: per spec S6 this fires ONLY after the [QG][GATE-ML][SGS-CLOSURE] test
# email (T1-T7) is approved by Sanaa. DRY-RUN by default; pass --go to submit:
#   ./submit_piff_grid.sh          # print the qsub commands
#   ./submit_piff_grid.sh --go     # fire
#
# Model selection downstream: lowest val NLL (spec S3.3); eval_piff.py on the
# winner's best.pt produces the S4 milestone package. Job ids are collected in
# JOB_IDS so a follow-up `qsub -hold_jid <comma-list>` selection/eval job can
# chain without refactoring.
#
# I18 NOTE (deliberate, not an oversight): the three-job monitor unit
# (LIVE + FINALIZE per trainer) is wired by the submitting supervisor at
# grid-fire time, post [QG][GATE-ML] approval — this builder script carries
# the trainers only. Carry all job ids in the [QG][SUBMIT] email.

set -e

GO=0
[ "${1:-}" = "--go" ] && GO=1

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
BRANCH="$QG_ROOT/qg-sgs-closure"
SGE="$BRANCH/scripts/sge"
LOG_DIR="$BRANCH/logs"
QG_NOTIFY_EMAIL="${QG_NOTIFY_EMAIL:-sanaamz@mit.edu}"
mkdir -p "$LOG_DIR"

# hard-require the Step-0 artifacts + the branch venv before queueing anything
RUN0="$QG_ROOT/qg-simple-package-stable/src/qg/outputs/SGS_closure_ensemble/FPC-const"
[ -f "$RUN0/DATASET_MANIFEST.md" ] || { echo "ERROR: $RUN0/DATASET_MANIFEST.md missing (Step 0)"; exit 1; }
[ -f "$RUN0/DNS_LES_s4.npz" ] || { echo "ERROR: $RUN0/DNS_LES_s4.npz missing (Step 0)"; exit 1; }
[ -f "$QG_ROOT/qg-env-piff/bin/activate" ] || { echo "ERROR: qg-env-piff venv missing"; exit 1; }

if [ "$GO" = "1" ]; then
    run() { "$@"; }
    echo "=== Pi_FF 6-run grid: SUBMITTING ==="
else
    run() { echo "[DRY-RUN] $*"; }
    echo "=== Pi_FF 6-run grid: DRY-RUN (pass --go to submit) ==="
fi

# YAML/CLI floats in explicit-mantissa form per repo rule
JOB_IDS=()
for LR in 1.0e-4 3.0e-4 1.0e-3; do
    for WD in 1.0e-5 1.0e-4; do
        TAG="lr${LR}_wd${WD}"
        NAME="grid_${TAG}"
        # job name pF_<lr-exp><wd-exp> stays unique in qstat's 10 chars
        JN="pF_${LR/1.0e-/1e}_${WD/1.0e-/1e}"
        JN="${JN/3.0e-/3e}"
        OUT=$(run qsub -N "$JN" -q ibgpu.q -l gpu=1 \
            -o "$LOG_DIR/\$JOB_NAME.\$JOB_ID.log" -j y -cwd -V \
            -m ea -M "$QG_NOTIFY_EMAIL" \
            "$SGE/piff_train_job.sh" \
            --run-name "$NAME" --lr "$LR" --weight-decay "$WD")
        echo "$OUT"
        JID=$(echo "$OUT" | grep -oE '[0-9]+' | head -1)
        [ -n "$JID" ] && [ "$GO" = "1" ] && JOB_IDS+=("$JID")
    done
done

[ "${#JOB_IDS[@]}" -gt 0 ] && echo "JOB_IDS: $(IFS=,; echo "${JOB_IDS[*]}")  (use for -hold_jid selection/eval chaining)"
echo "=== grid done; monitor: qstat -u \$USER ==="
echo "  artifacts land in $BRANCH/ml_closure/runs_piff/grid_lr*_wd*/"
