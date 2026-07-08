#!/bin/bash
# train.sh - Submit train_v2.py to SGE on a GPU node, or run interactively.
#
# Usage:
#   ./train.sh --root-dir <path>                    # SGE GPU submission
#   ./train.sh --root-dir <path> --interactive       # local
#
# Common training flags forwarded:
#   --model {cnn,unet}    --input-fields omega_0 psi_0
#   --batch-size 4        --epochs 200            --lr 3e-4
#   --normalize           --num-workers 2          --base-channels 32
set -e
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR="$QG_ROOT/qg-simple-package-stable/src/qg"
SCRIPT_DIR="$QG_DIR/training"
INTERACTIVE=0
JOBNAME="train_closure_v2"
EXTRA_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --interactive)  INTERACTIVE=1; shift ;;
        --jobname)      JOBNAME="$2"; shift 2 ;;
        -h|--help)      sed -n '2,15p' "$0"; exit 0 ;;
        *)              EXTRA_ARGS+=("$1"); shift ;;
    esac
done
LOG_DIR="$QG_ROOT/qg-wiener-conditioning/logs"
mkdir -p "$LOG_DIR"
PYTHON_ARGS=("${EXTRA_ARGS[@]}")
if [ "$INTERACTIVE" -eq 1 ]; then
    source "$QG_ROOT/qg-env/bin/activate"
    cd "$SCRIPT_DIR"
    python -u train.py "${PYTHON_ARGS[@]}"
    exit 0
fi
JOB_LOG="$LOG_DIR/${JOBNAME}.log"
JOB_SCRIPT="$SCRIPT_DIR/train_v2_job.sh"
chmod +x "$JOB_SCRIPT" 2>/dev/null || true
# IMPORTANT: do NOT use -cwd here. We want the job's cwd to be SCRIPT_DIR
# (the training/ directory), not wherever the user happened to launch qsub
# from. The job script also `cd`s into SCRIPT_DIR as belt-and-suspenders.
QSUB_FLAGS=(
    -N "$JOBNAME"
    -o "$JOB_LOG"
    -e "$JOB_LOG"
    -j y -V
    -wd "$SCRIPT_DIR"
    -q "ibgpu.q"
    -l "gpu=1"
    -m ea
    -M "${QG_NOTIFY_EMAIL:-sanaamz@mit.edu}"
)
echo "submitting train_v2.py ($JOBNAME) -> $JOB_LOG"
echo "  workdir: $SCRIPT_DIR"
qsub "${QSUB_FLAGS[@]}" "$JOB_SCRIPT" "${PYTHON_ARGS[@]}"
echo
echo "watch: tail -f $JOB_LOG"