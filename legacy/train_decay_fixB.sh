#!/bin/bash
# train_decay_fixB.sh - retrain on EXISTING decay v2 dataset with the 6-channel
# input (Fix B). No source change, no dataset rebuild.
#
# Inputs: omega_0 psi_0 grad_psi_sq omega_x omega_y omega_m1
# Target: f_NN_target (default)
# Model:  unet, base_channels=32 (matches the original failed run)
#
# Run AFTER you have the by_time dataset built:
#   $SCRIPT_DIR/data/decaying_turbulence_dT_1em3_bytime/
# (or pass --root-dir to point at whatever dataset you want).
#
# Usage:
#   ./train_decay_fixB.sh                            # SGE GPU submission
#   ./train_decay_fixB.sh --interactive              # local
#   ./train_decay_fixB.sh --root-dir /path/to/data
#   ./train_decay_fixB.sh --epochs 100 --lr 1e-4     # forwarded to train.py

set -e

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR="$QG_ROOT/qg-simple-package-stable/src/qg"
SCRIPT_DIR="$QG_DIR/training"

DATASET_ROOT="$SCRIPT_DIR/data/decaying_turbulence_dT_1em3_bytime"
RUN_NAME="fixB_6chan_$(date -u +%Y%m%d_%H%M%S)"
JOBNAME="train_decay_fixB"
INTERACTIVE=0
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --interactive)  INTERACTIVE=1; shift ;;
        --root-dir)     DATASET_ROOT="$2"; shift 2 ;;
        --jobname)      JOBNAME="$2"; shift 2 ;;
        --run-name)     RUN_NAME="$2"; shift 2 ;;
        *)              EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [ ! -d "$DATASET_ROOT" ]; then
    echo "ERROR: dataset root not found: $DATASET_ROOT"
    exit 1
fi

TRAIN_ARGS=(
    --root-dir "$DATASET_ROOT"
    --run-name "$RUN_NAME"
    --model unet
    --input-fields omega_0 psi_0 grad_psi_sq omega_x omega_y omega_m1
    --target-field f_NN_target
    --batch-size 4
    --epochs 200
    --lr 3e-4
    --weight-decay 1e-4
    --lr-schedule cosine
    --base-channels 32
    --kernel 3
    --normalize
    --num-workers 2
    --print-every 1
    "${EXTRA_ARGS[@]}"
)

echo "==================================================================="
echo " train decay (Fix B: 6 input channels)                             "
echo "==================================================================="
echo "  dataset    : $DATASET_ROOT"
echo "  run name   : $RUN_NAME"
echo "  inputs     : omega_0 psi_0 grad_psi_sq omega_x omega_y omega_m1"
echo "  target     : f_NN_target"
echo "  jobname    : $JOBNAME"
echo "==================================================================="
echo

TRAIN_SH="$SCRIPT_DIR/train.sh"
if [ ! -f "$TRAIN_SH" ]; then
    echo "ERROR: train.sh not found at $TRAIN_SH"
    exit 1
fi

CMD=("$TRAIN_SH" --jobname "$JOBNAME")
if [ "$INTERACTIVE" -eq 1 ]; then
    CMD+=(--interactive)
fi
CMD+=("${TRAIN_ARGS[@]}")

echo "Running:"
printf "  %s\n" "${CMD[@]}"
echo
"${CMD[@]}"
