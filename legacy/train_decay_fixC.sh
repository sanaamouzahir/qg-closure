#!/bin/bash
# train_decay_fixC.sh - retrain on the REBUILT richN dataset with 8 input
# channels including precomputed N_0 and N_dot_0_anal.
#
# Inputs: omega_0 psi_0 grad_psi_sq omega_x omega_y omega_m1 N_0 N_dot_0_anal
# Target: f_NN_target (default)
# Model:  unet, base_channels=32
#
# REQUIRES: the richN dataset built by build_training_data_decay_richN.sh,
# which depends on the patched build_training_data.py and dataset.py.
#
# Usage:
#   ./train_decay_fixC.sh                            # SGE GPU submission
#   ./train_decay_fixC.sh --interactive              # local
#   ./train_decay_fixC.sh --root-dir /path/to/data
#   ./train_decay_fixC.sh --epochs 100 --lr 1e-4     # forwarded to train.py

set -e

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR="$QG_ROOT/qg-simple-package-stable/src/qg"
SCRIPT_DIR="$QG_DIR/training"

DATASET_ROOT="$SCRIPT_DIR/data/decaying_turbulence_dT_1em3_richN"
RUN_NAME="fixC_8chan_$(date -u +%Y%m%d_%H%M%S)"
JOBNAME="train_decay_fixC"
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
    echo "       run build_training_data_decay_richN.sh first."
    exit 1
fi

# Sanity-check that the dataset actually has N_0 fields (i.e. was built with
# the patched build script). Read one sample and check keys.
SANITY_CHECK=$(python -c "
import numpy as np
from pathlib import Path
samples = sorted(Path('$DATASET_ROOT/samples').glob('sample_*.npz'))
if not samples:
    print('NO_SAMPLES'); raise SystemExit(0)
keys = sorted(np.load(samples[0]).files)
need = {'N_0', 'N_dot_0_anal'}
missing = need - set(keys)
print('OK' if not missing else f'MISSING:{missing}')
")
if [ "$SANITY_CHECK" != "OK" ]; then
    echo "ERROR: dataset $DATASET_ROOT is missing N_0/N_dot_0_anal fields."
    echo "       Result: $SANITY_CHECK"
    echo "       This dataset was NOT built with the patched build_training_data.py."
    exit 1
fi
echo "Dataset sanity check: OK (N_0 + N_dot_0_anal present)"

TRAIN_ARGS=(
    --root-dir "$DATASET_ROOT"
    --run-name "$RUN_NAME"
    --model unet
    --input-fields omega_0 psi_0 grad_psi_sq omega_x omega_y omega_m1 N_0 N_dot_0_anal
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
echo " train decay (Fix C: 8 input channels including N_0, N_dot_0_anal) "
echo "==================================================================="
echo "  dataset    : $DATASET_ROOT"
echo "  run name   : $RUN_NAME"
echo "  inputs     : omega_0 psi_0 grad_psi_sq omega_x omega_y omega_m1 N_0 N_dot_0_anal"
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
