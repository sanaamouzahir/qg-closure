#!/bin/bash
# train_decay_fixD_v2.sh -- train the Fix D v2 closure NN.
#
# 6 input channels:
#   omega_0, omega_m1, omega_m2,    (vorticity at t=0, -dT, -2dT)
#   psi_0,   psi_m1,   psi_m2       (streamfunction at same times)
#
# All saved directly to npz by build_training_data_fixD_v2.py.
#
# Architecture: BilinearClosureNet (model_fixD.BilinearClosureNet)
#   - 5 conv blocks, ~370K params
#   - Receptive field: 7 pixels
#   - 1 GLU activation for bilinear Jacobian products
#   - Circular padding throughout
#
# Rationale: with psi given as input at all 3 time levels, the network
# never has to learn the global inverse-Laplacian operator. All operations
# in the closure formula are local: gradients (3x3), bilinear products
# (GLU), and one Laplacian (3x3). No UNet hierarchy needed.
#
# Target: f_NN_target = (1/12) * [L*Ndot - 5*Nddot]   (Fix D's target convention)

set -e

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR="$QG_ROOT/qg-simple-package-stable/src/qg"
TRAINING_DIR="$QG_DIR/training"

ROOT_DIR="${ROOT_DIR:-$TRAINING_DIR/data/decaying_turbulence_dT_1em3_fixD_v2_float64_dealiased}"
if [ ! -d "$ROOT_DIR" ]; then
    echo "ERROR: $ROOT_DIR does not exist."
    echo "Build it first:  ./submit_build_fixD_v2.sh"
    exit 1
fi
if [ ! -f "$ROOT_DIR/manifest.json" ]; then
    echo "ERROR: $ROOT_DIR/manifest.json missing -- build incomplete?"
    exit 1
fi

RUN_NAME="fixD_v2_6chan_$(date -u +%Y%m%d_%H%M%S)"
JOBNAME="train_decay_fixD_v2"

# Delete any stale norm_stats (different channel count = different shape)
if [ -f "$ROOT_DIR/norm_stats.npz" ]; then
    n_in=$(python3 -c "
import numpy as np
try:
    s = np.load('$ROOT_DIR/norm_stats.npz')
    print(len(s['input_mean']))
except Exception:
    print(0)
" 2>/dev/null || echo 0)
    if [ "$n_in" != "6" ]; then
        echo "[train_decay_fixD_v2] removing stale norm_stats.npz (had $n_in channels, need 6)"
        rm -f "$ROOT_DIR/norm_stats.npz"
    fi
fi

"$TRAINING_DIR/train_v2.sh" \
    --jobname "$JOBNAME" \
    --root-dir "$ROOT_DIR" \
    --run-name "$RUN_NAME" \
    --model bilinear_closure \
    --input-fields \
        omega_0 omega_m1 omega_m2 \
        psi_0   psi_m1   psi_m2 \
    --target-field f_NN_target \
    --batch-size 4 \
    --epochs 200 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --lr-schedule cosine \
    --hidden-channels 64 \
    --kernel 3 \
    --normalize \
    --num-workers 8 \
    --print-every 1
