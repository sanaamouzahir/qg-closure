#!/bin/bash
# train_ft_cheap_deriv.sh -- train the cheap N-derivative closure NN.
#
# 6 input channels:
#   omega_0, omega_m1, omega_m2,    (vorticity at t=0, -dT, -2dT)
#   psi_0,   psi_m1,   psi_m2       (streamfunction at same times)
#
# Architecture: CheapDerivClosureNet (model_deriv_closure.py)
#   - time-FD stencil   -> {f, f_dot, f_ddot} for omega and psi
#   - depthwise spatial-derivative kernels (init central diff)
#   - 9 Jacobian features J(psi^i, omega^j)
#   - 1x1 mixing conv    -> [Ndot, Nddot, N3dot]
#   - O(1e2) params; FLOPs far below one spectral Jacobian.
#
# Targets (multi-channel, the LOCAL N-derivatives; the L^k brackets are
# assembled spectrally at inference, never learned):
#   N_dot_0_anal, N_ddot_0_anal, N_3dot_0_anal   (R3 + R4)
#
# Loss: per-channel relative L2 (channels span orders of magnitude, so a flat
# loss would let the largest dominate). NO --normalize: the relative loss is the
# normalizer, and the dataset rejects --normalize for multi-channel targets.
#
# --dealias-pred: predictions are projected onto the 2/3 band (same cutoff as the
# solver/builder) BEFORE the loss, so the model is scored only on the resolved
# band -- not penalized for top-band aliasing it cannot represent (no FFT in the
# forward). REQUIRES a dealiased dataset; the deployed model stays FFT-free and
# the rollout still applies the (free) --dealias-nn mask.
set -e
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR="$QG_ROOT/qg-simple-package-stable/src/qg"
TRAINING_DIR="$QG_DIR/training"
# Defaults to the DEALIASED forced-turbulence set (the mv target of the held pack
# job). Override with e.g.
#   ROOT_DIR=.../decaying_turbulence_dT_1em3_fixD_v2_float64_dealiased ./train_ft_cheap_deriv.sh
ROOT_DIR="${ROOT_DIR:-$TRAINING_DIR/data/_4snap_staging/forced_turbulence_dT_1em3}"
if [ ! -d "$ROOT_DIR" ]; then
    echo "ERROR: $ROOT_DIR does not exist."
    exit 1
fi
if [ ! -f "$ROOT_DIR/manifest.json" ]; then
    echo "ERROR: $ROOT_DIR/manifest.json missing -- build incomplete?"
    exit 1
fi
RUN_NAME="cheap_deriv_6chan_$(date -u +%Y%m%d_%H%M%S)"
JOBNAME="train_ft_cheap_deriv"
"$TRAINING_DIR/train_v2.sh" \
    --jobname "$JOBNAME" \
    --root-dir "$ROOT_DIR" \
    --run-name "$RUN_NAME" \
    --model cheap_deriv \
    --input-fields \
        omega_0 omega_m1 omega_m2 omega_m3 psi_0 psi_m1 psi_m2 psi_m3 \
    --target-fields \
        N_dot_0_anal N_ddot_0_anal N_3dot_0_anal \
    --out-orders 3 \
    --refine-channels 0 \
    --loss rel_l2 \
    --dealias-pred \
    --batch-size 4 \
    --epochs 200 \
    --lr 5e-5 \
    --weight-decay 1e-4 \
    --lr-schedule cosine \
    --kernel 3 \
    --num-workers 8 \
    --print-every 1
   