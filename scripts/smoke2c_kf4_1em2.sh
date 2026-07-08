#!/bin/bash
# smoke2c_kf4_1em2.sh -- disambiguation probe: is the closure-arm blowup
# tier-specific (CFL/dT) or generic? kf4 @ dT=1e-2 (in-pool), mid IC row 820.
# Run on a compute node (qrsh -q ibgpu.q -l gpu=1).
set -e
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
source "$QG_ROOT/qg-env/bin/activate"
export MPLBACKEND=Agg
cd "$QG_ROOT/qg-wiener-conditioning/training"
exec python -u rollout_aposteriori.py \
    --root-dir data/ensemble_N5_7lag/FRC-kf4/sweep_dT_1em2 \
    --ckpt data/ensemble_N5_7lag/training_runs/deriv7_filtered_lr5e-5/best.pt \
    --ic-index 820 --K 20 --n-steps 16 --n-checkpoints 4 \
    --arms bare,closure --diag --device cuda \
    --tag smoke2c --out-dir "$QG_ROOT/qg-wiener-conditioning/diagnostics/Results/apost_smoke2"
