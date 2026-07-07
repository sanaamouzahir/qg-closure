#!/bin/bash
# smoke_rollout_apost.sh -- tiny-horizon end-to-end smoke of rollout_aposteriori.py.
# Run on a compute node (qrsh -q ibgpu.q -l gpu=1). Uses the existing
# deriv7_filtered_lr5e-5 checkpoint; 12 coarse steps, K=20 truth.
set -e
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
source "$QG_ROOT/qg-env/bin/activate"
export MPLBACKEND=Agg
cd "$QG_ROOT/qg-wiener-conditioning/training"
exec python -u rollout_aposteriori.py \
    --root-dir data/ensemble_N5_7lag/FRC-b2/sweep_dT_5em3 \
    --ckpt data/ensemble_N5_7lag/training_runs/deriv7_filtered_lr5e-5/best.pt \
    --ic-index 0 --K 20 --n-steps 12 --n-checkpoints 4 \
    --arms bare,r3only,closure --device cuda \
    --tag smoke --out-dir "$QG_ROOT/qg-wiener-conditioning/diagnostics/Results/apost_smoke" "$@"
