#!/bin/bash
# smoke2_val_ic.sh -- leakage check (physics-sanity flag): repeat SMOKE-2a/2b
# from VAL-split rows (kf4@1.5e-2 row 837; b2@5e-3 row 934). If the early
# closure gain matches the train-row runs, the 4-8x gain is not memorization.
# Run on a compute node (qrsh -q ibgpu.q -l gpu=1).
set -e
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
source "$QG_ROOT/qg-env/bin/activate"
export MPLBACKEND=Agg
cd "$QG_ROOT/qg-wiener-conditioning/training"
CKPT=data/ensemble_N5_7lag/training_runs/deriv7_filtered_lr5e-5/best.pt
OUT="$QG_ROOT/qg-wiener-conditioning/diagnostics/Results/apost_smoke2"

echo "================ SMOKE-2a-val: kf4 @ 1.5e-2, VAL ic 837 ================"
python -u rollout_aposteriori.py \
    --root-dir data/ensemble_N5_7lag/FRC-kf4/sweep_dT_1p5em2 \
    --ckpt "$CKPT" --ic-index 837 --K 20 --n-steps 16 --n-checkpoints 4 \
    --arms bare,closure --device cuda \
    --tag smoke2a_val --out-dir "$OUT"

echo "================ SMOKE-2b-val: b2 @ 5e-3, VAL ic 934 ================"
python -u rollout_aposteriori.py \
    --root-dir data/ensemble_N5_7lag/FRC-b2/sweep_dT_5em3 \
    --ckpt "$CKPT" --ic-index 934 --K 20 --n-steps 12 --n-checkpoints 4 \
    --arms bare,closure --device cuda \
    --tag smoke2b_val --out-dir "$OUT"
