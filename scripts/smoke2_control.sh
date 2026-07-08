#!/bin/bash
# smoke2_control.sh -- SMOKE-2a/2b control smokes (2026-07-08 work order):
#   2a: kf4 @ dT=1.5e-2 (in-distribution, truncation-dominated tier),
#       mid-filtered-train IC row 820 -- expect closure BEATS bare.
#   2b: FRC-b2 @ 5e-3 exactly as the original smoke BUT mid-filtered-train
#       IC row 964 (original used row 0, which the quiescent filter DROPS
#       from every split) -- IC-pathology falsification test.
# Run on a compute node (qrsh -q ibgpu.q -l gpu=1). Both with --diag.
set -e
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
source "$QG_ROOT/qg-env/bin/activate"
export MPLBACKEND=Agg
cd "$QG_ROOT/qg-wiener-conditioning/training"
CKPT=data/ensemble_N5_7lag/training_runs/deriv7_filtered_lr5e-5/best.pt
OUT="$QG_ROOT/qg-wiener-conditioning/diagnostics/Results/apost_smoke2"

echo "================ SMOKE-2a: kf4 @ 1.5e-2, ic 820 ================"
python -u rollout_aposteriori.py \
    --root-dir data/ensemble_N5_7lag/FRC-kf4/sweep_dT_1p5em2 \
    --ckpt "$CKPT" --ic-index 820 --K 20 --n-steps 16 --n-checkpoints 4 \
    --arms bare,r3only,closure --diag --device cuda \
    --tag smoke2a --out-dir "$OUT"

echo "================ SMOKE-2b: b2 @ 5e-3, ic 964 ================"
python -u rollout_aposteriori.py \
    --root-dir data/ensemble_N5_7lag/FRC-b2/sweep_dT_5em3 \
    --ckpt "$CKPT" --ic-index 964 --K 20 --n-steps 12 --n-checkpoints 4 \
    --arms bare,r3only,closure --diag --device cuda \
    --tag smoke2b --out-dir "$OUT"
