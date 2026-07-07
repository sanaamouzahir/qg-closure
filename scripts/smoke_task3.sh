#!/bin/bash
# smoke_task3.sh -- validates the three a-posteriori diagnostics (3a/3b/3c)
# end-to-end against the tiny rollout_apost smoke outputs.
# Run on a compute node (qrsh -q ibgpu.q -l gpu=1).
set -e
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
W=$QG_ROOT/qg-wiener-conditioning
source "$QG_ROOT/qg-env/bin/activate"
export MPLBACKEND=Agg
RES=$W/diagnostics/Results/apost_smoke

echo "== 3a accuracy =="
cd "$W/diagnostics"
python -u aposteriori_accuracy.py --npz "$RES/rollout_apost_smoke.npz" \
    --Lx 12.566370614359172 --out-dir "$RES"

echo "== 3c stability =="
python -u aposteriori_stability.py --npz "$RES/rollout_apost_smoke.npz" \
    --tag smoke --out-dir "$RES"

echo "== 3b walltime (tiny) =="
cd "$W/training"
python -u benchmark_walltime_closure.py \
    --root-dir data/ensemble_N5_7lag/FRC-b2/sweep_dT_5em3 \
    --ckpt data/ensemble_N5_7lag/training_runs/deriv7_filtered_lr5e-5/best.pt \
    --K 20 --bench-steps 8 --fine-bench-steps 40 --horizon-steps 12 \
    --accuracy-json "$RES/rollout_apost_smoke.json" \
    --tag smoke --out-dir "$RES" --device cuda

echo "== ALL 3 PASS =="
ls -la "$RES"
