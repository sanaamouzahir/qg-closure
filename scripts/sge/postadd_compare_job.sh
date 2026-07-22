#!/bin/bash
#$ -S /bin/bash
#$ -q all.q
#$ -j y
#$ -cwd
set -uo pipefail
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
W=$QG_ROOT/qg-wiener-conditioning
D=$QG_ROOT/qg-simple-package-stable/src/qg/training/data/ensemble_N5_7lag/training_runs
OUT=$W/diagnostics/Results/apost_postadd_20260720
REP=$W/diagnostics/Results/apost_opt2_rep_20260711/FRC-kf4_ic912
CK=$D/rollout_ft_w31p3_certv2/best.pt
source $QG_ROOT/qg-env/bin/activate
cd $W/training
mkdir -p $OUT
for MODE in folded postadd; do
  echo "=== closure-apply=$MODE ==="
  python -u rollout_aposteriori.py \
    --root-dir data/ensemble_N5_7lag/FRC-kf4/sweep_dT_5em3 \
    --ckpt $CK --ic-index 912 --K 500 --n-steps 64 --n-checkpoints 24 \
    --arms bare,r3anal,closure --closure-apply $MODE --device cpu \
    --nn-float64 --out-dir $OUT --tag cmp_$MODE \
    --load-refs $REP/apost_refs_ic912_5em3_h64.npz || echo "RC_FAIL_$MODE"
done
echo "[postadd] done"
