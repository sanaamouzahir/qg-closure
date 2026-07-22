#!/bin/bash
#$ -S /bin/bash
#$ -q all.q
#$ -j y
#$ -cwd
set -uo pipefail
QG=/gdata/projects/ml_scope/Closure_modeling/QG-closure
W=$QG/qg-wiener-conditioning
D=$QG/qg-simple-package-stable/src/qg/training/data/ensemble_N5_7lag/training_runs
CK=$D/rollout_ft_w31p3_certv2/best.pt
R=data/ensemble_N5_7lag/FRC-kf4/sweep_dT_5em3
OUT=$W/diagnostics/Results/wiener_amp_20260721
source $QG/qg-env/bin/activate
cd $W/training
mkdir -p $OUT
echo "############ UNIT-CHECK GATE ############"
python -u ../diagnostics/nn_amplification.py --ckpt $CK --root-dir $R --ic-index 912 --Delta-T 5.0e-3 --unit-checks-only --device cpu || { echo "UNIT CHECKS FAILED -- no rho reported"; exit 3; }
echo "############ TEST B/C: rho at truth + NN-augmented states ############"
python -u ../diagnostics/nn_amplification.py --ckpt $CK --root-dir $R --ic-index 912 --Delta-T 5.0e-3 --dev-steps 0,10,20,30 --device cpu --out $OUT/nnamp_kf4_912 || echo "BC_FAIL"
echo "############ nddot-depth probe: 7 (baseline) vs 4 ############"
for K in 7 4; do
  echo "=== nddot-depth $K ==="
  python -u rollout_aposteriori.py --root-dir $R --ckpt $CK --ic-index 912 --Delta-T 5.0e-3 --arms bare,r3anal,closure --no-truth --nddot-depth $K --scalars-every 1 --device cpu --tag d$K --out-dir $OUT/nddot || echo "NDDOT_FAIL_$K"
done
echo "[wiener_amp] done"
