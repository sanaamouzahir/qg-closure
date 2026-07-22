#!/bin/bash
#$ -S /bin/bash
#$ -q all.q
#$ -j y
#$ -cwd
set -uo pipefail
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
W=$QG_ROOT/qg-wiener-conditioning
D=$QG_ROOT/qg-simple-package-stable/src/qg/training/data/ensemble_N5_7lag/training_runs
OUT=$W/diagnostics/Results/apost_timefd_20260720
REP=$W/diagnostics/Results/apost_opt2_rep_20260711/FRC-kf4_ic912
CK=$D/rollout_ft_w31p3_certv2/best.pt
source $QG_ROOT/qg-env/bin/activate
cd $W/training
mkdir -p $OUT
echo "############ TEST B: drop Nddot (the 1/dt^2-amplified term) ############"
echo "# hypothesis: N-ddot time-FD amplifies self-history error by ~1/dt^2 (4e4 at dt=5e-3)."
echo "# if that is the loop, dropping Nddot should delay or remove the blowup."
python -u rollout_aposteriori.py --root-dir data/ensemble_N5_7lag/FRC-kf4/sweep_dT_5em3 \
  --ckpt $CK --ic-index 912 --K 500 --n-steps 64 --n-checkpoints 24 \
  --arms bare,closure --drop-nddot --device cpu --nn-float64 \
  --out-dir $OUT --tag dropNddot \
  --load-refs $REP/apost_refs_ic912_5em3_h64.npz || echo RC_FAIL_dropNddot
echo "############ TEST A: dT sweep, stability only (--no-truth) ############"
echo "# 1/dt^k amplification WEAKENS as dt grows -> under the time-FD hypothesis"
echo "# the blowup STEP should come LATER at larger dT (opposite of a physics"
echo "# instability, which worsens with dT)."
for R in sweep_dT_5em3:5.0e-3 sweep_dT_1em2:1.0e-2 sweep_dT_1p5em2:1.5e-2; do
  SR=${R%%:*}; DT=${R##*:}
  RD=data/ensemble_N5_7lag/FRC-kf4/$SR
  [ -d "$RD" ] || { echo "SKIP $SR (missing)"; continue; }
  echo "=== dT=$DT ($SR) ==="
  python -u rollout_aposteriori.py --root-dir $RD \
    --ckpt $CK --ic-index 912 --K 500 --n-steps 64 --n-checkpoints 8 \
    --arms bare,closure --no-truth --device cpu --nn-float64 \
    --out-dir $OUT --tag dtsweep_${SR} || echo RC_FAIL_$SR
done
echo "[timefd] done"
