#!/bin/bash
#$ -N apw23spec
#$ -q ibgpu.q
#$ -l gpu=1
#$ -j y
#$ -o /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/$JOB_NAME.$JOB_ID.log
#$ -e /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/$JOB_NAME.$JOB_ID.err
#$ -cwd
#$ -m ea
#$ -M sanaamz@mit.edu
# apost_world23_job.sh -- Sanaa full wiener mandate 2026-07-09:
#  PART 1: per-shell spectral error profile eps(k) of predicted [Ndot,Nddot,
#          N3dot] for cond ep63 (frozen) + standing uncond control
#          (deriv7_filtered_floor0.1) + the rollout ckpt (deriv7_filtered_
#          lr5e-5), val samples across (member,dt), 512^2 members.
#  PART 3: the 2/3-WORLD rollout -- --world-mask-radius 2/3 replaces the
#          solver's sqrt2 radial mask for the WHOLE harness (RK4 truth
#          REGENERATED under the 2/3 mask -> NEW ref files, originals kept;
#          bare + analytic r3anal + both NN ckpts, full variant, 3 dT).
# Same member/IC/horizon as the sqrt2-world ladders (kf4, IC 837, M=16).
set -e
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
WT=$QG_ROOT/qg-wiener-conditioning
OUT=$WT/diagnostics/Results/apost_ladder_20260709_third23
SPECOUT=$WT/diagnostics/Results/spectral_error_profile_20260709
CKPT_U=data/ensemble_N5_7lag/training_runs/deriv7_filtered_lr5e-5/best.pt
CKPT_C=data/ensemble_N5_7lag/training_runs/deriv7_cond_local_v2/frozen_eval_20260709/best.pt
CKPT_CTRL=data/ensemble_N5_7lag/training_runs/deriv7_filtered_floor0.1/best.pt
ROOT=data/ensemble_N5_7lag/FRC-kf4
W23=0.6666666666666667
source $QG_ROOT/qg-env/bin/activate
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR=$QG_ROOT/.mplcache; mkdir -p "$MPLCONFIGDIR"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd $WT/training
mkdir -p "$OUT" "$SPECOUT"
echo "[w23] host=$HOSTNAME date=$(date -u +%FT%TZ)"
python -u -c 'import sys, torch; ok = torch.cuda.is_available(); print("[gpu-check]", ok, flush=True); sys.exit(0 if ok else 2)'

echo "===================== PART 1: spectral error profile ====================="
python -u $WT/diagnostics/spectral_error_profile.py \
    --ckpt cond=$CKPT_C --ckpt control=$CKPT_CTRL --ckpt rollout_unc=$CKPT_U \
    --roots data/ensemble_N5_7lag/FRC-b1/sweep_dT_* \
            data/ensemble_N5_7lag/FRC-b2/sweep_dT_* \
            data/ensemble_N5_7lag/FRC-kf4/sweep_dT_* \
            data/ensemble_N5_7lag/FRC-combo/sweep_dT_* \
            data/ensemble_N5_7lag/FRC-Re25k/sweep_dT_5em3 \
            data/ensemble_N5_7lag/FRC-Re25k/sweep_dT_1em2 \
    --n-samples 6 --device cuda \
    --out $SPECOUT/spectral_error_profile.npz

echo "===================== PART 3: the 2/3 world ====================="
run_w23 () {
    local tag=$1 sweep=$2 K=$3; shift 3
    echo "===================== W23 $tag : sweep=$sweep K=$K ====================="
    python -u rollout_aposteriori.py \
        --root-dir "$ROOT/$sweep" \
        --ckpt "$CKPT_U" --ckpt2 "$CKPT_C" \
        --ic-index 837 --K "$K" --n-steps 16 --n-checkpoints 24 \
        --arms bare,r3anal,closure --log-sigma \
        --world-mask-radius $W23 --save-refs \
        --device cuda --out-dir "$OUT" --tag "$tag" "$@"
}
run_w23 w23_1p5em2 sweep_dT_1p5em2 1500
run_w23 w23_1em2   sweep_dT_1em2   1000
run_w23 w23_5em3   sweep_dT_5em3    500

echo "===================== CONSOLIDATE (one npz per case) ====================="
python -u $WT/diagnostics/consolidate_apost_cases.py --dir "$OUT" \
    --tags w23_1p5em2,w23_1em2,w23_5em3 \
    --arm-labels closure=uncond,closure2=cond,r3anal=analytic --delete-intermediates
echo "[w23] done $(date -u +%FT%TZ)"
