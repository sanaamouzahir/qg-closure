#!/bin/bash
# apost_opt2_job.sh - OPTION-2 verdict ladder: rerun the a-posteriori M=16
# ladder (kf4, IC 837) with the rollout_ft_opt2_cond best ckpt (ep33,
# lambda=1e-3 annulus hinge) through the IDENTICAL code path as the
# 2026-07-09 4-arm table and the 07-10 rollsmoke. Truth refs REUSED.
# Success = 1.5e-2 stable at 16 steps AND 5e-3 final ratio >= pre-FT 0.72x.
# Pattern: apost_rollsmoke_job.sh (single-ckpt variant).
#
# Submit: qsub -q ibgpu.q -l gpu=1 -N apost_opt2 \
#   -o logs/apost_opt2.$JOB_ID.log -j y -cwd -V scripts/sge/apost_opt2_job.sh

#$ -S /bin/bash
#$ -cwd
#$ -V

set -e
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
WT=$QG_ROOT/qg-wiener-conditioning
OUT=$WT/diagnostics/Results/apost_opt2_20260711
LADDER=$WT/diagnostics/Results/apost_ladder_20260709
REFS15=$WT/diagnostics/Results/apost_smoke3/apost_refs_ladderrefs.npz
CKPT=data/ensemble_N5_7lag/training_runs/rollout_ft_opt2_cond/best.pt
ROOT=data/ensemble_N5_7lag/FRC-kf4
source $QG_ROOT/qg-env/bin/activate
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR=$QG_ROOT/.mplcache; mkdir -p "$MPLCONFIGDIR"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd $WT/training
mkdir -p "$OUT"
echo "[apost_opt2] host=$HOSTNAME date=$(date -u +%FT%TZ)"
python -u -c 'import sys, torch; ok = torch.cuda.is_available(); print("[gpu-check]", ok, flush=True); sys.exit(0 if ok else 2)'

run_case () {
    local tag=$1 sweep=$2 K=$3 refs=$4
    echo "===================== APOST_OPT2 $tag : sweep=$sweep K=$K ====================="
    python -u rollout_aposteriori.py \
        --root-dir "$ROOT/$sweep" \
        --ckpt "$CKPT" \
        --ic-index 837 --K "$K" --n-steps 16 --n-checkpoints 24 \
        --arms bare,closure --log-sigma \
        --device cuda --out-dir "$OUT" --tag "$tag" --load-refs "$refs"
}
run_case opt2_1p5em2 sweep_dT_1p5em2 1500 "$REFS15"
run_case opt2_1em2   sweep_dT_1em2   1000 "$LADDER/apost_refs_full_1em2.npz"
run_case opt2_5em3   sweep_dT_5em3    500 "$LADDER/apost_refs_full_5em3.npz"

echo "===================== CONSOLIDATE (one npz per case) ====================="
python -u $WT/diagnostics/consolidate_apost_cases.py --dir "$OUT" \
    --tags opt2_1p5em2,opt2_1em2,opt2_5em3 \
    --arm-labels closure=opt2_cond --delete-intermediates
echo "[apost_opt2] done $(date -u +%FT%TZ)"
