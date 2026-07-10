#!/bin/bash
# apost_rollsmoke_job.sh - the approved success metric for the rollout-loss
# smokes: rerun the a-posteriori ladder (kf4, IC 837, 16 steps) with the two
# smoke checkpoints (rollout_ft_phys / rollout_ft_cond) through the IDENTICAL
# code path as the 2026-07-09 4-arm table. Truth refs REUSED (no RK4
# recompute). Pattern: apost_matrix_job.sh.
#
# Submit: qsub -q ibgpu.q -l gpu=1 -N apost_rlsm \
#   -o logs/apost_rlsm.$JOB_ID.log -j y -cwd -V scripts/sge/apost_rollsmoke_job.sh

#$ -S /bin/bash
#$ -cwd
#$ -V

set -e
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
WT=$QG_ROOT/qg-wiener-conditioning
OUT=$WT/diagnostics/Results/apost_rollout_smoke
LADDER=$WT/diagnostics/Results/apost_ladder_20260709
REFS15=$WT/diagnostics/Results/apost_smoke3/apost_refs_ladderrefs.npz
CKPT_P=data/ensemble_N5_7lag/training_runs/rollout_ft_phys/best.pt
CKPT_C=data/ensemble_N5_7lag/training_runs/rollout_ft_cond/best.pt
ROOT=data/ensemble_N5_7lag/FRC-kf4
source $QG_ROOT/qg-env/bin/activate
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR=$QG_ROOT/.mplcache; mkdir -p "$MPLCONFIGDIR"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd $WT/training
mkdir -p "$OUT"
echo "[rollsmoke] host=$HOSTNAME date=$(date -u +%FT%TZ)"
python -u -c 'import sys, torch; ok = torch.cuda.is_available(); print("[gpu-check]", ok, flush=True); sys.exit(0 if ok else 2)'

run_case () {
    local tag=$1 sweep=$2 K=$3 refs=$4
    echo "===================== ROLLSMOKE $tag : sweep=$sweep K=$K ====================="
    python -u rollout_aposteriori.py \
        --root-dir "$ROOT/$sweep" \
        --ckpt "$CKPT_P" --ckpt2 "$CKPT_C" \
        --ic-index 837 --K "$K" --n-steps 16 --n-checkpoints 24 \
        --arms bare,closure --log-sigma \
        --device cuda --out-dir "$OUT" --tag "$tag" --load-refs "$refs"
}
run_case full_1p5em2 sweep_dT_1p5em2 1500 "$REFS15"
run_case full_1em2   sweep_dT_1em2   1000 "$LADDER/apost_refs_full_1em2.npz"
run_case full_5em3   sweep_dT_5em3    500 "$LADDER/apost_refs_full_5em3.npz"

echo "===================== CONSOLIDATE (one npz per case) ====================="
python -u $WT/diagnostics/consolidate_apost_cases.py --dir "$OUT" \
    --tags full_1p5em2,full_1em2,full_5em3 \
    --arm-labels closure=rollout_phys,closure2=rollout_cond --delete-intermediates
echo "[rollsmoke] done $(date -u +%FT%TZ)"
