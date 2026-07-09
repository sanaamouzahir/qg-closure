#!/bin/bash
#$ -N apmx0709
#$ -q ibgpu.q
#$ -l gpu=1
#$ -j y
#$ -o /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/$JOB_NAME.$JOB_ID.log
#$ -e /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/$JOB_NAME.$JOB_ID.err
#$ -cwd
#$ -m ea
#$ -M sanaamz@mit.edu
# apost_matrix_job.sh -- 2026-07-09 ladder-rerun matrix (Sanaa green light):
# 2 ckpts (UNCOND deriv7_filtered_lr5e-5 / COND deriv7_cond_local_v2 FROZEN
# epoch-63 copy) x 2 variants (full / --drop-nddot) x 3 Delta_T (1.5e-2 refs
# REUSED from apost_smoke3 ladderrefs; 1e-2 and 5e-3 refs generated here at
# h_fine = 1e-5). Member FRC-kf4 (beta=1) 512^2, IC packed row 837, M=16
# coarse steps, gamma=1, NO remediation -- a blowup IS the result.
# Both ckpts ride one invocation (--ckpt/--ckpt2 share bare arm + truth);
# consolidation splits them into one npz per case and deletes intermediates.
set -e
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
WT=$QG_ROOT/qg-wiener-conditioning
OUT=$WT/diagnostics/Results/apost_ladder_20260709
REFS15=$WT/diagnostics/Results/apost_smoke3/apost_refs_ladderrefs.npz
CKPT_U=data/ensemble_N5_7lag/training_runs/deriv7_filtered_lr5e-5/best.pt
CKPT_C=data/ensemble_N5_7lag/training_runs/deriv7_cond_local_v2/frozen_eval_20260709/best.pt
ROOT=data/ensemble_N5_7lag/FRC-kf4
source $QG_ROOT/qg-env/bin/activate
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR=$QG_ROOT/.mplcache; mkdir -p "$MPLCONFIGDIR"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd $WT/training
mkdir -p "$OUT"
echo "[matrix] host=$HOSTNAME date=$(date -u +%FT%TZ)"
python -u -c 'import sys, torch; ok = torch.cuda.is_available(); print("[gpu-check]", ok, flush=True); sys.exit(0 if ok else 2)'

run_case () {
    local tag=$1 sweep=$2 K=$3 refflag=$4; shift 4
    echo "===================== MATRIX $tag : sweep=$sweep K=$K $refflag $* ====================="
    python -u rollout_aposteriori.py \
        --root-dir "$ROOT/$sweep" \
        --ckpt "$CKPT_U" --ckpt2 "$CKPT_C" \
        --ic-index 837 --K "$K" --n-steps 16 --n-checkpoints 24 \
        --arms bare,closure --log-sigma \
        --device cuda --out-dir "$OUT" --tag "$tag" $refflag "$@"
}

# Delta_T = 1.5e-2 : reuse saved ladder refs (never regenerate)
run_case full_1p5em2 sweep_dT_1p5em2 1500 "--load-refs $REFS15"
run_case drop_1p5em2 sweep_dT_1p5em2 1500 "--load-refs $REFS15" --drop-nddot

# Delta_T = 1e-2 : generate refs once (K=1000 -> h_fine=1e-5), reuse for drop
run_case full_1em2 sweep_dT_1em2 1000 "--save-refs"
run_case drop_1em2 sweep_dT_1em2 1000 "--load-refs $OUT/apost_refs_full_1em2.npz" --drop-nddot

# Delta_T = 5e-3 : generate refs once (K=500 -> h_fine=1e-5), reuse for drop
run_case full_5em3 sweep_dT_5em3 500 "--save-refs"
run_case drop_5em3 sweep_dT_5em3 500 "--load-refs $OUT/apost_refs_full_5em3.npz" --drop-nddot

echo "===================== CONSOLIDATE (one npz per case) ====================="
python -u $WT/diagnostics/consolidate_apost_cases.py --dir "$OUT" \
    --tags full_1p5em2,drop_1p5em2,full_1em2,drop_1em2,full_5em3,drop_5em3 \
    --arm-labels closure=uncond,closure2=cond --delete-intermediates
echo "[matrix] done $(date -u +%FT%TZ)"
