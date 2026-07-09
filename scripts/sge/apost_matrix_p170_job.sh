#!/bin/bash
#$ -N apmxp170
#$ -q ibgpu.q
#$ -l gpu=1
#$ -j y
#$ -o /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/$JOB_NAME.$JOB_ID.log
#$ -e /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/$JOB_NAME.$JOB_ID.err
#$ -cwd
#$ -m ea
#$ -M sanaamz@mit.edu
# apost_matrix_p170_job.sh -- 2026-07-09 follow-up (Sanaa ruling: solver mask
# untouched; remediate on the correction alone + analytic-closure question):
#   (A) NN matrix rerun WITH --nn-project-radius (alias-safe 2/3 -> mode
#       ~170.67 at 512^2): 2 ckpts x {full, --drop-nddot} x 3 Delta_T.
#   (B) ANALYTIC closure arms (r3anal, exact chain-rule derivs, no NN) at
#       each Delta_T, standard dealias (ceiling reference) AND with the
#       projection (annulus-isolation).
# Same member/IC/horizon as apost_ladder_20260709; ALL truth refs REUSED.
# COND ckpt = frozen_eval_20260709 (epoch 63; job 1827306 best.pt unchanged
# since the freeze -- best_val plateau).
set -e
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
WT=$QG_ROOT/qg-wiener-conditioning
OUT=$WT/diagnostics/Results/apost_ladder_20260709_p170
REFS15=$WT/diagnostics/Results/apost_smoke3/apost_refs_ladderrefs.npz
REFS10=$WT/diagnostics/Results/apost_ladder_20260709/apost_refs_full_1em2.npz
REFS05=$WT/diagnostics/Results/apost_ladder_20260709/apost_refs_full_5em3.npz
CKPT_U=data/ensemble_N5_7lag/training_runs/deriv7_filtered_lr5e-5/best.pt
CKPT_C=data/ensemble_N5_7lag/training_runs/deriv7_cond_local_v2/frozen_eval_20260709/best.pt
ROOT=data/ensemble_N5_7lag/FRC-kf4
source $QG_ROOT/qg-env/bin/activate
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR=$QG_ROOT/.mplcache; mkdir -p "$MPLCONFIGDIR"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd $WT/training
mkdir -p "$OUT"
echo "[matrix-p170] host=$HOSTNAME date=$(date -u +%FT%TZ)"
python -u -c 'import sys, torch; ok = torch.cuda.is_available(); print("[gpu-check]", ok, flush=True); sys.exit(0 if ok else 2)'

run_nn () {   # NN matrix leg: both ckpts, alias-safe projection ON
    local tag=$1 sweep=$2 K=$3 refs=$4; shift 4
    echo "===================== P170-NN $tag : sweep=$sweep K=$K $* ====================="
    python -u rollout_aposteriori.py \
        --root-dir "$ROOT/$sweep" \
        --ckpt "$CKPT_U" --ckpt2 "$CKPT_C" \
        --ic-index 837 --K "$K" --n-steps 16 --n-checkpoints 24 \
        --arms bare,closure --log-sigma --nn-project-radius \
        --device cuda --out-dir "$OUT" --tag "$tag" --load-refs "$refs" "$@"
}
run_anal () { # analytic leg: r3anal (exact derivs, no NN); model loaded, unused
    local tag=$1 sweep=$2 K=$3 refs=$4; shift 4
    echo "===================== P170-ANALYTIC $tag : sweep=$sweep K=$K $* ====================="
    python -u rollout_aposteriori.py \
        --root-dir "$ROOT/$sweep" \
        --ckpt "$CKPT_U" \
        --ic-index 837 --K "$K" --n-steps 16 --n-checkpoints 24 \
        --arms bare,r3anal --log-sigma \
        --device cuda --out-dir "$OUT" --tag "$tag" --load-refs "$refs" "$@"
}

# (A) NN matrix with alias-safe projection
run_nn pf_1p5em2 sweep_dT_1p5em2 1500 "$REFS15"
run_nn pd_1p5em2 sweep_dT_1p5em2 1500 "$REFS15" --drop-nddot
run_nn pf_1em2   sweep_dT_1em2   1000 "$REFS10"
run_nn pd_1em2   sweep_dT_1em2   1000 "$REFS10" --drop-nddot
run_nn pf_5em3   sweep_dT_5em3    500 "$REFS05"
run_nn pd_5em3   sweep_dT_5em3    500 "$REFS05" --drop-nddot

# (B) analytic closure: standard dealias (ceiling), then projected (annulus)
run_anal an_1p5em2  sweep_dT_1p5em2 1500 "$REFS15"
run_anal anp_1p5em2 sweep_dT_1p5em2 1500 "$REFS15" --nn-project-radius
run_anal an_1em2    sweep_dT_1em2   1000 "$REFS10"
run_anal anp_1em2   sweep_dT_1em2   1000 "$REFS10" --nn-project-radius
run_anal an_5em3    sweep_dT_5em3    500 "$REFS05"
run_anal anp_5em3   sweep_dT_5em3    500 "$REFS05" --nn-project-radius

echo "===================== CONSOLIDATE (one npz per case) ====================="
python -u $WT/diagnostics/consolidate_apost_cases.py --dir "$OUT" \
    --tags pf_1p5em2,pd_1p5em2,pf_1em2,pd_1em2,pf_5em3,pd_5em3,an_1p5em2,anp_1p5em2,an_1em2,anp_1em2,an_5em3,anp_5em3 \
    --arm-labels closure=uncond,closure2=cond,r3anal=analytic --delete-intermediates
echo "[matrix-p170] done $(date -u +%FT%TZ)"
