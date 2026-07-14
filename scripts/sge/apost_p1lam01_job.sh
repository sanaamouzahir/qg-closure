#!/bin/bash
# apost_p1lam01_job.sh - PSW3 verdict ladder: the M=16 a-posteriori ladder on the
# rollout_ft_p1_lam01 BEST ckpt (ep18, val 1.3148e-04; the P1 lambda-sweep WINNER,
# vn-lambda 0.1, trained on the SMALL ensemble kf4+FRC-256; Sanaa order 2026-07-14).
# IDENTICAL code path/flags as apost_opt2_job.sh / apost_opt2_rep_job.sh; the
# ONLY change vs the ep33 runs is the ckpt. Cases = the full comparison grid:
#   kf4 IC837 (the 07-11 headline: ep33 gave 16.6x / 0.50x / 0.33x)
#   + the 9 replication (member,IC) pairs of job 1830720
# ALL truth refs REUSED (--load-refs; hard-fail if missing -- no silent
# recompute, the truth leg is the only expensive one). 30 rungs, each rung =
# bare + closure arms at 16 coarse steps => minutes, not hours.
# Question: does per-stride weighting improve 1.5e-2/1e-2 accuracy without
# losing the 5e-3 gain?
#
# Submit: qsub -q ibgpu.q -l gpu=1 -N apost_p1lam01 \
#   -o logs/apost_p1lam01.\$JOB_ID.log -j y -cwd -V scripts/sge/apost_p1lam01_job.sh

#$ -S /bin/bash
#$ -cwd
#$ -V

set -e
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
WT=$QG_ROOT/qg-wiener-conditioning
OUT=$WT/diagnostics/Results/apost_p1lam01_20260714
REP=$WT/diagnostics/Results/apost_opt2_rep_20260711
LADDER=$WT/diagnostics/Results/apost_ladder_20260709
REFS15_837=$WT/diagnostics/Results/apost_smoke3/apost_refs_ladderrefs.npz
CKPT=data/ensemble_N5_7lag/training_runs/rollout_ft_p1_lam01/best.pt
source $QG_ROOT/qg-env/bin/activate
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR=$QG_ROOT/.mplcache; mkdir -p "$MPLCONFIGDIR"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd $WT/training
mkdir -p "$OUT"
echo "[apost_p1lam01] host=$HOSTNAME date=$(date -u +%FT%TZ)"
python -u -c 'import sys, torch; ok = torch.cuda.is_available(); print("[gpu-check]", ok, flush=True); sys.exit(0 if ok else 2)'

run_case () {
    local member=$1 ic=$2 tag=$3 sweep=$4 K=$5 refs=$6
    local cdir=$OUT/${member}_ic${ic}
    mkdir -p "$cdir"
    if [[ ! -f "$refs" ]]; then
        echo "[apost_p1lam01] FATAL: missing refs $refs (refs are reuse-only in this job)" >&2
        exit 3
    fi
    echo "===================== APOST_P1LAM01 $member ic=$ic $tag : K=$K ====================="
    python -u rollout_aposteriori.py \
        --root-dir "data/ensemble_N5_7lag/$member/$sweep" \
        --ckpt "$CKPT" \
        --ic-index "$ic" --K "$K" --n-steps 16 --n-checkpoints 24 \
        --arms bare,closure --log-sigma \
        --device cuda --out-dir "$cdir" --tag "$tag" --load-refs "$refs"
}

consolidate () {
    local member=$1 ic=$2
    echo "===================== CONSOLIDATE $member ic=$ic ====================="
    python -u $WT/diagnostics/consolidate_apost_cases.py \
        --dir "$OUT/${member}_ic${ic}" \
        --tags "ic${ic}_1p5em2,ic${ic}_1em2,ic${ic}_5em3" \
        --arm-labels "closure=p1lam01_${member#FRC-}_ic${ic}" \
        --delete-intermediates
}

# IC837 legs DROPPED 2026-07-14: their ref files (apost_smoke3/, apost_ladder
# full refs) no longer exist on disk; the 9 replication pairs below have
# complete per-case refs (verified) and include ALL THREE held-out combo ICs.

# --- the 9 replication (member,IC) pairs (refs saved by job 1830720)
for spec in "FRC-kf4 532" "FRC-kf4 912" "FRC-kf4 1356" \
            "FRC-256 549" "FRC-256 933" "FRC-256 1357" \
            "FRC-combo 527" "FRC-combo 884" "FRC-combo 1355"; do
    set -- $spec; member=$1; ic=$2
    rdir=$REP/${member}_ic${ic}
    run_case "$member" "$ic" "ic${ic}_1p5em2" sweep_dT_1p5em2 1500 "$rdir/apost_refs_ic${ic}_1p5em2.npz"
    run_case "$member" "$ic" "ic${ic}_1em2"   sweep_dT_1em2   1000 "$rdir/apost_refs_ic${ic}_1em2.npz"
    run_case "$member" "$ic" "ic${ic}_5em3"   sweep_dT_5em3    500 "$rdir/apost_refs_ic${ic}_5em3.npz"
    consolidate "$member" "$ic"
done

echo "===================== MERGE SUMMARY ====================="
first=1
for f in "$OUT"/*/ladder_matrix_summary.csv; do
    if [[ $first == 1 ]]; then cat "$f"; first=0; else tail -n +2 "$f"; fi
done > "$OUT/ladder_matrix_summary_ALL.csv"
echo "[apost_p1lam01] merged summary -> $OUT/ladder_matrix_summary_ALL.csv"
echo "[apost_p1lam01] done $(date -u +%FT%TZ)"
