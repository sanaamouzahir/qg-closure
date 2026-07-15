#!/bin/bash
# apost_indist_condv2_job.sh - IN-DISTRIBUTION pure-conditioned ladder
# (coordinator order 2026-07-14): the M=16 a-posteriori ladder with the PURE
# a-priori conditioned closure (deriv7_cond_local_v2/best.pt, ep63, model
# cond_local, BEFORE any rollout stability fine-tune) + bare, on the SAME 6
# IN-DISTRIBUTION (member,IC) pairs as the session-14c true-closure job and
# the p1lam01 NN job (kf4 532/912/1356, FRC-256 549/933/1357). Same 3 dT.
# ALL truth refs REUSED from apost_opt2_rep_20260711 (--load-refs; hard-fail
# if missing -- no silent recompute), IDENTICAL to 14c.
#
# DECISIVE EXPERIMENT: gives the four-way bare / p1-NN (existing CSV) / cond_v2
# / true-closure. Question: does the pure conditioned model, BEFORE the
# stability fine-tune degraded its a-priori Nddot and reintroduced the U-shape,
# sit closer to the true-closure ceiling on the STABLE draws? Blowups on some
# draws are EXPECTED (cond_v2 has no stability FT) -- that is the stability
# cost we are quantifying.
#
# Code path/flags IDENTICAL to apost_indist_trueclosure_job.sh; the ONLY
# changes: CKPT -> cond_local_v2/best.pt, arms bare,closure (not bare,r3anal),
# out/label -> condv2. Model name is read from the ckpt config (cond_local),
# no --model flag needed.
#
# Submit: qsub -q ibgpu.q -l gpu=1 -N apost_indist_condv2 \
#   -o logs/apost_indist_condv2.\$JOB_ID.log -j y -cwd -V \
#   scripts/sge/apost_indist_condv2_job.sh

#$ -S /bin/bash
#$ -cwd
#$ -V

set -e
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
WT=$QG_ROOT/qg-wiener-conditioning
OUT=$WT/diagnostics/Results/apost_indist_condv2_20260714
REP=$WT/diagnostics/Results/apost_opt2_rep_20260711
CKPT=data/ensemble_N5_7lag/training_runs/deriv7_cond_local_v2/best.pt
source $QG_ROOT/qg-env/bin/activate
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR=$QG_ROOT/.mplcache; mkdir -p "$MPLCONFIGDIR"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd $WT/training
mkdir -p "$OUT"
echo "[apost_indist_condv2] host=$HOSTNAME date=$(date -u +%FT%TZ)"
python -u -c 'import sys, torch; ok = torch.cuda.is_available(); print("[gpu-check]", ok, flush=True); sys.exit(0 if ok else 2)'

run_case () {
    local member=$1 ic=$2 tag=$3 sweep=$4 K=$5 refs=$6
    local cdir=$OUT/${member}_ic${ic}
    mkdir -p "$cdir"
    if [[ ! -f "$refs" ]]; then
        echo "[apost_indist_condv2] FATAL: missing refs $refs (refs are reuse-only in this job)" >&2
        exit 3
    fi
    echo "===================== APOST_INDIST_CONDV2 $member ic=$ic $tag : K=$K ====================="
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
        --arm-labels "closure=condv2_${member#FRC-}_ic${ic}" \
        --delete-intermediates
}

# --- the 6 IN-DISTRIBUTION pairs (refs saved by job 1830720)
for spec in "FRC-kf4 532" "FRC-kf4 912" "FRC-kf4 1356" \
            "FRC-256 549" "FRC-256 933" "FRC-256 1357"; do
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
echo "[apost_indist_condv2] merged summary -> $OUT/ladder_matrix_summary_ALL.csv"
echo "[apost_indist_condv2] done $(date -u +%FT%TZ)"
