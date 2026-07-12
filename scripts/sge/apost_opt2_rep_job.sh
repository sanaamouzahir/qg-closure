#!/bin/bash
# apost_opt2_rep_job.sh - REPLICATION of the OPTION-2 verdict ladder
# (session 9, job 1830550: kf4 IC837 M=16 -> blow-ups cured, 5e-3 16.6x).
# Same ckpt (rollout_ft_opt2_cond best = ep33), IDENTICAL code path and
# flags as apost_opt2_job.sh / the 07-09 4-arm table; only (member, IC)
# varies. 3 ICs x 3 members x 3 dT = 27 cases:
#   FRC-kf4  ICs 532  912 1356   (fine-tune member; IC837 = the landed run)
#   FRC-256  ICs 549  933 1357   (fine-tune member, 256^2)
#   FRC-combo ICs 527  884 1355  (HELD OUT of the rollout fine-tune)
# ICs are val-split rows common to all three sweeps of each member,
# >= 30 rows apart (max_anchors=3 -> distinct deep windows) and away
# from 837. Truth refs: NEW per (member, IC, dT) with --save-refs on
# first pass, --load-refs on any rerun (refs are the expensive leg,
# ~218 s/IC at 512^2, ~55 s at 256^2; kept for future ckpt ladders).
# K per dT keeps h_fine = 1.0e-5 (accuracy-run rule): 1500/1000/500.
#
# Submit: qsub -q ibgpu.q -l gpu=1 -N apost_rep \
#   -o logs/apost_rep.$JOB_ID.log -j y -cwd -V scripts/sge/apost_opt2_rep_job.sh

#$ -S /bin/bash
#$ -cwd
#$ -V

set -e
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
WT=$QG_ROOT/qg-wiener-conditioning
OUT=$WT/diagnostics/Results/apost_opt2_rep_20260711
CKPT=data/ensemble_N5_7lag/training_runs/rollout_ft_opt2_cond/best.pt
source $QG_ROOT/qg-env/bin/activate
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR=$QG_ROOT/.mplcache; mkdir -p "$MPLCONFIGDIR"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd $WT/training
mkdir -p "$OUT"
echo "[apost_rep] host=$HOSTNAME date=$(date -u +%FT%TZ)"
python -u -c 'import sys, torch; ok = torch.cuda.is_available(); print("[gpu-check]", ok, flush=True); sys.exit(0 if ok else 2)'

run_case () {
    local member=$1 ic=$2 tag=$3 sweep=$4 K=$5
    local cdir=$OUT/${member}_ic${ic}
    mkdir -p "$cdir"
    local refs=$cdir/apost_refs_${tag}.npz
    local reffl="--save-refs"
    [[ -f "$refs" ]] && reffl="--load-refs $refs"
    echo "===================== APOST_REP $member ic=$ic $tag : K=$K ($reffl) ====================="
    python -u rollout_aposteriori.py \
        --root-dir "data/ensemble_N5_7lag/$member/$sweep" \
        --ckpt "$CKPT" \
        --ic-index "$ic" --K "$K" --n-steps 16 --n-checkpoints 24 \
        --arms bare,closure --log-sigma \
        --device cuda --out-dir "$cdir" --tag "$tag" $reffl
}

for spec in "FRC-kf4 532" "FRC-kf4 912" "FRC-kf4 1356" \
            "FRC-256 549" "FRC-256 933" "FRC-256 1357" \
            "FRC-combo 527" "FRC-combo 884" "FRC-combo 1355"; do
    set -- $spec; member=$1; ic=$2
    run_case "$member" "$ic" "ic${ic}_1p5em2" sweep_dT_1p5em2 1500
    run_case "$member" "$ic" "ic${ic}_1em2"   sweep_dT_1em2   1000
    run_case "$member" "$ic" "ic${ic}_5em3"   sweep_dT_5em3    500
    echo "===================== CONSOLIDATE $member ic=$ic ====================="
    python -u $WT/diagnostics/consolidate_apost_cases.py \
        --dir "$OUT/${member}_ic${ic}" \
        --tags "ic${ic}_1p5em2,ic${ic}_1em2,ic${ic}_5em3" \
        --arm-labels "closure=opt2_${member#FRC-}_ic${ic}" \
        --delete-intermediates
done

echo "===================== MERGE SUMMARY ====================="
first=1
for f in "$OUT"/*/ladder_matrix_summary.csv; do
    if [[ $first == 1 ]]; then cat "$f"; first=0; else tail -n +2 "$f"; fi
done > "$OUT/ladder_matrix_summary_ALL.csv"
echo "[apost_rep] merged summary -> $OUT/ladder_matrix_summary_ALL.csv"
echo "[apost_rep] done $(date -u +%FT%TZ)"
