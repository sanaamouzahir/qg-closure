#!/bin/bash
# apost_dissproj_finalize_job.sh - merge + verdict + figures for the P0
# dissipative-projection ladder. CPU-only; held on both ladder halves via
# -hold_jid. Merges per-case summary CSVs into _ALL files, then runs
# diagnostics/dissproj_ladder_report.py (pre-registered-bars verdict,
# merged A/B table, three figures + explainer dir).
#
# Submit: qsub -q all.q -N dissproj_fin -hold_jid <jobA>,<jobB> \
#           scripts/sge/apost_dissproj_finalize_job.sh
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -o /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/$JOB_NAME.o$JOB_ID
#$ -e /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/$JOB_NAME.e$JOB_ID
#$ -m ea
#$ -M sanaamz@mit.edu

set -e
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
WT=$QG_ROOT/qg-wiener-conditioning
NEW=$WT/diagnostics/Results/apost_dissproj_20260713
PNGS=$WT/diagnostics/pngs/dissipative_projection_ladder
source $QG_ROOT/qg-env/bin/activate
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR=$QG_ROOT/.mplcache; mkdir -p "$MPLCONFIGDIR"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd $WT/training
echo "[dissproj_fin] host=$HOSTNAME date=$(date -u +%FT%TZ)"

for suf in "" "_h64" "_h128"; do
    first=1
    out=$NEW/ladder_matrix_summary${suf}_ALL.csv
    for f in "$NEW"/*_ic*/ladder_matrix_summary${suf}.csv; do
        [[ -f "$f" ]] || continue
        if [[ $first == 1 ]]; then cat "$f"; first=0; else tail -n +2 "$f"; fi
    done > "$out"
    echo "[dissproj_fin] merged -> $out ($(($(wc -l < "$out") - 1)) rows)"
done

python -u $WT/diagnostics/dissproj_ladder_report.py \
    --dir "$NEW" --pngs "$PNGS"
echo "[dissproj_fin] done $(date -u +%FT%TZ)"
