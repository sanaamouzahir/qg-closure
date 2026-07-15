#!/bin/bash
# filter_new8_job.sh -- STAGE (c) of the DEC extension post-chain.
# Mandatory quiescent-window filter (rule 15) on the 8 newly sliced+resplit
# members. CPU-only -- runs on all.q. Output tee'd to a fixed-name log so the
# follow-on notify_filter job can parse per-member drop stats.
#$ -N deriv7_filter_new8
#$ -q all.q
#$ -j y
#$ -o /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-simple-package-stable/src/qg/logs/deriv7_filter_new8.$JOB_ID.log
#$ -cwd
set -euo pipefail

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR=$QG_ROOT/qg-simple-package-stable/src/qg
TRAIN_DIR=$QG_DIR/training
REPORT=$QG_DIR/logs/filter_new8_report.log

echo "[filter_new8] host=$HOSTNAME date=$(date -u +%FT%TZ)"
source "$QG_ROOT/qg-env/bin/activate"
cd "$TRAIN_DIR"

python -u filter_quiescent_windows.py \
    --sweeps data/ensemble_N5_7lag/FRC-b0/sweep_dT_* \
             data/ensemble_N5_7lag/FRC-b05/sweep_dT_* \
             data/ensemble_N5_7lag/FRC-b075/sweep_dT_* \
             data/ensemble_N5_7lag/FRC-b1/sweep_dT_* \
             data/ensemble_N5_7lag/DEC-base/sweep_dT_* \
             data/ensemble_N5_7lag/DEC-loRe/sweep_dT_* \
             data/ensemble_N5_7lag/DEC-hiRe/sweep_dT_* \
             data/ensemble_N5_7lag/DEC-512/sweep_dT_* \
    --frac 1e-2 --rough-min 1e-5 | tee "$REPORT"

echo "[filter_new8] done at $(date -u +%FT%TZ)"
