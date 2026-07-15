#!/bin/bash
# resplit_new8_job.sh -- STAGE (b) of the DEC extension post-chain.
# By-window resplit (rule 7) of the 8 newly sliced members' sweep_dT_* dirs.
# CPU-only (numpy only, no torch) -- runs on all.q, not the GPU queue.
#$ -N deriv7_resplit_new8
#$ -q all.q
#$ -j y
#$ -o /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-simple-package-stable/src/qg/logs/deriv7_resplit_new8.$JOB_ID.log
#$ -cwd
set -euo pipefail

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR=$QG_ROOT/qg-simple-package-stable/src/qg
TRAIN_DIR=$QG_DIR/training

echo "[resplit_new8] host=$HOSTNAME date=$(date -u +%FT%TZ)"
source "$QG_ROOT/qg-env/bin/activate"
cd "$TRAIN_DIR"

python -u resplit_by_window.py \
    --sweeps data/ensemble_N5_7lag/FRC-b0/sweep_dT_* \
             data/ensemble_N5_7lag/FRC-b05/sweep_dT_* \
             data/ensemble_N5_7lag/FRC-b075/sweep_dT_* \
             data/ensemble_N5_7lag/FRC-b1/sweep_dT_* \
             data/ensemble_N5_7lag/DEC-base/sweep_dT_* \
             data/ensemble_N5_7lag/DEC-loRe/sweep_dT_* \
             data/ensemble_N5_7lag/DEC-hiRe/sweep_dT_* \
             data/ensemble_N5_7lag/DEC-512/sweep_dT_* \
    --train-frac 0.70 --val-frac 0.15 --seed 0

echo "[resplit_new8] done at $(date -u +%FT%TZ)"
