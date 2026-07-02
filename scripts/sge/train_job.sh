#!/bin/bash
# train_job.sh - SGE worker for train.py
set -e
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
source "$QG_ROOT/qg-env/bin/activate"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

SCRIPT_DIR="$QG_ROOT/qg-simple-package-stable/src/qg/training"
cd "$SCRIPT_DIR"

echo "[train_job] hostname: $HOSTNAME"
echo "[train_job] date:     $(date -u +%FT%TZ)"
echo "[train_job] cuda dev: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "[train_job] cwd:      $PWD"
echo "[train_job] cmd:      python -u train.py $*"
echo "------------------------------------------------------------"

python -u train.py "$@"

echo "------------------------------------------------------------"
echo "[train_job] done at $(date -u +%FT%TZ)"
