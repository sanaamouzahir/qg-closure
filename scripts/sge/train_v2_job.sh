#!/bin/bash
# train_v2_job.sh - SGE worker for train_v2.py
set -e
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
source "$QG_ROOT/qg-env/bin/activate"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

SCRIPT_DIR="$QG_ROOT/qg-simple-package-stable/src/qg/training"
cd "$SCRIPT_DIR"

echo "[train_v2_job] hostname: $HOSTNAME"
echo "[train_v2_job] date:     $(date -u +%FT%TZ)"
echo "[train_v2_job] cuda dev: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "[train_v2_job] cwd:      $PWD"
echo "[train_v2_job] cmd:      python -u train.py $*"
echo "------------------------------------------------------------"

python -u train.py "$@"

echo "------------------------------------------------------------"
echo "[train_v2_job] done at $(date -u +%FT%TZ)"
