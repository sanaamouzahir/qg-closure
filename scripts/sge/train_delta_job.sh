#!/bin/bash
# train_delta_job.sh - SGE worker for train_delta.py (hybrid delta closure)
#$ -o /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-simple-package-stable/src/qg/logs/$JOB_NAME.$JOB_ID.log
#$ -j y
set -e
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
source "$QG_ROOT/qg-env/bin/activate"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

SCRIPT_DIR="$QG_ROOT/qg-simple-package-stable/src/qg/training"
cd "$SCRIPT_DIR"

echo "[train_delta_job] hostname: $HOSTNAME"
echo "[train_delta_job] date:     $(date -u +%FT%TZ)"
echo "[train_delta_job] cuda dev: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "[train_delta_job] cwd:      $PWD"
echo "[train_delta_job] cmd:      python -u train_delta.py $*"
echo "------------------------------------------------------------"

python -u train_delta.py "$@"

echo "------------------------------------------------------------"
echo "[train_delta_job] done at $(date -u +%FT%TZ)"