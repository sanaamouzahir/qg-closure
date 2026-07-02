#!/bin/bash
# train_deriv_job.sh - SGE worker for train_deriv.py (pre-6.1.2 derivative-loss closure)
#$ -o /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-simple-package-stable/src/qg/logs/$JOB_NAME.o$JOB_ID
#$ -j y
set -e
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
source "$QG_ROOT/qg-env/bin/activate"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

SCRIPT_DIR="$QG_ROOT/qg-simple-package-stable/src/qg/training"
cd "$SCRIPT_DIR"

echo "[train_deriv_job] hostname: $HOSTNAME"
echo "[train_deriv_job] date:     $(date -u +%FT%TZ)"
echo "[train_deriv_job] cuda dev: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "[train_deriv_job] cwd:      $PWD"
echo "[train_deriv_job] args:     $(printf '%s ' "$@" | grep -oE '\-\-[a-z-]+ [^ ]*' | grep -vE 'sweep-roots' )"
echo "------------------------------------------------------------"

python -u train_deriv.py "$@"

echo "------------------------------------------------------------"
echo "[train_deriv_job] done at $(date -u +%FT%TZ)"
