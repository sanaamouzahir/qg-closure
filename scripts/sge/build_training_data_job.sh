#!/bin/bash
# build_training_data_job.sh - SGE worker for build_training_data.py
# Args: forwarded verbatim to build_training_data.py.

set -e

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
source "$QG_ROOT/qg-env/bin/activate"

export MPLCONFIGDIR="$QG_ROOT/.mplcache"
mkdir -p "$MPLCONFIGDIR"
export MPLBACKEND=Agg

# Threading: solver work is GPU-dominated; CPU just hosts the loop.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

SCRIPT_DIR="$QG_ROOT/qg-simple-package-stable/src/qg/training/"
cd "$SCRIPT_DIR"

echo "[build_training_data_job] hostname: $HOSTNAME"
echo "[build_training_data_job] date:     $(date -u +%FT%TZ)"
echo "[build_training_data_job] python:   $(which python)"
echo "[build_training_data_job] cuda dev: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "[build_training_data_job] cwd:      $PWD"
echo "[build_training_data_job] cmd:      python -u build_training_data.py $*"
echo "----------------------------------------------------------------------"

python -u build_training_data.py "$@"

echo "----------------------------------------------------------------------"
echo "[build_training_data_job] done at $(date -u +%FT%TZ)"
