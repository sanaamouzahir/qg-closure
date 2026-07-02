#!/bin/bash
# prepare_npz_for_mmap_job.sh
#
# SGE worker that runs prepare_npz_for_mmap.py.
# Lives next to step1 in Convergence_studies/.

set -e

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
source "$QG_ROOT/qg-env/bin/activate"

# Tame thread fan-out; this is a write-bound job, not a compute one.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

SCRIPT_DIR="$QG_ROOT/qg-simple-package-stable/src/qg/Convergence_studies/Decaying_Turbulence/Step1"
cd "$SCRIPT_DIR"

echo "[prepare_npz_for_mmap_job] hostname: $HOSTNAME"
echo "[prepare_npz_for_mmap_job] date:     $(date -u +%FT%TZ)"
echo "[prepare_npz_for_mmap_job] cmd:      python -u prepare_npz_for_mmap.py $*"
echo "----------------------------------------------------------------------"

python -u prepare_npz_for_mmap.py "$@"

echo "----------------------------------------------------------------------"
echo "[prepare_npz_for_mmap_job] done at $(date -u +%FT%TZ)"
