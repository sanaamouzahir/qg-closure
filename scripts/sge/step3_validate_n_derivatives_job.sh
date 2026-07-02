#!/bin/bash
# step3_validate_n_derivatives_job.sh  (FLOW PAST CYLINDER)
#
# SGE worker that runs step3_validate_n_derivatives.py on the cluster.
# Submitted by submit_step3.sh; not intended to be run directly.
#
# Args: forwarded verbatim to step3_validate_n_derivatives.py.
#
# Lives at:
#   /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-simple-package-stable/src/qg/Convergence_studies/Decaying_Turbulence/Step3/

set -e

# ---- Environment ---------------------------------------------------------- #
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
source "$QG_ROOT/qg-env/bin/activate"

export MPLCONFIGDIR="$QG_ROOT/.mplcache"
mkdir -p "$MPLCONFIGDIR"
export MPLBACKEND=Agg

# Step 3 reads only ~3 snapshots per test triplet; small working set.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# ---- Run ------------------------------------------------------------------ #
SCRIPT_DIR="$QG_ROOT/qg-simple-package-stable/src/qg/Convergence_studies/Decaying_Turbulence/Step3"
cd "$SCRIPT_DIR"

echo "[step3_job:decay] hostname: $HOSTNAME"
echo "[step3_job:decay] date:     $(date -u +%FT%TZ)"
echo "[step3_job:decay] python:   $(which python)"
echo "[step3_job:decay] cwd:      $PWD"
echo "[step3_job:decay] cmd:      python -u step3_validate_n_derivatives.py $*"
echo "----------------------------------------------------------------------"

python -u step3_validate_n_derivatives.py "$@"

echo "----------------------------------------------------------------------"
echo "[step3_job:decay] done at $(date -u +%FT%TZ)"
