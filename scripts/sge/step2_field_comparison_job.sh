#!/bin/bash
# step2_field_comparison_job.sh  (FLOW PAST CYLINDER)
#
# SGE worker that runs step2_field_comparison.py on the cluster.
# Submitted by submit_step2.sh; not intended to be run directly.
#
# Args: forwarded verbatim to step2_field_comparison.py.
#
# Lives at:
#   /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-simple-package-stable/src/qg/Convergence_studies/Decaying_Turbulence/Step2/

set -e

# ---- Environment ---------------------------------------------------------- #
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
source "$QG_ROOT/qg-env/bin/activate"

export MPLCONFIGDIR="$QG_ROOT/.mplcache"
mkdir -p "$MPLCONFIGDIR"
export MPLBACKEND=Agg

# Step 2 reads only 3 snapshots per run; tiny working set.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

# ---- Run ------------------------------------------------------------------ #
SCRIPT_DIR="$QG_ROOT/qg-simple-package-stable/src/qg/Convergence_studies/Decaying_Turbulence/Step2"
cd "$SCRIPT_DIR"

echo "[step2_field_comparison_job:decay] hostname: $HOSTNAME"
echo "[step2_field_comparison_job:decay] date:     $(date -u +%FT%TZ)"
echo "[step2_field_comparison_job:decay] python:   $(which python)"
echo "[step2_field_comparison_job:decay] cwd:      $PWD"
echo "[step2_field_comparison_job:decay] cmd:      python -u step2_field_comparison.py $*"
echo "----------------------------------------------------------------------"

python -u step2_field_comparison.py "$@"

echo "----------------------------------------------------------------------"
echo "[step2_field_comparison_job:decay] done at $(date -u +%FT%TZ)"
