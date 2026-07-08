#!/bin/bash
# step1_convergence_job.sh  (FLOW PAST CYLINDER)
#
# SGE worker that runs step1_convergence_plot.py on the cluster.
# Submitted by submit_step1.sh; not intended to be run directly.
#
# Args: forwarded verbatim to step1_convergence_plot.py.
#
# Lives at:
#   /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-simple-package-stable/src/qg/Convergence_studies/Decaying_Turbulence/Step1/

#$ -o /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/$JOB_NAME.$JOB_ID.log
#$ -e /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/$JOB_NAME.$JOB_ID.err
set -e

# ---- Environment ---------------------------------------------------------- #
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
source "$QG_ROOT/qg-env/bin/activate"

export MPLCONFIGDIR="$QG_ROOT/.mplcache"
mkdir -p "$MPLCONFIGDIR"
export MPLBACKEND=Agg

# Threading: post-processing is FFT- and reduction-bound. 4 threads is plenty.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# ---- Run ------------------------------------------------------------------ #
SCRIPT_DIR="$QG_ROOT/qg-simple-package-stable/src/qg/Convergence_studies/Decaying_Turbulence/Step1"
cd "$SCRIPT_DIR"

echo "[step1_convergence_job:decay] hostname: $HOSTNAME"
echo "[step1_convergence_job:decay] date:     $(date -u +%FT%TZ)"
echo "[step1_convergence_job:decay] python:   $(which python)"
echo "[step1_convergence_job:decay] cwd:      $PWD"
echo "[step1_convergence_job:decay] cmd:      python -u step1_convergence_plot.py $*"
echo "----------------------------------------------------------------------"

python -u step1_convergence_plot.py "$@"

echo "----------------------------------------------------------------------"
echo "[step1_convergence_job:decay] done at $(date -u +%FT%TZ)"
