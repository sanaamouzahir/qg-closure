#!/bin/bash
# piff_step0_job.sh - Step-0 canonical artifacts for one run dir (CP-ML-1 S2):
# make_dataset_manifest.py writes DNS_LES_s<scale>.npz + U_of_t.npz +
# DATASET_MANIFEST.md. CPU job (FFT uv-build over ~300 LES frames), all.q.
#
# Usage:
#   LOGS=/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-sgs-closure/logs
#   qsub -N step0_<tag> -q all.q -o "$LOGS/\$JOB_NAME.\$JOB_ID.log" -j y -cwd -V \
#        piff_step0_job.sh <run_dir> [--scale 4] [--force]

#$ -S /bin/bash
#$ -cwd
#$ -V

set -e

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
BRANCH="$QG_ROOT/qg-sgs-closure"

source "$QG_ROOT/qg-env-piff/bin/activate"
export TMPDIR="$QG_ROOT/tmp"
export MPLCONFIGDIR="$QG_ROOT/mplcache"
export PYTHONUNBUFFERED=1

cd "$BRANCH/ml_closure"
echo "[piff_step0] host $HOSTNAME date $(date -u +%FT%TZ)"
echo "[piff_step0] cmd: python -u make_dataset_manifest.py $*"
echo "----------------------------------------------------------------------"
python -u make_dataset_manifest.py "$@"
echo "----------------------------------------------------------------------"
echo "[piff_step0] done at $(date -u +%FT%TZ)"
