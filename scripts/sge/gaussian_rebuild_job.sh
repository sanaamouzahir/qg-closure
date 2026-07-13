#!/bin/bash
# gaussian_rebuild_job.sh - all-Gaussian Pi_FF data rebuild for one member
# (Sanaa order 2026-07-13: CPU job, NOT GPU; outputs DNS_LES_s<N>_gaussian.npz
# in the member's own folder). One job per member, scales 4,2,8 sequential
# (training scale first).
#
# Usage:
#   LOGS=/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-sgs-closure/logs
#   qsub -N gRb_<tag> -q all.q -o "$LOGS/\$JOB_NAME.\$JOB_ID.log" -j y -cwd -V \
#        gaussian_rebuild_job.sh <member_dir> [args...]

#$ -S /bin/bash
#$ -cwd
#$ -V

set -e

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
BRANCH="$QG_ROOT/qg-sgs-closure"

# qg-env: has the editable qg package + torch (same env as compute_pi_ff_job.sh)
source "$QG_ROOT/qg-env/bin/activate"
export TMPDIR="$QG_ROOT/tmp"
export MPLCONFIGDIR="$QG_ROOT/mplcache"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}   # polite CPU share on all.q

cd "$BRANCH/ml_closure"
echo "[gauss_rebuild] host $HOSTNAME date $(date -u +%FT%TZ) threads $OMP_NUM_THREADS"
echo "[gauss_rebuild] cmd: python -u compute_pi_ff_gaussian_rebuild.py $*"
echo "----------------------------------------------------------------------"
python -u compute_pi_ff_gaussian_rebuild.py "$@"
echo "----------------------------------------------------------------------"
echo "[gauss_rebuild] done at $(date -u +%FT%TZ)"
