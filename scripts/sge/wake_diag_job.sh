#!/bin/bash
# wake_diag_job.sh - CPU worker for diagnostics/diagnostics_wake.py
# (task A wake package: phase-conditioned means, mean/rms wake maps,
# recirculation extent, probe statistics, phase-wheel coverage).
#
# Amendment 02 S3 (absolute): no .py on the frontend -- this worker is the
# batch vehicle for the module AND its --selftest. CPU-only (streamed
# slice-at-a-time npz reads; no GPU). Queue: all.q. Never the forbidden
# queue/memory-reservation flags (scripts/sge/CLAUDE.md).
#
# Usage (submit from the branch root so logs land in <branch>/logs/):
#   qsub -q all.q -N wake_<tag> \
#        -o "$PWD/logs/\$JOB_NAME.\$JOB_ID.log" \
#        -e "$PWD/logs/\$JOB_NAME.\$JOB_ID.err" \
#        scripts/sge/wake_diag_job.sh --run-dir <run-dir> [args...]
#
# All arguments are forwarded verbatim to diagnostics_wake.py.

#$ -S /bin/bash
#$ -cwd
#$ -V

set -e

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
BRANCH="$QG_ROOT/qg-sgs-closure"

source "$QG_ROOT/qg-env/bin/activate"
export TMPDIR="$QG_ROOT/tmp"
export MPLCONFIGDIR="$QG_ROOT/mplcache"
export PYTHONUNBUFFERED=1

echo "[wake_diag_job] hostname: $HOSTNAME"
echo "[wake_diag_job] date: $(date -u +%FT%TZ)"
echo "[wake_diag_job] args: $*"
echo "----------------------------------------------------------------------"

cd "$BRANCH/diagnostics"
python -u diagnostics_wake.py "$@"

echo "----------------------------------------------------------------------"
echo "[wake_diag_job] done at $(date -u +%FT%TZ)"
