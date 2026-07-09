#!/bin/bash
# shedding_job.sh - CPU worker for diagnostics/shedding_tracker.py
# (SGS-closure branch; AMENDMENT_01 SD, AMENDMENT_02 S4).
#
# Amendment 02 S3 (absolute): no .py executes on the frontend -- this worker
# is the batch vehicle for the tracker AND its --selftest. CPU-only (Welch/
# Hilbert on the dense scalar series; no GPU). Queue: all.q. Never the
# forbidden queue/memory-reservation flags (scripts/sge/CLAUDE.md; the only
# permitted GPU pair is documented in CLAUDE.md and does not apply here).
#
# Usage (submit from the branch root so logs land in <branch>/logs/;
# $SGE_O_WORKDIR is not usable in #$ lines -- pass -o/-e as qsub args):
#   qsub -q all.q -N shed_<tag> \
#        -o "$PWD/logs/\$JOB_NAME.\$JOB_ID.log" \
#        -e "$PWD/logs/\$JOB_NAME.\$JOB_ID.err" \
#        scripts/sge/shedding_job.sh <run-dir>/scalars.npz [args...]
#   qsub -q all.q -N shed_selftest ... scripts/sge/shedding_job.sh --selftest
#
# All arguments are forwarded verbatim to shedding_tracker.py.

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

echo "[shedding_job] hostname: $HOSTNAME"
echo "[shedding_job] date: $(date -u +%FT%TZ)"
echo "[shedding_job] args: $*"
echo "----------------------------------------------------------------------"

# diagnostics/ runs flat with sibling imports (repo rule 2 analogue)
cd "$BRANCH/diagnostics"
python -u shedding_tracker.py "$@"

echo "----------------------------------------------------------------------"
echo "[shedding_job] done at $(date -u +%FT%TZ)"
