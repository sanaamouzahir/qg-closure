#!/bin/bash
# audit_A_job.sh - CPU worker for diagnostics/audit_decorrelation.py
# (Audit A, Supervisor_simulation.md S8/A; runs at the FPC-const gate BEFORE
# the 4 modulated runs are submitted).
#
# Amendment 02 S3 (absolute): no .py executes on the frontend -- this worker
# is the batch vehicle for the audit AND its --selftest. CPU-only (ACF/
# cross-correlation/spatial-ACF reductions; no GPU). Queue: all.q. Never the
# forbidden queue/memory-reservation flags (scripts/sge/CLAUDE.md; the only
# permitted GPU pair is documented in CLAUDE.md and does not apply here).
#
# Usage (submit from the branch root so logs land in <branch>/logs/;
# $SGE_O_WORKDIR is not usable in #$ lines -- pass -o/-e as qsub args):
#   qsub -q all.q -N audA_<tag> \
#        -o "$PWD/logs/\$JOB_NAME.\$JOB_ID.log" \
#        -e "$PWD/logs/\$JOB_NAME.\$JOB_ID.err" \
#        scripts/sge/audit_A_job.sh --scalars <run-dir>/scalars.npz \
#        --shedding-summary <dir>/shedding_summary.npz --piff-dir <dir> [...]
#   qsub -q all.q -N audA_selftest ... scripts/sge/audit_A_job.sh --selftest
#
# All arguments are forwarded verbatim to audit_decorrelation.py (which
# imports its sibling shedding_tracker.py -- hence the cd below).

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

echo "[audit_A_job] hostname: $HOSTNAME"
echo "[audit_A_job] date: $(date -u +%FT%TZ)"
echo "[audit_A_job] args: $*"
echo "----------------------------------------------------------------------"

# diagnostics/ runs flat with sibling imports (repo rule 2 analogue)
cd "$BRANCH/diagnostics"
python -u audit_decorrelation.py "$@"

echo "----------------------------------------------------------------------"
echo "[audit_A_job] done at $(date -u +%FT%TZ)"
