#!/bin/bash
# piff_monitor_job.sh - I18 monitor sidecar for T4 (Pi_FF SVGP) trainings.
# LIVE:     qsub -N pMonL_<run> -q all.q -o <logs>/... -j y -cwd -V \
#                piff_monitor_job.sh <trainer_log> <run_name> <trainer_jid> [card]
# FINALIZE: same + -hold_jid <trainer_jid> -v QG_MONITOR_FINALIZE=1
# Reports spool to reporting/pending_mail (mseas relay — node-independent).
# CPU sidecar on all.q by repo convention; never a GPU job.

#$ -S /bin/bash
#$ -cwd
#$ -V

set -euo pipefail

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
BRANCH="$QG_ROOT/qg-sgs-closure"
source "$QG_ROOT/qg-env-piff/bin/activate"
export PYTHONUNBUFFERED=1

TRAINER_LOG="${1:?trainer_log}"
RUN_NAME="${2:?run_name}"
TRAINER_JID="${3:?trainer_job_id}"
CARD="${4:-$BRANCH/diagnostics/baseline_cards/SGS_piff_ens.json}"

echo "[piff_monitor] host $HOSTNAME mode ${QG_MONITOR_FINALIZE:-live} run $RUN_NAME jid $TRAINER_JID"
python -u "$BRANCH/diagnostics/monitor_piff.py" "$TRAINER_LOG" "$RUN_NAME" "$TRAINER_JID" "$CARD"
echo "[piff_monitor] done at $(date -u +%FT%TZ)"
