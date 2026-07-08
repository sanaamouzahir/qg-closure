#!/bin/bash
# monitor_training_job.sh -- monitoring is part of the submission, not an
# accessory (CHARTER v1.3 I18). Every training qsub is a THREE-job unit and
# sge-checker (G5) refuses submissions that are missing either monitor. Submit
# from the branch root so logs land in <branch>/logs/ (I12; pass -o/-e as qsub
# args -- $SGE_O_WORKDIR is not usable in #$ lines):
#
#   TRAIN=$(qsub -terse -q ibgpu.q -l gpu=1 -N <run>_train ... <train_job>.sh ...)
#   LIVE=$(qsub  -terse -N mon_live_<run> \
#                -o "$PWD/logs/\$JOB_NAME.\$JOB_ID.log" -e "$PWD/logs/\$JOB_NAME.\$JOB_ID.err" \
#                scripts/sge/monitor_training_job.sh <run_dir> <branch> $TRAIN [card] [trainer_log])
#   FINAL=$(qsub -terse -N mon_final_<run> -hold_jid $TRAIN -v QG_MONITOR_FINALIZE=1 \
#                -o "$PWD/logs/\$JOB_NAME.\$JOB_ID.log" -e "$PWD/logs/\$JOB_NAME.\$JOB_ID.err" \
#                scripts/sge/monitor_training_job.sh <run_dir> <branch> $TRAIN [card] [trainer_log])
#
#   run_dir      dir containing log.csv (.../training_runs/<run-name>)
#   branch       tag for the [QG][MONITOR][<branch>] subject
#   trainer_id   SGE id of the trainer; the live monitor exits when it leaves qstat
#   card         baseline card JSON (default diagnostics/baseline_cards/T1_deriv7.json
#                relative to $SGE_O_WORKDIR; I18d -- pass the template's card)
#   trainer_log  trainer stdout log path (optional; enables traceback detection)
#
# LIVE watches the run and emails [QG][MONITOR] at first val epoch, every 5
# epochs, and immediately on any trigger. FINAL (-hold_jid, finalize mode) is
# the safety net: silent if LIVE delivered a final verdict, otherwise it emits
# the postmortem. Monitor names must differ from the trainer's in the first 10
# chars (qstat truncates -- the 2026-07-08 triple-qdel lesson).
# The [QG][SUBMIT][log] email must carry ALL THREE job ids (I18a).
# CPU sidecar: all.q on purpose (repo convention for monitors; not a GPU job).
#$ -N monitor_training
#$ -q all.q
#$ -j y
#$ -cwd
set -uo pipefail

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
source "$QG_ROOT/qg-env/bin/activate"

RUN_DIR=$1
BRANCH=$2
JOB_ID=$3
CARD=${4:-$SGE_O_WORKDIR/diagnostics/baseline_cards/T1_deriv7.json}
TRAINER_LOG=${5:-}
MONITOR_PY=${MONITOR_PY:-$SGE_O_WORKDIR/diagnostics/monitor_training.py}

if [[ ! -f "$MONITOR_PY" ]]; then
    echo "[monitor-job] FATAL: monitor script not found: $MONITOR_PY" >&2
    exit 1
fi
if [[ ! -f "$CARD" ]]; then
    echo "[monitor-job] WARNING: baseline card not found: $CARD (I18d violation;" \
         "ORDER-INVERSION disabled)" >&2
    CARD=""
fi

ARGS=(--run-dir "$RUN_DIR" --branch "$BRANCH" --job-id "$JOB_ID"
      --email "${QG_NOTIFY_EMAIL:-sanaamz@mit.edu}" --interval 120 --cadence 5)
[[ -n "$CARD" ]] && ARGS+=(--baseline-card "$CARD")
[[ -n "$TRAINER_LOG" ]] && ARGS+=(--log "$TRAINER_LOG")
[[ -n "${QG_MONITOR_FINALIZE:-}" ]] && ARGS+=(--finalize)

python -u "$MONITOR_PY" "${ARGS[@]}"
