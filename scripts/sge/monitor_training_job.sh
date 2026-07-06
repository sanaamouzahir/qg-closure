#!/bin/bash
# monitor_training_job.sh -- low-resource watcher chained to every training job.
# Runs diagnostics/monitor_training.py CONCURRENTLY with the training job (do NOT
# -hold_jid it on the trainer -- it must watch the run live). sge-runner submits
# this immediately after any training submission (mandatory wiring):
#
#   qsub -N monitor_<run> scripts/sge/monitor_training_job.sh \
#        <run_dir> <branch> <training_job_id> [monitor_py]
#
#   run_dir         dir containing log.csv (…/training_runs/<run-name>)
#   branch          tag for the [QG][FLAG][<branch>] subject (e.g. free-time-fd, main)
#   training_job_id SGE id of the trainer; the monitor exits when it leaves qstat
#   monitor_py      optional explicit path to monitor_training.py
#                   (default: $SGE_O_WORKDIR/diagnostics/monitor_training.py)
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
MONITOR_PY=${4:-$SGE_O_WORKDIR/diagnostics/monitor_training.py}

if [[ ! -f "$MONITOR_PY" ]]; then
    echo "[monitor-job] FATAL: monitor script not found: $MONITOR_PY" >&2
    exit 1
fi

python "$MONITOR_PY" \
    --run-dir "$RUN_DIR" \
    --branch "$BRANCH" \
    --job-id "$JOB_ID" \
    --email "${QG_NOTIFY_EMAIL:-sanaamz@mit.edu}" \
    --interval 600
