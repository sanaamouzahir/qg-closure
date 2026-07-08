#!/bin/bash
# monitor_condlocal_job.sh -- decision-tree-D verdict for a train_deriv run,
# chained with -hold_jid so it fires when the training job leaves the queue
# (post-mortem parse of the full log; the live watch runs supervisor-side).
# CPU-only job -- submit WITHOUT gpu resources (all.q = repo convention for
# CPU sidecars/monitors), e.g.:
#   qsub -q all.q -N deriv7_cond_local_mon -hold_jid <trainid> \
#        -m ea -M $QG_NOTIFY_EMAIL scripts/sge/monitor_condlocal_job.sh \
#        --job-id <trainid> --log <trainlog> --run-dir <run_dir> --poll 30
#$ -o /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/$JOB_NAME.$JOB_ID.log
#$ -e /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/$JOB_NAME.$JOB_ID.err
set -e
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
source "$QG_ROOT/qg-env/bin/activate"
cd "$QG_ROOT/qg-wiener-conditioning"
exec python -u diagnostics/monitor_training.py "$@"
