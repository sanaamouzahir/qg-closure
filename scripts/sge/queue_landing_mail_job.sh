#!/bin/bash
# queue_landing_mail_job.sh - generic laptop-independent landing reporter:
# submitted with -hold_jid <job>, queues a pending_mail whose body is the
# tails/contents of the given files. $1=subject, rest=files to include.
# Submit: qsub -q all.q -N <name> -hold_jid <jid> -o logs/... -j y -cwd -V \
#         scripts/sge/queue_landing_mail_job.sh "<subject>" <file> [file...]

#$ -S /bin/bash
#$ -cwd
#$ -V

set -uo pipefail
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
PEND="$QG_ROOT/reporting/pending_mail"
SUBJ="${1:?subject}"; shift
f="$PEND/landing_$(date +%s)_$$.mail"
{
  echo "To: sanaamz@mit.edu"
  echo "Subject: $SUBJ"
  echo
  for x in "$@"; do
    echo "===== $x ====="
    if [ -f "$x" ]; then tail -60 "$x"; else echo "(missing)"; fi
    echo
  done
} > "$f"
echo "queued $f"
