#!/bin/bash
# notify_filter_new8_job.sh -- custom-content "(4) filter done" email.
# Held on deriv7_filter_new8. Parses filter_new8_report.log (tee'd by the
# filter stage) for per-member window counts + drop percentages.
#$ -N notify_filter_new8
#$ -q all.q
#$ -j y
#$ -o /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-simple-package-stable/src/qg/logs/notify_filter_new8.$JOB_ID.log
#$ -cwd
set -uo pipefail

QG_DIR=/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-simple-package-stable/src/qg
REPORT=$QG_DIR/logs/filter_new8_report.log
TO=sanaamz@mit.edu

BODY=$(mktemp)
{
  echo "8-member deep-build batch: filter stage (rule 15, mandatory) complete."
  echo
  if [[ -f "$REPORT" ]]; then
      echo "Per-member/dt window counts + drop percentages (from filter_quiescent_windows.py):"
      grep -E '^\[filter\]|windows=' "$REPORT"
  else
      echo "WARNING: filter report log not found at $REPORT -- check "
      echo "deriv7_filter_new8.<job_id>.log directly."
  fi
  echo
  echo "Pipeline complete through: build -> slice -> resplit -> filter."
  echo "No training was started (per instructions)."
} > "$BODY"

mail -s "[QG-closure] 8-member deep-build batch: FILTER DONE (per-member drop stats)" "$TO" < "$BODY"
cat "$BODY"
rm -f "$BODY"
echo "[notify_filter_new8] mailed $TO at $(date -u +%FT%TZ)"
