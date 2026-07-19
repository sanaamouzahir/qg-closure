#!/bin/bash
# notify_build_new8_job.sh -- custom-content "(1) builds done" email.
# Held on the 8 build_mmap_<member> job ids. Composes a body confirming the
# forcing-gate check per member (does NOT gate the pipeline itself -- that
# hard-stop lives inside slice_new8_job.sh; this is the supervisor ping).
#$ -N notify_build_new8
#$ -q all.q
#$ -j y
#$ -o /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-simple-package-stable/src/qg/logs/notify_build_new8.$JOB_ID.log
#$ -cwd
set -uo pipefail

QG_DIR=/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-simple-package-stable/src/qg
LOG_DIR=$QG_DIR/logs
TO=sanaamz@mit.edu
MEMBERS=(FRC-b0 FRC-b05 FRC-b075 FRC-b1 DEC-base DEC-loRe DEC-hiRe DEC-512)

BODY=$(mktemp)
{
  echo "8-member deep-build batch: build stage complete."
  echo "Members: ${MEMBERS[*]}"
  echo
  echo "Per-member forcing-gate check (from build_mmap_<member>.log):"
  for m in "${MEMBERS[@]}"; do
      log="$LOG_DIR/build_mmap_${m}.log"
      if [[ -f "$log" ]]; then
          line=$(grep -E '\[build\] (static forcing|no forcing)' "$log" | tail -1)
          done_line=$(grep -c '\[build\] done' "$log")
          status="OK"
          [[ "$done_line" == "0" ]] && status="INCOMPLETE (no [build] done line found)"
          echo "  $m: ${line:-<no forcing-gate line found>}   [$status]"
      else
          echo "  $m: LOG MISSING ($log)"
      fi
  done
  echo
  echo "Post-chain: slice -> resplit -> filter jobs are queued, held on these"
  echo "8 build job ids, and will fire automatically. Filter-stage report will"
  echo "follow in a separate email with per-member window counts/drop %."
} > "$BODY"

mail -s "[QG-closure] 8-member deep-build batch: BUILDS DONE" "$TO" < "$BODY"
cat "$BODY"
rm -f "$BODY"
echo "[notify_build_new8] mailed $TO at $(date -u +%FT%TZ)"
