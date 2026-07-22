#!/bin/bash
# Autonomous completion watcher for the J_NN fine-tune: waits for the job to
# leave the queue, parses the log, and spools a [QG][REPORT] email to Sanaa.
QG=/gdata/projects/ml_scope/Closure_modeling/QG-closure
JOB=1843955
L="$QG/qg-wiener-conditioning/logs/jnn_ft_v1.log"
PEND="$QG/reporting/pending_mail"
export PATH=/opt/rocks/bin:$PATH
while qstat 2>/dev/null | grep -qE "^ *$JOB "; do sleep 120; done
sleep 15
RHO=$(grep -E "J_NN rho max-per-window|\|G_eff\| max-per-window" "$L" 2>/dev/null)
EPS=$(grep -E "^ep +[0-9]" "$L" 2>/dev/null | tail -22)
EXITL=$(grep -E "TRAINER EXIT" "$L" 2>/dev/null | tail -1)
NANM=$(grep -ciE "NAN-ABORT|not finite|non-finite" "$L" 2>/dev/null)
NEPOCH=$(grep -cE "^ep +[0-9]" "$L" 2>/dev/null)
BEST=$(grep -iE "best|saved" "$L" 2>/dev/null | tail -3)
mkdir -p "$PEND"
MAILF="$PEND/jnn_ft_v1_$(date +%s).mail"
{
  echo "To: sanaamz@mit.edu"
  echo "Subject: [QG][REPORT] J_NN von Neumann fine-tune (jnn_ft_v1) -- result"
  echo
  echo "The --vn-mode jnn fine-tune (job $JOB) has finished."
  echo
  echo "WHAT THIS RUN IS: warm-start from cert-v2, 7 FRC roots, strides 1/2/3,"
  echo "20 epochs, lr 5e-5, vn-lambda 0.5, developed-steps 24. The stability"
  echo "penalty is the NETWORK'S OWN Jacobian amplification (amortized power"
  echo "iteration), replacing the analytic frozen-i*sigma certificate that Test A"
  echo "proved BLIND (it read ~1.0 while the real/measured amplification was ~1.8)."
  echo
  echo "HOW TO READ IT: the J_NN rho per epoch should DROP toward <=1 as the"
  echo "penalty trains stability in, WHILE the val loss stays low (accuracy kept)."
  echo "If rho drops and val holds -> the honest certificate works; next step is"
  echo "a rollout stability check + then shrinking the network."
  echo
  echo "=== J_NN rho per epoch (max-per-window mean/p95/max) ==="
  echo "${RHO:-（none captured）}"
  echo
  echo "=== last epochs (train/val loss) ==="
  echo "${EPS:-（none captured）}"
  echo
  echo "epochs completed: ${NEPOCH:-0}/20   nan/abort markers: ${NANM:-0}"
  echo "exit line: ${EXITL:-（none — may have been killed）}"
  echo "checkpoints: ${BEST:-（none）}"
  echo "full log: $L"
} > "$MAILF"
echo "[jnn_ft_watch] emailed result to $MAILF at $(date -u +%FT%TZ)"
