#!/bin/bash
# postlap_eval_job.sh -- Sanaa standing order 2026-07-16 evening: the moment a
# lap training lands, run the eval diagnostics and email her the result
# directories. Submitted with -hold_jid <trainer>; CPU only (diagnostics rule).
# args: GEOM (fpc|cape)
#$ -S /bin/bash
#$ -q all.q
#$ -j y
#$ -cwd
set -uo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

G=${1:?usage: postlap_eval_job.sh fpc|cape}
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
B=$QG_ROOT/qg-sgs-closure
SP=$QG_ROOT/reporting/pending_mail
source "$QG_ROOT/qg-env-piff/bin/activate"
cd "$B/ml_closure"

CKPT="runs_piff/piff_${G}_gjs_lap/best.pt"
CONF="conf_piff_${G}_gjs_lap.yaml"
[ -f "$CKPT" ] || { echo "no best.pt -- trainer died before a checkpoint"; exit 1; }

echo "== eval_piff (a-priori eval package, S4)"
python -u eval_piff.py --ckpt "$CKPT" --config "$CONF" --device cpu \
    || echo "EVAL_PIFF_FAIL"
echo "== mean-prediction suite"
python -u diagnose_mean_prediction.py --ckpt "$CKPT" --config "$CONF" \
    --device cpu --report-run "lap_eval_${G}" || echo "MPD_FAIL"
echo "== error-tails suite"
python -u diagnose_error_tails.py --ckpt "$CKPT" --config "$CONF" \
    --device cpu --report-run "lap_tails_${G}" || echo "ETD_FAIL"

mkdir -p "$SP"
{
echo "To: ${QG_NOTIFY_EMAIL:-sanaamz@mit.edu}"
echo "Subject: [QG][LANDED][sgs-closure] lap $G training DONE -- eval diagnostics complete, directories inside"
echo
echo "PARAMETERS: piff_${G}_gjs_lap (warm from ylp75, +|lap omega| log1p ARD"
echo "feature); evals on best.pt, val split, filtered ylp75 targets, CPU."
echo
echo "RESULT DIRECTORIES:"
echo "  a-priori eval pkg : $B/ml_closure/runs_piff/piff_${G}_gjs_lap/eval/"
echo "  mean-prediction   : $B/ml_closure/runs_piff/piff_${G}_gjs_lap/mean_prediction_diag/"
echo "  error-tails       : $B/ml_closure/runs_piff/piff_${G}_gjs_lap/error_tails_diag/"
echo "  figures           : $B/ml_closure/pngs/{mean_prediction_diag,error_tails_diag}/piff_${G}_gjs_lap/"
echo "  git summaries     : reports/lap_eval_${G}/ + reports/lap_tails_${G}/ (pushed)"
echo
echo "BASELINE TO BEAT (ylp75): FPC R2 .83-.90 relRMSE .32-.41 / CAPE .89-.98."
echo
echo "NEXT: compare summary_all_members.csv vs the ylp75 tables (same suites,"
echo "same members); near-wall rel-error + sigma-at-events rerun is the"
echo "follow-up once you rule the headline numbers are worth it."
} > "$SP/$(date +%Y%m%dT%H%M%S)_lap_${G}_landed.mail"
echo "[postlap] done"
