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

G=${1:?usage: postlap_eval_job.sh fpc|cape [snapshot]}
SNAP=${2:-}
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
B=$QG_ROOT/qg-sgs-closure
SP=$QG_ROOT/reporting/pending_mail
source "$QG_ROOT/qg-env-piff/bin/activate"
cd "$B/ml_closure"

RD="runs_piff/piff_${G}_gjs_lap"
CKPT="$RD/best.pt"
CONF="conf_piff_${G}_gjs_lap.yaml"
[ -f "$CKPT" ] || { echo "no best.pt -- trainer died before a checkpoint"; exit 1; }
TAG=""
MAXF=""
if [ -n "$SNAP" ]; then
    # mid-training snapshot (Sanaa 2026-07-16): copy best.pt so the live
    # trainer cannot race the read; outputs suffixed to keep the final
    # post-training evals collision-free
    TAG="_snap$(date +%H%M)"
    cp "$CKPT" "$RD/best${TAG}.pt"
    CKPT="$RD/best${TAG}.pt"
fi
if [ "$SNAP" = "quick" ]; then
    # QUICK LOOK (Sanaa 2026-07-16 evening): 2 members, 8 frames each --
    # minutes not hours; the full-resolution pass runs separately.
    TAG="${TAG}q"
    MAXF="--max-frames 8"
    QCONF="conf_piff_${G}_gjs_lap_quick.yaml"
    python - "$CONF" "$QCONF" "$G" <<'PY'
import sys, yaml
conf, out, g = sys.argv[1:4]
c = yaml.safe_load(open(conf))
keep = {'fpc': ('FPC-const', 'FPC-sine'),
        'cape': ('FPCape-tel', 'FPCape-sine')}[g]
c['data']['runs'] = [r for r in c['data']['runs']
                     if any(r.rstrip('/').endswith(k) for k in keep)]
yaml.safe_dump(c, open(out, 'w'), sort_keys=False)
print('[quick] members:', [r.split('/')[-1] for r in c['data']['runs']])
PY
    CONF="$QCONF"
fi

echo "== eval_piff (a-priori eval package, S4)"
python -u eval_piff.py --ckpt "$CKPT" --config "$CONF" --device cpu \
    --outdir "$RD/eval${TAG}" || echo "EVAL_PIFF_FAIL"
echo "== mean-prediction suite"
python -u diagnose_mean_prediction.py --ckpt "$CKPT" --config "$CONF" \
    --device cpu $MAXF --outdir "$RD/mean_prediction_diag${TAG}" \
    --fig-dir "pngs/mean_prediction_diag/piff_${G}_gjs_lap${TAG}" \
    --report-run "lap_eval_${G}${TAG}" || echo "MPD_FAIL"
echo "== error-tails suite"
python -u diagnose_error_tails.py --ckpt "$CKPT" --config "$CONF" \
    --device cpu $MAXF --outdir "$RD/error_tails_diag${TAG}" \
    --fig-dir "pngs/error_tails_diag/piff_${G}_gjs_lap${TAG}" \
    --report-run "lap_tails_${G}${TAG}" || echo "ETD_FAIL"

mkdir -p "$SP"
{
if [ -n "$SNAP" ]; then
  echo "To: ${QG_NOTIFY_EMAIL:-sanaamz@mit.edu}"
  echo "Subject: [QG][MONITOR][sgs-closure] lap $G MID-TRAINING SNAPSHOT evals done -- directories inside"
else
  echo "To: ${QG_NOTIFY_EMAIL:-sanaamz@mit.edu}"
  echo "Subject: [QG][LANDED][sgs-closure] lap $G training DONE -- eval diagnostics complete, directories inside"
fi
echo
echo "PARAMETERS: piff_${G}_gjs_lap (warm from ylp75, +|lap omega| log1p ARD"
echo "feature); evals on best${TAG}.pt, val split, filtered ylp75 targets, CPU."
echo
echo "RESULT DIRECTORIES (fields truth/pred/error panels are in eval${TAG};"
echo "biggest-error locations + exceedance-count distributions in error_tails):"
echo "  a-priori eval pkg : $B/ml_closure/$RD/eval${TAG}/"
echo "  mean-prediction   : $B/ml_closure/$RD/mean_prediction_diag${TAG}/"
echo "  error-tails       : $B/ml_closure/$RD/error_tails_diag${TAG}/"
echo "  figures           : $B/ml_closure/pngs/{mean_prediction_diag,error_tails_diag}/piff_${G}_gjs_lap${TAG}/"
echo "  git summaries     : reports/lap_eval_${G}${TAG}/ + reports/lap_tails_${G}${TAG}/ (pushed)"
echo
echo "BASELINE TO BEAT (ylp75): FPC R2 .83-.90 relRMSE .32-.41 / CAPE .89-.98."
echo
echo "NEXT: compare summary_all_members.csv vs the ylp75 tables (same suites,"
echo "same members); near-wall rel-error + sigma-at-events rerun is the"
echo "follow-up once you rule the headline numbers are worth it."
} > "$SP/$(date +%Y%m%dT%H%M%S)_lap_${G}_landed.mail"
echo "[postlap] done"
