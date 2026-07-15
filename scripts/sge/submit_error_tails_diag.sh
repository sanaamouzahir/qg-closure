#!/bin/bash
# submit_error_tails_diag.sh -- Sanaa order 2026-07-15 (follow-up): WHERE the
# huge errors live, HOW MANY pixels carry them, zero-closure padding check,
# Re-constant nan explanation. One CPU job per geometry (diagnostics never on
# the GPU queue -- 2026-07-15 ruling). [fable-authored]
#
# Same contract as submit_mean_prediction_diag.sh: raw logs -> <branch>/logs/
# (I23a); summaries -> reports/error_tails_{fpc,cape}/ pushed by the driver
# (I23b); QG_DIGEST_RUN start/fail hook in piff_tool_job.sh. GREEN diagnostics.
#
# DAY-MODE (I21c): ssh mseas "cd <branch> && git pull --rebase && \
#     scripts/sge/submit_error_tails_diag.sh"
set -euo pipefail
[[ -d /opt/rocks/bin ]] && export PATH=/opt/rocks/bin:$PATH

BRANCH=$(git rev-parse --show-toplevel)
EMAIL=${QG_NOTIFY_EMAIL:-sanaamz@mit.edu}
mkdir -p "$BRANCH/logs"
cd "$BRANCH/ml_closure"

SMOKE=${1:-}
EXTRA=()
[[ "$SMOKE" == "--smoke" ]] && EXTRA=(--max-frames 6)

for GEOM in fpc cape; do
    CKPT="runs_piff/piff_${GEOM}_gjs_ylp75/best.pt"
    CONF="conf_piff_${GEOM}_gjs_ylp75.yaml"
    [[ -e "$CKPT" ]] || { echo "MISSING $CKPT" >&2; exit 1; }
    JID=$(qsub -terse -q all.q -m ea -M "$EMAIL" \
        -N "etd_${GEOM}" -j y \
        -o "$BRANCH/logs/etd_${GEOM}.\$JOB_ID.log" -cwd -V \
        -v "QG_DIGEST_RUN=error_tails_${GEOM},OMP_NUM_THREADS=8,MKL_NUM_THREADS=8,OPENBLAS_NUM_THREADS=8" \
        ../scripts/sge/piff_tool_job.sh diagnose_error_tails.py \
        --ckpt "$CKPT" --config "$CONF" --split val --device cpu \
        --report-run "error_tails_${GEOM}" ${EXTRA[@]+"${EXTRA[@]}"})
    JID=${JID%%.*}
    echo "etd_${GEOM}: job $JID  raw log $BRANCH/logs/etd_${GEOM}.$JID.log"
done
echo "summaries land in reports/error_tails_{fpc,cape}/ (git, pushed)"
echo "REMINDER: send [QG][SUBMIT][log] carrying both job ids (I12/6.3)."
