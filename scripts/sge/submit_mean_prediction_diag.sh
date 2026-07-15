#!/bin/bash
# submit_mean_prediction_diag.sh -- Sanaa order 2026-07-15: per-member MEAN-
# prediction diagnostics (R2/RMSE/signed extrema, Re-error, error-location,
# spatial+temporal autocorrelations, pred-vs-truth) on the ylp75 production
# candidates, FILTERED fields (the confs' gaussian_jonly_ylp75 variant).
# [fable-authored]
#
# One CPU job per geometry (fpc, cape) -- Sanaa ruling 2026-07-15: diagnostics
# NEVER run on the GPU queue; ibgpu.q is for training/simulation/builds only.
# Worker = piff_tool_job.sh; driver = ml_closure/diagnose_mean_prediction.py.
# CHARTER v1.4:
#   raw logs   -> <branch>/logs/ (I23a, never committed)
#   summaries  -> reports/mean_prediction_diag_<geom>/ (pushed by the driver
#                 via diagnostics/digest_writer.py, I22b/I23b -- phone-readable)
#   heavy figs -> ml_closure/pngs/mean_prediction_diag/<model>/<member>/
#   yaml/csv   -> runs_piff/<model>/mean_prediction_diag/<member>/
# Diagnostics are GREEN-tier (charter sec. 1); no monitor unit is required
# (G5's three-job rule binds TRAINING submissions).
#
# DAY-MODE submission path (I21c), one bounded call from the local agent:
#   ssh mseas "cd /gdata/.../QG-closure/qg-sgs-closure && git pull --rebase && \
#              scripts/sge/submit_mean_prediction_diag.sh"
set -euo pipefail

# mseas default git (1.8.3.1) cannot parse linked worktrees; the 2.9.2 at
# /opt/rocks/bin can (2026-07-15 incident). Harmless no-op elsewhere.
[[ -d /opt/rocks/bin ]] && export PATH=/opt/rocks/bin:$PATH

BRANCH=$(git rev-parse --show-toplevel)
EMAIL=${QG_NOTIFY_EMAIL:-sanaamz@mit.edu}
mkdir -p "$BRANCH/logs"
cd "$BRANCH/ml_closure"

SMOKE=${1:-}   # pass --smoke for a 6-frame-per-member dry run first
EXTRA=()
[[ "$SMOKE" == "--smoke" ]] && EXTRA=(--max-frames 6)

for GEOM in fpc cape; do
    CKPT="runs_piff/piff_${GEOM}_gjs_ylp75/best.pt"
    CONF="conf_piff_${GEOM}_gjs_ylp75.yaml"
    [[ -e "$CKPT" ]] || { echo "MISSING $CKPT" >&2; exit 1; }
    [[ -e "$CONF" ]] || { echo "MISSING $CONF" >&2; exit 1; }
    JID=$(qsub -terse -q all.q -m ea -M "$EMAIL" \
        -N "mpd_${GEOM}" -j y \
        -o "$BRANCH/logs/mpd_${GEOM}.\$JOB_ID.log" -cwd -V \
        -v "QG_DIGEST_RUN=mean_prediction_diag_${GEOM},OMP_NUM_THREADS=8,MKL_NUM_THREADS=8,OPENBLAS_NUM_THREADS=8" \
        ../scripts/sge/piff_tool_job.sh diagnose_mean_prediction.py \
        --ckpt "$CKPT" --config "$CONF" --split val --device cpu \
        --report-run "mean_prediction_diag_${GEOM}" ${EXTRA[@]+"${EXTRA[@]}"})
    JID=${JID%%.*}
    echo "mpd_${GEOM}: job $JID  raw log $BRANCH/logs/mpd_${GEOM}.$JID.log"
done
echo "summaries land in reports/mean_prediction_diag_{fpc,cape}/ (git, pushed)"
echo "REMINDER: send [QG][SUBMIT][log] carrying both job ids (I12/6.3)."
