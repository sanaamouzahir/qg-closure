#!/bin/bash
# submit_wallv2.sh - wallv2 SGS closure generation (Sanaa GO 2026-07-18):
# per geometry (fpc, cape) one chained unit
#
#   trainer  piff_train_job.sh --config conf_piff_<g>_gjs_wallv2.yaml
#            --run-name piff_<g>_gjs_wallv2        (ibgpu.q gpu=1; mirrors the
#            piff grid-point submission; digest start/done/fail wired inside
#            piff_train_job.sh)
#   -> CPU diagnostics on all.q via piff_tool_job.sh (diagnostics never run on
#      the GPU queue, Sanaa ruling 2026-07-15), -hold_jid on the trainer:
#        diagnose_mean_prediction.py   (r2 per member)
#        diagnose_error_tails.py       (worst0.1pct_SS_share + extreme_events.csv)
#      then, held on error_tails (needs its extreme_events.csv):
#        diagnose_sigma_at_events.py   x5 members (z_true_median)
#   -> gate_piff_events.py vs the ylp75 baseline run, held on ALL diagnostics,
#      -m ea mail. Exit 0 PASS / 2 PASS-conditional / 3 REGRESSED — a non-PASS
#      gate shows as a failed job + fail digest (intended: it must be loud).
#
# I18 NOTE (as submit_piff_grid.sh): LIVE/FINALIZE monitor units are wired by
# the submitting supervisor at fire time; this builder carries the work jobs.
# Day-mode submission via the I21c ssh sequence ONLY.
# Dry-run by default; pass --go to submit.

set -euo pipefail

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
BRANCH="$QG_ROOT/qg-sgs-closure"
ML="$BRANCH/ml_closure"
LOGS="$BRANCH/logs"
QG_NOTIFY_EMAIL="${QG_NOTIFY_EMAIL:-sanaamz@mit.edu}"

GEOMS=(fpc cape)
members_of() {
    case "$1" in
        fpc)  echo "FPC-const FPC-sine FPC-ramp FPC-ou FPC-telS-A" ;;
        cape) echo "FPCape-const FPCape-sine FPCape-ramp FPCape-ou FPCape-tel" ;;
        *)    echo "unknown geometry $1" >&2; return 1 ;;
    esac
}

GO=0
while [ $# -gt 0 ]; do
    case "$1" in
        --go) GO=1 ;;
        *) echo "ERROR: unknown arg '$1' (only --go accepted)" >&2; exit 2 ;;
    esac
    shift
done

# ---- preflight ----------------------------------------------------------- #
[ -f "$QG_ROOT/qg-env-piff/bin/activate" ] || { echo "ERROR: qg-env-piff venv missing" >&2; exit 1; }
for g in "${GEOMS[@]}"; do
    RN="piff_${g}_gjs_wallv2"
    BASE="piff_${g}_gjs_ylp75"
    for f in "$ML/conf_piff_${g}_gjs_wallv2.yaml" "$ML/gate_piff_events.py"; do
        [ -e "$f" ] || { echo "MISSING: $f" >&2; exit 1; }
    done
    # baseline gate inputs (old paths are results-tree symlinks — still valid)
    for d in "$ML/runs_piff/$BASE/error_tails_diag" \
             "$ML/runs_piff/$BASE/mean_prediction_diag"; do
        [ -e "$d" ] || { echo "MISSING baseline diagnostics: $d" >&2; exit 1; }
    done
    # guard on the run DIR, not best.pt: a partially-trained/interrupted dir
    # (no best.pt yet) must not be silently reused (G5 audit 2026-07-19)
    [ -d "$ML/runs_piff/$RN" ] && \
        { echo "EXISTS: runs_piff/$RN — refusing to clobber" >&2; exit 1; }
done
echo "[preflight] confs + gate + baseline diagnostics present; run dirs free"

if [ "$GO" -ne 1 ]; then
    echo "DRY RUN - per geometry: 1 GPU trainer (ibgpu.q gpu=1) -> 2 CPU diag"
    echo "(all.q) -> 5 sigma-at-events (all.q) -> 1 gate vs ylp75 (all.q, -m ea)."
    echo "Full qsub commands previewed below; re-run with --go to submit."
fi

# dry-run-aware qsub: previews the full command when GO=0 (G5 audit
# 2026-07-19, submit_piff_grid.sh run() parity); job-id capture gets "DRY"
qsub_go() {
    if [ "$GO" -eq 1 ]; then qsub "$@"; else
        echo "[DRY-RUN] qsub $*" >&2; echo "DRY"; fi
}

cd "$BRANCH"
mkdir -p "$LOGS"

for g in "${GEOMS[@]}"; do
    RN="piff_${g}_gjs_wallv2"
    CONF="conf_piff_${g}_gjs_wallv2.yaml"
    BASE="piff_${g}_gjs_ylp75"

    # ---- GPU trainer (mirrors the grid-point submission) ------------------ #
    TRAIN=$(qsub_go -terse -q ibgpu.q -l gpu=1 -N "pWv2_$g" -j y -cwd -V \
            -m ea -M "$QG_NOTIFY_EMAIL" \
            -o "$LOGS/pWv2_$g.\$JOB_ID.log" \
            scripts/sge/piff_train_job.sh --config "$CONF" --run-name "$RN")
    echo "trainer $RN: $TRAIN"

    # ---- CPU diagnostics (all.q), held on the trainer --------------------- #
    MP=$(qsub_go -terse -q all.q -N "pWv2mp_$g" -hold_jid "$TRAIN" -j y -cwd -V \
         -v QG_DIGEST_RUN="mean_prediction_wallv2_$g" \
         -o "$LOGS/pWv2mp_$g.\$JOB_ID.log" \
         scripts/sge/piff_tool_job.sh diagnose_mean_prediction.py \
         --ckpt "runs_piff/$RN/best.pt" --config "$CONF" --device cpu \
         --report-run "mean_prediction_wallv2_$g")
    ET=$(qsub_go -terse -q all.q -N "pWv2et_$g" -hold_jid "$TRAIN" -j y -cwd -V \
         -v QG_DIGEST_RUN="error_tails_wallv2_$g" \
         -o "$LOGS/pWv2et_$g.\$JOB_ID.log" \
         scripts/sge/piff_tool_job.sh diagnose_error_tails.py \
         --ckpt "runs_piff/$RN/best.pt" --config "$CONF" --device cpu \
         --report-run "error_tails_wallv2_$g")
    echo "diag $RN: mean_prediction $MP error_tails $ET"

    # ---- sigma-at-events per member, held on error_tails ------------------ #
    HOLD_ALL="$MP,$ET"
    for m in $(members_of "$g"); do
        SFX="${m#*-}"
        SIG=$(qsub_go -terse -q all.q -N "pWv2sg_${g}_${SFX}" -hold_jid "$ET" \
              -j y -cwd -V -v QG_DIGEST_RUN="sigma_events_wallv2_$g" \
              -o "$LOGS/pWv2sg_${g}_${SFX}.\$JOB_ID.log" \
              scripts/sge/piff_tool_job.sh diagnose_sigma_at_events.py \
              --ckpt "runs_piff/$RN/best.pt" --config "$CONF" \
              --member "$m" \
              --events "runs_piff/$RN/error_tails_diag/$m/extreme_events.csv" \
              --report-run "sigma_events_wallv2_$g")
        echo "sigma $m: $SIG"
        HOLD_ALL="$HOLD_ALL,$SIG"
    done

    # ---- event gate vs ylp75 baseline, held on everything ----------------- #
    GATE=$(qsub_go -terse -q all.q -N "pWv2gt_$g" -hold_jid "$HOLD_ALL" -j y -cwd -V \
           -m ea -M "$QG_NOTIFY_EMAIL" \
           -v QG_DIGEST_RUN="gate_wallv2_$g" \
           -o "$LOGS/pWv2gt_$g.\$JOB_ID.log" \
           scripts/sge/piff_tool_job.sh gate_piff_events.py \
           --new-run "runs_piff/$RN" --baseline-run "runs_piff/$BASE" \
           --expect-members "$(members_of "$g" | tr ' ' ',')" \
           --report-run "gate_wallv2_$g")
    echo "gate $RN vs $BASE: $GATE"
done

echo "[submit_wallv2] both geometry chains submitted."
