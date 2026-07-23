#!/bin/bash
# submit_mtgp.sh - MULTITASK (coregionalized) SGS closure (Sanaa GO 2026-07-21):
# per geometry (fpc, cape) ONE chained unit, ONE trainer (not two).
#
# MOTIVATION (measured, not assumed). The two-band approach (separate near +
# far specialists blended by a partition of unity) SOLVED the near-wall band
# (r2 ~.98) but REGRESSED the far field (far-only specialist worse than the
# single lap model -- telS-A far 48% err vs lap 30%), because band restriction
# removed the cross-region context. The fix: ONE GP with TWO CORRELATED TASKS
# trained on the WHOLE field -- task 0 = near-wall closure (sdf <= 1.25 D),
# task 1 = far-field closure (sdf > 1.25 D) -- so an IndexKernel (ICM)
# coregionalization matrix gives each task its own OUTPUT SCALE (with per-task
# y-standardization handling the ~40x near/far amplitude gap) AND a LEARNED
# cross-task correlation, so far predictions are informed by near-wall data
# through the off-diagonal. Everything else is the plain `lap` recipe.
#
# JOB GRAPH per geometry g (GEOM = flow_past_cylinder | flow_past_cape,
# MODEL = piff_<g>_gjs_mtgp, RES = results/$GEOM/$MODEL):
#
#   [GPU ibgpu.q gpu=1]  TR  piff_train_job.sh --config conf_piff_<g>_gjs_mtgp.yaml
#     with auto-wired LIVE + FINALIZE piff monitors (I18 / NaN policy - NEVER
#     optional; train_piff.py additionally hard-aborts in-process on 2
#     consecutive non-finite epochs -> NAN_ABORT.txt + exit 9)
#   -hold_jid TR   (all all.q, --device cpu, ckpt = runs_piff/$MODEL/best.pt,
#                   eval config = conf_piff_<g>_gjs_mtgp.yaml: whole field, the
#                   model derives each pixel's task from input channel 3)
#     MP    diagnose_mean_prediction.py  --outdir $RES/mean_prediction
#     ET    diagnose_error_tails.py      --outdir $RES/error_tails
#     PFA   plot_fields_assess.py        --per-member 2  --outdir $RES/eval_assess
#     CBM   compare_band_metrics.py      --per-member 2  --out-csv $RES/band_metrics/mtgp.csv
#     RPL   replot_eval_fields.py        --per-member 3  --outdir $RES/eval
#   -hold_jid ET    SIG x5 members  diagnose_sigma_at_events.py (consumes ET's extreme_events.csv)
#   (independent, no hold) CBM_{ylp75,lap,wallv2,blended}: the SAME band metrics
#     for the baselines, each under ITS OWN trained config/ckpt ->
#     $RES/band_metrics/{ylp75,lap,wallv2,blended}.csv (table columns; absent = n/a)
#   -hold_jid EVERYTHING
#     GATE  mtgp_gate_job.sh: gate_piff_events.py vs ylp75 (--expect-members)
#           + make_band_table.py (verdict + table: rows NEAR/FAR/ALL x columns
#           ylp75/lap/wallv2/blended/MTGP) + a [QG][REPORT] mail carrying the
#           full report contract, -m ea
#   -hold_jid GATE
#     MIG   migrate_results_tree.sh - results-standard sweep-up
#
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
geomdir_of() {
    case "$1" in
        fpc)  echo "flow_past_cylinder" ;;
        cape) echo "flow_past_cape" ;;
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
for f in "$ML/gate_piff_events.py" "$ML/make_band_table.py" \
         "$ML/model_piff.py" "$ML/piff_model_loader.py" \
         "$ML/compare_band_metrics.py" "$ML/train_piff.py" \
         "$BRANCH/scripts/sge/mtgp_gate_job.sh" \
         "$BRANCH/scripts/sge/piff_train_job.sh" \
         "$BRANCH/scripts/sge/piff_tool_job.sh" \
         "$BRANCH/scripts/sge/piff_monitor_job.sh" \
         "$BRANCH/scripts/sge/migrate_results_tree.sh"; do
    [ -e "$f" ] || { echo "MISSING: $f" >&2; exit 1; }
done
for g in "${GEOMS[@]}"; do
    [ -e "$ML/conf_piff_${g}_gjs_mtgp.yaml" ] || \
        { echo "MISSING: $ML/conf_piff_${g}_gjs_mtgp.yaml" >&2; exit 1; }
    # anti-clobber on the RUN DIR (a partially-trained dir must not be reused)
    [ -d "$ML/runs_piff/piff_${g}_gjs_mtgp" ] && \
        { echo "EXISTS: runs_piff/piff_${g}_gjs_mtgp - refusing to clobber" >&2; exit 1; }
    # baseline gate inputs (consumed by the GATE job hours later; warn only)
    for d in "$ML/runs_piff/piff_${g}_gjs_ylp75/error_tails_diag" \
             "$ML/runs_piff/piff_${g}_gjs_ylp75/mean_prediction_diag"; do
        [ -e "$d" ] || echo "WARNING: baseline diagnostics absent ($d) - regenerate before the gate fires" >&2
    done
    # table baselines are OPTIONAL: make_band_table prints n/a rows for missing CSVs
    for b in ylp75 lap wallv2; do
        [ -f "$ML/runs_piff/piff_${g}_gjs_${b}/best.pt" ] || \
            echo "[preflight] WARNING: runs_piff/piff_${g}_gjs_${b}/best.pt absent - the '$b' column of the $g band table will read n/a"
    done
    [ -f "$ML/runs_piff/piff_${g}_gjs_blended/blended.pt" ] || \
        echo "[preflight] WARNING: runs_piff/piff_${g}_gjs_blended/blended.pt absent - the 'blended' column of the $g band table will read n/a"
done
echo "[preflight] confs + tools present; run dirs free"

if [ "$GO" -ne 1 ]; then
    echo "DRY RUN - per geometry: 1 GPU trainer (ibgpu.q gpu=1, +2 monitors) ->"
    echo "5 mtgp diagnostics + up to 4 baseline band metrics (all.q) -> 5"
    echo "sigma-at-events (all.q) -> gate+table+mail (all.q, -m ea) -> results-"
    echo "tree migration. Full qsub commands previewed below; re-run with --go."
fi

# dry-run-aware qsub: previews the full command when GO=0
qsub_go() {
    if [ "$GO" -eq 1 ]; then qsub "$@"; else
        echo "[DRY-RUN] qsub $*" >&2; echo "DRY"; fi
}

cd "$BRANCH"
mkdir -p "$LOGS"

for g in "${GEOMS[@]}"; do
    GEOM=$(geomdir_of "$g")
    MODEL="piff_${g}_gjs_mtgp"
    BASE="piff_${g}_gjs_ylp75"
    CONF="conf_piff_${g}_gjs_mtgp.yaml"
    EVALCONF="$CONF"                              # whole field; task from channel 3
    CKPT="runs_piff/$MODEL/best.pt"
    RES="results/$GEOM/$MODEL"

    # ---- GPU trainer: the single multitask model --------------------------- #
    TR=$(qsub_go -terse -q ibgpu.q -l gpu=1 -N "pMt_$g" -j y -cwd -V \
         -m ea -M "$QG_NOTIFY_EMAIL" \
         -o "$LOGS/pMt_$g.\$JOB_ID.log" \
         scripts/sge/piff_train_job.sh --config "$CONF" --run-name "$MODEL")
    echo "trainer $MODEL: $TR"

    # I18 monitors wired HERE, never "by the supervisor at fire time" (the
    # 2026-07-19 wallv2 NaN burn is why). LIVE catches nan/inversion/stall while
    # running; FINALIZE postmortems after exit. train_piff.py additionally
    # hard-aborts in-process on 2 consecutive non-finite epochs (exit 9).
    TLOG="$LOGS/pMt_$g.$TR.log"
    MONL=$(qsub_go -terse -q all.q -N "pMtmL_$g" -j y -cwd -V \
           -o "$LOGS/pMtmL_$g.\$JOB_ID.log" \
           scripts/sge/piff_monitor_job.sh "$TLOG" "$MODEL" "$TR")
    MONF=$(qsub_go -terse -q all.q -N "pMtmF_$g" -hold_jid "$TR" \
           -v QG_MONITOR_FINALIZE=1 -j y -cwd -V \
           -o "$LOGS/pMtmF_$g.\$JOB_ID.log" \
           scripts/sge/piff_monitor_job.sh "$TLOG" "$MODEL" "$TR")
    echo "monitors $MODEL: live $MONL finalize $MONF"

    # ---- diagnostics ladder on the multitask model (all.q, cpu) ------------ #
    MP=$(qsub_go -terse -q all.q -N "pMtmp_$g" -hold_jid "$TR" -j y -cwd -V \
         -v QG_DIGEST_RUN="mean_prediction_mtgp_$g" \
         -o "$LOGS/pMtmp_$g.\$JOB_ID.log" \
         scripts/sge/piff_tool_job.sh diagnose_mean_prediction.py \
         --ckpt "$CKPT" --config "$EVALCONF" --device cpu \
         --outdir "$RES/mean_prediction" --fig-dir "$RES/mean_prediction" \
         --report-run "mean_prediction_mtgp_$g")
    ET=$(qsub_go -terse -q all.q -N "pMtet_$g" -hold_jid "$TR" -j y -cwd -V \
         -v QG_DIGEST_RUN="error_tails_mtgp_$g" \
         -o "$LOGS/pMtet_$g.\$JOB_ID.log" \
         scripts/sge/piff_tool_job.sh diagnose_error_tails.py \
         --ckpt "$CKPT" --config "$EVALCONF" --device cpu \
         --outdir "$RES/error_tails" --fig-dir "$RES/error_tails" \
         --report-run "error_tails_mtgp_$g")
    echo "diag $MODEL: mean_prediction $MP error_tails $ET"

    PFA=$(qsub_go -terse -q all.q -N "pMtpfa_$g" -hold_jid "$TR" -j y -cwd -V \
          -v QG_DIGEST_RUN="fields_assess_mtgp_$g" \
          -o "$LOGS/pMtpfa_$g.\$JOB_ID.log" \
          scripts/sge/piff_tool_job.sh plot_fields_assess.py \
          --ckpt "$CKPT" --config "$EVALCONF" --device cpu \
          --per-member 2 --outdir "$RES/eval_assess")
    CBM=$(qsub_go -terse -q all.q -N "pMtcbm_$g" -hold_jid "$TR" -j y -cwd -V \
          -v QG_DIGEST_RUN="band_metrics_mtgp_$g" \
          -o "$LOGS/pMtcbm_$g.\$JOB_ID.log" \
          scripts/sge/piff_tool_job.sh compare_band_metrics.py \
          --ckpt "$CKPT" --config "$EVALCONF" --device cpu \
          --per-member 2 --model-tag mtgp \
          --out-csv "$RES/band_metrics/mtgp.csv")
    RPL=$(qsub_go -terse -q all.q -N "pMtrpl_$g" -hold_jid "$TR" -j y -cwd -V \
          -v QG_DIGEST_RUN="replot_mtgp_$g" \
          -o "$LOGS/pMtrpl_$g.\$JOB_ID.log" \
          scripts/sge/piff_tool_job.sh replot_eval_fields.py \
          --ckpt "$CKPT" --config "$EVALCONF" --device cpu \
          --per-member 3 --outdir "$RES/eval")
    echo "diag $MODEL: fields_assess $PFA band_metrics $CBM replot $RPL"

    HOLD_ALL="$MP,$ET,$PFA,$CBM,$RPL"

    # ---- sigma-at-events per member, held on error_tails ------------------- #
    for m in $(members_of "$g"); do
        SFX="${m#*-}"
        SIG=$(qsub_go -terse -q all.q -N "pMts${g}${SFX}" -hold_jid "$ET" \
              -j y -cwd -V -v QG_DIGEST_RUN="sigma_events_mtgp_$g" \
              -o "$LOGS/pMts${g}${SFX}.\$JOB_ID.log" \
              scripts/sge/piff_tool_job.sh diagnose_sigma_at_events.py \
              --ckpt "$CKPT" --config "$EVALCONF" \
              --member "$m" \
              --events "$RES/error_tails/$m/extreme_events.csv" \
              --outdir "$RES/sigma_at_events" \
              --report-run "sigma_events_mtgp_$g")
        echo "sigma $m: $SIG"
        HOLD_ALL="$HOLD_ALL,$SIG"
    done

    # ---- baseline band metrics for the table ------------------------------- #
    # ylp75/lap/wallv2 each under ITS OWN trained config; blended under the lap
    # config (NO data.band, ungated) via its blended.pt handle. Missing ckpts
    # were warned about in preflight and become n/a rows in the table.
    for b in ylp75 lap wallv2; do
        BCK="runs_piff/piff_${g}_gjs_${b}/best.pt"
        [ -f "$ML/$BCK" ] || continue
        BJ=$(qsub_go -terse -q all.q -N "pMtb${g}${b}" -j y -cwd -V \
             -v QG_DIGEST_RUN="band_metrics_${b}_$g" \
             -o "$LOGS/pMtb${g}${b}.\$JOB_ID.log" \
             scripts/sge/piff_tool_job.sh compare_band_metrics.py \
             --ckpt "$BCK" --config "conf_piff_${g}_gjs_${b}.yaml" --device cpu \
             --per-member 2 --model-tag "$b" \
             --out-csv "$RES/band_metrics/${b}.csv")
        echo "baseline band metrics $b: $BJ"
        HOLD_ALL="$HOLD_ALL,$BJ"
    done
    BLND="runs_piff/piff_${g}_gjs_blended/blended.pt"
    if [ -f "$ML/$BLND" ]; then
        BJ=$(qsub_go -terse -q all.q -N "pMtb${g}blend" -j y -cwd -V \
             -v QG_DIGEST_RUN="band_metrics_blended_$g" \
             -o "$LOGS/pMtb${g}blend.\$JOB_ID.log" \
             scripts/sge/piff_tool_job.sh compare_band_metrics.py \
             --ckpt "$BLND" --config "conf_piff_${g}_gjs_lap.yaml" --device cpu \
             --per-member 2 --model-tag blended \
             --out-csv "$RES/band_metrics/blended.csv")
        echo "baseline band metrics blended: $BJ"
        HOLD_ALL="$HOLD_ALL,$BJ"
    fi

    # ---- gate + table + [QG][REPORT] mail, held on everything -------------- #
    GATE=$(qsub_go -terse -q all.q -N "pMtgt_$g" -hold_jid "$HOLD_ALL" -j y -cwd -V \
           -m ea -M "$QG_NOTIFY_EMAIL" \
           -v QG_DIGEST_RUN="gate_mtgp_$g" \
           -o "$LOGS/pMtgt_$g.\$JOB_ID.log" \
           scripts/sge/mtgp_gate_job.sh "$g" "$GEOM" "$MODEL" "$BASE" \
           "$(members_of "$g" | tr ' ' ',')")
    echo "gate $MODEL vs $BASE: $GATE"

    # ---- results-standard sweep-up ---------------------------------------- #
    MIG=$(qsub_go -terse -q all.q -N "pMtmig_$g" -hold_jid "$GATE" -j y -cwd -V \
          -o "$LOGS/pMtmig_$g.\$JOB_ID.log" \
          scripts/sge/migrate_results_tree.sh)
    echo "results migration ($g): $MIG"
done

echo "[submit_mtgp] both geometry chains submitted."
