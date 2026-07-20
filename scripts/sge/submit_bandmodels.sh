#!/bin/bash
# submit_bandmodels.sh - TWO-BAND SGS closure generation (Sanaa GO 2026-07-20):
# per geometry (fpc, cape) one chained unit.
#
# MOTIVATION (measured, not assumed): one model cannot serve both regions.
# Near-wall targets have RMS ~3-6.6, far-field ~0.08-0.42 (40x gap), while a
# single SVGP has ONE y-standardization, ONE inducing set and ONE ARD geometry,
# so capacity goes to whichever regime dominates. Measured trade, same
# frames/truth/scale: wallv2 NEAR r2 .954 err 21% / FAR .874 35%; lap NEAR
# .907 30% / FAR .934 25%. Goal: beat BOTH specialists simultaneously.
#
# BOTH specialists use the plain `lap` recipe and differ ONLY by data.band, so
# band-vs-pooled is the one variable under test (spec revision 2026-07-20:
# tail-bias and the wall gate are redundant under band restriction; their code
# paths stay guarded and default-off for a later ablation).
#
# JOB GRAPH per geometry g (GEOM = flow_past_cylinder | flow_past_cape,
# MODEL = piff_<g>_gjs_blended, RES = results/$GEOM/$MODEL):
#
#   [GPU ibgpu.q gpu=1]  TR_near  piff_train_job.sh --config conf_piff_<g>_gjs_bandnear.yaml
#                        TR_far   piff_train_job.sh --config conf_piff_<g>_gjs_bandfar.yaml
#     each with auto-wired LIVE + FINALIZE piff monitors (I18 / NaN policy -
#     NEVER optional; train_piff.py additionally hard-aborts in-process on 2
#     consecutive non-finite epochs -> NAN_ABORT.txt + exit 9)
#   -hold_jid TR_near,TR_far
#     [all.q] MAN   write_blended_manifest.py -> runs_piff/$MODEL/{blended_manifest.yaml,blended.pt}
#   -hold_jid MAN   (all all.q, --device cpu, blended ckpt = runs_piff/$MODEL/blended.pt,
#                    eval config = conf_piff_<g>_gjs_lap.yaml: NO data.band, no wall gate)
#     MP    diagnose_mean_prediction.py  --outdir $RES/mean_prediction
#     ET    diagnose_error_tails.py      --outdir $RES/error_tails
#     PFA   plot_fields_assess.py        --per-member 2  --outdir $RES/eval_assess
#     CBM   compare_band_metrics.py      --per-member 2  --out-csv $RES/band_metrics/blended.csv
#     RPL   replot_eval_fields.py        --per-member 3  --outdir $RES/eval
#   -hold_jid ET    SIG x5 members  diagnose_sigma_at_events.py (consumes ET's extreme_events.csv)
#   (independent, no hold) CBM_ylp75 / CBM_lap / CBM_wallv2: the SAME band metrics
#     for the three baselines, each under ITS OWN trained config ->
#     $RES/band_metrics/{ylp75,lap,wallv2}.csv (table columns; absent = n/a column)
#   -hold_jid EVERYTHING
#     GATE  bandmodels_gate_job.sh: gate_piff_events.py vs ylp75 (--expect-members)
#           + make_band_table.py (plain-English verdict + the table: rows
#           NEAR/FAR/ALL x columns ylp75/lap/wallv2/BLENDED) + a [QG][REPORT] mail
#           carrying the full report contract: verdict in words, the TABLE INLINE,
#           an absolute-path artifact list by suite, and the explicit results-
#           standard confirmation (2026-07-19 + 2026-07-20 contracts), -m ea
#   -hold_jid GATE
#     MIG   migrate_results_tree.sh - results-standard sweep-up
#
# Results standard (Sanaa 2026-07-17, "everything in one place"):
#   ml_closure/results/<geometry>/<model>/<suite>/<member>/ - set by explicit
#   --outdir everywhere the tools support it; migrate_results_tree.sh sweeps up
#   the rest (incl. the two specialists' own artifacts) and does the member-dir
#   renaming to plain modulation names.
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
         "$ML/write_blended_manifest.py" "$ML/model_piff_blended.py" \
         "$ML/piff_model_loader.py" "$ML/compare_band_metrics.py" \
         "$BRANCH/scripts/sge/bandmodels_gate_job.sh" \
         "$BRANCH/scripts/sge/migrate_results_tree.sh"; do
    [ -e "$f" ] || { echo "MISSING: $f" >&2; exit 1; }
done
for g in "${GEOMS[@]}"; do
    BASE="piff_${g}_gjs_ylp75"
    for f in "$ML/conf_piff_${g}_gjs_bandnear.yaml" \
             "$ML/conf_piff_${g}_gjs_bandfar.yaml" \
             "$ML/conf_piff_${g}_gjs_lap.yaml"; do
        [ -e "$f" ] || { echo "MISSING: $f" >&2; exit 1; }
    done
    # baseline gate inputs (old paths are results-tree symlinks — still valid)
    for d in "$ML/runs_piff/$BASE/error_tails_diag" \
             "$ML/runs_piff/$BASE/mean_prediction_diag"; do
        # WARN, do not block (2026-07-20): the baselines are consumed only by
        # the GATE job hours later, and gate_piff_events.py already hard-fails
        # REGRESSED on missing artifacts. Blocking here would idle 4 GPUs while
        # CPU diagnostics regenerate in parallel.
        [ -e "$d" ] || echo "WARNING: baseline diagnostics absent ($d) � regenerate before the gate fires" >&2
    done
    # anti-clobber on the RUN DIRS, not best.pt: a partially-trained/interrupted
    # dir (no best.pt yet) must not be silently reused (G5 audit 2026-07-19)
    for RN in "piff_${g}_gjs_bandnear" "piff_${g}_gjs_bandfar" \
              "piff_${g}_gjs_blended"; do
        [ -d "$ML/runs_piff/$RN" ] && \
            { echo "EXISTS: runs_piff/$RN — refusing to clobber" >&2; exit 1; }
    done
    # table baselines are OPTIONAL: make_band_table prints n/a rows for missing
    # CSVs and never fails on them — warn, never block
    for b in ylp75 lap wallv2; do
        [ -f "$ML/runs_piff/piff_${g}_gjs_${b}/best.pt" ] || \
            echo "[preflight] WARNING: runs_piff/piff_${g}_gjs_${b}/best.pt absent — the '$b' column of the $g band table will read n/a"
    done
done
echo "[preflight] confs + tools + baseline diagnostics present; run dirs free"

if [ "$GO" -ne 1 ]; then
    echo "DRY RUN - per geometry: 2 GPU trainers (ibgpu.q gpu=1, each +2 monitors)"
    echo "-> 1 manifest job (all.q) -> 5 blended diagnostics + 3 baseline band"
    echo "metrics (all.q) -> 5 sigma-at-events (all.q) -> gate+table+mail (all.q,"
    echo "-m ea) -> results-tree migration. Full qsub commands previewed below;"
    echo "re-run with --go to submit."
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
    GEOM=$(geomdir_of "$g")
    NEAR_RN="piff_${g}_gjs_bandnear"
    FAR_RN="piff_${g}_gjs_bandfar"
    MODEL="piff_${g}_gjs_blended"
    BASE="piff_${g}_gjs_ylp75"
    EVALCONF="conf_piff_${g}_gjs_lap.yaml"       # NO data.band, no wall gate
    CKPT="runs_piff/$MODEL/blended.pt"
    RES="results/$GEOM/$MODEL"

    # ---- GPU trainers: the two band specialists ---------------------------- #
    TR_IDS=""
    for band in near far; do
        RN="piff_${g}_gjs_band${band}"
        CONF="conf_piff_${g}_gjs_band${band}.yaml"
        TID=$(qsub_go -terse -q ibgpu.q -l gpu=1 -N "pBnd${band}_$g" -j y -cwd -V \
              -m ea -M "$QG_NOTIFY_EMAIL" \
              -o "$LOGS/pBnd${band}_$g.\$JOB_ID.log" \
              scripts/sge/piff_train_job.sh --config "$CONF" --run-name "$RN")
        echo "trainer $RN: $TID"

        # I18 monitors wired HERE, never "by the supervisor at fire time" —
        # the 2026-07-19 wallv2 NaN burn (100 ep x 2 GPUs, no monitor
        # attached) is why. LIVE catches nan/inversion/stall while running;
        # FINALIZE postmortems after exit. train_piff.py additionally hard-
        # aborts in-process on 2 consecutive non-finite epochs (exit 9).
        TLOG="$LOGS/pBnd${band}_$g.$TID.log"
        MONL=$(qsub_go -terse -q all.q -N "pBmL${band}_$g" -j y -cwd -V \
               -o "$LOGS/pBmL${band}_$g.\$JOB_ID.log" \
               scripts/sge/piff_monitor_job.sh "$TLOG" "$RN" "$TID")
        MONF=$(qsub_go -terse -q all.q -N "pBmF${band}_$g" -hold_jid "$TID" \
               -v QG_MONITOR_FINALIZE=1 -j y -cwd -V \
               -o "$LOGS/pBmF${band}_$g.\$JOB_ID.log" \
               scripts/sge/piff_monitor_job.sh "$TLOG" "$RN" "$TID")
        echo "monitors $RN: live $MONL finalize $MONF"
        TR_IDS="${TR_IDS:+$TR_IDS,}$TID"
    done

    # ---- pair the specialists into the blended handle ---------------------- #
    MAN=$(qsub_go -terse -q all.q -N "pBndman_$g" -hold_jid "$TR_IDS" -j y -cwd -V \
          -v QG_DIGEST_RUN="blend_manifest_$g" \
          -o "$LOGS/pBndman_$g.\$JOB_ID.log" \
          scripts/sge/piff_tool_job.sh write_blended_manifest.py \
          --near "runs_piff/$NEAR_RN/best.pt" --far "runs_piff/$FAR_RN/best.pt" \
          --geometry "$g" --outdir "runs_piff/$MODEL" \
          --overlap-lo 1.0 --overlap-hi 1.5 --eval-config "$EVALCONF")
    echo "manifest $MODEL: $MAN"

    # ---- diagnostics ladder on the BLENDED model (all.q, cpu) -------------- #
    MP=$(qsub_go -terse -q all.q -N "pBndmp_$g" -hold_jid "$MAN" -j y -cwd -V \
         -v QG_DIGEST_RUN="mean_prediction_band_$g" \
         -o "$LOGS/pBndmp_$g.\$JOB_ID.log" \
         scripts/sge/piff_tool_job.sh diagnose_mean_prediction.py \
         --ckpt "$CKPT" --config "$EVALCONF" --device cpu \
         --outdir "$RES/mean_prediction" --fig-dir "$RES/mean_prediction" \
         --report-run "mean_prediction_band_$g")
    ET=$(qsub_go -terse -q all.q -N "pBndet_$g" -hold_jid "$MAN" -j y -cwd -V \
         -v QG_DIGEST_RUN="error_tails_band_$g" \
         -o "$LOGS/pBndet_$g.\$JOB_ID.log" \
         scripts/sge/piff_tool_job.sh diagnose_error_tails.py \
         --ckpt "$CKPT" --config "$EVALCONF" --device cpu \
         --outdir "$RES/error_tails" --fig-dir "$RES/error_tails" \
         --report-run "error_tails_band_$g")
    echo "diag $MODEL: mean_prediction $MP error_tails $ET"

    PFA=$(qsub_go -terse -q all.q -N "pBndpfa_$g" -hold_jid "$MAN" -j y -cwd -V \
          -v QG_DIGEST_RUN="fields_assess_band_$g" \
          -o "$LOGS/pBndpfa_$g.\$JOB_ID.log" \
          scripts/sge/piff_tool_job.sh plot_fields_assess.py \
          --ckpt "$CKPT" --config "$EVALCONF" --device cpu \
          --per-member 2 --outdir "$RES/eval_assess")
    CBM=$(qsub_go -terse -q all.q -N "pBndcbm_$g" -hold_jid "$MAN" -j y -cwd -V \
          -v QG_DIGEST_RUN="band_metrics_band_$g" \
          -o "$LOGS/pBndcbm_$g.\$JOB_ID.log" \
          scripts/sge/piff_tool_job.sh compare_band_metrics.py \
          --ckpt "$CKPT" --config "$EVALCONF" --device cpu \
          --per-member 2 --model-tag blended \
          --out-csv "$RES/band_metrics/blended.csv")
    RPL=$(qsub_go -terse -q all.q -N "pBndrpl_$g" -hold_jid "$MAN" -j y -cwd -V \
          -v QG_DIGEST_RUN="replot_band_$g" \
          -o "$LOGS/pBndrpl_$g.\$JOB_ID.log" \
          scripts/sge/piff_tool_job.sh replot_eval_fields.py \
          --ckpt "$CKPT" --config "$EVALCONF" --device cpu \
          --per-member 3 --outdir "$RES/eval")
    echo "diag $MODEL: fields_assess $PFA band_metrics $CBM replot $RPL"

    HOLD_ALL="$MP,$ET,$PFA,$CBM,$RPL"

    # ---- sigma-at-events per member, held on error_tails ------------------- #
    for m in $(members_of "$g"); do
        SFX="${m#*-}"
        SIG=$(qsub_go -terse -q all.q -N "pBs${g}${SFX}" -hold_jid "$ET" \
              -j y -cwd -V -v QG_DIGEST_RUN="sigma_events_band_$g" \
              -o "$LOGS/pBs${g}${SFX}.\$JOB_ID.log" \
              scripts/sge/piff_tool_job.sh diagnose_sigma_at_events.py \
              --ckpt "$CKPT" --config "$EVALCONF" \
              --member "$m" \
              --events "$RES/error_tails/$m/extreme_events.csv" \
              --outdir "$RES/sigma_at_events" \
              --report-run "sigma_events_band_$g")
        echo "sigma $m: $SIG"
        HOLD_ALL="$HOLD_ALL,$SIG"
    done

    # ---- baseline band metrics for the table (each under ITS OWN config) --- #
    # independent of the trainers: they score already-trained ckpts. Missing
    # ckpts were warned about in preflight and become n/a rows in the table.
    for b in ylp75 lap wallv2; do
        BCK="runs_piff/piff_${g}_gjs_${b}/best.pt"
        [ -f "$ML/$BCK" ] || continue
        BJ=$(qsub_go -terse -q all.q -N "pBb${g}${b}" -j y -cwd -V \
             -v QG_DIGEST_RUN="band_metrics_${b}_$g" \
             -o "$LOGS/pBb${g}${b}.\$JOB_ID.log" \
             scripts/sge/piff_tool_job.sh compare_band_metrics.py \
             --ckpt "$BCK" --config "conf_piff_${g}_gjs_${b}.yaml" --device cpu \
             --per-member 2 --model-tag "$b" \
             --out-csv "$RES/band_metrics/${b}.csv")
        echo "baseline band metrics $b: $BJ"
        HOLD_ALL="$HOLD_ALL,$BJ"
    done

    # ---- gate + three-band table + [QG][REPORT] mail, held on everything --- #
    GATE=$(qsub_go -terse -q all.q -N "pBndgt_$g" -hold_jid "$HOLD_ALL" -j y -cwd -V \
           -m ea -M "$QG_NOTIFY_EMAIL" \
           -v QG_DIGEST_RUN="gate_bandmodels_$g" \
           -o "$LOGS/pBndgt_$g.\$JOB_ID.log" \
           scripts/sge/bandmodels_gate_job.sh "$g" "$GEOM" "$MODEL" "$BASE" \
           "$(members_of "$g" | tr ' ' ',')")
    echo "gate $MODEL vs $BASE: $GATE"

    # ---- results-standard sweep-up ---------------------------------------- #
    MIG=$(qsub_go -terse -q all.q -N "pBndmig_$g" -hold_jid "$GATE" -j y -cwd -V \
          -o "$LOGS/pBndmig_$g.\$JOB_ID.log" \
          scripts/sge/migrate_results_tree.sh)
    echo "results migration ($g): $MIG"
done

echo "[submit_bandmodels] both geometry chains submitted."
