#!/bin/bash
# submit_phaseCapeA_wave.sh - FPCape production wave, CAPE-A rerun (Sanaa
# ruling 2026-07-11 chat): the same five cases FPCape-{const,sine,ramp,ou,tel}
# at 2048^2 with dt HALVED to 1.25e-4, after the three-arm smoke verdict
# (capeSmk P/Y/D, jobs 1830334/35/40) showed 2048^2 @ 2.5e-4 dt-unstable for
# the cape (penalty exonerated) and 2048^2 @ 1.25e-4 clean to T=3.
#
# Differences vs submit_phaseCape_wave.sh (the failed 2.5e-4 wave):
#   * dt 1.25e-4 is baked into phaseCape_job.sh (CAPE-A comment there);
#   * inlet tables = the dt1p25e-4_T120 set (generated + validated on
#     2026-07-11: bitwise dt-consistent with the 2.5e-4 tables at shared
#     times); table load enforces dt match, so the old tables CANNOT be
#     reused silently;
#   * qg.time.save_rate=3600 keeps dt_save = 0.45 at the halved dt;
#   * +qg.diag.flush_every=500 (recorder atomic rewrite every 5000 steps,
#     ~8 min wall) so the NaN-guard now armed in phaseCape_job.sh sees a
#     fresh record; a blowup costs minutes, not 27 GPU-h;
#   * NO conditional tables job: this script REFUSES to submit if any
#     dt1p25e-4 table is missing (they are cheap; regenerate via
#     training/modulation.py, seed 20260707).
#
# Everything else verbatim from the approved wave: cape geometry/penalty
# 1.025 from conf/scenario/flow_past_cape.yaml, nu 6.4443e-4, T=120, f64
# solve / f32 write, recorder rate 10, approved cape lee-probe set,
# +qg.diag.length=1.0, shed tracker chained per case (-hold_jid, t-min 30).
#
# Cost: ~27 h/member on one GPU (smoke capeSmkD: 24k steps in ~40 min =>
# 960k steps ~ 26.7 h), five members co-scheduled as slots allow.
#
# DRY-RUN by default; pass --go to submit:
#   ./submit_phaseCapeA_wave.sh          # print the qsub commands
#   ./submit_phaseCapeA_wave.sh --go     # fire
# GPU rule: exactly -q ibgpu.q -l gpu=1 (the only permitted pair).

set -e

GO=0
[ "${1:-}" = "--go" ] && GO=1

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR="$QG_ROOT/qg-simple-package-stable/src/qg"
BRANCH="$QG_ROOT/qg-sgs-closure"
SGE="$BRANCH/scripts/sge"
TABLES_DIR="$QG_DIR/outputs/SGS_closure_ensemble/tables"
LOG_DIR="$BRANCH/logs"

QG_NOTIFY_EMAIL="${QG_NOTIFY_EMAIL:-sanaamz@mit.edu}"
mkdir -p "$LOG_DIR"

if [ "$GO" = "1" ]; then
    run() { "$@"; }
    echo "=== FPCape wave CAPE-A (dt 1.25e-4): SUBMITTING ==="
else
    run() { echo "[DRY-RUN] $*"; }
    echo "=== FPCape wave CAPE-A (dt 1.25e-4): DRY-RUN (pass --go to submit) ==="
fi

CAPE_DIAG='+qg.diag.length=1.0 +qg.diag.probes=[[6.0265,4.0],[7.0265,4.0],[8.0265,4.0],[6.0265,4.5],[6.0265,3.5],[6.0265,2.0]]'

# ---- 1. hard-require the dt1p25e-4 tables -------------------------------- #
declare -A SIG=( [const]=const [sine]=sine [ramp]=ramp [ou]=ou [tel]=telegraph )
declare -A SHORT=( [const]=co [sine]=si [ramp]=ra [ou]=ou [tel]=te )
for mod in const sine ramp ou tel; do
    TABLE="$TABLES_DIR/${SIG[$mod]}_dt1p25e-4_T120.npz"
    [ -f "$TABLE" ] || { echo "ERROR: missing $TABLE -- regenerate with training/modulation.py (dt 1.25e-4, T 120, t-wait 30, seed 20260707)"; exit 1; }
done
echo "[OK] all 5 dt1p25e-4_T120 inlet tables present"

# ---- 2. one sim + chained shedding tracker per case ---------------------- #
for mod in const sine ramp ou tel; do
    RUN_REL="outputs/SGS_closure_ensemble/FPCape-$mod"
    RUN_DIR="$QG_DIR/$RUN_REL"
    TABLE="$TABLES_DIR/${SIG[$mod]}_dt1p25e-4_T120.npz"
    if [ -f "$RUN_DIR/DNS_FR.npz" ]; then
        echo "[SKIP] $RUN_DIR/DNS_FR.npz exists -- never overwrite (charter S8)"
        continue
    fi
    # job names sgsCapA_* + shed_CA*: unique within qstat's 10-char display
    OUT=$(run qsub -N "sgsCapA_${SHORT[$mod]}" \
          -o "$LOG_DIR/\$JOB_NAME.\$JOB_ID.log" -j y -cwd -V \
          -q ibgpu.q -l gpu=1 -m ea -M "$QG_NOTIFY_EMAIL" \
          "$SGE/phaseCape_job.sh" \
          qg.grid.Nx=2048 qg.grid.Ny=2048 qg.time.save_rate=3600 \
          +qg.bc.inlet_table="$TABLE" \
          +qg.diag.scalar_rate=10 +qg.diag.flush_every=500 \
          +qg.diag.out="$RUN_DIR/scalars.npz" \
          $CAPE_DIAG \
          hydra.run.dir="$RUN_REL")
    echo "$OUT"
    SIM_ID=$(echo "$OUT" | grep -oE '[0-9]+' | head -1)
    HOLD_SIM=""
    [ -n "$SIM_ID" ] && [ "$GO" = "1" ] && HOLD_SIM="-hold_jid $SIM_ID"
    run qsub -q all.q -N "shed_CA${SHORT[$mod]}" $HOLD_SIM \
         -o "$LOG_DIR/\$JOB_NAME.\$JOB_ID.log" -j y -cwd -V \
         -m ea -M "$QG_NOTIFY_EMAIL" \
         "$SGE/shedding_job.sh" "$RUN_DIR/scalars.npz" \
         --outdir "$RUN_DIR/shedding" --t-min 30.0 --tag "FPCape-$mod"
done

echo "=== CAPE-A wave done; monitor: qstat -u \$USER ==="
echo "  runs land in $QG_DIR/outputs/SGS_closure_ensemble/FPCape-{const,sine,ramp,ou,tel}/"
echo "  each: 960000 steps (~27 h), ~267 snapshots at dt_save 0.45, NaN-guard armed"
