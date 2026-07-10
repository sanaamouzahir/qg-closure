#!/bin/bash
# submit_phaseCape_wave.sh - FPCape production wave (Sanaa order 2026-07-10):
# the SAME five cases as the FPC ensemble -- FPCape-{const,sine,ramp,ou,tel}
# -- for FLOW PAST A CAPE at 2048^2 (Sanaa's 2026-07-10 order supersedes the
# charter S4.1 cape row's 1024^2).
#
# Commons identical to the FPC waves (submit_phaseB_wave{1,2}.sh precedent):
# 2048^2, L=8pi, dt 2.5e-4, T=120, T_wait=30, nu 6.4443e-4 (brief rule 5:
# never change nu between cases), f64 solve / f32 write, save_rate 1800
# (dt_save 0.45, Sanaa-approved audit-A relaxation), recorder rate 10 with
# PER-RUN +qg.diag.out (d48fcae rule), DNS_FR never-overwrite guard.
#
# Cape-specific recorder overrides (approved cape lee-probe proposal,
# BRANCH_LOG 2026-07-07 "pass as qg.diag.probes + qg.diag.length=1.0"):
#   +qg.diag.length=1.0                          L_cape = 1 (Re_cape mid 3103)
#   +qg.diag.probes=[[6.0265,4.0],[7.0265,4.0],[8.0265,4.0],
#                    [6.0265,4.5],[6.0265,3.5],[6.0265,2.0]]
#   (wake x_c+{1,2,3}L at tip height 4.0; cross-stream pair x_c+1L, 4.0+-0.5L;
#    6th recirculation probe -- cape wake is bottom-attached/asymmetric)
#
# Inlet tables: the FPC dt2p5e-4_T120 tables are pure U(t) series (geometry-
# independent by construction: modulation.py has no geometry inputs; bc.py
# const_x_flow reads U[n] only) -- REUSED verbatim. Tables job submitted only
# if any table is missing.
#
# Chain per case: sim (ibgpu.q gpu=1) -> shedding tracker (all.q, -hold_jid,
# --t-min 30, --tag FPCape-<case>).
#
# DRY-RUN by default; pass --go to submit:
#   ./submit_phaseCape_wave.sh          # print the qsub commands, submit nothing
#   ./submit_phaseCape_wave.sh --go     # fire
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
mkdir -p "$LOG_DIR" "$TABLES_DIR"

if [ "$GO" = "1" ]; then
    run() { "$@"; }
    echo "=== FPCape wave: SUBMITTING ==="
else
    run() { echo "[DRY-RUN] $*"; }
    echo "=== FPCape wave: DRY-RUN (pass --go to submit) ==="
fi

CAPE_DIAG='+qg.diag.length=1.0 +qg.diag.probes=[[6.0265,4.0],[7.0265,4.0],[8.0265,4.0],[6.0265,4.5],[6.0265,3.5],[6.0265,2.0]]'

# ---- 1. tables (reuse FPC tables; conditional regeneration only) --------- #
HOLD_TAB=""
NEED_MOD=0
for sig in sine ramp ou telegraph; do
    [ -f "$TABLES_DIR/${sig}_dt2p5e-4_T120.npz" ] || NEED_MOD=1
done
if [ "$NEED_MOD" = "1" ]; then
    OUT=$(run qsub -q all.q -N pCape_tab \
          -o "$LOG_DIR/\$JOB_NAME.\$JOB_ID.log" -j y -cwd -V \
          "$SGE/phaseB_make_tables_wave2.sh")
    echo "$OUT"
    if [ "$GO" = "1" ]; then
        TAB_ID=$(echo "$OUT" | grep -oE '[0-9]+' | head -1)
        [ -n "$TAB_ID" ] && HOLD_TAB="-hold_jid $TAB_ID"
    fi
else
    echo "[SKIP] all 4 modulated tables exist (reused from the FPC waves)"
fi
if [ ! -f "$TABLES_DIR/const_dt2p5e-4_T120.npz" ]; then
    OUT=$(run qsub -q all.q -N pCape_tabc \
          -o "$LOG_DIR/\$JOB_NAME.\$JOB_ID.log" -j y -cwd -V \
          "$SGE/phaseB_make_tables.sh")
    echo "$OUT"
    if [ "$GO" = "1" ]; then
        TABC_ID=$(echo "$OUT" | grep -oE '[0-9]+' | head -1)
        [ -n "$TABC_ID" ] && HOLD_TAB="$HOLD_TAB -hold_jid $TABC_ID"
    fi
else
    echo "[SKIP] const table exists (reused from the FPC waves)"
fi

# ---- 2. one sim + chained shedding tracker per case ---------------------- #
declare -A SIG=( [const]=const [sine]=sine [ramp]=ramp [ou]=ou [tel]=telegraph )
declare -A SHORT=( [const]=co [sine]=si [ramp]=ra [ou]=ou [tel]=te )

for mod in const sine ramp ou tel; do
    RUN_REL="outputs/SGS_closure_ensemble/FPCape-$mod"
    RUN_DIR="$QG_DIR/$RUN_REL"
    TABLE="$TABLES_DIR/${SIG[$mod]}_dt2p5e-4_T120.npz"
    if [ -f "$RUN_DIR/DNS_FR.npz" ]; then
        echo "[SKIP] $RUN_DIR/DNS_FR.npz exists -- never overwrite (charter S8)"
        continue
    fi
    # job names sgsCape_co/si/ra/ou/te + shed_Cp*: unique within qstat's
    # 10-char display (be2a0b0 lesson)
    OUT=$(run qsub -N "sgsCape_${SHORT[$mod]}" $HOLD_TAB \
          -o "$LOG_DIR/\$JOB_NAME.\$JOB_ID.log" -j y -cwd -V \
          -q ibgpu.q -l gpu=1 -m ea -M "$QG_NOTIFY_EMAIL" \
          "$SGE/phaseCape_job.sh" \
          qg.grid.Nx=2048 qg.grid.Ny=2048 qg.time.save_rate=1800 \
          +qg.bc.inlet_table="$TABLE" \
          +qg.diag.scalar_rate=10 +qg.diag.out="$RUN_DIR/scalars.npz" \
          $CAPE_DIAG \
          hydra.run.dir="$RUN_REL")
    echo "$OUT"
    SIM_ID=$(echo "$OUT" | grep -oE '[0-9]+' | head -1)
    HOLD_SIM=""
    [ -n "$SIM_ID" ] && [ "$GO" = "1" ] && HOLD_SIM="-hold_jid $SIM_ID"
    run qsub -q all.q -N "shed_Cp${SHORT[$mod]}" $HOLD_SIM \
         -o "$LOG_DIR/\$JOB_NAME.\$JOB_ID.log" -j y -cwd -V \
         -m ea -M "$QG_NOTIFY_EMAIL" \
         "$SGE/shedding_job.sh" "$RUN_DIR/scalars.npz" \
         --outdir "$RUN_DIR/shedding" --t-min 30.0 --tag "FPCape-$mod"
done

echo "=== FPCape wave done; monitor: qstat -u \$USER ==="
echo "  runs land in $QG_DIR/outputs/SGS_closure_ensemble/FPCape-{const,sine,ramp,ou,tel}/"
echo "  each: DNS_FR.npz + scalars.npz + shedding/; ~480000 steps, ~267 snapshots"
