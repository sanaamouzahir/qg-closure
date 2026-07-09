#!/bin/bash
# submit_phaseC_tier.sh - Phase-C convergence tier (charter S5 + Amendment 02
# + Sanaa 2026-07-08 FIVE-GRID directive + full green light 2026-07-09).
# FPC geometry, MOD-const (Re=3900), shared developed-flow IC (true time
# t=29.97, frame 111), T=60 (production time t in [29.97, 89.97]), float64
# solve / float32 at write.
#
# Job graph:
#   cnv_tab   (all.q)             const inlet tables at each tier dt, T=60
#   cnv_ics   (all.q)             spectral-regrid ICs 2048->{256,512,1024,4096}
#                                 + charter round-trip check
#   7 sims    (ibgpu.q gpu=1, hold cnv_tab[,cnv_ics]):
#     GRID STUDY, dt=1.25e-4 fixed (save_rate 2160 = 0.27 t.u.):
#       cnv_N0256  cnv_N0512  cnv_N1024  cnv_N2048  cnv_N4096
#     DT STUDY, N=2048 fixed ((2048, 1.25e-4) shared with the grid study):
#       cnv_dt5em4 (dt=5.0e-4, save_rate 540)
#       cnv_dt2p5e (dt=2.5e-4, save_rate 1080)
#
# FIXED-PHYSICS RULE (charter S5.2): Brinkman penalty and sponge eta at the
# FIXED PHYSICAL VALUE from the baseline convention eta = factor*dt
# (obstacle.py:57, bc.py Sponge): eta_phys = 1.25 * 2.5e-4 = 3.125e-4 for
# BOTH. Explicit per-dt factor overrides so eta_phys never scales with dt
# or grid:  dt=1.25e-4 -> 2.5   dt=2.5e-4 -> 1.25   dt=5.0e-4 -> 0.625.
#
# WALLTIME RISK: cnv_N4096 projected ~11 h (480k steps at ~12 it/s scaled
# from the 2048^2 calibration). Charter S5.3: report its projected finish
# time within 12 h of submission.
#
# Release guard, same two-step convention as Gate 1 / Phase B:
#   TIERC_RELEASE=1 QG_NOTIFY_EMAIL=... ./submit_phaseC_tier.sh

set -e

echo "HOLD: set TIERC_RELEASE=1 to submit (green light recorded 2026-07-09)"
[ "${TIERC_RELEASE:-0}" = "1" ] || exit 1

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR="$QG_ROOT/qg-simple-package-stable/src/qg"
BRANCH="$QG_ROOT/qg-sgs-closure"
SGE="$BRANCH/scripts/sge"
CONV_REL="outputs/SGS_closure_ensemble/convergence"    # hydra cwd = $QG_DIR
CONV_DIR="$QG_DIR/$CONV_REL"
TABLES_DIR="$QG_DIR/outputs/SGS_closure_ensemble/tables"
ICS_DIR="$CONV_DIR/ics"
SRC_IC="$QG_DIR/outputs/SGS_closure_ensemble/FPC-const/restart_ic_t30.npy"
LOG_DIR="$BRANCH/logs"

if [ -z "${QG_NOTIFY_EMAIL:-}" ]; then
    echo "ERROR: QG_NOTIFY_EMAIL unset"; exit 1
fi
[ -f "$SRC_IC" ] || { echo "ERROR: shared IC missing: $SRC_IC"; exit 1; }
mkdir -p "$LOG_DIR" "$CONV_DIR"

echo "=== Phase-C convergence tier: 5-grid study + dt study (7 sims) ==="

# 1. Inlet tables (skip job if all three exist)
HOLD_TAB=""
if [ -f "$TABLES_DIR/const_dt1p25e-4_T60.npz" ] && \
   [ -f "$TABLES_DIR/const_dt2p5e-4_T60.npz" ]  && \
   [ -f "$TABLES_DIR/const_dt5p0e-4_T60.npz" ]; then
    echo "[SKIP] all three T=60 const tables exist"
else
    OUT=$(qsub -q all.q -N cnv_tab \
          -o "$LOG_DIR/cnv_tab.\$JOB_ID.log" -j y -cwd -V \
          "$SGE/phaseC_make_tables.sh")
    echo "$OUT"
    HOLD_TAB=$(echo "$OUT" | grep -oE '[0-9]+' | head -1)
fi

# 2. Regridded ICs (skip job if all four exist)
HOLD_ICS=""
if [ -f "$ICS_DIR/ic_t29p97_N256.npy" ] && [ -f "$ICS_DIR/ic_t29p97_N512.npy" ] && \
   [ -f "$ICS_DIR/ic_t29p97_N1024.npy" ] && [ -f "$ICS_DIR/ic_t29p97_N4096.npy" ]; then
    echo "[SKIP] all four regridded ICs exist"
else
    OUT=$(qsub -q all.q -N cnv_ics \
          -o "$LOG_DIR/cnv_ics.\$JOB_ID.log" -j y -cwd -V \
          "$SGE/phaseC_regrid_job.sh")
    echo "$OUT"
    HOLD_ICS=$(echo "$OUT" | grep -oE '[0-9]+' | head -1)
fi

# 3. The 7 sims (GPU; the ONLY permitted pair -q ibgpu.q -l gpu=1)
# submit_one <jobname> <N> <dt> <save_rate> <eta_factor> <ic_path> <table> <runid>
submit_one () {
    local NAME=$1 N=$2 DT=$3 SR=$4 ETA=$5 IC=$6 TABLE=$7 RUNID=$8
    local RUN_DIR="$CONV_DIR/$RUNID"
    if [ -f "$RUN_DIR/DNS_FR.npz" ]; then
        echo "[SKIP] $RUN_DIR/DNS_FR.npz exists — never overwrite (charter S8)"
        return
    fi
    local HOLD_LIST=""
    [ -n "$HOLD_TAB" ] && HOLD_LIST="$HOLD_TAB"
    if [ "$IC" != "$SRC_IC" ] && [ -n "$HOLD_ICS" ]; then
        HOLD_LIST="${HOLD_LIST:+$HOLD_LIST,}$HOLD_ICS"
    fi
    local HOLD_OPT=()
    [ -n "$HOLD_LIST" ] && HOLD_OPT=(-hold_jid "$HOLD_LIST")
    local OUT
    OUT=$(qsub -N "$NAME" "${HOLD_OPT[@]}" \
          -o "$LOG_DIR/$NAME.\$JOB_ID.log" -j y -cwd -V \
          -q ibgpu.q -l gpu=1 -m ea -M "$QG_NOTIFY_EMAIL" \
          "$SGE/phaseC_job.sh" \
          qg.grid.Nx="$N" qg.grid.Ny="$N" \
          qg.time.dt="$DT" qg.time.save_rate="$SR" \
          qg.pde.penalty="$ETA" qg.bc.sponge="$ETA" \
          qg.ic.function=from_file +qg.ic.path="$IC" \
          +qg.bc.inlet_table="$TABLE" \
          +qg.diag.scalar_rate=10 +qg.diag.out="$RUN_DIR/scalars.npz" \
          hydra.run.dir="$CONV_REL/$RUNID")
    echo "$OUT  [$NAME: ${N}^2 dt=$DT sr=$SR eta_factor=$ETA]"
}

T1="$TABLES_DIR/const_dt1p25e-4_T60.npz"
T2="$TABLES_DIR/const_dt2p5e-4_T60.npz"
T5="$TABLES_DIR/const_dt5p0e-4_T60.npz"

# GRID STUDY (dt=1.25e-4, save_rate 2160, eta factor 2.5)
submit_one cnv_N0256 256  1.25e-4 2160 2.5 "$ICS_DIR/ic_t29p97_N256.npy"  "$T1" N0256_dt1p25e-4
submit_one cnv_N0512 512  1.25e-4 2160 2.5 "$ICS_DIR/ic_t29p97_N512.npy"  "$T1" N0512_dt1p25e-4
submit_one cnv_N1024 1024 1.25e-4 2160 2.5 "$ICS_DIR/ic_t29p97_N1024.npy" "$T1" N1024_dt1p25e-4
submit_one cnv_N2048 2048 1.25e-4 2160 2.5 "$SRC_IC"                      "$T1" N2048_dt1p25e-4
submit_one cnv_N4096 4096 1.25e-4 2160 2.5 "$ICS_DIR/ic_t29p97_N4096.npy" "$T1" N4096_dt1p25e-4

# DT STUDY (N=2048; (2048, 1.25e-4) above is the shared member)
submit_one cnv_dt5em4 2048 5.0e-4 540  0.625 "$SRC_IC" "$T5" N2048_dt5p0e-4
submit_one cnv_dt2p5e 2048 2.5e-4 1080 1.25  "$SRC_IC" "$T2" N2048_dt2p5e-4

echo "=== submitted; monitor hints ==="
echo "  qstat -u \$USER"
echo "  tail -f $LOG_DIR/cnv_N4096.<id>.log   (walltime risk: ~11 h at ~12 it/s;"
echo "                                          REPORT projected finish <12 h of submit)"
echo "  480k steps at dt=1.25e-4; 240k at 2.5e-4; 120k at 5.0e-4 (T=60)"
echo "  At landing: audit_resolution.py (Audit B, 5-grid-aware) + S5.4 analysis"
echo "  -> [QG][CONV][SGS-CLOSURE] report (carry true IC time t=29.97, the"
echo "     round-trip number from cnv_ics log, and eta_phys=3.125e-4)."
