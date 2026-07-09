#!/bin/bash
# submit_phaseB_wave2.sh - Phase-B wave 2: the 4 MODULATED FPC production runs
# (MOD-sine / MOD-ramp / MOD-ou / MOD-tel), released by Audit A + Sanaa's
# 2026-07-09 approvals: full-ensemble green light AND dt_save = 0.45
# (save_rate 1800; audit-A RELAX 0.5 would violate phase binning: T_sh 3.80
# -> 7.60 bins < 8; 0.45 gives 8.45, incommensurate; 0.475 = exactly 8.00 =
# the S5 trap).
#
# Charter S4.1 commons: FPC 2048^2, L=8pi, dt 2.5e-4, T=120, T_wait=30,
# nu 6.4443e-4, f64 solve / f32 write, recorder rate 10 w/ per-run diag.out
# (d48fcae rule), tables generated AT the solver dt (no interpolation),
# ou/telegraph seed 20260707 (charter S3.1).
#
# Release: PHASEB_RELEASE=1 QG_NOTIFY_EMAIL=... ./submit_phaseB_wave2.sh
# GPU rule: exactly -q ibgpu.q -l gpu=1. Never ibamd.q, never h_vmem.

set -e
echo "HOLD: set PHASEB_RELEASE=1 to submit (approvals recorded in DECISIONS 2026-07-09)"
[ "${PHASEB_RELEASE:-0}" = "1" ] || exit 1

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR="$QG_ROOT/qg-simple-package-stable/src/qg"
BRANCH="$QG_ROOT/qg-sgs-closure"
SGE="$BRANCH/scripts/sge"
TABLES_DIR="$QG_DIR/outputs/SGS_closure_ensemble/tables"
LOG_DIR="$BRANCH/logs"

if [ -z "${QG_NOTIFY_EMAIL:-}" ]; then
    echo "ERROR: QG_NOTIFY_EMAIL unset"; exit 1
fi
mkdir -p "$LOG_DIR" "$TABLES_DIR"

# ---- 1. tables job (all 4 signals, dt = solver dt, T=120, t-wait 30) ---- #
NEED_TABLES=0
for sig in sine ramp ou telegraph; do
    [ -f "$TABLES_DIR/${sig}_dt2p5e-4_T120.npz" ] || NEED_TABLES=1
done
HOLD_TAB=""
if [ "$NEED_TABLES" = "1" ]; then
    OUT=$(qsub -q all.q -N pB2_tab \
          -o "$LOG_DIR/pB2_tab.\$JOB_ID.log" -j y -cwd -V \
          "$SGE/phaseB_make_tables_wave2.sh")
    echo "$OUT"
    TAB_ID=$(echo "$OUT" | grep -oE '[0-9]+' | head -1)
    HOLD_TAB="-hold_jid $TAB_ID"
else
    echo "[SKIP] all 4 tables exist"
fi

# ---- 2. one sim + chained shedding tracker per modulation --------------- #
# save_rate 1800 = dt_save 0.45 (Sanaa-approved audit-A relaxation)
declare -A SIG=( [sine]=sine [ramp]=ramp [ou]=ou [tel]=telegraph )
declare -A SHORT=( [sine]=si [ramp]=ra [ou]=ou [tel]=te )

for mod in sine ramp ou tel; do
    RUN_REL="outputs/SGS_closure_ensemble/FPC-$mod"
    RUN_DIR="$QG_DIR/$RUN_REL"
    TABLE="$TABLES_DIR/${SIG[$mod]}_dt2p5e-4_T120.npz"
    if [ -f "$RUN_DIR/DNS_FR.npz" ]; then
        echo "[SKIP] $RUN_DIR/DNS_FR.npz exists — never overwrite (charter S8)"
        continue
    fi
    OUT=$(qsub -N "sgs_FPC_$mod" $HOLD_TAB \
          -o "$LOG_DIR/sgs_FPC_$mod.\$JOB_ID.log" -j y -cwd -V \
          -q ibgpu.q -l gpu=1 -m ea -M "$QG_NOTIFY_EMAIL" \
          "$SGE/phaseB_job.sh" \
          qg.grid.Nx=2048 qg.grid.Ny=2048 qg.time.save_rate=1800 \
          +qg.bc.inlet_table="$TABLE" \
          +qg.diag.scalar_rate=10 +qg.diag.out="$RUN_DIR/scalars.npz" \
          hydra.run.dir="$RUN_REL")
    echo "$OUT"
    SIM_ID=$(echo "$OUT" | grep -oE '[0-9]+' | head -1)
    qsub -q all.q -N "shed_FPC${SHORT[$mod]}" -hold_jid "$SIM_ID" \
         -o "$LOG_DIR/shed_FPC${SHORT[$mod]}.\$JOB_ID.log" -j y -cwd -V \
         -m ea -M "$QG_NOTIFY_EMAIL" \
         "$SGE/shedding_job.sh" "$RUN_DIR/scalars.npz" \
         --outdir "$RUN_DIR/shedding" --t-min 30.0 --tag "FPC-$mod"
done

echo "=== wave 2 submitted; monitor: qstat -u \$USER ==="
echo "  runs land in $QG_DIR/outputs/SGS_closure_ensemble/FPC-{sine,ramp,ou,tel}/"
echo "  each: DNS_FR.npz + U_of_t table + scalars.npz + shedding/"
