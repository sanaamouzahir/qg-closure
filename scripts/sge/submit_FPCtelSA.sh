#!/bin/bash
# submit_FPCtelSA.sh - FPC-telS rerun at dt HALVED to 1.25e-4 (FPC-A, the
# CAPE-A playbook applied to the cylinder; authorized under Sanaa's 07-12
# autonomy window, chat).
#
# Postmortem basis (2026-07-12, job 1830422 killed by the NaN-guard):
#   * FPC-telS (smoothed switches) blew at t=66.81 -- 12.3 time units INTO
#     the long Re=5600 dwell [54.47, 68.86], 2.05 before the next switch;
#     the original hard-switch FPC-tel blew at t=68.6 inside the SAME dwell.
#     Switch-impulse hypothesis disproven.
#   * Signature = cape dt-edge: enstrophy Z exploded 3.4 -> 7.6e191 over
#     0.07 time units while E stayed flat at 4.314 (grid-scale blowup).
#   * Re 5600 => U_inlet 2.87 vs 2.0 at Re 3900, the max the clean members
#     FPC-{const,sine,ramp,ou} ever saw at 2.5e-4. Cape precedent: 2048^2
#     dt-unstable at 2.5e-4, clean at 1.25e-4 (penalty exonerated).
#
# What changes vs submit_FPCtelS.sh (job 1830422):
#   * worker phaseB_A_job.sh (dt 1.25e-4 baked; NaN-guard verbatim);
#   * inlet table telegraphS20_dt1p25e-4_T120.npz: --switch-smooth-steps is
#     in SOLVER STEPS, so 20 steps at 1.25e-4 keeps the same PHYSICAL ramp
#     0.0025 as 10 steps at 2.5e-4; switch times + seed 20260707 unchanged.
#     Validated 2026-07-12: bitwise equal to telegraphS10_dt2p5e-4 at ALL
#     480001 shared times (ramp windows included), ramp duration 0.0025;
#   * qg.time.save_rate=3600 keeps dt_save 0.45 at the halved dt (CAPE-A);
#   * output dir FPC-telS-A -- the partial FPC-telS (healthy t<66.81) KEPT.
#
# Cost: 960000 steps at ~8.3 it/s (job 1830422 measured) ~ 32 h on one GPU.
#
# DRY-RUN by default; pass --go to submit.
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
    echo "=== FPC-telS-A (smoothed telegraph, dt 1.25e-4): SUBMITTING ==="
else
    run() { echo "[DRY-RUN] $*"; }
    echo "=== FPC-telS-A (smoothed telegraph, dt 1.25e-4): DRY-RUN (pass --go to submit) ==="
fi

TABLE="$TABLES_DIR/telegraphS20_dt1p25e-4_T120.npz"
[ -f "$TABLE" ] || { echo "ERROR: missing $TABLE -- regenerate with training/modulation.py --signal telegraph --dt 1.25e-4 --switch-smooth-steps 20 --seed 20260707"; exit 1; }

RUN_REL="outputs/SGS_closure_ensemble/FPC-telS-A"
RUN_DIR="$QG_DIR/$RUN_REL"
if [ -f "$RUN_DIR/DNS_FR.npz" ]; then
    echo "ERROR: $RUN_DIR/DNS_FR.npz exists -- never overwrite (charter S8)"; exit 1
fi

OUT=$(run qsub -N sgsB_teSA \
      -o "$LOG_DIR/\$JOB_NAME.\$JOB_ID.log" -j y -cwd -V \
      -q ibgpu.q -l gpu=1 -m ea -M "$QG_NOTIFY_EMAIL" \
      "$SGE/phaseB_A_job.sh" \
      qg.grid.Nx=2048 qg.grid.Ny=2048 qg.time.save_rate=3600 \
      +qg.bc.inlet_table="$TABLE" \
      +qg.diag.scalar_rate=10 +qg.diag.flush_every=500 \
      +qg.diag.out="$RUN_DIR/scalars.npz" \
      hydra.run.dir="$RUN_REL")
echo "$OUT"
SIM_ID=$(echo "$OUT" | grep -oE '[0-9]+' | head -1)
HOLD_SIM=""
[ -n "$SIM_ID" ] && [ "$GO" = "1" ] && HOLD_SIM="-hold_jid $SIM_ID"
run qsub -q all.q -N shed_CteSA $HOLD_SIM \
     -o "$LOG_DIR/\$JOB_NAME.\$JOB_ID.log" -j y -cwd -V \
     -m ea -M "$QG_NOTIFY_EMAIL" \
     "$SGE/shedding_job.sh" "$RUN_DIR/scalars.npz" \
     --outdir "$RUN_DIR/shedding" --t-min 30.0 --tag "FPC-telS-A"

echo "=== FPC-telS-A done; monitor: qstat -u \$USER ==="
