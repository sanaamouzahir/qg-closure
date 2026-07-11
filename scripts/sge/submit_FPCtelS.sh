#!/bin/bash
# submit_FPCtelS.sh - FPC-tel rerun with SMOOTHED telegraph switches (Sanaa GO
# 2026-07-11 chat on the SGS report QUESTIONS item 1, option (b) with the
# recommended 5-10 dt switch smoothing -- 10 dt implemented).
#
# What changed vs the failed FPC-tel (blew silently at t=68.6, 43% NaN):
#   * inlet table telegraphS10_dt2p5e-4_T120.npz: every level jump -- incl.
#     the T_wait entry jump Re_mid->Re_max at t=30 that produced the first
#     Cd~1378 penalty impulse -- replaced by a linear ramp over 10 solver
#     steps (max per-step dRe 3400 -> 340; switch TIMES unchanged, table
#     equals the hard one outside the 7 ramp windows). Tests the
#     penalty-impulse hypothesis AND yields a full-length dataset.
#   * NaN-guard now armed in phaseB_job.sh (+qg.diag.flush_every=500 so the
#     guard sees a fresh record; blowup costs minutes, not ~11 GPU-h).
#   * output dir FPC-telS -- the partial FPC-tel (healthy t<68.6) is KEPT.
#
# Everything else verbatim from the wave-2 production commons: 2048^2,
# dt 2.5e-4, T=120, nu 6.4443e-4, f64 solve / f32 write, save_rate 1800
# (dt_save 0.45), recorder rate 10, chained shed tracker (t-min 30).
# Cost: ~11 h on one GPU (wave-2 precedent).
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
    echo "=== FPC-telS (smoothed telegraph): SUBMITTING ==="
else
    run() { echo "[DRY-RUN] $*"; }
    echo "=== FPC-telS (smoothed telegraph): DRY-RUN (pass --go to submit) ==="
fi

TABLE="$TABLES_DIR/telegraphS10_dt2p5e-4_T120.npz"
[ -f "$TABLE" ] || { echo "ERROR: missing $TABLE -- regenerate with training/modulation.py --signal telegraph --switch-smooth-steps 10"; exit 1; }

RUN_REL="outputs/SGS_closure_ensemble/FPC-telS"
RUN_DIR="$QG_DIR/$RUN_REL"
if [ -f "$RUN_DIR/DNS_FR.npz" ]; then
    echo "ERROR: $RUN_DIR/DNS_FR.npz exists -- never overwrite (charter S8)"; exit 1
fi

OUT=$(run qsub -N sgsB_teS \
      -o "$LOG_DIR/\$JOB_NAME.\$JOB_ID.log" -j y -cwd -V \
      -q ibgpu.q -l gpu=1 -m ea -M "$QG_NOTIFY_EMAIL" \
      "$SGE/phaseB_job.sh" \
      qg.grid.Nx=2048 qg.grid.Ny=2048 qg.time.save_rate=1800 \
      +qg.bc.inlet_table="$TABLE" \
      +qg.diag.scalar_rate=10 +qg.diag.flush_every=500 \
      +qg.diag.out="$RUN_DIR/scalars.npz" \
      hydra.run.dir="$RUN_REL")
echo "$OUT"
SIM_ID=$(echo "$OUT" | grep -oE '[0-9]+' | head -1)
HOLD_SIM=""
[ -n "$SIM_ID" ] && [ "$GO" = "1" ] && HOLD_SIM="-hold_jid $SIM_ID"
run qsub -q all.q -N shed_CteS $HOLD_SIM \
     -o "$LOG_DIR/\$JOB_NAME.\$JOB_ID.log" -j y -cwd -V \
     -m ea -M "$QG_NOTIFY_EMAIL" \
     "$SGE/shedding_job.sh" "$RUN_DIR/scalars.npz" \
     --outdir "$RUN_DIR/shedding" --t-min 30.0 --tag "FPC-telS"

echo "=== FPC-telS done; monitor: qstat -u \$USER ==="
