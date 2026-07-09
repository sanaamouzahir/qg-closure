#!/bin/bash
# submit_phaseB_wave1.sh - Phase-B WAVE 1 (post-Gate-1 approval, Sanaa
# 2026-07-09; charter S4 + Amendment 02): FPC MOD-const production run with
# its hold chain.
#
#   phaseB_tab   (all.q)              const inlet table dt=2.5e-4 T=120
#   sgs_FPC_const(ibgpu.q gpu=1,      2048^2, save_rate 1080, recorder on,
#                 hold phaseB_tab)      per-run diag.out, ~480000 steps
#   shed_FPCc    (all.q,              shedding_tracker on scalars.npz,
#                 hold sgs_FPC_const)   --t-min 30 (production window)
#
# NOT in this wave (recorded in BRANCH_LOG/DECISIONS):
#   - Pi_FF s{2,4,8} + audit_A: submitted AT LANDING — audit A's decision
#     rule (theory doc S8/A) needs Pi_FF tau_int at s=4, and Pi_FF needs the
#     landed DNS_FR (+ conditional mmap prep, charter S6.1).
#   - The 4 modulated runs: held behind Audit A (Amendment 02 S5).
#   - Gate D-1 Re=200: HELD + [QG][FLAG][SGS] sent (T=120 at U=0.1026 is
#     9.8 convective units / 1.4 shedding periods — undeveloped by
#     construction; awaiting Sanaa's ruling).
#   - Convergence tier: held behind the t=30 IC extract from this run.
#
# Release guard, same two-step convention as Gate 1:
#   PHASEB_RELEASE=1 QG_NOTIFY_EMAIL=... ./submit_phaseB_wave1.sh

set -e

echo "HOLD: set PHASEB_RELEASE=1 to submit (Gate-1 approval recorded 7a6eb3b)"
[ "${PHASEB_RELEASE:-0}" = "1" ] || exit 1

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR="$QG_ROOT/qg-simple-package-stable/src/qg"
BRANCH="$QG_ROOT/qg-sgs-closure"
SGE="$BRANCH/scripts/sge"
RUN_ROOT_REL="outputs/SGS_closure_ensemble/FPC-const"   # hydra cwd = $QG_DIR
RUN_DIR="$QG_DIR/$RUN_ROOT_REL"
TABLE="$QG_DIR/outputs/SGS_closure_ensemble/tables/const_dt2p5e-4_T120.npz"
LOG_DIR="$BRANCH/logs"

if [ -z "${QG_NOTIFY_EMAIL:-}" ]; then
    echo "ERROR: QG_NOTIFY_EMAIL unset"; exit 1
fi
mkdir -p "$LOG_DIR"

if [ -f "$RUN_DIR/DNS_FR.npz" ]; then
    echo "[SKIP] $RUN_DIR/DNS_FR.npz already exists — never overwrite (charter S8)"
    exit 0
fi

echo "=== Phase-B wave 1: FPC-const production ==="

# 1. Inlet table (skip if already generated)
HOLD_TAB=""
if [ -f "$TABLE" ]; then
    echo "[SKIP] table exists: $TABLE"
else
    OUT=$(qsub -q all.q -N phaseB_tab \
          -o "$LOG_DIR/phaseB_tab.\$JOB_ID.log" -j y -cwd -V \
          "$SGE/phaseB_make_tables.sh")
    echo "$OUT"
    TAB_ID=$(echo "$OUT" | grep -oE '[0-9]+' | head -1)
    HOLD_TAB="-hold_jid $TAB_ID"
fi

# 2. Production run (GPU; the ONLY permitted pair)
OUT=$(qsub -N sgs_FPC_const $HOLD_TAB \
      -o "$LOG_DIR/sgs_FPC_const.\$JOB_ID.log" -j y -cwd -V \
      -q ibgpu.q -l gpu=1 -m ea -M "$QG_NOTIFY_EMAIL" \
      "$SGE/phaseB_job.sh" \
      qg.grid.Nx=2048 qg.grid.Ny=2048 qg.time.save_rate=1080 \
      +qg.bc.inlet_table="$TABLE" \
      +qg.diag.scalar_rate=10 +qg.diag.out="$RUN_DIR/scalars.npz" \
      hydra.run.dir="$RUN_ROOT_REL")
echo "$OUT"
SIM_ID=$(echo "$OUT" | grep -oE '[0-9]+' | head -1)

# 3. Shedding tracker, chained on the sim (scalars-only; safe CLI)
qsub -q all.q -N shed_FPCc -hold_jid "$SIM_ID" \
     -o "$LOG_DIR/shed_FPCc.\$JOB_ID.log" -j y -cwd -V \
     -m ea -M "$QG_NOTIFY_EMAIL" \
     "$SGE/shedding_job.sh" "$RUN_DIR/scalars.npz" \
     --outdir "$RUN_DIR/shedding" --t-min 30.0 --tag FPC-const

echo "=== submitted; monitor hints ==="
echo "  qstat -u \$USER"
echo "  tail -f $LOG_DIR/sgs_FPC_const.<id>.log   (it/s; ~480000 steps)"
echo "  ls -la $RUN_DIR                            (scalars.npz liveness,"
echo "                                              flush every 2000 rows)"
echo "  At landing: Pi_FF s{2,4,8} -> audit_A -> [QG][RUN][SGS] report."
