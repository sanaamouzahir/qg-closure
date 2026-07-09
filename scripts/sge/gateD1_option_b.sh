#!/bin/bash
# gateD1_option_b.sh - Gate D-1 (AMENDMENT_01 SF) per OPTION B, approved by
# Sanaa 2026-07-09 via chat with EXPLICIT dt sign-off ([red-approved],
# DECISIONS.md; deviation from the chartered 2.5e-4 solver dt):
#
#   FPC const Re=200, 1024^2, dt=2.5e-3 (CFL_adv = 0.0105 at U=0.1026),
#   T=1500, transient discard T_wait=250 -> ~20 post-transient shedding
#   periods (T_sh(200) ~ 62.5 t.u.), ~102 post-transient convective units.
#
# save_rate derivation (theory doc S3.2 + S5, recomputed at Re=200):
#   phase binning: dt_save <= T_sh/8 = 7.8 t.u.
#   commensurability: dt_save 7.8 gives T_sh/dt_save = 8.01 (~integer trap);
#   chosen dt_save = 7.0 t.u. -> T_sh/dt_save = 8.93 (~40 deg/period phase
#   advance, full wheel in ~9 periods). save_rate = 7.0/2.5e-3 = 2800.
#   -> ~215 snapshots, ~0.9 GB f32 at 1024^2.
# Scalars: rate 10 = 0.025 t.u. -> ~2500 samples per shedding period.
# Table: const --re-const 200 at THE RUN dt (bc no-interpolation rule).
#
# This is a self-contained submitter (three-job hold chain, gate1/phaseB
# pattern): gd1_tab (all.q) -> sgs_gd1_re200 (ibgpu.q gpu=1) -> shed_gd1
# (all.q, --gate-d1 mode). Reuses phaseB_job.sh? NO - dt/T commons differ;
# the run payload is inlined here via qg_job-style worker exec of run_qg.py
# through gateD1_run_job.sh (sibling).
#
# Usage: GATED1_RELEASE=1 QG_NOTIFY_EMAIL=... ./gateD1_option_b.sh

set -e

echo "HOLD: set GATED1_RELEASE=1 to submit (option-B approval in DECISIONS.md)"
[ "${GATED1_RELEASE:-0}" = "1" ] || exit 1

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR="$QG_ROOT/qg-simple-package-stable/src/qg"
BRANCH="$QG_ROOT/qg-sgs-closure"
SGE="$BRANCH/scripts/sge"
RUN_ROOT_REL="outputs/SGS_closure_gateD1/FPC-re200"
RUN_DIR="$QG_DIR/$RUN_ROOT_REL"
TABLE="$QG_DIR/outputs/SGS_closure_gateD1/tables/const_re200_dt2p5e-3_T1500.npz"
LOG_DIR="$BRANCH/logs"

[ -n "${QG_NOTIFY_EMAIL:-}" ] || { echo "ERROR: QG_NOTIFY_EMAIL unset"; exit 1; }
mkdir -p "$LOG_DIR"

if [ -f "$RUN_DIR/DNS_FR.npz" ]; then
    echo "[SKIP] $RUN_DIR/DNS_FR.npz exists — never overwrite (charter S8)"
    exit 0
fi

# 1. Table at the run dt
HOLD_TAB=""
if [ -f "$TABLE" ]; then
    echo "[SKIP] table exists: $TABLE"
else
    OUT=$(qsub -q all.q -N gd1_tab \
          -o "$LOG_DIR/gd1_tab.\$JOB_ID.log" -j y -cwd -V \
          "$SGE/gateD1_make_table.sh" "$TABLE")
    echo "$OUT"
    HOLD_TAB="-hold_jid $(echo "$OUT" | grep -oE '[0-9]+' | head -1)"
fi

# 2. The run (GPU; the ONLY permitted pair)
OUT=$(qsub -N sgs_gd1_re200 $HOLD_TAB \
      -o "$LOG_DIR/sgs_gd1_re200.\$JOB_ID.log" -j y -cwd -V \
      -q ibgpu.q -l gpu=1 -m ea -M "$QG_NOTIFY_EMAIL" \
      "$SGE/gateD1_run_job.sh" \
      qg.grid.Nx=1024 qg.grid.Ny=1024 qg.time.save_rate=2800 \
      +qg.bc.inlet_table="$TABLE" \
      +qg.diag.scalar_rate=10 +qg.diag.out="$RUN_DIR/scalars.npz" \
      hydra.run.dir="$RUN_ROOT_REL")
echo "$OUT"
SIM_ID=$(echo "$OUT" | grep -oE '[0-9]+' | head -1)

# 3. Shedding tracker in Gate D-1 mode (auto band 0.6-1.6 x St_ref*U/D)
qsub -q all.q -N shed_gd1 -hold_jid "$SIM_ID" \
     -o "$LOG_DIR/shed_gd1.\$JOB_ID.log" -j y -cwd -V \
     -m ea -M "$QG_NOTIFY_EMAIL" \
     "$SGE/shedding_job.sh" "$RUN_DIR/scalars.npz" \
     --outdir "$RUN_DIR/shedding" --t-min 250.0 --gate-d1 --tag gateD1-re200

echo "=== submitted; monitor: tail -f $LOG_DIR/sgs_gd1_re200.<id>.log ==="
echo "  Gate D-1 targets (2D literature, the ONE permitted comparison):"
echo "  mean Cd ~ 1.3-1.4, St ~ 0.195-0.20, rms Cl ~ 0.4-0.7."
