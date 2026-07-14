#!/bin/bash
# submit_ood_chirp.sh - the OOD generalization cases (Sanaa order 2026-07-14:
# "we are simply going to generate another case and use that as the OOD case"
# — replaces the killed LOMO ladders 1833331/1833354; NO training involved).
# One NEW inlet modulation per geometry: 'chirp' — amplitude band identical
# to the trained sine (Re_mid +- Re_amp), frequency sweeping 1x -> 3x the
# sine rate across t in [30, 120]. OOD in time structure, not in Re range.
#
# Per geometry: FR sim + chained shedding tracker + chained video rerender
# (video-rerender-mandatory rule). dt matches THE TRAINING MEMBERS of each
# geometry: FPC 2.5e-4 (phaseB commons), FPCape 1.25e-4 (CAPE-A baked into
# phaseCape_job.sh; save_rate 3600 = same dt_save 0.45). Tables pre-generated
# 2026-07-14 by training/modulation.py (chirp_dt{2p5e-4,1p25e-4}_T120.npz).
# Step-0 packaging (Pi_FF gaussian jonly + manifest) fires AFTER landing.
#
# Cost: ~11 h (FPC) + ~22 h (cape, halved dt) on 1 GPU each.
# Release: OOD_RELEASE=1 QG_NOTIFY_EMAIL=... ./submit_ood_chirp.sh
# GPU rule: exactly -q ibgpu.q -l gpu=1. Never ibamd.q, never h_vmem.

set -e
echo "HOLD: set OOD_RELEASE=1 to submit"
[ "${OOD_RELEASE:-0}" = "1" ] || exit 1

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR="$QG_ROOT/qg-simple-package-stable/src/qg"
BRANCH="$QG_ROOT/qg-sgs-closure"
SGE="$BRANCH/scripts/sge"
TABLES_DIR="$QG_DIR/outputs/SGS_closure_ensemble/tables"
LOG_DIR="$BRANCH/logs"

if [ -z "${QG_NOTIFY_EMAIL:-}" ]; then
    echo "ERROR: QG_NOTIFY_EMAIL unset"; exit 1
fi
mkdir -p "$LOG_DIR"

CAPE_DIAG='+qg.diag.length=1.0 +qg.diag.probes=[[6.0265,4.0],[7.0265,4.0],[8.0265,4.0],[6.0265,4.5],[6.0265,3.5],[6.0265,2.0]]'

for tbl in chirp_dt2p5e-4_T120.npz chirp_dt1p25e-4_T120.npz; do
    [ -f "$TABLES_DIR/$tbl" ] || { echo "ERROR: missing $TABLES_DIR/$tbl"; exit 1; }
done

# ---- FPC-chirp (phaseB commons: dt 2.5e-4 baked; save_rate 1800) --------- #
RUN_REL="outputs/SGS_closure_ensemble/FPC-chirp"
RUN_DIR="$QG_DIR/$RUN_REL"
if [ -f "$RUN_DIR/DNS_FR.npz" ]; then
    echo "[SKIP] $RUN_DIR/DNS_FR.npz exists — never overwrite (charter S8)"
else
    SIM=$(qsub -terse -N sgs_FPC_ch -o "$LOG_DIR/sgs_FPC_ch.\$JOB_ID.log" -j y -cwd -V \
          -q ibgpu.q -l gpu=1 -m ea -M "$QG_NOTIFY_EMAIL" \
          "$SGE/phaseB_job.sh" \
          qg.grid.Nx=2048 qg.grid.Ny=2048 qg.time.save_rate=1800 \
          +qg.bc.inlet_table="$TABLES_DIR/chirp_dt2p5e-4_T120.npz" \
          +qg.diag.scalar_rate=10 +qg.diag.out="$RUN_DIR/scalars.npz" \
          hydra.run.dir="$RUN_REL")
    SHED=$(qsub -terse -q all.q -N shed_FPCch -hold_jid "$SIM" \
          -o "$LOG_DIR/shed_FPCch.\$JOB_ID.log" -j y -cwd -V \
          "$SGE/shedding_job.sh" "$RUN_DIR/scalars.npz" \
          --outdir "$RUN_DIR/shedding" --t-min 30.0 --tag FPC-chirp)
    VID=$(qsub -terse -q all.q -N vid_FPCch -hold_jid "$SIM" \
          -o "$LOG_DIR/vid_FPCch.\$JOB_ID.log" -j y -cwd -V \
          "$SGE/rerender_videos_job.sh" "$RUN_DIR")
    echo "FPC-chirp: sim $SIM shed $SHED video $VID"
fi

# ---- FPCape-chirp (phaseCape: dt 1.25e-4 baked; save_rate 3600) ---------- #
RUN_REL="outputs/SGS_closure_ensemble/FPCape-chirp"
RUN_DIR="$QG_DIR/$RUN_REL"
if [ -f "$RUN_DIR/DNS_FR.npz" ]; then
    echo "[SKIP] $RUN_DIR/DNS_FR.npz exists — never overwrite (charter S8)"
else
    SIM=$(qsub -terse -N sgsCape_ch -o "$LOG_DIR/sgsCape_ch.\$JOB_ID.log" -j y -cwd -V \
          -q ibgpu.q -l gpu=1 -m ea -M "$QG_NOTIFY_EMAIL" \
          "$SGE/phaseCape_job.sh" \
          qg.grid.Nx=2048 qg.grid.Ny=2048 qg.time.save_rate=3600 \
          +qg.bc.inlet_table="$TABLES_DIR/chirp_dt1p25e-4_T120.npz" \
          +qg.diag.scalar_rate=10 +qg.diag.flush_every=500 \
          +qg.diag.out="$RUN_DIR/scalars.npz" \
          $CAPE_DIAG \
          hydra.run.dir="$RUN_REL")
    SHED=$(qsub -terse -q all.q -N shed_CAch -hold_jid "$SIM" \
          -o "$LOG_DIR/shed_CAch.\$JOB_ID.log" -j y -cwd -V \
          "$SGE/shedding_job.sh" "$RUN_DIR/scalars.npz" \
          --outdir "$RUN_DIR/shedding" --t-min 30.0 --tag FPCape-chirp)
    VID=$(qsub -terse -q all.q -N vid_CAch -hold_jid "$SIM" \
          -o "$LOG_DIR/vid_CAch.\$JOB_ID.log" -j y -cwd -V \
          "$SGE/rerender_videos_job.sh" "$RUN_DIR")
    echo "FPCape-chirp: sim $SIM shed $SHED video $VID"
fi

echo "=== OOD chirp cases submitted; runs land in outputs/SGS_closure_ensemble/{FPC,FPCape}-chirp/ ==="
