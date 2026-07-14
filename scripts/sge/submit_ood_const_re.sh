#!/bin/bash
# submit_ood_const_re.sh - OOD-Re constant cases (Sanaa order 2026-07-14,
# OOD list item 1): FPC + FPCape at const Re 6000 (above the trained band
# max 5600) and const Re 400 (far below min 2200). 4 FR sims + chained
# shedding + video rerender each.
#
# PRIORITY CONTRACT (Sanaa): "these jobs should always have less priority
# than other jobs I run." All jobs here submit with -p -1023 (the lowest
# user priority) so the scheduler NEVER starts them while any normal-
# priority job is waiting. HONEST LIMIT: SGE cannot PAUSE a running GPU
# sim in a way that frees GPU memory for a new job (a suspended CUDA
# process keeps its allocation). If a running OOD sim is in the way,
# qdel it — DNS snapshots are written as the run progresses, and it can
# be resubmitted to continue-from-snapshot via extract_restart_ic (or
# simply rerun; they are "run whenever" by design).
#
# dt matches the training members per geometry: FPC 2.5e-4 (phaseB baked),
# cape 1.25e-4 (phaseCape baked). save_rate 1800 / 3600 (= dt_save 0.45).
# Re 400 caveat (flagged, run as ordered): U = 0.205 -> T_shed ~ 29, so
# T = 120 holds only ~3 shedding periods after t_wait; statistics thin.
#
# Release: OODRE_RELEASE=1 QG_NOTIFY_EMAIL=... ./submit_ood_const_re.sh
# GPU rule: exactly -q ibgpu.q -l gpu=1. Never ibamd.q, never h_vmem.

set -e
echo "HOLD: set OODRE_RELEASE=1 to submit"
[ "${OODRE_RELEASE:-0}" = "1" ] || exit 1

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR="$QG_ROOT/qg-simple-package-stable/src/qg"
BRANCH="$QG_ROOT/qg-sgs-closure"
SGE="$BRANCH/scripts/sge"
TABLES_DIR="$QG_DIR/outputs/SGS_closure_ensemble/tables"
LOG_DIR="$BRANCH/logs"
PRIO="-p -1023"

if [ -z "${QG_NOTIFY_EMAIL:-}" ]; then
    echo "ERROR: QG_NOTIFY_EMAIL unset"; exit 1
fi
mkdir -p "$LOG_DIR"

CAPE_DIAG='+qg.diag.length=1.0 +qg.diag.probes=[[6.0265,4.0],[7.0265,4.0],[8.0265,4.0],[6.0265,4.5],[6.0265,3.5],[6.0265,2.0]]'

submit_case () {
    local geom=$1 re=$2
    local short job tbl run_rel run_dir worker extra srate
    if [ "$geom" = FPC ]; then
        tbl="$TABLES_DIR/constRe${re}_dt2p5e-4_T120.npz"
        worker="$SGE/phaseB_job.sh"; extra=""; srate=1800
    else
        tbl="$TABLES_DIR/constRe${re}_dt1p25e-4_T120.npz"
        worker="$SGE/phaseCape_job.sh"; extra="+qg.diag.flush_every=500 $CAPE_DIAG"; srate=3600
    fi
    [ -f "$tbl" ] || { echo "ERROR: missing $tbl"; exit 1; }
    run_rel="outputs/SGS_closure_ensemble/${geom}-constRe${re}"
    run_dir="$QG_DIR/$run_rel"
    if [ -f "$run_dir/DNS_FR.npz" ]; then
        echo "[SKIP] $run_dir/DNS_FR.npz exists — never overwrite (charter S8)"
        return
    fi
    short="${geom:0:3}R${re:0:2}"          # e.g. FPCR60 / FPCR40 (cape: FPC->FPC? use geom-specific)
    [ "$geom" = FPCape ] && short="CaR${re:0:2}"
    job=$(qsub -terse -N "ood_${short}" $PRIO \
          -o "$LOG_DIR/ood_${short}.\$JOB_ID.log" -j y -cwd -V \
          -q ibgpu.q -l gpu=1 -m ea -M "$QG_NOTIFY_EMAIL" \
          "$worker" \
          qg.grid.Nx=2048 qg.grid.Ny=2048 qg.time.save_rate=$srate \
          +qg.bc.inlet_table="$tbl" \
          +qg.diag.scalar_rate=10 +qg.diag.out="$run_dir/scalars.npz" \
          $extra \
          hydra.run.dir="$run_rel")
    qsub -q all.q -N "shd_${short}" $PRIO -hold_jid "$job" \
         -o "$LOG_DIR/shd_${short}.\$JOB_ID.log" -j y -cwd -V \
         "$SGE/shedding_job.sh" "$run_dir/scalars.npz" \
         --outdir "$run_dir/shedding" --t-min 30.0 --tag "${geom}-constRe${re}" > /dev/null
    qsub -q all.q -N "vid_${short}" $PRIO -hold_jid "$job" \
         -o "$LOG_DIR/vid_${short}.\$JOB_ID.log" -j y -cwd -V \
         "$SGE/rerender_videos_job.sh" "$run_dir" > /dev/null
    echo "${geom}-constRe${re}: sim $job (+shed/video chained, all -p -1023)"
}

submit_case FPC 6000
submit_case FPC 400
submit_case FPCape 6000
submit_case FPCape 400

echo "=== OOD const-Re cases queued at LOWEST priority; they never outrank a normal job ==="
