#!/bin/bash
# submit_decaying_turb_sweep.sh - Submit a dt-sensitivity sweep for the
# decaying-turbulence scenario.
#
# Decaying turbulence has NO forcing, NO obstacle, NO sponge -- a clean
# QG-with-Brinkman=0 setup. Used as an unambiguous reference for the
# closure validation: any artifact we see in the cylinder sweep that does
# NOT show up here is sponge/Brinkman-induced.
#
# Usage:
#   ./submit_decaying_turb_sweep.sh
#
# Submits 7 GPU jobs at 256^2 for T=60, with dt values:
#   {2e-3, 1e-3, 5e-4, 2.5e-4, 1.25e-4, 2e-5, 1e-5}
# matching the cylinder sweep dt set so we can compare timestep-by-timestep.
# For each run save_rate is set such that dt*save_rate = 0.05, giving 1200
# snapshots per run.
#
# Outputs land in:
#   outputs/decaying_turb_dt_sweep/dt_<tag>/

set -e

# ---------------------------------------------------------------------- #
# Sweep configuration                                                    #
# ---------------------------------------------------------------------- #

# (dt, save_rate) pairs: dt * save_rate = 0.05 in every case
# save_rate = 0.05 / dt
# Format: "dt_value save_rate label"
DT_CONFIGS=(
    "2.0e-3      25  2em3"
    "1.0e-3      50  1em3"
    "5.0e-4     100  5em4"
    "2.5e-4     200  2p5em4"
    "1.25e-4    400  1p25em4"
    "2.0e-5    2500  2em5"
    "1.0e-5    5000  1em5"
)

# Common run parameters
GRID_NX=256
GRID_NY=256
T_FINAL=60
SCENARIO=decaying_turbulence

# Paths
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
SWEEP_ROOT="$QG_ROOT/qg-simple-package-stable/src/qg/outputs/decaying_turb_dt_sweep"
SUBMIT_SCRIPT="$QG_ROOT/qg-simple-package-stable/src/qg/submit_qg.sh"

mkdir -p "$SWEEP_ROOT"

# ---------------------------------------------------------------------- #
# Submit jobs                                                            #
# ---------------------------------------------------------------------- #

echo "==================================================================="
echo "Decaying turbulence dt-sensitivity sweep"
echo "  scenario : $SCENARIO"
echo "  grid     : ${GRID_NX} x ${GRID_NY}"
echo "  T        : $T_FINAL"
echo "  dt set   : 2e-3, 1e-3, 5e-4, 2.5e-4, 1.25e-4, 2e-5, 1e-5"
echo "  outputs  : $SWEEP_ROOT/dt_<tag>/"
echo "==================================================================="
echo

TOTAL=0

for CONFIG in "${DT_CONFIGS[@]}"; do
    read -r DT SAVE_RATE DT_LABEL <<< "$CONFIG"

    OUT_DIR="$SWEEP_ROOT/dt_${DT_LABEL}"
    JOBNAME="decay_dt${DT_LABEL}_T${T_FINAL}_${GRID_NX}_gpu"

    mkdir -p "$OUT_DIR"

    echo "-----  dt=$DT, save_rate=$SAVE_RATE  -----"
    echo "  jobname : $JOBNAME"
    echo "  out_dir : $OUT_DIR"

    "$SUBMIT_SCRIPT" "$JOBNAME" --gpu -- \
        scenario="$SCENARIO" \
        qg.grid.Nx="$GRID_NX" \
        qg.grid.Ny="$GRID_NY" \
        qg.time.T="$T_FINAL" \
        qg.time.dt="$DT" \
        qg.time.save_rate="$SAVE_RATE" \
        hydra.run.dir="$OUT_DIR"

    TOTAL=$((TOTAL + 1))
    echo
done

echo "==================================================================="
echo "Submitted $TOTAL jobs."
echo
echo "Monitor with:"
echo "  qstat -u \$USER"
echo
echo "Tail any single log:"
echo "  tail -f $QG_ROOT/qg-simple-package-stable/src/qg/logs/decay_dt<tag>_T${T_FINAL}_${GRID_NX}_gpu.log"
echo
echo "Outputs will land in:"
echo "  $SWEEP_ROOT/dt_<tag>/"
echo "==================================================================="
