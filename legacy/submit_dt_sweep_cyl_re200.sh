#!/bin/bash
# submit_cyl_sweep_v2.sh - Submit a dt-sensitivity sweep for the
# flow-past-cylinder scenario, with adjusted sponge coefficient at small
# dt to keep eta_sponge = sponge*dt approximately constant.
#
# Why?
# In v1 we used sponge=5.0 for all dt. The solver computes
# eta_sponge = sponge * dt, so eta shrinks as dt -> 0, making the AB2
# explicit treatment of the boundary sponge stiffer at smaller dt. This
# caused wrap-around / energy-spike artifacts at dt < 5e-4 in v1.
#
# v1 with sponge=5.0 was clean for dt >= 5e-4 (giving eta_sponge >= 2.5e-3).
# We keep that for those dt and scale sponge up for smaller dt to maintain
# eta_sponge = 2.5e-3 throughout.
#
# We do NOT change penalty (Brinkman): the obstacle-interior penalty
# was not the source of the v1 artifacts.
#
# Usage:
#   ./submit_cyl_sweep_v2.sh
#
# Outputs land in:
#   outputs/cylinder_dt_sweep_re200_v2/dt_<tag>/

set -e

# ---------------------------------------------------------------------- #
# Sweep configuration                                                    #
# ---------------------------------------------------------------------- #

# Target eta_sponge for dt < 5e-4 (matching the v1 dt=5e-4 working point):
#   eta_sponge = sponge * dt = 0.0025
# For dt >= 5e-4 we use the v1 sponge=5.0 value as-is.

# Format: "dt save_rate sponge label"
DT_CONFIGS=(
    "2.0e-3      25      5.00     2em3"
    "1.0e-3      50      5.00     1em3"
    "5.0e-4     100      5.00     5em4"
    "2.5e-4     200     10.00     2p5em4"
    "1.25e-4    400     20.00     1p25em4"
    "2.0e-5    2500    125.00     2em5"
    "1.0e-5    5000    250.00     1em5"
)

# Common run parameters
GRID_NX=512
GRID_NY=512
T_FINAL=50
SCENARIO=flow_past_cylinder_sponge

# Paths
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
SWEEP_ROOT="$QG_ROOT/qg-simple-package-stable/src/qg/outputs/cylinder_dt_sweep_re200_v2"
SUBMIT_SCRIPT="$QG_ROOT/qg-simple-package-stable/src/qg/submit_qg.sh"

mkdir -p "$SWEEP_ROOT"

# ---------------------------------------------------------------------- #
# Submit jobs                                                            #
# ---------------------------------------------------------------------- #

echo "==================================================================="
echo "Cylinder dt-sensitivity sweep v2 (constant eta_sponge for small dt)"
echo "  scenario : $SCENARIO"
echo "  grid     : ${GRID_NX} x ${GRID_NY}"
echo "  T        : $T_FINAL"
echo "  dt set   : 2e-3, 1e-3, 5e-4, 2.5e-4, 1.25e-4, 2e-5, 1e-5"
echo "  sponge   : 5.0 for dt >= 5e-4; scaled to keep eta_sponge=0.0025 below"
echo "  penalty  : unchanged from YAML default (1.25, Brinkman)"
echo "  outputs  : $SWEEP_ROOT/dt_<tag>/"
echo "==================================================================="
echo

TOTAL=0

for CONFIG in "${DT_CONFIGS[@]}"; do
    read -r DT SAVE_RATE SPONGE DT_LABEL <<< "$CONFIG"

    OUT_DIR="$SWEEP_ROOT/dt_${DT_LABEL}"
    JOBNAME="cyl2_dt${DT_LABEL}_T${T_FINAL}_${GRID_NX}_gpu"

    mkdir -p "$OUT_DIR"

    ETA_SPONGE=$(awk -v s="$SPONGE" -v d="$DT" 'BEGIN { printf "%.4e", s * d }')

    echo "-----  dt=$DT, save_rate=$SAVE_RATE, sponge=$SPONGE (eta=$ETA_SPONGE)  -----"
    echo "  jobname : $JOBNAME"
    echo "  out_dir : $OUT_DIR"

    "$SUBMIT_SCRIPT" "$JOBNAME" --gpu -- \
        scenario="$SCENARIO" \
        qg.grid.Nx="$GRID_NX" \
        qg.grid.Ny="$GRID_NY" \
        qg.time.T="$T_FINAL" \
        qg.time.dt="$DT" \
        qg.time.save_rate="$SAVE_RATE" \
        qg.bc.sponge="$SPONGE" \
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
echo "  tail -f $QG_ROOT/qg-simple-package-stable/src/qg/logs/cyl2_dt<tag>_T${T_FINAL}_${GRID_NX}_gpu.log"
echo
echo "After completion, run the Step1/2/3 pipeline pointing to:"
echo "  $SWEEP_ROOT"
echo "==================================================================="