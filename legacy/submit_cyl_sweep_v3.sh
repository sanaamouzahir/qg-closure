#!/bin/bash
# submit_cyl_sweep_v3.sh - Submit a dt-sensitivity sweep for the
# flow-past-cylinder scenario with BOTH eta_sponge AND eta_brinkman held
# constant across the sweep (= 2.5e-3 everywhere).
#
# v3 vs v2:
#   v2 fixed eta_sponge = 2.5e-3 across all dt (got rid of the wrap-around
#       artifacts), but left penalty=1.25 fixed so eta_brinkman = 1.25 * dt
#       still varied with dt.
#   v3 ALSO fixes eta_brinkman = 2.5e-3 across all dt.
#
# Why?
# In v2 the obstacle was effectively softer at large dt and stiffer at
# small dt. By Angot-Bruneau-Fabrie the velocity error from finite-eta
# Brinkman penalization is O(sqrt(eta)) globally, so this dt-coupled eta
# pollutes temporal-convergence analysis with an O(sqrt(dt)) Brinkman
# model error that masks AB2CN2's true 2nd-order truncation.
#
# In v3 every run solves the SAME PDE - only the time-stepping changes -
# so we should recover clean slope-2 in the convergence plots.
#
# Usage:
#   ./submit_cyl_sweep_v3.sh
#   ./submit_cyl_sweep_v3.sh --dry-run
#
# Outputs land in:
#   outputs/cylinder_dt_sweep_re200_v3/dt_<tag>/

set -e

# ---------------------------------------------------------------------- #
# Sweep configuration                                                    #
# ---------------------------------------------------------------------- #

# Target eta = sponge*dt = penalty*dt = 0.0025 for ALL dt.
# (Matches v2's working sponge eta and the v1 dt_max=2e-3 effective Brinkman eta.)
# Both penalty and sponge multipliers are precomputed so eta is exactly 2.5e-3.

# Format: "dt save_rate penalty sponge label"
DT_CONFIGS=(
    "2.0e-3      25      1.25      1.25     2em3"
    "1.0e-3      50      2.50      2.50     1em3"
    "5.0e-4     100      5.00      5.00     5em4"
    "2.5e-4     200     10.00     10.00     2p5em4"
    "1.25e-4    400     20.00     20.00     1p25em4"
    "2.0e-5    2500    125.00    125.00     2em5"
    "1.0e-5    5000    250.00    250.00     1em5"
)

# Common run parameters
GRID_NX=512
GRID_NY=512
T_FINAL=50
SCENARIO=flow_past_cylinder_sponge

# Paths
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
SWEEP_ROOT="$QG_ROOT/qg-simple-package-stable/src/qg/outputs/cylinder_dt_sweep_re200_v3"
SUBMIT_SCRIPT="$QG_ROOT/qg-simple-package-stable/src/qg/submit_qg.sh"

DRY_RUN=0
while [ "$#" -ge 1 ]; do
    case "$1" in
        --dry-run)    DRY_RUN=1; shift ;;
        --sweep-root) SWEEP_ROOT="$2"; shift 2 ;;
        -h|--help)    sed -n '2,29p' "$0"; exit 0 ;;
        *)            echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$SWEEP_ROOT"

# ---------------------------------------------------------------------- #
# Submit jobs                                                            #
# ---------------------------------------------------------------------- #

echo "==================================================================="
echo "Cylinder dt-sensitivity sweep v3 (CONSTANT eta_sponge AND eta_brinkman)"
echo "  scenario : $SCENARIO"
echo "  grid     : ${GRID_NX} x ${GRID_NY}"
echo "  T        : $T_FINAL"
echo "  dt set   : 2e-3, 1e-3, 5e-4, 2.5e-4, 1.25e-4, 2e-5, 1e-5"
echo "  sponge   : scaled to keep eta_sponge   = 0.0025 across all dt"
echo "  penalty  : scaled to keep eta_brinkman = 0.0025 across all dt"
echo "  outputs  : $SWEEP_ROOT/dt_<tag>/"
echo "  dry_run  : $DRY_RUN"
echo "==================================================================="
echo

TOTAL=0

for CONFIG in "${DT_CONFIGS[@]}"; do
    read -r DT SAVE_RATE PENALTY SPONGE DT_LABEL <<< "$CONFIG"

    OUT_DIR="$SWEEP_ROOT/dt_${DT_LABEL}"
    JOBNAME="cyl3_dt${DT_LABEL}_T${T_FINAL}_${GRID_NX}_gpu"

    mkdir -p "$OUT_DIR"

    ETA_BRINK=$(awk -v p="$PENALTY" -v d="$DT" 'BEGIN { printf "%.4e", p * d }')
    ETA_SPONGE=$(awk -v s="$SPONGE"  -v d="$DT" 'BEGIN { printf "%.4e", s * d }')

    echo "-----  dt=$DT, save_rate=$SAVE_RATE, penalty=$PENALTY, sponge=$SPONGE  -----"
    echo "  eta_brinkman = $ETA_BRINK,  eta_sponge = $ETA_SPONGE"
    echo "  jobname      : $JOBNAME"
    echo "  out_dir      : $OUT_DIR"

    CMD=(
        "$SUBMIT_SCRIPT" "$JOBNAME" --gpu --
        scenario="$SCENARIO"
        qg.grid.Nx="$GRID_NX"
        qg.grid.Ny="$GRID_NY"
        qg.time.T="$T_FINAL"
        qg.time.dt="$DT"
        qg.time.save_rate="$SAVE_RATE"
        qg.pde.penalty="$PENALTY"
        qg.bc.sponge="$SPONGE"
        hydra.run.dir="$OUT_DIR"
    )

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  [dry-run] would submit:"
        printf "    %s\n" "${CMD[@]}"
    else
        "${CMD[@]}"
    fi

    TOTAL=$((TOTAL + 1))
    echo
done

echo "==================================================================="
if [ "$DRY_RUN" -eq 1 ]; then
    echo "Dry-run done. $TOTAL jobs would be submitted."
else
    echo "Submitted $TOTAL jobs."
fi
echo
echo "Monitor with:"
echo "  qstat -u \$USER"
echo
echo "Tail any single log:"
echo "  tail -f $QG_ROOT/qg-simple-package-stable/src/qg/logs/cyl3_dt<tag>_T${T_FINAL}_${GRID_NX}_gpu.log"
echo
echo "After completion, run the Step1 + ConvergencePlot pipeline on v3:"
echo "  cd $QG_ROOT/qg-simple-package-stable/src/qg"
echo "  ./Convergence_studies/Flow_Past_Cylinder/Step1/submit_step1.sh \\"
echo "      --sweep-root $SWEEP_ROOT --jobname step1_cyl_v3 --force-convert"
echo "  ./Convergence_studies/Flow_Past_Cylinder/ConvergencePlot/convergence_plot.sh \\"
echo "      --sweep-root $SWEEP_ROOT --jobname convergence_plot_cyl_v3 --t-start 25.0"
echo "==================================================================="
