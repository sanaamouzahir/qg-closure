#!/bin/bash
# submit_dt_sweep.sh - Submit a dt-sensitivity sweep for paper-faithful
# forced turbulence simulations.
#
# Usage:
#   ./submit_dt_sweep.sh [--beta0 | --beta20 | --both]
#
#   --beta0   Run sweep at beta = 0 (isotropic)                  [DEFAULT]
#   --beta20  Run sweep at beta = 20 (zonal jets)
#   --both    Run sweep at beta = 0 and beta = 20 (10 jobs total)
#
# Each sweep submits 5 GPU jobs at 1024^2 for T=50, with dt values
# {1e-3, 5e-4, 2.5e-4, 1e-4, 5e-5}. For each run save_rate is set such that
# dt*save_rate = 0.05, giving 1000 snapshots per run regardless of dt.
#
# Outputs land in:
#   outputs/forced_turbulence_dt_sweep/beta_<tag>/dt_<tag>/

set -e

# ---------------------------------------------------------------------- #
# Sweep configuration                                                    #
# ---------------------------------------------------------------------- #

# (dt, save_rate) pairs: dt * save_rate = 0.05 in every case
# Format: "dt_value save_rate label"
DT_CONFIGS=(
    "1.0e-3   50  1em3"     # 20x coarser than paper
    "5.0e-4  100  5em4"     # 10x coarser than paper
    "2.5e-4  200  2p5em4"   # 5x coarser than paper
    "1.0e-4  500  1em4"     # 2x coarser than paper
    "5.0e-5 1000  5em5"     # paper reference
)

# Common run parameters
GRID_NX=1024
GRID_NY=1024
T_FINAL=50
SCENARIO=forced_turbulence_paper

# Paths
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
SWEEP_ROOT="$QG_ROOT/qg-simple-package-stable/src/qg/outputs/forced_turbulence_dt_sweep"
SUBMIT_SCRIPT="$QG_ROOT/qg-simple-package-stable/src/qg/submit_qg.sh"

mkdir -p "$SWEEP_ROOT"

# ---------------------------------------------------------------------- #
# Parse args                                                             #
# ---------------------------------------------------------------------- #

MODE="beta0"
if [ "$#" -ge 1 ]; then
    case "$1" in
        --beta0)  MODE="beta0"  ;;
        --beta20) MODE="beta20" ;;
        --both)   MODE="both"   ;;
        -h|--help)
            sed -n '2,18p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown arg: $1"
            echo "Run with --help for usage."
            exit 1
            ;;
    esac
fi

case "$MODE" in
    beta0)  BETAS=(0.0)        ;;
    beta20) BETAS=(20.0)       ;;
    both)   BETAS=(0.0 20.0)   ;;
esac

echo "==================================================================="
echo "Forced turbulence dt-sensitivity sweep"
echo "  mode     : $MODE"
echo "  betas    : ${BETAS[*]}"
echo "  scenario : $SCENARIO"
echo "  grid     : ${GRID_NX} x ${GRID_NY}"
echo "  T        : $T_FINAL"
echo "  dt set   : 1e-3, 5e-4, 2.5e-4, 1e-4, 5e-5"
echo "  outputs  : $SWEEP_ROOT/beta_<tag>/dt_<tag>/"
echo "==================================================================="
echo

# ---------------------------------------------------------------------- #
# Submit jobs                                                            #
# ---------------------------------------------------------------------- #

TOTAL=0

for BETA in "${BETAS[@]}"; do
    BETA_TAG=$(echo "$BETA" | tr '.' 'p')   # 0.0 -> 0p0, 20.0 -> 20p0
    BETA_DIR="$SWEEP_ROOT/beta_${BETA_TAG}"
    mkdir -p "$BETA_DIR"

    echo "##### beta = $BETA #####"
    echo

    for CONFIG in "${DT_CONFIGS[@]}"; do
        # Parse the config line
        read -r DT SAVE_RATE DT_LABEL <<< "$CONFIG"

        OUT_DIR="$BETA_DIR/dt_${DT_LABEL}"
        JOBNAME="ft_b${BETA_TAG}_dt${DT_LABEL}_T${T_FINAL}_${GRID_NX}_gpu"

        mkdir -p "$OUT_DIR"

        echo "-----  beta=$BETA, dt=$DT, save_rate=$SAVE_RATE  -----"
        echo "  jobname : $JOBNAME"
        echo "  out_dir : $OUT_DIR"

        "$SUBMIT_SCRIPT" "$JOBNAME" --gpu -- \
            scenario="$SCENARIO" \
            qg.grid.Nx="$GRID_NX" \
            qg.grid.Ny="$GRID_NY" \
            qg.time.T="$T_FINAL" \
            qg.time.dt="$DT" \
            qg.time.save_rate="$SAVE_RATE" \
            qg.pde.B="$BETA" \
            hydra.run.dir="$OUT_DIR"

        TOTAL=$((TOTAL + 1))
        echo
    done
done

echo "==================================================================="
echo "Submitted $TOTAL jobs."
echo
echo "Monitor with:"
echo "  qstat -u \$USER"
echo
echo "Tail any single log:"
echo "  tail -f $QG_ROOT/qg-simple-package-stable/src/qg/logs/ft_b<beta>_dt<dt>_T${T_FINAL}_${GRID_NX}_gpu.log"
echo
case "$MODE" in
    beta0)
        echo "After beta=0 sweep is verified, submit beta=20 with:"
        echo "  ./submit_dt_sweep.sh --beta20"
        ;;
    beta20)
        echo "Beta=20 sweep submitted."
        ;;
    both)
        echo "Both beta=0 and beta=20 sweeps submitted."
        ;;
esac
echo "==================================================================="
