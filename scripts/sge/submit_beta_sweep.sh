#!/bin/bash
# submit_beta_sweep.sh - Submit a sweep of QG flow-past-cape simulations
# across multiple beta values.
#
# Usage:
#   ./submit_beta_sweep.sh [--paper-only | --remaining | --all]
#
#   --paper-only  Submit only beta in {0, 0.1, 1.0}              [DEFAULT]
#   --remaining   Submit only the 7 non-paper betas
#                 {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
#   --all         Submit all 10 betas {0, 0.1, ..., 1.0}
#
# Examples:
#   ./submit_beta_sweep.sh                # paper values first (recommended)
#   ./submit_beta_sweep.sh --remaining    # after paper values run cleanly
#   ./submit_beta_sweep.sh --all          # everything in one shot
#
# Each beta value gets its own qsub job named cape_beta<value>_T50_1024_gpu.
# All jobs run on GPU at 1024^2 for T=50 with seed=86 (paper default).
# Outputs land in outputs/cape_sweep/beta_<value>/ alongside per-run logs.

set -e

# ---------------------------------------------------------------------- #
# Sweep configuration                                                    #
# ---------------------------------------------------------------------- #

PAPER_BETAS=(0.0 0.1 1.0)
REMAINING_BETAS=(0.2 0.3 0.4 0.5 0.6 0.7 0.8)
ALL_BETAS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 1.0)

# Common run parameters for every beta
GRID_NX=1024
GRID_NY=1024
T_FINAL=100
SEED=86
SCENARIO=flow_past_cape

# Output/log root
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
SWEEP_ROOT="$QG_ROOT/qg-simple-package-stable/src/qg/outputs/cape_sweep"
SUBMIT_SCRIPT="$QG_ROOT/qg-simple-package-stable/src/qg/submit_qg.sh"

mkdir -p "$SWEEP_ROOT"

# ---------------------------------------------------------------------- #
# Parse args                                                             #
# ---------------------------------------------------------------------- #

MODE="paper-only"
if [ "$#" -ge 1 ]; then
    case "$1" in
        --paper-only) MODE="paper-only" ;;
        --remaining)  MODE="remaining"  ;;
        --all)        MODE="all"        ;;
        -h|--help)
            sed -n '2,20p' "$0"
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
    paper-only) BETAS=("${PAPER_BETAS[@]}") ;;
    remaining)  BETAS=("${REMAINING_BETAS[@]}") ;;
    all)        BETAS=("${ALL_BETAS[@]}") ;;
esac

echo "==================================================================="
echo "Beta sweep submission"
echo "  mode     : $MODE"
echo "  betas    : ${BETAS[*]}"
echo "  scenario : $SCENARIO"
echo "  grid     : ${GRID_NX} x ${GRID_NY}"
echo "  T        : $T_FINAL"
echo "  seed     : $SEED"
echo "  outputs  : $SWEEP_ROOT/beta_<value>/"
echo "==================================================================="
echo

# ---------------------------------------------------------------------- #
# Submit one job per beta                                                #
# ---------------------------------------------------------------------- #

for B in "${BETAS[@]}"; do
    # Use _ in directory/job names so qsub doesn't choke on dots.
    B_TAG=$(echo "$B" | tr '.' 'p')   # 0.1 -> 0p1, 1.0 -> 1p0
    JOBNAME="cape_beta${B_TAG}_T${T_FINAL}_${GRID_NX}_gpu"
    OUT_DIR="$SWEEP_ROOT/beta_${B_TAG}"

    mkdir -p "$OUT_DIR"

    echo "-----  beta = $B  -----"
    echo "  jobname : $JOBNAME"
    echo "  out_dir : $OUT_DIR"

    "$SUBMIT_SCRIPT" "$JOBNAME" --gpu -- \
        scenario="$SCENARIO" \
        qg.grid.Nx="$GRID_NX" \
        qg.grid.Ny="$GRID_NY" \
        qg.time.T="$T_FINAL" \
        qg.pde.B="$B" \
        qg.ic.seed="$SEED" \
        hydra.run.dir="$OUT_DIR"

    echo
done

echo "==================================================================="
echo "Submitted ${#BETAS[@]} jobs."
echo
echo "Monitor with:"
echo "  qstat -u \$USER"
echo
echo "Tail any single log:"
echo "  tail -f $QG_ROOT/qg-simple-package-stable/src/qg/logs/cape_beta<value>_T${T_FINAL}_${GRID_NX}_gpu.log"
echo
case "$MODE" in
    paper-only)
        echo "After verifying these 3 paper-value runs, submit the rest with:"
        echo "  ./submit_beta_sweep.sh --remaining"
        ;;
    remaining)
        echo "Combined with the paper-only run, this completes the 10-beta sweep."
        ;;
    all)
        echo "All 10 betas submitted."
        ;;
esac
echo "==================================================================="