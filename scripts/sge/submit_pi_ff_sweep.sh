#!/bin/bash
# submit_pi_ff_sweep.sh - Compute Pi_FF for the entire flow_past_cape beta sweep.
#
# Usage:
#   ./submit_pi_ff_sweep.sh [--paper-only | --remaining | --all]
#
#   --paper-only  Process only beta in {0, 0.1, 1.0}              [DEFAULT]
#   --remaining   Process only the 7 non-paper betas
#                 {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
#   --all         Process all 10 betas {0, 0.1, ..., 1.0}
#
# Each beta gets its own qsub job.
# All jobs use the paper filter parameters (scale=2, alpha=1.5) on GPU,
# and render diagnostic videos.
#
# Outputs land alongside DNS_FR.npz inside each beta_*/ directory:
#   - DNS_LES.npz                 (omega_bar + Pi_FF)
#   - DNS_LES_summary.yaml
#   - DNS_LES_*.mp4               (diagnostic videos)

set -e

# ---------------------------------------------------------------------- #
# Sweep configuration                                                    #
# ---------------------------------------------------------------------- #

PAPER_BETAS=(0.0 0.1 1.0)
REMAINING_BETAS=(0.2 0.3 0.4 0.5 0.6 0.7 0.8)
ALL_BETAS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 1.0)

# Pi_FF computation parameters
SCALE=2
ALPHA=1.5
DEVICE=cuda

# Paths
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
SWEEP_ROOT="$QG_ROOT/qg-simple-package-stable/src/qg/outputs/cape_sweep"
SUBMIT_SCRIPT="$QG_ROOT/qg-simple-package-stable/src/qg/submit_pi_ff.sh"

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
echo "Pi_FF sweep submission"
echo "  mode    : $MODE"
echo "  betas   : ${BETAS[*]}"
echo "  scale   : $SCALE"
echo "  alpha   : $ALPHA"
echo "  device  : $DEVICE"
echo "  outputs : $SWEEP_ROOT/beta_<value>/DNS_LES.npz"
echo "==================================================================="
echo

# ---------------------------------------------------------------------- #
# Submit one job per beta                                                #
# ---------------------------------------------------------------------- #

SKIPPED=0
SUBMITTED=0

for B in "${BETAS[@]}"; do
    B_TAG=$(echo "$B" | tr '.' 'p')
    OUT_DIR="$SWEEP_ROOT/beta_${B_TAG}"
    JOBNAME="pi_beta${B_TAG}_s${SCALE}_a${ALPHA//./p}"

    # Sanity check: input must exist
    if [ ! -f "$OUT_DIR/DNS_FR.npz" ]; then
        echo "-----  beta = $B  -----"
        echo "  [SKIP] $OUT_DIR/DNS_FR.npz not found"
        echo
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo "-----  beta = $B  -----"
    echo "  jobname : $JOBNAME"
    echo "  in_dir  : $OUT_DIR"

    if [ "$DEVICE" = "cuda" ]; then
        "$SUBMIT_SCRIPT" "$JOBNAME" --gpu -- \
            "$OUT_DIR" --scale "$SCALE" --alpha "$ALPHA" --device cuda
    else
        "$SUBMIT_SCRIPT" "$JOBNAME" -- \
            "$OUT_DIR" --scale "$SCALE" --alpha "$ALPHA" --device cpu
    fi

    SUBMITTED=$((SUBMITTED + 1))
    echo
done

echo "==================================================================="
echo "Submitted: $SUBMITTED jobs"
if [ "$SKIPPED" -gt 0 ]; then
    echo "Skipped:   $SKIPPED (DNS_FR.npz missing)"
fi
echo
echo "Monitor with:"
echo "  qstat -u \$USER"
echo
echo "Tail any single log:"
echo "  tail -f $QG_ROOT/qg-simple-package-stable/src/qg/logs/pi_beta<value>_s${SCALE}_a${ALPHA//./p}.log"
echo
case "$MODE" in
    paper-only)
        echo "After verifying these 3 paper-value jobs, submit the rest with:"
        echo "  ./submit_pi_ff_sweep.sh --remaining"
        ;;
    remaining)
        echo "Combined with the paper-only run, this completes the 10-beta Pi_FF sweep."
        ;;
    all)
        echo "All 10 Pi_FF jobs submitted."
        ;;
esac
echo "==================================================================="
