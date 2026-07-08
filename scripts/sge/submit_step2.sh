#!/bin/bash
# submit_step2.sh  (FLOW PAST CYLINDER)
#
# Submit the step 2 field-comparison plot as a CPU SGE job (or run
# interactively).
#
# Step 2 is much smaller than step 1 -- it reads only 3 snapshots per
# run, total memory footprint < 100 MB. So one phase only, no conversion
# step (assumes prepare_npz_for_mmap.py has already been run by step 1).
#
# By default uses decaying-turbulence defaults; override with flags as needed:
#
#   ./submit_step2.sh                                # auto, decaying-turbulence defaults
#   ./submit_step2.sh --interactive                  # run on the login node
#   ./submit_step2.sh --sweep-root /path/...
#   ./submit_step2.sh --convergence-csv /path/...    # uses step1's csv to pick times
#   ./submit_step2.sh --t-early 5 --t-middle 25 --t-late 45
#   ./submit_step2.sh --early-threshold 0.05 --middle-threshold 0.30 ...
#
# Any other args are forwarded to step2_field_comparison.py.

set -e

# ---- Defaults ------------------------------------------------------------- #
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR="$QG_ROOT/qg-simple-package-stable/src/qg"
CONV_DIR="$QG_DIR/Convergence_studies/Decaying_Turbulence/Step2"
STEP1_DIR="$QG_DIR/Convergence_studies/Decaying_Turbulence/Step1"

JOBNAME="step2_field_decay"
SWEEP_ROOT="$QG_DIR/outputs/decaying_turb_dt_sweep"
OUT_DIR="$CONV_DIR/figures"
# Default: ingest step 1's csv if it exists (so times are auto-picked from
# convergence diagnostics rather than from decaying-turb fallbacks).
DEFAULT_CSV="$STEP1_DIR/figures/step1_convergence_decay.csv"
INTERACTIVE=0

EXTRA_ARGS=()
USER_PROVIDED_CSV=0

# ---- Parse flags ---------------------------------------------------------- #
while [ $# -gt 0 ]; do
    case "$1" in
        --interactive)        INTERACTIVE=1; shift ;;
        --jobname)            JOBNAME="$2"; shift 2 ;;
        --sweep-root)         SWEEP_ROOT="$2"; shift 2 ;;
        --out-dir)            OUT_DIR="$2"; shift 2 ;;
        --convergence-csv)    EXTRA_ARGS+=(--convergence-csv "$2"); USER_PROVIDED_CSV=1; shift 2 ;;
        --t-early)            EXTRA_ARGS+=(--t-early "$2"); shift 2 ;;
        --t-middle)           EXTRA_ARGS+=(--t-middle "$2"); shift 2 ;;
        --t-late)             EXTRA_ARGS+=(--t-late "$2"); shift 2 ;;
        --early-threshold)    EXTRA_ARGS+=(--early-threshold "$2"); shift 2 ;;
        --middle-threshold)   EXTRA_ARGS+=(--middle-threshold "$2"); shift 2 ;;
        --late-threshold)     EXTRA_ARGS+=(--late-threshold "$2"); shift 2 ;;
        -h|--help)            sed -n '2,22p' "$0"; exit 0 ;;
        *)                    EXTRA_ARGS+=("$1"); shift ;;
    esac
done

mkdir -p "$OUT_DIR"
LOG_DIR="$QG_ROOT/qg-wiener-conditioning/logs"
mkdir -p "$LOG_DIR"

# Inject the default csv only if the user didn't pass one explicitly AND it exists
if [ "$USER_PROVIDED_CSV" -eq 0 ] && [ -f "$DEFAULT_CSV" ]; then
    EXTRA_ARGS=(--convergence-csv "$DEFAULT_CSV" "${EXTRA_ARGS[@]}")
    echo "(using step 1's convergence csv: $DEFAULT_CSV)"
elif [ "$USER_PROVIDED_CSV" -eq 0 ]; then
    echo "(no step 1 convergence csv found; decaying-turb fallback times will be used)"
fi

# ---- Sanity check the sweep root ------------------------------------------ #
if [ ! -d "$SWEEP_ROOT" ]; then
    echo "ERROR: sweep root does not exist:"
    echo "  $SWEEP_ROOT"
    echo "Pass --sweep-root <dir> if your data lives elsewhere."
    exit 1
fi

# Verify .npy exists for at least the 3 displayed runs (coarsest, middle, ref)
HAS_DATA=0
for sub in dt_2em3 dt_2p5em4 dt_1em5; do
    if [ -f "$SWEEP_ROOT/$sub/DNS_FR_omega.npy" ]; then
        HAS_DATA=1
        break
    fi
done
if [ "$HAS_DATA" -eq 0 ]; then
    echo "ERROR: no DNS_FR_omega.npy found in $SWEEP_ROOT/dt_*/"
    echo "Run step 1's submit_step1.sh first (it converts npz -> npy)."
    exit 1
fi

# ---- Run ------------------------------------------------------------------ #
PYTHON_ARGS=(
    --sweep-root "$SWEEP_ROOT"
    --out-dir    "$OUT_DIR"
    "${EXTRA_ARGS[@]}"
)

if [ "$INTERACTIVE" -eq 1 ]; then
    echo "==========================================================="
    echo "[interactive] step2_field_comparison.py (decaying turbulence)"
    echo "==========================================================="
    source "$QG_ROOT/qg-env/bin/activate"
    export MPLCONFIGDIR="$QG_ROOT/.mplcache"
    mkdir -p "$MPLCONFIGDIR"
    export MPLBACKEND=Agg
    export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
    cd "$CONV_DIR"
    python -u step2_field_comparison.py "${PYTHON_ARGS[@]}"
    echo
    echo "Results:"
    ls -la "$OUT_DIR"
else
    LOG="$LOG_DIR/${JOBNAME}.log"
    JOB_SCRIPT="$CONV_DIR/step2_field_comparison_job.sh"
    chmod +x "$JOB_SCRIPT" 2>/dev/null || true

    echo "==========================================================="
    echo "submitting step 2 (decaying turbulence field comparison)"
    echo "  jobname:    $JOBNAME"
    echo "  log:        $LOG"
    echo "  sweep root: $SWEEP_ROOT"
    echo "  out dir:    $OUT_DIR"
    echo "==========================================================="

    QSUB_FLAGS=(
        -N "$JOBNAME"
        -o "$LOG"
        -e "$LOG"
        -j y
        -cwd
        -V
        -q "ibfdr.q"
    )

    qsub "${QSUB_FLAGS[@]}" "$JOB_SCRIPT" "${PYTHON_ARGS[@]}"
    echo
    cat <<EOF
==========================================================
Watch progress with:
  tail -f $LOG

Output figure will appear in:
  $OUT_DIR/step2_fields_decay.png
==========================================================
EOF
fi
