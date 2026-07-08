#!/bin/bash
# submit_step3.sh  (FLOW PAST CYLINDER)
#
# Submit step 3 (analytical chain-rule N_dot/N_ddot validation) as a CPU
# SGE job (or run interactively).
#
# Step 3 is small -- it reads 3 snapshots per test, total memory < 200 MB.
# No mmap conversion phase needed if step 1 has already been run (which
# created the DNS_FR_omega.npy files). If step 1 hasn't run, this script
# will not auto-convert; do that via Step1's submit_step1.sh first.
#
# Usage:
#   ./submit_step3.sh                                 # auto, decaying-turbulence defaults
#   ./submit_step3.sh --interactive                   # run on the login node
#   ./submit_step3.sh --sweep-root /path/...
#   ./submit_step3.sh --ref-subdir dt_2em5            # if dt_1em5 not done
#   ./submit_step3.sh --n-snapshots 10
#   ./submit_step3.sh --skip-test-a
#   ./submit_step3.sh --skip-test-b
#
# Any other args are forwarded to step3_validate_n_derivatives.py.

set -e

# ---- Defaults ------------------------------------------------------------- #
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR="$QG_ROOT/qg-simple-package-stable/src/qg"
CONV_DIR="$QG_DIR/Convergence_studies/Decaying_Turbulence/Step3"

JOBNAME="step3_n_derivs_decay"
SWEEP_ROOT="$QG_DIR/outputs/decaying_turb_dt_sweep"
OUT_DIR="$CONV_DIR/figures"
INTERACTIVE=0

EXTRA_ARGS=()

# ---- Parse flags ---------------------------------------------------------- #
while [ $# -gt 0 ]; do
    case "$1" in
        --interactive)        INTERACTIVE=1; shift ;;
        --jobname)            JOBNAME="$2"; shift 2 ;;
        --sweep-root)         SWEEP_ROOT="$2"; shift 2 ;;
        --out-dir)            OUT_DIR="$2"; shift 2 ;;
        --ref-subdir)         EXTRA_ARGS+=(--ref-subdir "$2"); shift 2 ;;
        --n-snapshots)        EXTRA_ARGS+=(--n-snapshots "$2"); shift 2 ;;
        --n-starts)           EXTRA_ARGS+=(--n-starts "$2"); shift 2 ;;
        --h-fine)             EXTRA_ARGS+=(--h-fine "$2"); shift 2 ;;
        --h-ultrafine)        EXTRA_ARGS+=(--h-ultrafine "$2"); shift 2 ;;
        --skip-test-a)        EXTRA_ARGS+=(--skip-test-a); shift ;;
        --skip-test-b)        EXTRA_ARGS+=(--skip-test-b); shift ;;
        --skip-test-c)        EXTRA_ARGS+=(--skip-test-c); shift ;;
        -h|--help)            sed -n '2,22p' "$0"; exit 0 ;;
        *)                    EXTRA_ARGS+=("$1"); shift ;;
    esac
done

mkdir -p "$OUT_DIR"
LOG_DIR="$QG_ROOT/qg-wiener-conditioning/logs"
mkdir -p "$LOG_DIR"

# ---- Sanity check the sweep root ------------------------------------------ #
if [ ! -d "$SWEEP_ROOT" ]; then
    echo "ERROR: sweep root does not exist:"
    echo "  $SWEEP_ROOT"
    echo "Pass --sweep-root <dir> if your data lives elsewhere."
    exit 1
fi

# Default ref subdir is dt_1em5; check that it (or whichever the user picked)
# has been converted to .npy. If not, point them at Step1.
USER_REF=""
for ((i=0; i<${#EXTRA_ARGS[@]}; i++)); do
    if [ "${EXTRA_ARGS[$i]}" = "--ref-subdir" ]; then
        USER_REF="${EXTRA_ARGS[$((i+1))]}"
        break
    fi
done
REF="${USER_REF:-dt_1em5}"

if [ ! -f "$SWEEP_ROOT/$REF/DNS_FR_omega.npy" ]; then
    echo "ERROR: $SWEEP_ROOT/$REF/DNS_FR_omega.npy not found."
    echo "  Step 3 needs the .npy converted form."
    echo "  Run Step 1's submit_step1.sh first (it does the npz -> npy conversion)."
    echo "  Or run: cd ../Step1 && ./submit_step1.sh --convert-only"
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
    echo "[interactive] step3_validate_n_derivatives.py (decaying turb)"
    echo "==========================================================="
    source "$QG_ROOT/qg-env/bin/activate"
    export MPLCONFIGDIR="$QG_ROOT/.mplcache"
    mkdir -p "$MPLCONFIGDIR"
    export MPLBACKEND=Agg
    export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
    cd "$CONV_DIR"
    python -u step3_validate_n_derivatives.py "${PYTHON_ARGS[@]}"
    echo
    echo "Results:"
    ls -la "$OUT_DIR"
else
    LOG="$LOG_DIR/${JOBNAME}.log"
    JOB_SCRIPT="$CONV_DIR/step3_validate_n_derivatives_job.sh"
    chmod +x "$JOB_SCRIPT" 2>/dev/null || true

    echo "==========================================================="
    echo "submitting step 3 (decaying turbulence N-derivative validation)"
    echo "  jobname:    $JOBNAME"
    echo "  log:        $LOG"
    echo "  sweep root: $SWEEP_ROOT"
    echo "  ref subdir: $REF"
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

Output figures will appear in:
  $OUT_DIR/step3_test_a_single_mode.png
  $OUT_DIR/step3_test_b_2d_in_the_wild_cyl.png
  $OUT_DIR/step3_test_b_field_example_cyl.png
==========================================================
EOF
fi
