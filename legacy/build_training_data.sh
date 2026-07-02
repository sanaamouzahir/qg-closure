#!/bin/bash
# build_training_data.sh - Submit build_training_data.py to SGE on the cluster
# (or run interactively with --interactive).
#
# Usage:
#   ./build_training_data.sh --scenario decaying_turbulence
#   ./build_training_data.sh --scenario flow_past_cylinder
#   ./build_training_data.sh --scenario decaying_turbulence --interactive
#
# Common flags forwarded to build_training_data.py:
#   Single batch (default):
#     --batch-index 0
#   Ensemble (recommended for chaotic decaying turb):
#     --n-batches 20            # use first 20 batches
#     --batches "0,1,2,3"       # use a specific list
#   Per-batch seed budget:
#     --n-seeds 500
#   Stencil + truth:
#     --Delta-T 1e-3   --h-fine 1e-5   --h-ultrafine 5e-6
#   Time range for seeds:
#     --t-start 5.0    --t-end 55.0
#   Train/val/test split:
#     --split-mode auto    # by_batch if multi-batch, by_time if single batch
#   Compute:
#     --device cuda    --dtype float64

set -e

# ---- Defaults ------------------------------------------------------------- #
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR="$QG_ROOT/qg-simple-package-stable/src/qg"
SCRIPT_DIR="$QG_DIR/training"

SCENARIO=""
INTERACTIVE=0
JOBNAME=""

# Per-scenario defaults.
# Point at the ORIGINAL DNS_FR.npz (which contains ALL batches + times).
# The DNS_FR_omega.npy / DNS_FR_times.npy pair only contains a single batch
# (whichever was extracted by prepare_npz_for_mmap.py for the convergence
# study) so they're NOT useful here for ensemble training.
DECAY_OMEGA="$QG_DIR/outputs/decaying_turb_dt_sweep/dt_1em5/DNS_FR.npz"
DECAY_TIMES=""   # will be pulled from the npz's 'times' key
DECAY_YAML="$QG_DIR/conf/scenario/decaying_turbulence.yaml"

CYL_OMEGA="$QG_DIR/outputs/cylinder_dt_sweep_re200_v3/dt_1em5/DNS_FR.npz"
CYL_TIMES=""
CYL_YAML="$QG_DIR/conf/scenario/flow_past_cylinder_sponge.yaml"

OUT_DIR_DEFAULT="$SCRIPT_DIR/data"

EXTRA_ARGS=()
SOURCE_OMEGA=""
SOURCE_TIMES=""
SOURCE_YAML=""
OUT_DIR=""

# ---- Parse flags ---------------------------------------------------------- #
while [ $# -gt 0 ]; do
    case "$1" in
        --scenario)        SCENARIO="$2"; shift 2 ;;
        --source-omega)    SOURCE_OMEGA="$2"; shift 2 ;;
        --source-times)    SOURCE_TIMES="$2"; shift 2 ;;
        --source-yaml)     SOURCE_YAML="$2"; shift 2 ;;
        --out-dir)         OUT_DIR="$2"; shift 2 ;;
        --interactive)     INTERACTIVE=1; shift ;;
        --jobname)         JOBNAME="$2"; shift 2 ;;
        -h|--help)         sed -n '2,15p' "$0"; exit 0 ;;
        *)                 EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ---- Resolve scenario-specific defaults ----------------------------------- #
if [ -z "$SCENARIO" ]; then
    echo "ERROR: --scenario {decaying_turbulence|flow_past_cylinder} is required"
    exit 1
fi

case "$SCENARIO" in
    decaying_turbulence)
        [ -z "$SOURCE_OMEGA" ] && SOURCE_OMEGA="$DECAY_OMEGA"
        [ -z "$SOURCE_TIMES" ] && SOURCE_TIMES="$DECAY_TIMES"
        [ -z "$SOURCE_YAML"  ] && SOURCE_YAML="$DECAY_YAML"
        [ -z "$JOBNAME"      ] && JOBNAME="build_td_decay"
        ;;
    flow_past_cylinder)
        [ -z "$SOURCE_OMEGA" ] && SOURCE_OMEGA="$CYL_OMEGA"
        [ -z "$SOURCE_TIMES" ] && SOURCE_TIMES="$CYL_TIMES"
        [ -z "$SOURCE_YAML"  ] && SOURCE_YAML="$CYL_YAML"
        [ -z "$JOBNAME"      ] && JOBNAME="build_td_cyl"
        ;;
    *)
        echo "ERROR: unknown --scenario '$SCENARIO'"
        exit 1
        ;;
esac

[ -z "$OUT_DIR" ] && OUT_DIR="$OUT_DIR_DEFAULT"
mkdir -p "$OUT_DIR"

# ---- Pre-flight checks ---------------------------------------------------- #
if [ ! -f "$SOURCE_OMEGA" ]; then
    echo "ERROR: source omega not found at:"
    echo "  $SOURCE_OMEGA"
    echo "Pass --source-omega <path> if your data is elsewhere."
    exit 1
fi
if [ -n "$SOURCE_TIMES" ] && [ ! -f "$SOURCE_TIMES" ]; then
    echo "ERROR: --source-times specified but file not found:"
    echo "  $SOURCE_TIMES"
    exit 1
fi
if [ ! -f "$SOURCE_YAML" ]; then
    echo "ERROR: source YAML not found at:"
    echo "  $SOURCE_YAML"
    exit 1
fi

LOG_DIR="$QG_DIR/logs"
mkdir -p "$LOG_DIR"

# ---- Build python args ---------------------------------------------------- #
PYTHON_ARGS=(
    --scenario      "$SCENARIO"
    --source-omega  "$SOURCE_OMEGA"
    --source-yaml   "$SOURCE_YAML"
    --out-dir       "$OUT_DIR"
)
if [ -n "$SOURCE_TIMES" ]; then
    PYTHON_ARGS+=(--source-times "$SOURCE_TIMES")
fi
PYTHON_ARGS+=("${EXTRA_ARGS[@]}")

echo "==========================================================="
echo "build_training_data: $SCENARIO"
echo "  source omega: $SOURCE_OMEGA"
echo "  source times: $SOURCE_TIMES"
echo "  source yaml:  $SOURCE_YAML"
echo "  out dir:      $OUT_DIR"
echo "  extra args:   ${EXTRA_ARGS[*]}"
echo "==========================================================="

# ---- Interactive mode ----------------------------------------------------- #
if [ "$INTERACTIVE" -eq 1 ]; then
    echo "[interactive] running on login node (CPU/GPU as configured)"
    source "$QG_ROOT/qg-env/bin/activate"
    export MPLCONFIGDIR="$QG_ROOT/.mplcache"
    mkdir -p "$MPLCONFIGDIR"
    export MPLBACKEND=Agg
    cd "$SCRIPT_DIR"
    python -u build_training_data.py "${PYTHON_ARGS[@]}"
    exit 0
fi

# ---- SGE submission ------------------------------------------------------- #
JOB_LOG="$LOG_DIR/${JOBNAME}.log"
JOB_SCRIPT="$SCRIPT_DIR/build_training_data_job.sh"

if [ ! -f "$JOB_SCRIPT" ]; then
    echo "ERROR: job script not found:"
    echo "  $JOB_SCRIPT"
    echo "Did you copy build_training_data_job.sh into $SCRIPT_DIR?"
    exit 1
fi
chmod +x "$JOB_SCRIPT" 2>/dev/null || true
if head -1 "$JOB_SCRIPT" | grep -q $'\r'; then
    echo "ERROR: $JOB_SCRIPT has CRLF line endings."
    echo "Fix: sed -i 's/\\r\$//' $JOB_SCRIPT"
    exit 1
fi

QSUB_FLAGS=(
    -N "$JOBNAME"
    -o "$JOB_LOG"
    -e "$JOB_LOG"
    -j y
    -cwd
    -V
    -q "ibgpu.q"
    -l "h_vmem=32G"
    -l "gpu=1"
)
echo "submitting to SGE:"
echo "  jobname: $JOBNAME"
echo "  log:     $JOB_LOG"
echo "  script:  $JOB_SCRIPT"
qsub "${QSUB_FLAGS[@]}" "$JOB_SCRIPT" "${PYTHON_ARGS[@]}"

cat <<EOF

Watch progress with:
  tail -f $JOB_LOG

Check job status:
  qstat -u \$USER

Output HDF5 will land in:
  $OUT_DIR
EOF