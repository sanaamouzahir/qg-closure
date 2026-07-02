#!/bin/bash
# build_training_data_decay_richN.sh - Rebuild the decay v2 training data
# with the patched build_training_data.py + dataset.py that save:
#   - N_0:           nonlinear RHS at t^0
#   - N_dot_0_anal:  analytical dN/dt at t^0
# in addition to the existing fields. Also uses --split-mode by_time so train
# and val draw from the same trajectories (different times), which is the
# right test for the deployment scenario.
#
# Output: $OUT_DIR/decaying_turbulence_dT_1em3_richN/
#   - manifest.json
#   - split.npz
#   - norm_stats.npz
#   - samples/sample_<N>.npz   each with 13 fields (now includes N_0,
#                              N_dot_0_anal)
#
# Usage:
#   ./build_training_data_decay_richN.sh                 # SGE submission
#   ./build_training_data_decay_richN.sh --interactive   # local
#   ./build_training_data_decay_richN.sh --dry-run

set -e

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR="$QG_ROOT/qg-simple-package-stable/src/qg"
SCRIPT_DIR="$QG_DIR/training"

# Source data: decay v2 reference run (smallest dt). Adjust if your decay v2
# reference is in a different location.
SOURCE_OMEGA="$QG_DIR/outputs/decaying_turb_dt_sweep/dt_1em5/DNS_FR_omega.npy"
SOURCE_TIMES="$QG_DIR/outputs/decaying_turb_dt_sweep/dt_1em5/DNS_FR_times.npy"
SOURCE_YAML="$QG_DIR/conf/scenario/decaying_turbulence.yaml"

OUT_DIR="$SCRIPT_DIR/data"
DATASET_NAME="decaying_turbulence_dT_1em3_richN"

# Match the params that produced the original by_batch dataset
DELTA_T=1.0e-3
H_FINE=1.0e-5
H_ULTRAFINE=5.0e-6
N_BATCHES=20
N_SEEDS=500
T_START=5.0
T_END=55.0
DEVICE=cuda
DTYPE=float64

JOBNAME="build_td_decay_richN"
INTERACTIVE=0
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --interactive)   INTERACTIVE=1; shift ;;
        --dry-run)       DRY_RUN=1; shift ;;
        --jobname)       JOBNAME="$2"; shift 2 ;;
        --source-omega)  SOURCE_OMEGA="$2"; shift 2 ;;
        --out-dir)       OUT_DIR="$2"; shift 2 ;;
        --dataset-name)  DATASET_NAME="$2"; shift 2 ;;
        --n-batches)     N_BATCHES="$2"; shift 2 ;;
        --n-seeds)       N_SEEDS="$2"; shift 2 ;;
        --t-start)       T_START="$2"; shift 2 ;;
        --t-end)         T_END="$2"; shift 2 ;;
        -h|--help)       sed -n '2,22p' "$0"; exit 0 ;;
        *)               echo "unknown arg: $1"; exit 1 ;;
    esac
done

if [ ! -f "$SOURCE_OMEGA" ]; then
    echo "ERROR: source omega not found: $SOURCE_OMEGA"
    exit 1
fi

mkdir -p "$OUT_DIR"
LOG_DIR="$QG_DIR/logs"
mkdir -p "$LOG_DIR"

EXPECTED_AUTONAME_DIR="$OUT_DIR/decaying_turbulence_dT_1em3"
TARGET_DIR="$OUT_DIR/$DATASET_NAME"

if [ -d "$TARGET_DIR" ]; then
    echo "ERROR: target dir already exists: $TARGET_DIR"
    echo "       remove it first if you want a fresh rebuild."
    exit 1
fi

# If by_batch dataset is in the way, move it aside
if [ -e "$EXPECTED_AUTONAME_DIR" ]; then
    if [ -f "$EXPECTED_AUTONAME_DIR/manifest.json" ]; then
        EXISTING_MODE=$(python -c "import json; print(json.load(open('$EXPECTED_AUTONAME_DIR/manifest.json')).get('split_mode','?'))" 2>/dev/null || echo '?')
        echo "Existing dataset at $EXPECTED_AUTONAME_DIR (split_mode=$EXISTING_MODE)"
        BACKUP_NAME="${EXPECTED_AUTONAME_DIR}_${EXISTING_MODE}_backup_$(date +%Y%m%d_%H%M%S)"
        echo "Renaming to $BACKUP_NAME"
        mv "$EXPECTED_AUTONAME_DIR" "$BACKUP_NAME"
    else
        echo "ERROR: $EXPECTED_AUTONAME_DIR exists but has no manifest.json. Inspect manually."
        exit 1
    fi
fi

PY_ARGS=(
    --scenario decaying_turbulence
    --source-omega "$SOURCE_OMEGA"
    --source-times "$SOURCE_TIMES"
    --source-yaml "$SOURCE_YAML"
    --out-dir "$OUT_DIR"
    --Delta-T "$DELTA_T"
    --h-fine "$H_FINE"
    --h-ultrafine "$H_ULTRAFINE"
    --n-batches "$N_BATCHES"
    --n-seeds "$N_SEEDS"
    --t-start "$T_START"
    --t-end "$T_END"
    --split-mode by_time
    --device "$DEVICE"
    --dtype "$DTYPE"
)

echo "==================================================================="
echo " Rebuild decay training data with N_0 + N_dot_0_anal               "
echo "==================================================================="
echo "  source       : $SOURCE_OMEGA"
echo "  out-dir      : $TARGET_DIR"
echo "  Delta_T      : $DELTA_T  (h_fine=$H_FINE, K=$(awk "BEGIN{print int($DELTA_T/$H_FINE)}"))"
echo "  n_batches    : $N_BATCHES   ($N_SEEDS seeds each)"
echo "  t-range      : [$T_START, $T_END]"
echo "  split-mode   : by_time"
echo "  device       : $DEVICE,  dtype: $DTYPE"
echo "==================================================================="
echo

if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] would run:"
    echo "  python build_training_data.py ${PY_ARGS[@]}"
    echo "  mv $EXPECTED_AUTONAME_DIR -> $TARGET_DIR"
    exit 0
fi

if [ "$INTERACTIVE" -eq 1 ]; then
    source "$QG_ROOT/qg-env/bin/activate"
    cd "$SCRIPT_DIR"
    python -u build_training_data.py "${PY_ARGS[@]}"
    if [ -d "$EXPECTED_AUTONAME_DIR" ] && [ ! -e "$TARGET_DIR" ]; then
        mv "$EXPECTED_AUTONAME_DIR" "$TARGET_DIR"
        echo "renamed $EXPECTED_AUTONAME_DIR -> $TARGET_DIR"
    fi
    exit 0
fi

# SGE submission
TMP_JOB="$LOG_DIR/${JOBNAME}.run.sh"
cat > "$TMP_JOB" <<EOF
#!/bin/bash
set -e
QG_ROOT=$QG_ROOT
source "\$QG_ROOT/qg-env/bin/activate"
export MPLCONFIGDIR="\$QG_ROOT/.mplcache"
mkdir -p "\$MPLCONFIGDIR"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
cd "$SCRIPT_DIR"
echo "[$JOBNAME] hostname: \$HOSTNAME"
echo "[$JOBNAME] date:     \$(date -u +%FT%TZ)"
echo "[$JOBNAME] cwd:      \$PWD"
echo "------------------------------------------------------------"
$(printf '%q ' python -u build_training_data.py "${PY_ARGS[@]}")
if [ -d "$EXPECTED_AUTONAME_DIR" ] && [ ! -e "$TARGET_DIR" ]; then
    mv "$EXPECTED_AUTONAME_DIR" "$TARGET_DIR"
    echo "renamed $EXPECTED_AUTONAME_DIR -> $TARGET_DIR"
fi
echo "------------------------------------------------------------"
echo "[$JOBNAME] done at \$(date -u +%FT%TZ)"
EOF
chmod +x "$TMP_JOB"

JOB_LOG="$LOG_DIR/${JOBNAME}.log"
QSUB_FLAGS=(
    -N "$JOBNAME"
    -o "$JOB_LOG"
    -e "$JOB_LOG"
    -j y
    -V
    -wd "$SCRIPT_DIR"
    -q "ibgpu.q"
    -l "gpu=1"
)

qsub "${QSUB_FLAGS[@]}" "$TMP_JOB"
echo "submitted -> $JOB_LOG"
echo "Watch with:  tail -f $JOB_LOG"