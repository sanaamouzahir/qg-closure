#!/bin/bash
# build_training_data_decay_bytime.sh - Rebuild the decay v2 training dataset
# with --split-mode by_time, while keeping all other parameters identical to
# the original by_batch build. Same source data, just a different split policy.
#
# Why: by_batch (current) holds out entire IC realizations for val/test, so
# train and val sample disjoint regions of the decay attractor and the model
# tests extrapolation across realizations. by_time uses all 20 batches in
# train+val+test, but holds out the last seed-times within each batch -- a
# more sensible test for the actual deployment scenario (closure applied to
# already-seen flow regimes at later times).
#
# Output: $OUT_DIR/decaying_turbulence_dT_1em3_bytime/
#         (does NOT overwrite the original $OUT_DIR/decaying_turbulence_dT_1em3/)
#
# Usage:
#   ./build_training_data_decay_bytime.sh                 # SGE submission
#   ./build_training_data_decay_bytime.sh --interactive   # local
#   ./build_training_data_decay_bytime.sh --dry-run

set -e

# ---- Paths ---------------------------------------------------------------- #
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR="$QG_ROOT/qg-simple-package-stable/src/qg"
SCRIPT_DIR="$QG_DIR/training"

# Source data: decay v2 (Re ~ 1e5) reference run
SOURCE_OMEGA="$QG_DIR/outputs/decaying_turb_dt_sweep/dt_1em5/DNS_FR.npz"
SOURCE_YAML="$QG_DIR/conf/scenario/decaying_turbulence.yaml"

# Output directory: NEW name so we don't clobber the original by_batch dataset
OUT_DIR="$SCRIPT_DIR/data"
DATASET_NAME="decaying_turbulence_dT_1em3_bytime"

# Default build params (must match the original by_batch build to keep
# everything else apples-to-apples). Adjust if your original used different
# values.
DELTA_T=1.0e-3
H_FINE=1.0e-5
H_ULTRAFINE=5.0e-6
N_BATCHES=20
N_SEEDS=500
T_START=5.0
T_END=55.0
DEVICE=cuda
DTYPE=float64

JOBNAME="build_td_decay_bytime"
INTERACTIVE=0
DRY_RUN=0

# ---- Parse flags ---------------------------------------------------------- #
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
        -h|--help)       sed -n '2,30p' "$0"; exit 0 ;;
        *)               echo "unknown arg: $1"; exit 1 ;;
    esac
done

if [ ! -f "$SOURCE_OMEGA" ]; then
    echo "ERROR: source omega not found:"
    echo "  $SOURCE_OMEGA"
    exit 1
fi

mkdir -p "$OUT_DIR"
LOG_DIR="$QG_ROOT/qg-wiener-conditioning/logs"
mkdir -p "$LOG_DIR"

# ---- Banner --------------------------------------------------------------- #
echo "==================================================================="
echo " Rebuild decay training data with --split-mode by_time            "
echo "==================================================================="
echo "  source       : $SOURCE_OMEGA"
echo "  out-dir      : $OUT_DIR/$DATASET_NAME"
echo "  Delta_T      : $DELTA_T"
echo "  h_fine       : $H_FINE  (K = $(awk "BEGIN{print int($DELTA_T/$H_FINE)}"))"
echo "  h_ultrafine  : $H_ULTRAFINE"
echo "  n_batches    : $N_BATCHES   ($N_SEEDS seeds each)"
echo "  t-range      : [$T_START, $T_END]"
echo "  split-mode   : by_time"
echo "  device       : $DEVICE"
echo "  dtype        : $DTYPE"
echo "==================================================================="
echo

# ---- Build the python args ----------------------------------------------- #
PY_ARGS=(
    --scenario decaying_turbulence
    --source-omega "$SOURCE_OMEGA"
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

# Note: build_training_data.py auto-names the output dir from --Delta-T,
# producing "<scenario>_dT_<tag>". To get our distinct "_bytime" suffix we
# rename after the build finishes (unless build_training_data.py supports a
# direct override -- check by adding an explicit --dataset-name later if
# desired).
EXPECTED_AUTONAME_DIR="$OUT_DIR/decaying_turbulence_dT_1em3"
TARGET_DIR="$OUT_DIR/$DATASET_NAME"

if [ -d "$TARGET_DIR" ]; then
    echo "WARN: target dir already exists: $TARGET_DIR"
    echo "      remove it first if you want a fresh rebuild."
    exit 1
fi

# Define a wrapper that runs the build, then renames the output dir.
WRAPPER_CMD=$(cat <<EOF
set -e
cd "$SCRIPT_DIR"
python -u build_training_data.py ${PY_ARGS[@]}
if [ -d "$EXPECTED_AUTONAME_DIR" ] && [ ! -e "$TARGET_DIR" ]; then
    mv "$EXPECTED_AUTONAME_DIR" "$TARGET_DIR"
    echo "renamed $EXPECTED_AUTONAME_DIR -> $TARGET_DIR"
elif [ ! -d "$EXPECTED_AUTONAME_DIR" ]; then
    echo "ERROR: build_training_data.py did not produce $EXPECTED_AUTONAME_DIR"
    exit 1
fi
EOF
)

if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] would run:"
    echo "  python build_training_data.py ${PY_ARGS[@]}"
    echo "  mv $EXPECTED_AUTONAME_DIR $TARGET_DIR"
    exit 0
fi

# IMPORTANT: $EXPECTED_AUTONAME_DIR collision risk -- if your original
# by_batch build is in that exact directory, this rename will fail.
# Move it aside before rebuilding:
if [ -e "$EXPECTED_AUTONAME_DIR" ]; then
    if [ -d "$EXPECTED_AUTONAME_DIR" ] && [ -f "$EXPECTED_AUTONAME_DIR/manifest.json" ]; then
        echo "Detected existing by_batch dataset at:"
        echo "  $EXPECTED_AUTONAME_DIR"
        # Read split_mode from manifest to confirm which one this is
        EXISTING_MODE=$(python -c "import json; print(json.load(open('$EXPECTED_AUTONAME_DIR/manifest.json')).get('split_mode','?'))" 2>/dev/null || echo '?')
        echo "  (its split_mode = $EXISTING_MODE)"
        if [ "$EXISTING_MODE" = "by_batch" ]; then
            echo "  Renaming it to ${EXPECTED_AUTONAME_DIR}_bybatch to preserve it..."
            mv "$EXPECTED_AUTONAME_DIR" "${EXPECTED_AUTONAME_DIR}_bybatch"
            echo "  done."
        else
            echo "  ERROR: existing dataset's split_mode is '$EXISTING_MODE', not 'by_batch'."
            echo "  Manually rename or remove it first to avoid clobbering."
            exit 1
        fi
    else
        echo "ERROR: $EXPECTED_AUTONAME_DIR exists but has no manifest.json."
        echo "Manually inspect/remove it first."
        exit 1
    fi
fi

# ---- Submit / run --------------------------------------------------------- #
if [ "$INTERACTIVE" -eq 1 ]; then
    echo "[interactive] running build + rename..."
    source "$QG_ROOT/qg-env/bin/activate"
    eval "$WRAPPER_CMD"
    echo "DONE."
    exit 0
fi

# SGE submission. Use the existing build_training_data_job.sh worker, but
# we need the rename step to run AFTER python returns. Easiest: write a tiny
# shell script for the job to execute.
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
SCRIPT_DIR="$SCRIPT_DIR"
cd "\$SCRIPT_DIR"
echo "[build_td_bytime] hostname: \$HOSTNAME"
echo "[build_td_bytime] date:     \$(date -u +%FT%TZ)"
echo "[build_td_bytime] cuda dev: \${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "[build_td_bytime] cwd:      \$PWD"
echo "------------------------------------------------------------"
python -u build_training_data.py ${PY_ARGS[@]}
if [ -d "$EXPECTED_AUTONAME_DIR" ] && [ ! -e "$TARGET_DIR" ]; then
    mv "$EXPECTED_AUTONAME_DIR" "$TARGET_DIR"
    echo "renamed $EXPECTED_AUTONAME_DIR -> $TARGET_DIR"
fi
echo "------------------------------------------------------------"
echo "[build_td_bytime] done at \$(date -u +%FT%TZ)"
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

echo "submitting build_training_data.py rebuild ($JOBNAME) -> $JOB_LOG"
qsub "${QSUB_FLAGS[@]}" "$TMP_JOB"
echo
echo "watch with:  tail -f $JOB_LOG"
