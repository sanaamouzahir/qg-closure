#!/bin/bash
# submit_build_fixD_v2_K100.sh -- build dealiased K=100 training dataset.
#
# Uses float64 DNS_FR_omega.npy / DNS_FR_times.npy (produced by
# extract_omega_from_dns_npy.py from the raw DNS.npy of the dt_2em5 run).
# These bypass the float32 truncation that dataset.py inflicts on DNS_FR.npz.
#
# Usage:
#   ./submit_build_fixD_v2_K100.sh                # SGE GPU submission
#   ./submit_build_fixD_v2_K100.sh --interactive  # local
#   ./submit_build_fixD_v2_K100.sh --dry-run

set -e

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR="$QG_ROOT/qg-simple-package-stable/src/qg"
TRAINING_DIR="$QG_DIR/training"
LOG_DIR="$QG_ROOT/qg-wiener-conditioning/logs"
mkdir -p "$LOG_DIR"

JOBNAME="build_td_decay_fixD_v2_K100_dealiased_f64"
JOB_LOG="$LOG_DIR/${JOBNAME}.log"

INTERACTIVE=0
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --interactive)  INTERACTIVE=1; shift ;;
        --dry-run)      DRY_RUN=1; shift ;;
        -h|--help)      sed -n '2,30p' "$0"; exit 0 ;;
        *)              echo "unknown arg: $1"; exit 1 ;;
    esac
done

# Source: float64 npy files produced by extract_omega_from_dns_npy.py
# from the raw solver DNS.npy at dt_2em5.
SOURCE_OMEGA="$QG_DIR/outputs/decaying_turb_dt_sweep_float64/dt_2em5/DNS_FR_omega.npy"
SOURCE_TIMES="$QG_DIR/outputs/decaying_turb_dt_sweep_float64/dt_2em5/DNS_FR_times.npy"
SOURCE_YAML="$QG_DIR/conf/scenario/decaying_turbulence.yaml"
OUT_DIR="$TRAINING_DIR/data"

if [ ! -f "$SOURCE_OMEGA" ]; then
    echo "ERROR: $SOURCE_OMEGA not found -- run extract_omega_from_dns_npy.py first"
    exit 1
fi
if [ ! -f "$SOURCE_TIMES" ]; then
    echo "ERROR: $SOURCE_TIMES not found -- run extract_omega_from_dns_npy.py first"
    exit 1
fi
if [ ! -f "$SOURCE_YAML" ]; then
    echo "ERROR: $SOURCE_YAML not found"
    exit 1
fi

# Sanity check: verify source omega is actually float64.
# If the npy was generated via dataset.py (which truncates to float32),
# the whole pipeline would silently regress.
python -c "
import numpy as np
arr = np.load('$SOURCE_OMEGA', mmap_mode='r')
print(f'src omega: shape={arr.shape}, dtype={arr.dtype}')
assert arr.dtype == np.float64, f'expected float64, got {arr.dtype} -- aborting'
" || { echo "source omega is not float64 -- aborting"; exit 1; }

# Build args
PY_ARGS=(
    --scenario       decaying_turbulence
    --source-omega   "$SOURCE_OMEGA"
    --source-times   "$SOURCE_TIMES"
    --source-yaml    "$SOURCE_YAML"
    --out-dir        "$OUT_DIR"
    --Delta-T        1.0e-3
    --h-fine         1.0e-5
    --h-ultrafine    5.0e-6
    --n-seeds        500
    --t-start        5.0
    --n-batches      20
    --device         cuda
    --dtype          float64
    --split-mode     by_time
)

# Force the output subdir name to be ..._dealiased
# build_training_data_fixD_v2.py defaults to <scenario>_dT_<tag>; we rename
# after the fact to keep the dealiased + float64 version distinct from older
# aliased / float32-target builds.
EXPECTED_SUBDIR="$OUT_DIR/decaying_turbulence_dT_1em3"
TARGET_SUBDIR="$OUT_DIR/decaying_turbulence_dT_1em3_fixD_v2_float64_dealiased"

if [ -d "$TARGET_SUBDIR" ]; then
    echo "WARNING: $TARGET_SUBDIR already exists. Move/delete it first or rename."
    exit 1
fi
if [ -d "$EXPECTED_SUBDIR" ]; then
    echo "WARNING: $EXPECTED_SUBDIR already exists. The build will fail because"
    echo "         the script expects to create this directory fresh."
    echo "         Move it out of the way (rename to *_oldB or similar) first."
    exit 1
fi

echo "==================================================================="
echo " Build Fix D K=100 dealiased float64 training dataset             "
echo "==================================================================="
echo "  source-omega : $SOURCE_OMEGA"
echo "  source-times : $SOURCE_TIMES"
echo "  source-yaml  : $SOURCE_YAML"
echo "  out-dir      : $OUT_DIR"
echo "  expected     : $EXPECTED_SUBDIR (will be renamed -> $TARGET_SUBDIR)"
echo "  Delta_T      : 1e-3"
echo "  h_fine       : 1e-5  (K = 100)"
echo "  n_seeds      : 500 per batch"
echo "  n_batches    : 20"
echo "  total seeds  : 10000"
echo "  split-mode   : by_time"
echo "==================================================================="
echo

if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] cd $TRAINING_DIR"
    echo "[dry-run] python -u build_training_data_fixD_v2.py ${PY_ARGS[*]}"
    echo "[dry-run] mv $EXPECTED_SUBDIR $TARGET_SUBDIR"
    exit 0
fi

run_build() {
    cd "$TRAINING_DIR"
    python -u build_training_data_fixD_v2.py "${PY_ARGS[@]}"
    if [ -d "$EXPECTED_SUBDIR" ] && [ ! -d "$TARGET_SUBDIR" ]; then
        mv "$EXPECTED_SUBDIR" "$TARGET_SUBDIR"
        echo "renamed $EXPECTED_SUBDIR -> $TARGET_SUBDIR"
    fi
}

if [ "$INTERACTIVE" -eq 1 ]; then
    source "$QG_ROOT/qg-env/bin/activate"
    export MPLCONFIGDIR="$QG_ROOT/.mplcache"
    export CUDA_VISIBLE_DEVICES=5
    mkdir -p "$MPLCONFIGDIR"
    export MPLBACKEND=Agg
    run_build
    exit 0
fi

TMP_JOB="$LOG_DIR/${JOBNAME}.run.sh"
cat > "$TMP_JOB" <<EOF
#!/bin/bash
set -e
QG_ROOT=$QG_ROOT
source "\$QG_ROOT/qg-env/bin/activate"
export MPLCONFIGDIR="\$QG_ROOT/.mplcache"
mkdir -p "\$MPLCONFIGDIR"
export MPLBACKEND=Agg
echo "[$JOBNAME] hostname: \$HOSTNAME"
echo "[$JOBNAME] date:     \$(date -u +%FT%TZ)"
echo "[$JOBNAME] cuda dev: \${CUDA_VISIBLE_DEVICES:-?}"
echo "------------------------------------------------------------"
cd "$TRAINING_DIR"
$(printf '%q ' python -u build_training_data_fixD_v2.py "${PY_ARGS[@]}")
if [ -d "$EXPECTED_SUBDIR" ] && [ ! -d "$TARGET_SUBDIR" ]; then
    mv "$EXPECTED_SUBDIR" "$TARGET_SUBDIR"
    echo "renamed $EXPECTED_SUBDIR -> $TARGET_SUBDIR"
fi
echo "------------------------------------------------------------"
echo "[$JOBNAME] done at \$(date -u +%FT%TZ)"
EOF
chmod +x "$TMP_JOB"

echo "submitting $JOBNAME to ibgpu.q -> $JOB_LOG"
qsub -N "$JOBNAME" \
     -o "$JOB_LOG" -e "$JOB_LOG" -j y -V \
     -wd "$TRAINING_DIR" \
     -q "ibgpu.q" -l "gpu=1" \
     "$TMP_JOB"

echo
echo "watch with:  tail -f $JOB_LOG"
echo "qstat -u \$USER"