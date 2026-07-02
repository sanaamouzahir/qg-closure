#!/bin/bash
# submit_build_fixD.sh -- build the Fix D training dataset.
#
# Differences vs the prior build:
#   1) New target convention: f_NN_target = (1/12)[L*Ndot - 5*Nddot]
#      (Delta_T-independent physics bracket; not S^{-1}*e_NN as before)
#   2) Stores all analytical scaffolding fields as INPUT options:
#        N_0, N_dot_0_anal, N_ddot_0_anal,
#        L_omega, L2_omega, L3_omega, L_N, L2_N
#   3) Stores f_NN_target_from_e as a diagnostic (should ~match f_NN_target
#      up to higher-order Taylor truncation).
#
# Source: 20-batch decay reference run at dt=1e-5
# Output: $QG_DIR/training/data/decaying_turbulence_dT_1em3_fixD/
# Split:  by_time (default for n_batches=1) or by_batch otherwise
#
# Usage:
#   ./submit_build_fixD.sh                # SGE GPU submission
#   ./submit_build_fixD.sh --interactive  # local
#   ./submit_build_fixD.sh --dry-run

set -e

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR="$QG_ROOT/qg-simple-package-stable/src/qg"
TRAINING_DIR="$QG_DIR/training"
LOG_DIR="$QG_DIR/logs"
mkdir -p "$LOG_DIR"

JOBNAME="build_td_decay_fixD"
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

# Source: same as Fix B/C build -- the 20-batch decay reference (NPZ archive)
# DNS_FR.npz contains all 20 batches + times key in one file.
# (NOT DNS_FR_omega.npy, which is a different single-batch file)
SOURCE_OMEGA="$QG_DIR/outputs/decaying_turb_dt_sweep/dt_1em5/DNS_FR.npz"
SOURCE_TIMES=""    # not needed: NPZ contains 'times' key
SOURCE_YAML="$QG_DIR/conf/scenario/decaying_turbulence.yaml"
OUT_DIR="$TRAINING_DIR/data"

if [ ! -f "$SOURCE_OMEGA" ]; then
    echo "ERROR: $SOURCE_OMEGA not found"
    exit 1
fi
if [ ! -f "$SOURCE_YAML" ]; then
    echo "ERROR: $SOURCE_YAML not found"
    exit 1
fi

# Build args
#
# Source is DNS_FR.npz (20 batches in one NPZ archive). Use n_batches=20.
# The build script auto-detects the 'times' key in the NPZ, so --source-times
# isn't needed (and would error if pointed at a missing path).
PY_ARGS=(
    --scenario       decaying_turbulence
    --source-omega   "$SOURCE_OMEGA"
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
# If SOURCE_TIMES is non-empty (i.e., source is .npy), include it.
if [ -n "$SOURCE_TIMES" ]; then
    PY_ARGS=( "${PY_ARGS[@]}" --source-times "$SOURCE_TIMES" )
fi

# Force the output subdir name to be ..._fixD
# build_training_data_fixD.py defaults to <scenario>_dT_<tag>; we override.
# Looking at the source, --out-dir is the parent dir; subdir is constructed
# automatically. So we'll rename after the fact.
EXPECTED_SUBDIR="$OUT_DIR/decaying_turbulence_dT_1em3"
TARGET_SUBDIR="$OUT_DIR/decaying_turbulence_dT_1em3_fixD"

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
echo " Build Fix D training dataset                                      "
echo "==================================================================="
echo "  source-omega : $SOURCE_OMEGA"
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
    echo "[dry-run] python -u build_training_data_fixD.py ${PY_ARGS[*]}"
    echo "[dry-run] mv $EXPECTED_SUBDIR $TARGET_SUBDIR"
    exit 0
fi

run_build() {
    cd "$TRAINING_DIR"
    python -u build_training_data_fixD.py "${PY_ARGS[@]}"
    if [ -d "$EXPECTED_SUBDIR" ] && [ ! -d "$TARGET_SUBDIR" ]; then
        mv "$EXPECTED_SUBDIR" "$TARGET_SUBDIR"
        echo "renamed $EXPECTED_SUBDIR -> $TARGET_SUBDIR"
    fi
}

if [ "$INTERACTIVE" -eq 1 ]; then
    source "$QG_ROOT/qg-env/bin/activate"
    export MPLCONFIGDIR="$QG_ROOT/.mplcache"
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
$(printf '%q ' python -u build_training_data_fixD.py "${PY_ARGS[@]}")
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