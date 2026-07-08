#!/bin/bash
# train_decay_fixD_v2_annealing.sh
#
# Re-train the bilinear closure NN using simulated-annealing LR schedule
# (cosine annealing with warm restarts + per-restart peak damping).
#
# Same architecture / inputs / target / dataset as train_decay_fixD_v2.sh,
# only the LR schedule changes:
#
#   schedule:           simulated_annealing
#   initial cycle len:  20 epochs
#   cycle multiplier:   2  (so cycles are 20, 40, 80, ...)
#   peak LR damping:    0.7 per restart  (0.7^k after k restarts)
#   floor LR (eta_min): 1e-6
#
# Defaults to the OLD float32-bug dataset so we can compare apples to apples
# against the run we're trying to beat.  Override ROOT_DIR to point at the
# new float64 build instead, e.g.:
#
#   ROOT_DIR=$QG_DIR/training/data/decaying_turbulence_dT_1em3_fixD_v2_float64 \
#       ./train_decay_fixD_v2_annealing.sh
#
# Output: $ROOT_DIR/training_runs/fixD_v2_annealing_<timestamp>/

set -e

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR="$QG_ROOT/qg-simple-package-stable/src/qg"
TRAINING_DIR="$QG_DIR/training"

# Default to OLD f32-bug dataset (same as the 19% plateau baseline).
ROOT_DIR="${ROOT_DIR:-$TRAINING_DIR/data/decaying_turbulence_dT_1em3_fixD_v2_OLD_f32bug}"

if [ ! -d "$ROOT_DIR" ]; then
    echo "ERROR: $ROOT_DIR does not exist."
    exit 1
fi
if [ ! -f "$ROOT_DIR/manifest.json" ]; then
    echo "ERROR: $ROOT_DIR/manifest.json missing."
    exit 1
fi

RUN_NAME="fixD_v2_annealing_$(date -u +%Y%m%d_%H%M%S)"
JOBNAME="train_decay_fixD_v2_annealing"

# Clear stale norm stats with wrong channel count
if [ -f "$ROOT_DIR/norm_stats.npz" ]; then
    n_in=$(python3 -c "
import numpy as np
try:
    s = np.load('$ROOT_DIR/norm_stats.npz')
    print(len(s['input_mean']))
except Exception:
    print(0)
" 2>/dev/null || echo 0)
    if [ "$n_in" != "6" ]; then
        echo "[train_decay_fixD_v2_annealing] removing stale norm_stats.npz (had $n_in channels, need 6)"
        rm -f "$ROOT_DIR/norm_stats.npz"
    fi
fi

# Use train_v2.sh as the submission wrapper, but force it to invoke
# train_v2_annealing.py instead of train_v2.py.  The simplest way: temporarily
# alias train_v2_annealing as if it were train_v2 in a sed-edited copy.
#
# Instead, we just call train_v2_annealing.py directly through a small inline
# qsub.  This avoids editing train_v2.sh.

LOG_DIR="$QG_ROOT/qg-wiener-conditioning/logs"
mkdir -p "$LOG_DIR"
JOB_LOG="$LOG_DIR/${JOBNAME}.log"

PY_ARGS=(
    --root-dir       "$ROOT_DIR"
    --run-name       "$RUN_NAME"
    --model          bilinear_closure
    --input-fields   omega_0 omega_m1 omega_m2 psi_0 psi_m1 psi_m2
    --target-field   f_NN_target
    --batch-size     4
    --epochs         200
    --lr             3.0e-4
    --weight-decay   1.0e-4
    --lr-schedule    simulated_annealing
    --warm-restart-T0     20
    --warm-restart-Tmult  2
    --lr-min         1.0e-6
    --sa-damping     0.7
    --hidden-channels 64
    --kernel         3
    --normalize
    --num-workers    2
    --print-every    1
)

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
echo "[$JOBNAME] root-dir: $ROOT_DIR"
echo "[$JOBNAME] run-name: $RUN_NAME"
echo "------------------------------------------------------------"
cd "$TRAINING_DIR"
$(printf '%q ' python -u train_v2_annealing.py "${PY_ARGS[@]}")
echo "------------------------------------------------------------"
echo "[$JOBNAME] done at \$(date -u +%FT%TZ)"
EOF
chmod +x "$TMP_JOB"

echo "submitting $JOBNAME -> $JOB_LOG"
qsub -N "$JOBNAME" \
     -o "$JOB_LOG" -e "$JOB_LOG" -j y -V \
     -wd "$TRAINING_DIR" \
     -q "ibgpu.q" -l "gpu=1" \
     -m ea -M "${QG_NOTIFY_EMAIL:-sanaamz@mit.edu}" \
     "$TMP_JOB"

echo
echo "watch with:  tail -f $JOB_LOG"
echo "log.csv:     $ROOT_DIR/training_runs/$RUN_NAME/log.csv"
