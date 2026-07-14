#!/bin/bash
# piff_lomo_fpc_gjs_job.sh - FIRST-EVER cylinder LOMO ladder, on the current
# production recipe (gaussian_jonly + structural sigma + upstream mask +
# ORDER-3 conditioning; conf_lomo_fpc_gjs_*). Context (Sanaa question
# 2026-07-14): FPC generalization was previously tested only by the 1-to-4
# cross-eval of the single-member model (prod_ext150 on the other members);
# leave-one-out on the 5-member ensemble was never run. COLD folds — no warm
# start (the gjs ckpt saw all 5 members; warm would leak the holdout).
# Five sequential folds, one GPU job, ~9-13 h (5 x 100 ep x ~65-95 s).
# Fold tag telSA maps to member FPC-telS-A.
#
# Usage: qsub -N pLomoFgjs -q ibgpu.q -l gpu=1 -o <logs>/... -j y -cwd -V \
#             piff_lomo_fpc_gjs_job.sh

#$ -S /bin/bash
#$ -cwd
#$ -V

set -e

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
BRANCH="$QG_ROOT/qg-sgs-closure"

source "$QG_ROOT/qg-env-piff/bin/activate"
export TMPDIR="$QG_ROOT/tmp"
export MPLCONFIGDIR="$QG_ROOT/mplcache"
export PYTHONUNBUFFERED=1

mkdir -p "$QG_ROOT/cache/torch" "$QG_ROOT/cache/triton" "$QG_ROOT/cache/nvrtc"
export TORCH_EXTENSIONS_DIR="$QG_ROOT/cache/torch"
export TRITON_CACHE_DIR="$QG_ROOT/cache/triton"
export PYTORCH_KERNEL_CACHE_PATH="$QG_ROOT/cache/nvrtc"

if command -v nvidia-smi >/dev/null 2>&1; then
    IDLE_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t',' -k2 -n | head -1 | awk -F',' '{gsub(/ /,""); print $1}')
    export CUDA_VISIBLE_DEVICES="$IDLE_GPU"
    echo "[piff_lomo_gjs] selected GPU $IDLE_GPU on $HOSTNAME"
fi

cd "$BRANCH/ml_closure"
for m in const sine ramp ou telSA; do
    CONF="conf_lomo_fpc_gjs_$m.yaml"
    RN="fpc_lomo_gjs_$m"
    if [ -e "runs_piff/$RN/best.pt" ]; then
        echo "[piff_lomo_gjs] $RN exists — skipping (resume-safe)"
        continue
    fi
    echo "[piff_lomo_gjs] ===== fold $m ($(date -u +%FT%TZ)) ====="
    python -u train_piff.py --config "$CONF" --run-name "$RN"
    python -u eval_piff.py --ckpt "runs_piff/$RN/best.pt" --config "$CONF"
done
echo "[piff_lomo_gjs] all folds done at $(date -u +%FT%TZ)"
