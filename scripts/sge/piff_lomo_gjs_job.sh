#!/bin/bash
# piff_lomo_gjs_job.sh - cape LOMO ladder REDO on the current production recipe
# (gaussian_jonly + structural sigma + ORDER-3 conditioning; conf_lomo_cape_gjs_*).
# Replaces the sharp-era ladder 1832241, killed 2026-07-13 in the ratified
# sharp-fleet kill (DECISIONS.md line 131: "redo post-gaussian"). COLD folds —
# no warm start (the gjs ckpt saw all 5 members; warm would leak the holdout).
# Five sequential folds, one GPU job, ~9-13 h (5 x 100 ep x ~65-95 s).
# Run names cape_lomo_gjs_<m> (the stale sharp partial cape_lomo_const is left
# on disk for the record and must NOT satisfy the resume-skip).
#
# Usage: qsub -N pLomoGjs -q ibgpu.q -l gpu=1 -o <logs>/... -j y -cwd -V \
#             piff_lomo_gjs_job.sh

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
for m in const sine ramp ou tel; do
    CONF="conf_lomo_cape_gjs_$m.yaml"
    RN="cape_lomo_gjs_$m"
    if [ -e "runs_piff/$RN/best.pt" ]; then
        echo "[piff_lomo_gjs] $RN exists — skipping (resume-safe)"
        continue
    fi
    echo "[piff_lomo_gjs] ===== fold $m ($(date -u +%FT%TZ)) ====="
    python -u train_piff.py --config "$CONF" --run-name "$RN"
    python -u eval_piff.py --ckpt "runs_piff/$RN/best.pt" --config "$CONF"
done
echo "[piff_lomo_gjs] all folds done at $(date -u +%FT%TZ)"
