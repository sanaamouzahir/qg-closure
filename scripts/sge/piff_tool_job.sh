#!/bin/bash
# piff_tool_job.sh - run one ml_closure tool script on GPU: $1 = script name
# (e.g. calibrate_piff.py), rest = its args. Same env prep as piff_eval_job.sh.
#
# Usage:
#   LOGS=/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-sgs-closure/logs
#   qsub -N <name> -q ibgpu.q -l gpu=1 \
#        -o "$LOGS/\$JOB_NAME.\$JOB_ID.log" -j y -cwd -V \
#        piff_tool_job.sh calibrate_piff.py --ckpt runs_piff/<name>/best.pt

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
    echo "[piff_tool] selected GPU $IDLE_GPU on $HOSTNAME"
fi

TOOL="${1:?usage: piff_tool_job.sh <script.py> [args...]}"
shift

cd "$BRANCH/ml_closure"
echo "[piff_tool] host $HOSTNAME date $(date -u +%FT%TZ)"
echo "[piff_tool] cmd: python -u $TOOL $*"
echo "----------------------------------------------------------------------"
python -u "$TOOL" "$@"
echo "----------------------------------------------------------------------"
echo "[piff_tool] done at $(date -u +%FT%TZ)"
