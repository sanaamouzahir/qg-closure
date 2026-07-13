#!/bin/bash
# piff_eval_job.sh - S4 a priori evaluation package on a trained checkpoint.
# Runs eval_piff.py (calibration + field figures + summary.yaml) on the ckpt
# passed as $1 (path relative to ml_closure/ or absolute). Budget < 30 min.
#
# Usage (absolute -o path — do not rely on submit cwd):
#   LOGS=/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-sgs-closure/logs
#   qsub -N piff_eval -q ibgpu.q -l gpu=1 \
#        -o "$LOGS/\$JOB_NAME.\$JOB_ID.log" -j y -cwd -V \
#        piff_eval_job.sh runs_piff/<name>/best.pt

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
    echo "[piff_eval] selected GPU $IDLE_GPU on $HOSTNAME"
fi

CKPT="${1:?usage: piff_eval_job.sh <ckpt path> [extra eval_piff.py args: --config ... --outdir ...]}"
shift

cd "$BRANCH/ml_closure"
echo "[piff_eval] host $HOSTNAME date $(date -u +%FT%TZ) ckpt $CKPT extra: $*"
python -u eval_piff.py --ckpt "$CKPT" "$@"
echo "----------------------------------------------------------------------"
echo "[piff_eval] done at $(date -u +%FT%TZ)"
