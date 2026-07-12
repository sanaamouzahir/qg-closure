#!/bin/bash
# piff_train_job.sh - one Pi_FF grid point (ML SPEC 01 S3.2): forwards args
# verbatim to train_piff.py. Submitted 6x by submit_piff_grid.sh
# (lr x weight-decay grid), one GPU job each: -q ibgpu.q -l gpu=1 ONLY.
#
# Usage:
#   qsub -N piff_<tag> -q ibgpu.q -l gpu=1 \
#        -o logs/\$JOB_NAME.\$JOB_ID.log -j y -cwd -V \
#        piff_train_job.sh --run-name <name> --lr 3.0e-4 --weight-decay 1.0e-4

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
    echo "[piff_train] selected GPU $IDLE_GPU on $HOSTNAME"
fi

cd "$BRANCH/ml_closure"
echo "[piff_train] host $HOSTNAME date $(date -u +%FT%TZ)"
echo "[piff_train] cmd: python -u train_piff.py $*"
echo "----------------------------------------------------------------------"
python -u train_piff.py "$@"
echo "----------------------------------------------------------------------"
echo "[piff_train] done at $(date -u +%FT%TZ)"
