#!/bin/bash
# piff_t6arm_job.sh - one arm of the 3-arm T6 discrimination (orchestrator
# ruling 2026-07-12 under Sanaa's autonomy window). Forwards args verbatim to
# ml_closure/t6_arm.py. GPU job: -q ibgpu.q -l gpu=1 ONLY.
#
# Usage (absolute -o path — do not rely on submit cwd):
#   LOGS=/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-sgs-closure/logs
#   qsub -N pT6_A -q ibgpu.q -l gpu=1 \
#        -o "$LOGS/\$JOB_NAME.\$JOB_ID.log" -j y -cwd -V \
#        piff_t6arm_job.sh --arm A --epochs 150
#   qsub -N pT6_B ... piff_t6arm_job.sh --arm B --n-inducing 1024
#   qsub -N pT6_C ... piff_t6arm_job.sh --arm C --likelihood studentt

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
    echo "[piff_t6arm] selected GPU $IDLE_GPU on $HOSTNAME"
fi

cd "$BRANCH/ml_closure"
echo "[piff_t6arm] host $HOSTNAME date $(date -u +%FT%TZ)"
echo "[piff_t6arm] cmd: python -u t6_arm.py $*"
echo "----------------------------------------------------------------------"
python -u t6_arm.py "$@"
echo "----------------------------------------------------------------------"
echo "[piff_t6arm] done at $(date -u +%FT%TZ)"
