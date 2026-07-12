#!/bin/bash
# piff_smoke_job.sh - T7: one full Pi_FF pipeline smoke on GPU (ML SPEC 01 S5).
# 2 epochs end-to-end on FPC-const s=4: train_piff.py (all per-epoch logs,
# curves.png, residual-PDF kurtosis after the warmup epoch) then eval_piff.py
# on best.pt (calibration + field figures + summary.yaml). Budget < 30 min.
#
# Usage (absolute -o path — do not rely on submit cwd):
#   LOGS=/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-sgs-closure/logs
#   qsub -N piff_smoke -q ibgpu.q -l gpu=1 \
#        -o "$LOGS/\$JOB_NAME.\$JOB_ID.log" -j y -cwd -V piff_smoke_job.sh

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
    echo "[piff_smoke] selected GPU $IDLE_GPU on $HOSTNAME"
fi

cd "$BRANCH/ml_closure"
RUN_NAME="smoke_T7_$(date -u +%Y%m%d_%H%M)"
echo "[piff_smoke] host $HOSTNAME date $(date -u +%FT%TZ) run $RUN_NAME"
echo "----------------------------------------------------------------------"
python -u train_piff.py --run-name "$RUN_NAME" --epochs 2
python -u eval_piff.py --ckpt "runs_piff/$RUN_NAME/best.pt"
echo "----------------------------------------------------------------------"
echo "[piff_smoke] T7 done at $(date -u +%FT%TZ); artifacts in ml_closure/runs_piff/$RUN_NAME/"
