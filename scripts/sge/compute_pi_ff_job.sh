#!/bin/bash
# compute_pi_ff_job.sh - Submit a Pi_FF computation as a batch job.
#
# Usage:
#   qsub -N <jobname> -o <stdout.log> -e <stderr.log> [scheduler-flags] \
#        compute_pi_ff_job.sh <save_path> [--scale N] [--alpha A] [--device cuda] [--no-videos]
#
# All <args...> are passed verbatim to compute_pi_ff.py.
#
# Example (single):
#   qsub -N pi_beta0p0 -o logs/pi_beta0p0.log -e logs/pi_beta0p0.err -j y \
#        -l gpu=0 -q ibgpu.q \
#        compute_pi_ff_job.sh outputs/cape_sweep/beta_0p0 --scale 2 --alpha 1.5 --device cuda
#
# This script handles:
#   1. venv activation
#   2. cache redirection off $HOME (torch / triton / nvrtc)
#   3. idle-GPU auto-selection
#   4. running compute_pi_ff.py with the passed args

#$ -S /bin/bash
#$ -cwd
#$ -V

set -e

# ---- 1. Activate venv & set caches ------------------------------------- #
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure

source "$QG_ROOT/qg-env/bin/activate"

export TMPDIR=/tmp
export PIP_CACHE_DIR="$QG_ROOT/pip-cache"
export MPLCONFIGDIR="$QG_ROOT/mplcache"
export PATH="$QG_ROOT/qg-env/bin-extra:$PATH"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# ---- 1b. Redirect torch / triton / nvrtc kernel caches off $HOME ------- #
mkdir -p "$QG_ROOT/cache/torch" "$QG_ROOT/cache/triton" "$QG_ROOT/cache/nvrtc"
export TORCH_EXTENSIONS_DIR="$QG_ROOT/cache/torch"
export TRITON_CACHE_DIR="$QG_ROOT/cache/triton"
export PYTORCH_KERNEL_CACHE_PATH="$QG_ROOT/cache/nvrtc"

# ---- 2. Auto-pick an idle GPU ONLY when SGE assigned none --------------- #
# (sge-checker W1 2026-07-09: never clobber the scheduler's device grant)
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] || [ -n "${SGE_HGR_gpu:-}" ]; then
    echo "[pi_ff_job] using scheduler-assigned GPU(s): CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} SGE_HGR_gpu=${SGE_HGR_gpu:-}"
elif command -v nvidia-smi >/dev/null 2>&1; then
    IDLE_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t',' -k2 -n | head -1 | awk -F',' '{gsub(/ /,""); print $1}')
    export CUDA_VISIBLE_DEVICES="$IDLE_GPU"
    echo "[pi_ff_job] selected GPU $IDLE_GPU on $HOSTNAME"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total \
        --format=csv,noheader -i "$IDLE_GPU"
else
    echo "[pi_ff_job] no nvidia-smi on $HOSTNAME; running on CPU"
fi

# ---- 3. Run ------------------------------------------------------------ #
cd "$QG_ROOT/qg-simple-package-stable/src/qg"

echo "[pi_ff_job] hostname: $HOSTNAME"
echo "[pi_ff_job] date: $(date -u +%FT%TZ)"
echo "[pi_ff_job] python: $(which python)"
echo "[pi_ff_job] cmd: python -u compute_pi_ff.py $*"
echo "----------------------------------------------------------------------"

python -u compute_pi_ff.py "$@"

echo "----------------------------------------------------------------------"
echo "[pi_ff_job] done at $(date -u +%FT%TZ)"
