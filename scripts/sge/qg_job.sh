#!/bin/bash
# qg_job.sh - Submit a QG simulation as a batch job.
#
# Usage:
#   qsub -N <jobname> -o <stdout.log> -e <stderr.log> [scheduler-flags] qg_job.sh <args...>
#
# All <args...> are passed verbatim to run_qg.py. Examples:
#   qsub -N cape_512 -o /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/cape_512.log -e /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/cape_512.err -j y \
#        -q all.q \
#        qg_job.sh scenario=flow_past_cape qg.grid.Nx=512 qg.grid.Ny=512 \
#                  qg.time.T=20 hydra.run.dir=outputs/cape_v3_512_T20
#
#   qsub -N cape_gpu -o /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/cape_gpu.log -e /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/cape_gpu.err -j y \
#        -l gpu=1 -q ibgpu.q \
#        qg_job.sh scenario=flow_past_cape qg.grid.Nx=1024 qg.grid.Ny=1024 \
#                  +qg.grid.device=cuda qg.time.T=20 hydra.run.dir=outputs/cape_v3_1024_gpu
#
# The script handles env activation, idle-GPU selection, and progress flushing.

#$ -S /bin/bash
#$ -cwd                      # run in submission dir
#$ -V                        # inherit environment
#$ -o /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/$JOB_NAME.$JOB_ID.log
#$ -e /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/$JOB_NAME.$JOB_ID.err

set -e

# ---- 1. Activate venv & set caches -------------------------------------- #
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
source "$QG_ROOT/qg-env/bin/activate"
export TMPDIR="$QG_ROOT/tmp"
export PIP_CACHE_DIR="$QG_ROOT/pip-cache"
export MPLCONFIGDIR="$QG_ROOT/mplcache"
export PATH="$QG_ROOT/qg-env/bin-extra:$PATH"
export PYTHONUNBUFFERED=1

# ---- 2. If a GPU is requested, pick an idle one ------------------------- #
# Heuristic: if any arg contains "device=cuda", we're a GPU job.
IS_GPU=0
for a in "$@"; do
    case "$a" in
        *device=cuda*) IS_GPU=1 ;;
    esac
done

if [ "$IS_GPU" -eq 1 ]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[qg_job] ERROR: GPU job requested but nvidia-smi not found on $HOSTNAME"
        exit 2
    fi

    # Find the GPU with the LEAST memory used (idlest)
    IDLE_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t',' -k2 -n | head -1 | awk -F',' '{gsub(/ /,""); print $1}')
    export CUDA_VISIBLE_DEVICES="$IDLE_GPU"
    echo "[qg_job] selected GPU $IDLE_GPU (CUDA_VISIBLE_DEVICES=$IDLE_GPU) on $HOSTNAME"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total \
        --format=csv,noheader -i "$IDLE_GPU"
fi

# ---- 3. Run --------------------------------------------------------------- #
cd "$QG_ROOT/qg-simple-package-stable/src/qg"

echo "[qg_job] hostname: $HOSTNAME"
echo "[qg_job] date: $(date -u +%FT%TZ)"
echo "[qg_job] python: $(which python)"
echo "[qg_job] cmd: python -u run_qg.py $*"
echo "----------------------------------------------------------------------"

python -u run_qg.py "$@"

echo "----------------------------------------------------------------------"
echo "[qg_job] done at $(date -u +%FT%TZ)"
