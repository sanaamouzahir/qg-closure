#!/bin/bash
# run_guarantees_job.sh -- batch the full theoretical-guarantees sweep on a GPU.
#
# Submit FROM Theoretical_guarantees/ (so -cwd lands here and run_all_guarantees.sh
# + the diagnostics resolve, and ../data is reachable):
#   cd $QG_ROOT/qg-simple-package-stable/src/qg/training/Theoretical_guarantees
#   mkdir -p logs
#   qsub -N tg_all -q ibgpu.q -l gpu=1 -j y -o logs/tg_all.log run_guarantees_job.sh
#
# Matches qg_job.sh conventions: GPU = -q ibgpu.q -l gpu=1 ONLY (no ibamd.q, no h_vmem).

#$ -S /bin/bash
#$ -cwd                      # run in submission dir (Theoretical_guarantees/)
#$ -V                        # inherit environment

set -e

# ---- 1. venv + caches (same as qg_job.sh) ------------------------------- #
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
source "$QG_ROOT/qg-env/bin/activate"
export TMPDIR="$QG_ROOT/tmp"
export PIP_CACHE_DIR="$QG_ROOT/pip-cache"
export MPLCONFIGDIR="$QG_ROOT/mplcache"
export PATH="$QG_ROOT/qg-env/bin-extra:$PATH"
export PYTHONUNBUFFERED=1

# ---- 2. pick the idlest GPU (same heuristic as qg_job.sh) --------------- #
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[tg_job] ERROR: GPU job but nvidia-smi not found on $HOSTNAME"; exit 2
fi
IDLE_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | sort -t',' -k2 -n | head -1 | awk -F',' '{gsub(/ /,"");print $1}')
export CUDA_VISIBLE_DEVICES="$IDLE_GPU"
echo "[tg_job] selected GPU $IDLE_GPU on $HOSTNAME"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader -i "$IDLE_GPU"

# ---- 3. run the full sweep ---------------------------------------------- #
echo "[tg_job] hostname: $HOSTNAME"
echo "[tg_job] date:     $(date -u +%FT%TZ)"
echo "[tg_job] cwd:      $(pwd)"
echo "[tg_job] python:   $(which python)"
echo "----------------------------------------------------------------------"

bash run_all_guarantees.sh

echo "----------------------------------------------------------------------"
echo "[tg_job] done at $(date -u +%FT%TZ)"