#!/bin/bash
# gateD1_run_job.sh - GPU worker for the Gate D-1 option-B run. Same body
# as phaseB_job.sh but with the option-B commons (Sanaa [red-approved]
# 2026-07-09, DECISIONS.md: dt 2.5e-3 explicitly signed off; T=1500):
#   qg.time.dt=2.5e-3  qg.time.T=1500  qg.pde.nu=6.4443e-4
#   qg.grid.precision=float64
# Case specifics forwarded from "$@" (grid, save_rate, inlet_table, diag
# keys, hydra.run.dir). scenario= plain form (package-stable default trap).

#$ -S /bin/bash
#$ -cwd
#$ -V

set -e

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR="$QG_ROOT/qg-simple-package-stable/src/qg"

source "$QG_ROOT/qg-env/bin/activate"
export TMPDIR="$QG_ROOT/tmp"
export PIP_CACHE_DIR="$QG_ROOT/pip-cache"
export MPLCONFIGDIR="$QG_ROOT/mplcache"
export PATH="$QG_ROOT/qg-env/bin-extra:$PATH"
export PYTHONUNBUFFERED=1

mkdir -p "$QG_ROOT/cache/torch" "$QG_ROOT/cache/triton" "$QG_ROOT/cache/nvrtc"
export TORCH_EXTENSIONS_DIR="$QG_ROOT/cache/torch"
export TRITON_CACHE_DIR="$QG_ROOT/cache/triton"
export PYTORCH_KERNEL_CACHE_PATH="$QG_ROOT/cache/nvrtc"

if command -v nvidia-smi >/dev/null 2>&1; then
    IDLE_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t',' -k2 -n | head -1 | awk -F',' '{gsub(/ /,""); print $1}')
    export CUDA_VISIBLE_DEVICES="$IDLE_GPU"
    echo "[gateD1_job] selected GPU $IDLE_GPU on $HOSTNAME"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total \
        --format=csv,noheader -i "$IDLE_GPU"
fi

COMMON_OVERRIDES=(
    qg.time.dt=2.5e-3
    qg.time.T=1500
    qg.pde.nu=6.4443e-4
    qg.grid.precision=float64
)

cd "$QG_DIR"

echo "[gateD1_job] host $HOSTNAME  date $(date -u +%FT%TZ)"
echo "[gateD1_job] cmd: python -u run_qg.py scenario=flow_past_cylinder_sponge ${COMMON_OVERRIDES[*]} $*"
echo "----------------------------------------------------------------------"

python -u run_qg.py scenario=flow_past_cylinder_sponge "${COMMON_OVERRIDES[@]}" "$@"

echo "----------------------------------------------------------------------"
echo "[gateD1_job] done at $(date -u +%FT%TZ)"
