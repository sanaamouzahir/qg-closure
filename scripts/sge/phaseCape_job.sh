#!/bin/bash
# phaseCape_job.sh - Generic SGE GPU worker for the Phase-B CAPE production
# runs (FPCape wave, Sanaa order 2026-07-10: same five cases as the FPC
# ensemble, flow past cape, 2048^2). Copy of phaseB_job.sh with ONE change:
# scenario=flow_past_cape. Same commons (charter S4.1): f64 solve / f32 at
# write [dataset.py casts omega_FR to float32], scalars every 10 steps with
# PER-RUN diag.out (d48fcae lesson).
#
# Baked-in production commons (charter S4.1; nu identical to FPC per
# SUPERVISOR_BRIEF rule 5 "never change nu between cases"):
#   qg.time.dt=2.5e-4  qg.time.T=120  qg.pde.nu=6.4443e-4
#   qg.grid.precision=float64
# Cape geometry (mask x_center 0.2, y_base 0, x_scale 1, y_scale 4,
# x_support 2; bc const-outlet-vorticity-rtd, width 0.1, sponge 1.025;
# penalty 1.025) comes UNCHANGED from conf/scenario/flow_past_cape.yaml.
# Everything case-specific is forwarded verbatim from "$@" by the submitter:
#   qg.grid.Nx/Ny, qg.time.save_rate, +qg.bc.inlet_table,
#   +qg.diag.scalar_rate=10, +qg.diag.out=<run-dir>/scalars.npz,
#   +qg.diag.length=1.0 (REQUIRED: non-circular mask, L_cape=1),
#   +qg.diag.probes=[...] (approved cape lee-probe set, BRANCH_LOG
#   2026-07-07), hydra.run.dir.
#
# scenario= (NOT +scenario=): package-stable config has a scenario default
# (gate1_job.sh lesson, jobs 1828225-29).
#
# Usage:
#   qsub -N sgsCape_<modid> -q ibgpu.q -l gpu=1 [-hold_jid <tables-job>] \
#        phaseCape_job.sh qg.grid.Nx=2048 qg.grid.Ny=2048 \
#        qg.time.save_rate=1800 +qg.bc.inlet_table=<table.npz> \
#        +qg.diag.scalar_rate=10 +qg.diag.out=<run-dir>/scalars.npz \
#        +qg.diag.length=1.0 hydra.run.dir=<run-dir-rel>

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

# Pick the idle GPU (gate1_job.sh pattern)
if command -v nvidia-smi >/dev/null 2>&1; then
    IDLE_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t',' -k2 -n | head -1 | awk -F',' '{gsub(/ /,""); print $1}')
    export CUDA_VISIBLE_DEVICES="$IDLE_GPU"
    echo "[phaseCape_job] selected GPU $IDLE_GPU on $HOSTNAME"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total \
        --format=csv,noheader -i "$IDLE_GPU"
fi

COMMON_OVERRIDES=(
    qg.time.dt=2.5e-4
    qg.time.T=120
    qg.pde.nu=6.4443e-4
    qg.grid.precision=float64
)

cd "$QG_DIR"

echo "[phaseCape_job] host $HOSTNAME  date $(date -u +%FT%TZ)"
echo "[phaseCape_job] cmd: python -u run_qg.py scenario=flow_past_cape ${COMMON_OVERRIDES[*]} $*"
echo "----------------------------------------------------------------------"

python -u run_qg.py scenario=flow_past_cape "${COMMON_OVERRIDES[@]}" "$@"

echo "----------------------------------------------------------------------"
echo "[phaseCape_job] done at $(date -u +%FT%TZ)"
