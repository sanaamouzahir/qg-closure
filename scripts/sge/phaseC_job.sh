#!/bin/bash
# phaseC_job.sh - Generic SGE GPU worker for the Phase-C convergence tier
# (charter S5.3 as extended by Sanaa's 2026-07-08 five-grid directive:
# grids {256,512,1024,2048,4096}^2 at dt=1.25e-4 + dt study {5.0e-4,
# 2.5e-4, 1.25e-4} at 2048^2; MOD-const, shared t=29.97 IC, T=60).
#
# WHY A SEPARATE WORKER: phaseB_job.sh bakes qg.time.dt=2.5e-4 BEFORE "$@";
# hydra does not do last-wins on duplicate overrides, and the tier sweeps
# dt. This worker bakes only the dt-independent tier commons:
#   qg.time.T=60   qg.pde.nu=6.4443e-4   qg.grid.precision=float64
# Everything else is forwarded verbatim from the submitter:
#   qg.grid.Nx/Ny, qg.time.dt, qg.time.save_rate (0.27-t.u.-commensurate:
#   2160/1080/540 at dt 1.25e-4/2.5e-4/5.0e-4),
#   qg.pde.penalty + qg.bc.sponge (FIXED-PHYSICAL eta rule, charter S5.2:
#   factor = 3.125e-4/dt so eta_phys = 1.25*2.5e-4 stays constant),
#   qg.ic.function=from_file +qg.ic.path=<shared IC at run grid>
#   (+ form: 'path' is not in the scenario's ic block; the + append is the
#   empirically proven pattern on this config tree — FPC-const landed with
#   +qg.bc.inlet_table / +qg.diag.*; no landed run ever used plain ic.path),
#   +qg.bc.inlet_table=<const table at run dt>,
#   +qg.diag.scalar_rate=10 +qg.diag.out=<run-dir>/scalars.npz (d48fcae),
#   hydra.run.dir.
# f64 solve / f32-at-write per Amendment 02 S2 (dataset.py casts omega_FR).
#
# scenario= (NOT +scenario=): package-stable config has a scenario default
# (gate1_job.sh lesson, jobs 1828225-29).
#
# Usage (from submit_phaseC_tier.sh):
#   qsub -N cnv_N4096 -q ibgpu.q -l gpu=1 [-hold_jid tab,ics] \
#        phaseC_job.sh qg.grid.Nx=4096 qg.grid.Ny=4096 qg.time.dt=1.25e-4 \
#        qg.time.save_rate=2160 qg.pde.penalty=2.5 qg.bc.sponge=2.5 \
#        qg.ic.function=from_file +qg.ic.path=<ic.npy> \
#        +qg.bc.inlet_table=<table.npz> +qg.diag.scalar_rate=10 \
#        +qg.diag.out=<run-dir>/scalars.npz hydra.run.dir=<run-dir-rel>

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
    echo "[phaseC_job] selected GPU $IDLE_GPU on $HOSTNAME"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total \
        --format=csv,noheader -i "$IDLE_GPU"
fi

COMMON_OVERRIDES=(
    qg.time.T=60
    qg.pde.nu=6.4443e-4
    qg.grid.precision=float64
)

cd "$QG_DIR"

echo "[phaseC_job] host $HOSTNAME  date $(date -u +%FT%TZ)"
echo "[phaseC_job] cmd: python -u run_qg.py scenario=flow_past_cylinder_sponge ${COMMON_OVERRIDES[*]} $*"
echo "----------------------------------------------------------------------"

python -u run_qg.py scenario=flow_past_cylinder_sponge "${COMMON_OVERRIDES[@]}" "$@"

echo "----------------------------------------------------------------------"
echo "[phaseC_job] done at $(date -u +%FT%TZ)"
