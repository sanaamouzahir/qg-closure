#!/bin/bash
# Gate 1 (charter §3.4 + AMENDMENT_01 §C.3). Authored by sge-runner under
# branch-supervisor direction; TBDs filled 2026-07-07 after the Fable Phase A
# handoff (BRANCH_LOG 1b). Released for submission per Sanaa's resume order.
#
# gate1_job.sh - Generic SGE worker (qsub payload) for the Gate 1 flow-past-
# cylinder smokes (charter §3.4, AMENDMENT_01 §C.1/§C.3). Bakes in the common
# Gate-1 physical/numerical overrides; anything case-specific (the inlet-table
# and recorder keys, bc.inlet_velocity, hydra.run.dir, ...) is forwarded
# verbatim from "$@" by the submitter (submit_gate1_smokes.sh).
#
# Solver hooks (LIVE in qg-simple-package-stable, mirrored in solver_patches/
# sgs_hooks_2026-07-07.patch): NEW config keys are hydra APPEND overrides —
#   +qg.bc.inlet_table=<path/to/U_of_t.npz>   (bc.py, Flow.const_x_flow)
#   +qg.diag.scalar_rate=10                   (_output/scalars.py recorder)
# Absent keys => bit-identical legacy behavior (Gate-1 max|dw|=0.0 arm).
#
# Usage:
#   qsub -N sgs_gate1_<id> -q ibgpu.q -l gpu=1 -m ea -M $QG_NOTIFY_EMAIL \
#        gate1_job.sh +qg.bc.inlet_table=<path/to/table.npz> \
#        +qg.diag.scalar_rate=10 hydra.run.dir=outputs/SGS_closure_gate1/<id>
#
# Baked-in common Gate-1 overrides (charter §3.4):
#   qg.grid.Nx=512 qg.grid.Ny=512 qg.time.dt=2.5e-4 qg.time.T=15
#   qg.pde.nu=6.4443e-4 qg.grid.precision=float64

#$ -S /bin/bash
#$ -cwd
#$ -V

set -e

# ---- 1. Activate venv & set caches -------------------------------------- #
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR="$QG_ROOT/qg-simple-package-stable/src/qg"

source "$QG_ROOT/qg-env/bin/activate"
export TMPDIR="$QG_ROOT/tmp"
export PIP_CACHE_DIR="$QG_ROOT/pip-cache"
export MPLCONFIGDIR="$QG_ROOT/mplcache"
export PATH="$QG_ROOT/qg-env/bin-extra:$PATH"
export PYTHONUNBUFFERED=1

# ---- 1b. Redirect torch / triton / nvrtc kernel caches off $HOME ------- #
mkdir -p "$QG_ROOT/cache/torch" "$QG_ROOT/cache/triton" "$QG_ROOT/cache/nvrtc"
export TORCH_EXTENSIONS_DIR="$QG_ROOT/cache/torch"
export TRITON_CACHE_DIR="$QG_ROOT/cache/triton"
export PYTORCH_KERNEL_CACHE_PATH="$QG_ROOT/cache/nvrtc"

# ---- 2. Pick an idle GPU -------------------------------------------------- #
if command -v nvidia-smi >/dev/null 2>&1; then
    IDLE_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t',' -k2 -n | head -1 | awk -F',' '{gsub(/ /,""); print $1}')
    export CUDA_VISIBLE_DEVICES="$IDLE_GPU"
    echo "[gate1_job] selected GPU $IDLE_GPU (CUDA_VISIBLE_DEVICES=$IDLE_GPU) on $HOSTNAME"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total \
        --format=csv,noheader -i "$IDLE_GPU"
else
    echo "[gate1_job] WARNING: no nvidia-smi found on $HOSTNAME (GPU job expected one)"
fi

# ---- 3. Common Gate-1 overrides (charter §3.4; scientific notation always #
#         explicit-decimal-mantissa per project YAML rule) ---------------- #
# Dtype/precision key confirmed by grepping config.py / conf/config.yaml:
#   qg/config.py: GridConfig.precision : str
#   qg/conf/config.yaml: qg.grid.precision: float32 (default)
# Gate-1 charter rule: float64 for all runs on this branch — override below.
COMMON_OVERRIDES=(
    qg.grid.Nx=512
    qg.grid.Ny=512
    qg.time.dt=2.5e-4
    qg.time.T=15
    qg.pde.nu=6.4443e-4
    qg.grid.precision=float64
)

cd "$QG_DIR"

echo "[gate1_job] hostname: $HOSTNAME"
echo "[gate1_job] date: $(date -u +%FT%TZ)"
echo "[gate1_job] python: $(which python)"
echo "[gate1_job] cmd: python -u run_qg.py +scenario=flow_past_cylinder_sponge ${COMMON_OVERRIDES[*]} $*"
echo "----------------------------------------------------------------------"

python -u run_qg.py +scenario=flow_past_cylinder_sponge "${COMMON_OVERRIDES[@]}" "$@"

echo "----------------------------------------------------------------------"
echo "[gate1_job] done at $(date -u +%FT%TZ)"
