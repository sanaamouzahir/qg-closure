#!/bin/bash
# phaseB_job.sh - Generic SGE GPU worker for the Phase-B FPC production runs
# (charter S4.1/S4.2 as amended by Amendment 02: cylinder-only, f64 solve /
# f32-at-write [dataset.py casts omega_FR to float32], scalars every 10
# steps with PER-RUN diag.out — d48fcae lesson).
#
# Baked-in production commons (charter S4.1):
#   qg.time.dt=2.5e-4  qg.time.T=120  qg.pde.nu=6.4443e-4
#   qg.grid.precision=float64
# Everything case-specific is forwarded verbatim from "$@" by the submitter:
#   qg.grid.Nx/Ny, qg.time.save_rate (1080 on MOD-const runs per Amendment
#   02 S2 commensurability; 1000 on modulated runs), +qg.bc.inlet_table,
#   +qg.diag.scalar_rate=10, +qg.diag.out=<run-dir>/scalars.npz,
#   hydra.run.dir.
# Brinkman penalty/sponge: YAML eta = factor*dt convention KEPT for
# production (charter S4.1); resulting PHYSICAL eta values are recorded in
# the recorder meta and reported per-run (S4.3).
#
# scenario= (NOT +scenario=): package-stable config has a scenario default
# (gate1_job.sh lesson, jobs 1828225-29).
#
# Usage:
#   qsub -N sgs_FPC_<modid> -q ibgpu.q -l gpu=1 [-hold_jid <tables-job>] \
#        phaseB_job.sh qg.grid.Nx=2048 qg.grid.Ny=2048 qg.time.save_rate=1080 \
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
    echo "[phaseB_job] selected GPU $IDLE_GPU on $HOSTNAME"
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

echo "[phaseB_job] host $HOSTNAME  date $(date -u +%FT%TZ)"
echo "[phaseB_job] cmd: python -u run_qg.py scenario=flow_past_cylinder_sponge ${COMMON_OVERRIDES[*]} $*"
echo "----------------------------------------------------------------------"

# ---- NaN-guard (QUESTIONS item 3(i), Sanaa GO 2026-07-11) ----------------- #
# Same guard as phaseCape_job.sh: poll the atomically-rewritten scalars.npz,
# kill the solver on NaN (fail in minutes, not ~11 GPU-h), and fail LOUDLY
# (exit 99) if a completed record contains NaN (FPC-tel silent-NaN lesson).
SCALARS_OUT=""
for a in "$@"; do case "$a" in +qg.diag.out=*) SCALARS_OUT="${a#+qg.diag.out=}";; esac; done

nan_check() {  # exit 1 iff readable and containing NaN; unreadable = no verdict
    python - "$1" <<'PY'
import sys, numpy as np
try:
    z = np.load(sys.argv[1])
    bad = any(np.isnan(z[k]).any() for k in z.files if z[k].dtype.kind == 'f')
except Exception:
    sys.exit(0)
sys.exit(1 if bad else 0)
PY
}

python -u run_qg.py scenario=flow_past_cylinder_sponge "${COMMON_OVERRIDES[@]}" "$@" &
SOLVER_PID=$!

GUARD_PID=""
if [ -n "$SCALARS_OUT" ]; then
    (
        while sleep 300; do
            kill -0 "$SOLVER_PID" 2>/dev/null || exit 0
            [ -f "$SCALARS_OUT" ] || continue
            if ! nan_check "$SCALARS_OUT"; then
                echo "[NAN-GUARD] NaN in $SCALARS_OUT at $(date -u +%FT%TZ) -- killing solver (pid $SOLVER_PID)"
                kill "$SOLVER_PID" 2>/dev/null
                sleep 15
                kill -9 "$SOLVER_PID" 2>/dev/null
                exit 0
            fi
        done
    ) &
    GUARD_PID=$!
    echo "[phaseB_job] NaN-guard armed on $SCALARS_OUT (poll 300 s)"
else
    echo "[phaseB_job] WARNING: no +qg.diag.out= arg -- NaN-guard disarmed"
fi

set +e
wait "$SOLVER_PID"
RC=$?
if [ -n "$GUARD_PID" ]; then kill "$GUARD_PID" 2>/dev/null; fi
set -e

if [ "$RC" -ne 0 ]; then
    echo "[phaseB_job] solver exited $RC (NaN-abort or crash) at $(date -u +%FT%TZ)"
    exit "$RC"
fi
if [ -n "$SCALARS_OUT" ] && [ -f "$SCALARS_OUT" ]; then
    if ! nan_check "$SCALARS_OUT"; then
        echo "[NAN-GUARD] FINAL CHECK FAILED: NaN in completed record $SCALARS_OUT"
        exit 99
    fi
    echo "[NAN-GUARD] final check clean"
fi

echo "----------------------------------------------------------------------"
echo "[phaseB_job] done at $(date -u +%FT%TZ)"
