#!/bin/bash
# rollout_timed_pareto.sh
# Submit the on-the-fly timed rollout + Pareto (truth RK4 / bare / closure) to
# the GPU queue, or run interactively. Lives next to rollout_timed_pareto.py.
#
# Hardened: aborts if no CUDA GPU is visible (no silent CPU fallback), runs
# Python unbuffered (live progress), prints GPU diagnostics up front.
#
# Usage:
#   ./rollout_timed_pareto.sh [--interactive] [--jobname NAME] -- <py args>
# Example:
#   ./rollout_timed_pareto.sh --jobname rollout_ft -- \
#       --run-dir  $QG_DIR/training/data/forced_turbulence_dT_1em3/training_runs/cheap_deriv_6chan_20260604_181128 \
#       --root-dir $QG_DIR/training/data/forced_turbulence_dT_1em3 \
#       --n-steps 1000 --n-checkpoints 10 --ic-index 100 --device cuda --pareto
set -euo pipefail
INTERACTIVE=0
JOBNAME=rollout_timed
PYARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --interactive) INTERACTIVE=1; shift ;;
        --jobname)     JOBNAME="$2"; shift 2 ;;
        --) shift; while [[ $# -gt 0 ]]; do PYARGS+=("$1"); shift; done ;;
        *) PYARGS+=("$1"); shift ;;
    esac
done

# ---- Hard-coded paths ----
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
VENV=$QG_ROOT/qg-env
QG_DIR=$QG_ROOT/qg-simple-package-stable/src/qg
LOG_DIR=$QG_DIR/logs
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT=$HERE/rollout_timed_pareto.py
mkdir -p "$LOG_DIR"
[[ -f "$SCRIPT" ]]            || { echo "ERROR: cannot find $SCRIPT" >&2; exit 1; }
[[ -f "$VENV/bin/activate" ]]|| { echo "ERROR: venv not found at $VENV/bin/activate" >&2; exit 1; }

GPU_CHECK='import sys, torch
ok = torch.cuda.is_available()
print("[gpu-check] cuda_available =", ok, " device_count =", torch.cuda.device_count(), flush=True)
if ok:
    print("[gpu-check] device0 =", torch.cuda.get_device_name(0), flush=True)
else:
    print("[gpu-check] NO GPU VISIBLE -- aborting (no CPU fallback).", flush=True)
sys.exit(0 if ok else 2)'

if [[ "$INTERACTIVE" == "1" ]]; then
    echo "[rollout] interactive mode"
    source "$VENV/bin/activate"
    export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
    export MPLCONFIGDIR="$QG_ROOT/.mplcache"; mkdir -p "$MPLCONFIGDIR"
    cd "$HERE"
    nvidia-smi -L || true
    python -u -c "$GPU_CHECK"
    exec python -u "$SCRIPT" ${PYARGS[@]+"${PYARGS[@]}"}
fi

# Build the qsub job script
JOB_SCRIPT=$(mktemp /tmp/rollout_timed_job_XXXXXX.sh)
cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#\$ -N $JOBNAME
#\$ -q ibgpu.q
#\$ -l gpu=1
#\$ -j y
#\$ -o $LOG_DIR/${JOBNAME}.log
#\$ -cwd
set -e
source $VENV/bin/activate
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR=$QG_ROOT/.mplcache; mkdir -p "\$MPLCONFIGDIR"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd $HERE
echo "[rollout] host=\$HOSTNAME date=\$(date -u +%FT%TZ)"
echo "[rollout] CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L || echo "[rollout] nvidia-smi -L unavailable"
python -u -c '$GPU_CHECK'
echo "[rollout] GPU OK -- starting rollout"
python -u $SCRIPT ${PYARGS[@]+"${PYARGS[@]}"}
echo "[rollout] done \$(date -u +%FT%TZ)"
EOF
chmod +x "$JOB_SCRIPT"
echo "[rollout] submitting $JOBNAME ..."
qsub "$JOB_SCRIPT"
echo "[rollout] log: $LOG_DIR/${JOBNAME}.log"