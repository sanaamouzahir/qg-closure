#!/bin/bash
# rollout_perfect_closure.sh
# Submit the closure-CEILING test (analytic chain-rule derivatives, no NN) to the GPU
# queue, or run interactively. Lives next to rollout_perfect_closure.py. Mirrors
# rollout_timed_pareto.sh (same queue rules: -q ibgpu.q -l gpu=1 ONLY).
#
# Usage:
#   ./rollout_perfect_closure.sh [--interactive] [--jobname NAME] -- <py args>
# Example:
#   ./rollout_perfect_closure.sh --jobname perfect_ft_r4 -- \
#       --root-dir  $QG_DIR/training/data/_4snap_staging/forced_turbulence_dT_1em3 \
#       --load-refs $QG_DIR/training/data/forced_turbulence_dT_1em3/training_runs/cheap_deriv_6chan_20260604_181128/rollout_refs_ft.npz \
#       --r4 --r4-n3dot-coef 1 --dealias-nn --diag --pareto --device cuda
set -euo pipefail
INTERACTIVE=0
JOBNAME=rollout_perfect
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
SCRIPT=$HERE/rollout_perfect_closure.py
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
    echo "[perfect] interactive mode"
    source "$VENV/bin/activate"
    export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
    export MPLCONFIGDIR="$QG_ROOT/.mplcache"; mkdir -p "$MPLCONFIGDIR"
    cd "$HERE"
    nvidia-smi -L || true
    python -u -c "$GPU_CHECK"
    exec python -u "$SCRIPT" ${PYARGS[@]+"${PYARGS[@]}"}
fi

JOB_SCRIPT=$(mktemp /tmp/rollout_perfect_job_XXXXXX.sh)
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
echo "[perfect] host=\$HOSTNAME date=\$(date -u +%FT%TZ)"
echo "[perfect] CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L || echo "[perfect] nvidia-smi -L unavailable"
python -u -c '$GPU_CHECK'
echo "[perfect] GPU OK -- starting ceiling rollout"
python -u $SCRIPT ${PYARGS[@]+"${PYARGS[@]}"}
echo "[perfect] done \$(date -u +%FT%TZ)"
EOF
chmod +x "$JOB_SCRIPT"
echo "[perfect] submitting $JOBNAME ..."
qsub "$JOB_SCRIPT"
echo "[perfect] log: $LOG_DIR/${JOBNAME}.log"
