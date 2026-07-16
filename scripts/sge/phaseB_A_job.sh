#!/bin/bash
# phaseB_A_job.sh - FPC-A variant of phaseB_job.sh: identical worker with
# qg.time.dt HALVED to 1.25e-4 (the CAPE-A playbook applied to the cylinder).
#
# Why (postmortem 2026-07-12, FPC-telS job 1830422): both FPC-tel (hard
# switches) and FPC-telS (10-step smoothed switches) blew up DEEP inside the
# same long Re=5600 dwell (t in [54.47, 68.86]; onset t=66.81 resp. 68.6,
# each >1.3 time units from the nearest switch) with the cape signature --
# enstrophy Z exploding at the grid scale while E stays flat. The switch-
# impulse hypothesis is disproven; Re 5600 (U 2.87 vs 2.0 for the clean
# members' max Re 3900) sits past the 2048^2 @ 2.5e-4 advective dt-edge,
# exactly like the cape (capeSmk P/Y/D: dt-unstable 2.5e-4, clean 1.25e-4).
#
# Everything else verbatim from phaseB_job.sh (production commons charter
# S4.1, NaN-guard, GPU pick). Submitters must pass the dt1p25e-4 inlet table
# (table load enforces dt match) and qg.time.save_rate=3600 (dt_save 0.45).
#
# Usage:
#   qsub -N sgsB_<id> -q ibgpu.q -l gpu=1 \
#        phaseB_A_job.sh qg.grid.Nx=2048 qg.grid.Ny=2048 qg.time.save_rate=3600 \
#        +qg.bc.inlet_table=<table_dt1p25e-4.npz> +qg.diag.scalar_rate=10 \
#        +qg.diag.flush_every=500 +qg.diag.out=<run-dir>/scalars.npz \
#        hydra.run.dir=<run-dir-rel>

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
    echo "[phaseB_A_job] selected GPU $IDLE_GPU on $HOSTNAME"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total \
        --format=csv,noheader -i "$IDLE_GPU"
fi

COMMON_OVERRIDES=(
    qg.time.dt=1.25e-4
    qg.time.T=120
    qg.pde.nu=6.4443e-4
    qg.grid.precision=float64
)

cd "$QG_DIR"

echo "[phaseB_A_job] host $HOSTNAME  date $(date -u +%FT%TZ)"
echo "[phaseB_A_job] cmd: python -u run_qg.py scenario=flow_past_cylinder_sponge ${COMMON_OVERRIDES[*]} $*"
echo "----------------------------------------------------------------------"

# ---- NaN-guard (verbatim from phaseB_job.sh) ------------------------------ #
# Poll the atomically-rewritten scalars.npz, kill the solver on NaN (fail in
# minutes, not ~32 GPU-h), and fail LOUDLY (exit 99) if a completed record
# contains NaN (FPC-tel silent-NaN lesson).
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
                # Sanaa 2026-07-16: email on every NaN-kill (saves compute +
                # she knows immediately; spool works from network-less nodes)
                SP=/gdata/projects/ml_scope/Closure_modeling/QG-closure/reporting/pending_mail
                mkdir -p "$SP"
                printf 'To: %s\nSubject: [QG][MONITOR][sgs-closure] NaN-KILL: job %s (%s)\n\nThe NaN guard killed this simulation the moment NaN appeared in its\nscalar diagnostics -- no further compute wasted.\nrun dir: %s\nscalars: %s\ntime: %s\n\nNEXT: this parameter point is numerically unstable (or the run was\ncorrupted); it will NOT be retried automatically.\n' \
                    "${QG_NOTIFY_EMAIL:-sanaamz@mit.edu}" "${JOB_ID:-?}" "${JOB_NAME:-?}" \
                    "$PWD" "$SCALARS_OUT" "$(date -u +%FT%TZ)" \
                    > "$SP/$(date +%Y%m%dT%H%M%S)_nankill_${JOB_ID:-x}.mail"
                exit 0
            fi
        done
    ) &
    GUARD_PID=$!
    echo "[phaseB_A_job] NaN-guard armed on $SCALARS_OUT (poll 300 s)"
else
    echo "[phaseB_A_job] WARNING: no +qg.diag.out= arg -- NaN-guard disarmed"
fi

set +e
wait "$SOLVER_PID"
RC=$?
if [ -n "$GUARD_PID" ]; then kill "$GUARD_PID" 2>/dev/null; fi
set -e

if [ "$RC" -ne 0 ]; then
    echo "[phaseB_A_job] solver exited $RC (NaN-abort or crash) at $(date -u +%FT%TZ)"
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
echo "[phaseB_A_job] done at $(date -u +%FT%TZ)"
