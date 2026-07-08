#!/bin/bash
# build_training_data_ft.sh
# ----------------------------------------------------------------------------
# Build the FORCED-TURBULENCE temporal-closure dataset with
# build_training_data_fixD_v2.py.
#
#   operating step : DT = 1e-3 (largest stable step; 5e-3 and 1e-2 blow up)
#   learned targets: the raw N-time-derivatives  Ndot, Nddot, N3dot  (R3+R4),
#                    plus N4dot provisioned on disk (optional R5). The L^k
#                    weighting (incl. the nonlocal beta term) is assembled
#                    spectrally at inference, NOT learned.
#   inputs         : (omega_n, omega_{n-1}, omega_{n-2}, psi_n, psi_{n-1}, psi_{n-2})
#   forcing        : static  -0.1 cos(2x) + 0.1 cos(2y)  read from the YAML and
#                    folded into N (C==0, F==0 enforced by the python script).
#
# Source ICs (dt=1e-5 reference) are float32 on disk -- fine; they are seed
# snapshots, promoted to float64 internally. The build runs in float64.
#
# Usage:
#   ./build_training_data_ft.sh                 # submit to SGE (GPU)
#   ./build_training_data_ft.sh --interactive   # quick smoke test on this node
#   ./build_training_data_ft.sh --jobname ft1 -- --n-seeds 50   # extra build args
#
# Anything after `--` is forwarded verbatim to build_training_data_fixD_v2.py.
# ----------------------------------------------------------------------------
set -euo pipefail

# ---- Paths ---------------------------------------------------------------- #
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
VENV=$QG_ROOT/qg-env
QG_DIR=$QG_ROOT/qg-simple-package-stable/src/qg
SCRIPT_DIR=$QG_DIR/training
LOG_DIR=/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs
PY=build_training_data_mmap.py

# ---- Source data (20-member forced ensemble, dt=1e-5) --------------------- #
SRC_DIR=$QG_DIR/outputs/forced_turb_ensemble_n20_dt1em5_512
# Use the .npy pair produced by convert_npz_all_batches.py (keeps all batches).
SOURCE_OMEGA=$SRC_DIR/DNS_FR_omega.npy
SOURCE_TIMES=$SRC_DIR/DNS_FR_times.npy

# YAML MUST contain the qg.forcing block (A=-0.1,B=2,C=0,D=0.1,E=2,F=0) and
# nu/mu/B=1.0 so the static forcing + beta term are reconstructed correctly.
SOURCE_YAML=$QG_DIR/conf/scenario/forced_turbulence.yaml
OUT_DIR=$SCRIPT_DIR/data

# ---- Build knobs (override after `--`) ------------------------------------ #
SCENARIO=forced_turbulence
DELTA_T=1.0e-3
H_FINE=1.0e-5            # K = DT/h_fine = 100
H_ULTRAFINE=5.0e-6      # RK4 stencil warmup; raise to 1.0e-4 for ~20x faster warmup
N_SEEDS=500             # snapshots PER BATCH (capped at the unique snapshots in the t-range)
N_BATCHES=20            # ensemble members to use -- WITHOUT this the build uses batch 0 only!
T_START=15.0            # ensemble sampled over [15, ~30]; the run ends at t=29.95
SPLIT_MODE=by_batch     # hold out whole trajectories (needs >=3 batches)
DEVICE=cuda

# ---- CLI ------------------------------------------------------------------ #
INTERACTIVE=0
JOBNAME=build_td_ft
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --interactive) INTERACTIVE=1; shift ;;
        --jobname)     JOBNAME="$2"; shift 2 ;;
        --) shift; while [[ $# -gt 0 ]]; do EXTRA+=("$1"); shift; done ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

# ---- Pre-flight ----------------------------------------------------------- #
[[ -f "$VENV/bin/activate" ]] || { echo "ERROR: venv not at $VENV" >&2; exit 1; }
[[ -f "$SCRIPT_DIR/$PY"    ]] || { echo "ERROR: $PY not in $SCRIPT_DIR" >&2; exit 1; }
[[ -f "$SOURCE_OMEGA"      ]] || { echo "ERROR: source omega not found: $SOURCE_OMEGA" >&2; exit 1; }
[[ -f "$SOURCE_YAML"       ]] || { echo "ERROR: source YAML not found: $SOURCE_YAML" >&2; exit 1; }
mkdir -p "$LOG_DIR" "$OUT_DIR"

PYARGS=(
    --scenario     "$SCENARIO"
    --source-omega "$SOURCE_OMEGA"
    --source-yaml  "$SOURCE_YAML"
    --out-dir      "$OUT_DIR"
    --Delta-T      "$DELTA_T"
    --h-fine       "$H_FINE"
    --h-ultrafine  "$H_ULTRAFINE"
    --n-seeds      "$N_SEEDS"
    --n-batches    "$N_BATCHES"
    --t-start      "$T_START"
    --split-mode   "$SPLIT_MODE"
    --device       "$DEVICE"
)
[[ -n "$SOURCE_TIMES" ]] && PYARGS+=(--source-times "$SOURCE_TIMES")
PYARGS+=(${EXTRA[@]+"${EXTRA[@]}"})

# ---- Interactive ---------------------------------------------------------- #
if [[ "$INTERACTIVE" == "1" ]]; then
    echo "[ft-build] interactive"
    source "$VENV/bin/activate"
    export MPLCONFIGDIR="$QG_ROOT/.mplcache"; mkdir -p "$MPLCONFIGDIR"; export MPLBACKEND=Agg
    cd "$SCRIPT_DIR"
    exec python -u "$PY" "${PYARGS[@]}"
fi

# ---- SGE submission ------------------------------------------------------- #
# GPU queue ONLY: -q ibgpu.q -l gpu=1. No -l h_vmem, no other queue flags.
JOB_SCRIPT=$(mktemp /tmp/build_td_ft_XXXXXX.sh)
cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#\$ -N $JOBNAME
#\$ -q ibgpu.q
#\$ -l gpu=1
#\$ -j y
#\$ -o $LOG_DIR/\$JOB_NAME.\$JOB_ID.log
#\$ -e $LOG_DIR/\$JOB_NAME.\$JOB_ID.err
#\$ -cwd
set -e
source $VENV/bin/activate
export MPLCONFIGDIR=$QG_ROOT/.mplcache; mkdir -p "\$MPLCONFIGDIR"; export MPLBACKEND=Agg
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd $SCRIPT_DIR
echo "[ft-build] host \$HOSTNAME  gpu \${CUDA_VISIBLE_DEVICES:-none}  $(date -u +%FT%TZ)"
echo "[ft-build] cmd: python -u $PY ${PYARGS[*]}"
python -u $PY ${PYARGS[@]}
echo "[ft-build] done $(date -u +%FT%TZ)"
EOF
chmod +x "$JOB_SCRIPT"
echo "[ft-build] submitting $JOBNAME"
echo "[ft-build] log: $LOG_DIR/${JOBNAME}.*.log"
qsub "$JOB_SCRIPT"