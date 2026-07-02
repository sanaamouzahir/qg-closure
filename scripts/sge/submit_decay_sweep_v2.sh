#!/bin/bash
# submit_decay_sweep_v2.sh
#
# Decaying-turbulence dt-sensitivity sweep, RESTARTED from a developed-flow
# snapshot at t=10 (extracted from the v1 dt=1e-5 reference run), instead of
# from t=0.
#
# WHY:
#   v1 ran each dt from t=0 with the same RNG seed, so the IC at t=0 IS
#   identical across dts. But decaying turb is chaotic; tiny numerical
#   differences amplify exponentially. By t=10 the trajectories have
#   decorrelated, and at t=60 they are essentially independent realizations.
#   So convergence_plot does NOT show 2nd-order behavior because we're
#   comparing different chaotic realizations, not "same flow at different dt".
#
#   v2 fixes this: extract omega(t=10) from the v1 1e-5 run (where the flow
#   has already started decaying but is still vigorous), give all dts the
#   SAME starting field, and run each for ~10 more time units (until t=20).
#   Now any inter-dt differences are pure AB2CN2 truncation; convergence
#   should show clean slope-2.
#
# Single-batch IC (taken from batch 0 of the v1 reference) -- this restart
# sweep is for the DETERMINISTIC convergence test only, not for ensemble
# training data. The training-data sweep can keep using the original v1 data.
#
# Usage:
#   ./submit_decay_sweep_v2.sh
#   ./submit_decay_sweep_v2.sh --dry-run
#   ./submit_decay_sweep_v2.sh --t-star 5.0    # different restart time
#
# Outputs:
#   outputs/decaying_turb_dt_sweep_v2/dt_<tag>/

set -e

# ---------------------------------------------------------------------- #
# Defaults                                                               #
# ---------------------------------------------------------------------- #

T_STAR=10.0           # extract IC at this time
T_RUN=10.0            # how long to integrate after restart
T_FINAL=$(awk "BEGIN{print $T_STAR + $T_RUN}")  # passed to qg.time.T

# Format: "dt save_rate label"  (matches v1 dt grid)
DT_CONFIGS=(
    "2.0e-3      25      2em3"
    "1.0e-3      50      1em3"
    "5.0e-4     100      5em4"
    "2.5e-4     200      2p5em4"
    "1.25e-4    400      1p25em4"
    "2.0e-5    2500      2em5"
    "1.0e-5    5000      1em5"
)

GRID_NX=256
GRID_NY=256
SCENARIO=decaying_turbulence

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR="$QG_ROOT/qg-simple-package-stable/src/qg"
SUBMIT_SCRIPT="$QG_DIR/submit_qg.sh"

REF_DIR="$QG_DIR/outputs/decaying_turb_dt_sweep/dt_1em5"
RESTART_DIR="$QG_DIR/outputs/restart_ics/decay_v1_t${T_STAR%.*}"
SWEEP_ROOT="$QG_DIR/outputs/decaying_turb_dt_sweep_v2"

DRY_RUN=0
while [ "$#" -ge 1 ]; do
    case "$1" in
        --dry-run)     DRY_RUN=1; shift ;;
        --t-star)      T_STAR="$2"; T_FINAL=$(awk "BEGIN{print $T_STAR + $T_RUN}"); shift 2 ;;
        --t-run)       T_RUN="$2";  T_FINAL=$(awk "BEGIN{print $T_STAR + $T_RUN}"); shift 2 ;;
        --sweep-root)  SWEEP_ROOT="$2"; shift 2 ;;
        --restart-dir) RESTART_DIR="$2"; shift 2 ;;
        -h|--help)     sed -n '2,30p' "$0"; exit 0 ;;
        *)             echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$SWEEP_ROOT" "$RESTART_DIR"

RESTART_NPY="$RESTART_DIR/restart_omega.npy"

echo "==================================================================="
echo "Decaying turb dt-sensitivity sweep v2 (RESTART from developed flow)"
echo "  scenario      : $SCENARIO"
echo "  grid          : ${GRID_NX} x ${GRID_NY}"
echo "  reference     : $REF_DIR"
echo "  restart at    : t* = $T_STAR  (from ref dt=1e-5 run, batch 0)"
echo "  run for       : $T_RUN time units after restart"
echo "  T_final       : $T_FINAL  (absolute)"
echo "  outputs       : $SWEEP_ROOT/dt_<tag>/"
echo "  dry_run       : $DRY_RUN"
echo "==================================================================="
echo

# ---------------------------------------------------------------------- #
# Step A: Extract restart IC                                             #
# ---------------------------------------------------------------------- #

if [ ! -f "$RESTART_NPY" ]; then
    echo "Step A: extracting restart IC at t*=$T_STAR ..."
    REF_NPZ="$REF_DIR/DNS_FR.npz"
    REF_NPY="$REF_DIR/DNS_FR_omega.npy"
    REF_TIMES="$REF_DIR/DNS_FR_times.npy"

    if [ -f "$REF_NPZ" ]; then
        REF_OMEGA="$REF_NPZ"
        EXTRA_ARGS=()
    elif [ -f "$REF_NPY" ] && [ -f "$REF_TIMES" ]; then
        REF_OMEGA="$REF_NPY"
        EXTRA_ARGS=(--source-times "$REF_TIMES")
    else
        echo "ERROR: cannot find reference DNS in $REF_DIR"
        exit 1
    fi

    SCRIPT_DIR="$QG_DIR/RestartSweeps"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[dry-run] would run:"
        echo "  python $SCRIPT_DIR/extract_restart_ic.py \\"
        echo "      --source-omega $REF_OMEGA ${EXTRA_ARGS[*]} \\"
        echo "      --t-star $T_STAR --batch-index 0 \\"
        echo "      --out-dir $RESTART_DIR"
    else
        source "$QG_ROOT/qg-env/bin/activate"
        python "$SCRIPT_DIR/extract_restart_ic.py" \
            --source-omega "$REF_OMEGA" \
            "${EXTRA_ARGS[@]}" \
            --t-star "$T_STAR" --batch-index 0 \
            --out-dir "$RESTART_DIR"
    fi
    echo
else
    echo "Step A: restart IC already exists at $RESTART_NPY (skipping)"
    echo
fi

# ---------------------------------------------------------------------- #
# Step B: Submit per-dt jobs                                             #
# ---------------------------------------------------------------------- #

TOTAL=0

for CONFIG in "${DT_CONFIGS[@]}"; do
    read -r DT SAVE_RATE DT_LABEL <<< "$CONFIG"

    OUT_DIR="$SWEEP_ROOT/dt_${DT_LABEL}"
    JOBNAME="decay2_dt${DT_LABEL}_T${T_RUN%.*}_${GRID_NX}_gpu"

    mkdir -p "$OUT_DIR"

    echo "-----  dt=$DT, save_rate=$SAVE_RATE  -----"
    echo "  jobname : $JOBNAME"
    echo "  out_dir : $OUT_DIR"

    CMD=(
        "$SUBMIT_SCRIPT" "$JOBNAME" --gpu --
        scenario="$SCENARIO"
        qg.grid.Nx="$GRID_NX"
        qg.grid.Ny="$GRID_NY"
        qg.time.T="$T_FINAL"
        qg.time.dt="$DT"
        qg.time.save_rate="$SAVE_RATE"
        qg.ic.function="from_file"
        qg.ic.path="$RESTART_NPY"
        qg.ic.n_batch=1
        hydra.run.dir="$OUT_DIR"
    )

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  [dry-run] would submit:"
        printf "    %s\n" "${CMD[@]}"
    else
        "${CMD[@]}"
    fi

    TOTAL=$((TOTAL + 1))
    echo
done

echo "==================================================================="
if [ "$DRY_RUN" -eq 1 ]; then
    echo "Dry-run done. $TOTAL jobs would be submitted."
else
    echo "Submitted $TOTAL jobs."
fi
echo
echo "Monitor: qstat -u \$USER"
echo
echo "Once done, run convergence pipeline pointing at v2:"
echo "  ./Convergence_studies/Decaying_Turbulence/Step1/submit_step1.sh \\"
echo "      --sweep-root $SWEEP_ROOT --jobname step1_decay_v2 --force-convert"
echo "  ./Convergence_studies/Decaying_Turbulence/ConvergencePlot/convergence_plot.sh \\"
echo "      --sweep-root $SWEEP_ROOT --jobname convergence_plot_decay_v2 --t-start 0.0"
echo "==================================================================="
