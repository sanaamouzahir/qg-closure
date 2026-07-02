#!/bin/bash
# submit_cyl_sweep_v4.sh
#
# Cylinder dt-sensitivity sweep, RESTARTED from a fully-developed-flow snapshot
# instead of from t=0. Each dt-run starts with omega(t=35) extracted from the
# v3 dt=1e-5 reference run.
#
# WHY:
#   v1/v2/v3 ran each dt from t=0. Even with identical initial conditions,
#   the chaotic vortex shedding amplifies tiny numerical differences to O(1)
#   over 35+ time units of spinup. So the trajectories at t=50 across different
#   dt are not simply "same flow at different time-discretization" -- they are
#   different chaotic realizations.
#
#   v4 fixes this: extract omega(t=35) from the v3 1e-5 run, give all dts the
#   SAME starting field, and run each for ~10 more time units. Now any
#   inter-dt differences are pure AB2CN2 truncation, and convergence_plot
#   should show clean slope-2.
#
# Plus: same fixed-eta treatment as v3 (eta_brinkman = eta_sponge = 2.5e-3
# constant across all dts).
#
# Usage:
#   ./submit_cyl_sweep_v4.sh
#   ./submit_cyl_sweep_v4.sh --dry-run
#   ./submit_cyl_sweep_v4.sh --t-star 30.0   # override the restart time
#
# Outputs:
#   outputs/cylinder_dt_sweep_re200_v4/dt_<tag>/

set -e

# ---------------------------------------------------------------------- #
# Defaults                                                               #
# ---------------------------------------------------------------------- #

# Restart parameters
T_STAR=35.0           # extract IC at this time
T_RUN=10.0            # how long to integrate after restart
T_FINAL=$(awk "BEGIN{print $T_STAR + $T_RUN}")  # absolute final time (matches solver convention)

# Brinkman / sponge eta target (same as v3)
ETA_TARGET="2.5e-3"

# Format: "dt save_rate penalty sponge label"
# (penalty * dt = sponge * dt = 2.5e-3 in every case)
DT_CONFIGS=(
    "2.0e-3      25      1.25      1.25     2em3"
    "1.0e-3      50      2.50      2.50     1em3"
    "5.0e-4     100      5.00      5.00     5em4"
    "2.5e-4     200     10.00     10.00     2p5em4"
    "1.25e-4    400     20.00     20.00     1p25em4"
    "2.0e-5    2500    125.00    125.00     2em5"
    "1.0e-5    5000    250.00    250.00     1em5"
)

GRID_NX=512
GRID_NY=512
SCENARIO=flow_past_cylinder_sponge

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR="$QG_ROOT/qg-simple-package-stable/src/qg"
SUBMIT_SCRIPT="$QG_DIR/submit_qg.sh"

REF_DIR="$QG_DIR/outputs/cylinder_dt_sweep_re200_v3/dt_1em5"
RESTART_DIR="$QG_DIR/outputs/restart_ics/cyl_v3_t${T_STAR%.*}"
SWEEP_ROOT="$QG_DIR/outputs/cylinder_dt_sweep_re200_v4"

DRY_RUN=0
while [ "$#" -ge 1 ]; do
    case "$1" in
        --dry-run)     DRY_RUN=1; shift ;;
        --t-star)      T_STAR="$2"; T_FINAL=$(awk "BEGIN{print $T_STAR + $T_RUN}"); shift 2 ;;
        --t-run)       T_RUN="$2";  T_FINAL=$(awk "BEGIN{print $T_STAR + $T_RUN}"); shift 2 ;;
        --sweep-root)  SWEEP_ROOT="$2"; shift 2 ;;
        --restart-dir) RESTART_DIR="$2"; shift 2 ;;
        -h|--help)     sed -n '2,32p' "$0"; exit 0 ;;
        *)             echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$SWEEP_ROOT"
mkdir -p "$RESTART_DIR"

RESTART_NPY="$RESTART_DIR/restart_omega.npy"

echo "==================================================================="
echo "Cylinder dt-sensitivity sweep v4 (RESTART from developed flow)"
echo "  scenario      : $SCENARIO"
echo "  grid          : ${GRID_NX} x ${GRID_NY}"
echo "  reference     : $REF_DIR"
echo "  restart at    : t* = $T_STAR  (from ref dt=1e-5 run)"
echo "  run for       : $T_RUN time units after restart"
echo "  T_final       : $T_FINAL  (absolute, passed to qg.time.T)"
echo "  eta_target    : $ETA_TARGET (FIXED across all dts)"
echo "  outputs       : $SWEEP_ROOT/dt_<tag>/"
echo "  dry_run       : $DRY_RUN"
echo "==================================================================="
echo

# ---------------------------------------------------------------------- #
# Step A: Extract restart IC (only if not already done)                  #
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
        echo "  Looked for DNS_FR.npz or (DNS_FR_omega.npy + DNS_FR_times.npy)"
        exit 1
    fi

    SCRIPT_DIR="$QG_DIR/RestartSweeps"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[dry-run] would run:"
        echo "  python $SCRIPT_DIR/extract_restart_ic.py \\"
        echo "      --source-omega $REF_OMEGA \\"
        echo "      ${EXTRA_ARGS[*]} \\"
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
    read -r DT SAVE_RATE PENALTY SPONGE DT_LABEL <<< "$CONFIG"

    OUT_DIR="$SWEEP_ROOT/dt_${DT_LABEL}"
    JOBNAME="cyl4_dt${DT_LABEL}_T${T_RUN%.*}_${GRID_NX}_gpu"

    mkdir -p "$OUT_DIR"

    ETA_BRINK=$(awk -v p="$PENALTY" -v d="$DT" 'BEGIN { printf "%.4e", p * d }')
    ETA_SPONGE=$(awk -v s="$SPONGE"  -v d="$DT" 'BEGIN { printf "%.4e", s * d }')

    echo "-----  dt=$DT, save_rate=$SAVE_RATE, penalty=$PENALTY, sponge=$SPONGE  -----"
    echo "  eta_brinkman = $ETA_BRINK,  eta_sponge = $ETA_SPONGE"
    echo "  jobname      : $JOBNAME"
    echo "  out_dir      : $OUT_DIR"

    # Override IC via Hydra: ic.function='from_file' + ic.path=...
    CMD=(
        "$SUBMIT_SCRIPT" "$JOBNAME" --gpu --
        scenario="$SCENARIO"
        qg.grid.Nx="$GRID_NX"
        qg.grid.Ny="$GRID_NY"
        qg.time.T="$T_FINAL"
        qg.time.dt="$DT"
        qg.time.save_rate="$SAVE_RATE"
        qg.pde.penalty="$PENALTY"
        qg.bc.sponge="$SPONGE"
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
echo "Tail:    tail -f $QG_DIR/logs/cyl4_dt<tag>_T${T_RUN%.*}_${GRID_NX}_gpu.log"
echo
echo "Once done, run convergence pipeline pointing at v4:"
echo "  ./Convergence_studies/Flow_Past_Cylinder/Step1/submit_step1.sh \\"
echo "      --sweep-root $SWEEP_ROOT --jobname step1_cyl_v4 --force-convert"
echo "  ./Convergence_studies/Flow_Past_Cylinder/ConvergencePlot/convergence_plot.sh \\"
echo "      --sweep-root $SWEEP_ROOT --jobname convergence_plot_cyl_v4 --t-start 0.0"
echo "  (--t-start 0.0 is fine for v4: there's no spinup, all dts start"
echo "   from the same developed-flow IC.)"
echo "==================================================================="
