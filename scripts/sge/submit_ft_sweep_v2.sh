#!/bin/bash
# submit_ft_sweep_v2.sh
#
# PHASE B of the forced-turbulence pipeline: dt-sensitivity sweep RESTARTED
# from omega(t=10) of the reference run produced by submit_ft_reference.sh.
#
# Same restart logic as submit_cyl_sweep_v4.sh and submit_decay_sweep_v2.sh.
# Forced turbulence is also chaotic, so without restart we'd compare different
# realizations. With shared restart IC, all dts diverge only by AB2CN2
# truncation -> clean slope-2.
#
# Usage:
#   ./submit_ft_sweep_v2.sh                                # default reference path
#   ./submit_ft_sweep_v2.sh --reference-dir <path>         # custom reference
#   ./submit_ft_sweep_v2.sh --dry-run

set -e

# ---------------------------------------------------------------------- #
# Defaults                                                               #
# ---------------------------------------------------------------------- #

T_STAR=10.0           # restart time (= end of reference run)
T_RUN=5.0             # additional time after restart
T_FINAL=$(awk "BEGIN{print $T_STAR + $T_RUN}")  # qg.time.T

# Format: "dt save_rate label"
# (matches the resolution of the cylinder/decay sweeps for consistency)
DT_CONFIGS=(
    "1.0e-3      50      1em3"
    "5.0e-4     100      5em4"
    "2.5e-4     200      2p5em4"
    "1.0e-4     500      1em4"
    "5.0e-5    1000      5em5"
    "1.0e-5    5000      1em5"
)

GRID_NX=1024
GRID_NY=1024
SCENARIO=forced_turbulence

# MIT-recommended forcing + IC params (matched to reference run)
FORCING_A="-0.1"
FORCING_B=2
FORCING_C=0
FORCING_D="0.1"
FORCING_E=2
FORCING_F=0

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR="$QG_ROOT/qg-simple-package-stable/src/qg"
SUBMIT_SCRIPT="$QG_DIR/submit_qg.sh"

REF_DIR="$QG_DIR/outputs/forced_turb_reference_T10_dt1.0e-5_${GRID_NX}"
RESTART_DIR="$QG_DIR/outputs/restart_ics/ft_t${T_STAR%.*}"
SWEEP_ROOT="$QG_DIR/outputs/forced_turb_dt_sweep_v2"

DRY_RUN=0
while [ "$#" -ge 1 ]; do
    case "$1" in
        --dry-run)        DRY_RUN=1; shift ;;
        --t-star)         T_STAR="$2"; T_FINAL=$(awk "BEGIN{print $T_STAR + $T_RUN}"); shift 2 ;;
        --t-run)          T_RUN="$2";  T_FINAL=$(awk "BEGIN{print $T_STAR + $T_RUN}"); shift 2 ;;
        --reference-dir)  REF_DIR="$2"; shift 2 ;;
        --sweep-root)     SWEEP_ROOT="$2"; shift 2 ;;
        --restart-dir)    RESTART_DIR="$2"; shift 2 ;;
        -h|--help)        sed -n '2,18p' "$0"; exit 0 ;;
        *)                echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$SWEEP_ROOT" "$RESTART_DIR"

RESTART_NPY="$RESTART_DIR/restart_omega.npy"

echo "==================================================================="
echo "Forced turbulence: PHASE B dt-sensitivity sweep v2 (RESTART)"
echo "  scenario      : $SCENARIO"
echo "  grid          : ${GRID_NX} x ${GRID_NY}"
echo "  reference     : $REF_DIR"
echo "  restart at    : t* = $T_STAR"
echo "  run for       : $T_RUN time units after restart"
echo "  T_final       : $T_FINAL"
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
        echo "  Did Phase A finish?"
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
    JOBNAME="ft2_dt${DT_LABEL}_T${T_RUN%.*}_${GRID_NX}_gpu"

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
        qg.forcing.A="$FORCING_A"
        qg.forcing.B="$FORCING_B"
        qg.forcing.C="$FORCING_C"
        qg.forcing.D="$FORCING_D"
        qg.forcing.E="$FORCING_E"
        qg.forcing.F="$FORCING_F"
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
echo "Once done, run convergence pipeline pointing at v2."
echo "==================================================================="
