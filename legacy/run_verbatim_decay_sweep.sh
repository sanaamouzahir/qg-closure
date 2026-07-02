#!/bin/bash
# run_verbatim_decay_sweep.sh - run run_qg.py with the colleague's verbatim
# decay params, varying only dt. Each run uses a per-dt save_rate chosen so
# that ALL runs save snapshots at the SAME physical times.
#
# This matters when comparing very fine dts: with a fixed save_rate (in steps),
# different dts produce saves at different physical times, and the verbatim
# plot's [0, -2, 0] indexing compares snapshots that aren't at the same t.
# At fine dt the chaotic decorrelation between such snapshots dominates the
# error and the slope-2 convergence vanishes.
#
# Per-dt save_rate is chosen so dt * save_rate = TARGET_SAVE_DT = 0.01.
# All 8 dts give integer save_rates and produce the SAME number of snapshots
# (400) at the SAME physical times (t = 0.01, 0.02, ..., 4.00).
#
#   dt        save_rate  -> dt*save_rate
#   ---------------------------------------
#   1.0e-5    1000          0.01
#   2.5e-5    400           0.01
#   5.0e-5    200           0.01
#   1.0e-4    100           0.01
#   5.0e-4     20           0.01
#   1.0e-3     10           0.01
#   2.0e-3      5           0.01
#   5.0e-3      2           0.01
#
# Output:
#   $OUT_ROOT/<dt_tag>/qg_data.npy
#
# Usage:
#   ./run_verbatim_decay_sweep.sh                # SGE submission, one job per dt
#   ./run_verbatim_decay_sweep.sh --interactive  # local, sequential
#   ./run_verbatim_decay_sweep.sh --dry-run

set -e

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR="$QG_ROOT/qg-simple-package-stable/src/qg"
RUN_PY="$QG_DIR/run_qg.py"
OUT_ROOT="$QG_DIR/outputs/decay_verbatim_convergence_seed42_aligned_T1_Re100"

# Per-dt arrays (parallel: same index for each)
DT_VALUES=(   0.00001     0.000025      0.00005     0.0001     0.0005     0.001     0.002     0.005     )
DT_TAGS=(     dt_0.00001  dt_0.000025   dt_0.00005  dt_0001    dt_0005    dt_0010   dt_0020   dt_0050   )
SAVE_RATES=(  1000        400           200         100        20         10        5         2         )

INTERACTIVE=0
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --interactive)  INTERACTIVE=1; shift ;;
        --dry-run)      DRY_RUN=1; shift ;;
        --out-root)     OUT_ROOT="$2"; shift 2 ;;
        -h|--help)      sed -n '2,32p' "$0"; exit 0 ;;
        *)              echo "unknown arg: $1"; exit 1 ;;
    esac
done

if [ ! -f "$RUN_PY" ]; then
    echo "ERROR: $RUN_PY not found"
    exit 1
fi

mkdir -p "$OUT_ROOT"
LOG_DIR="$QG_DIR/logs"
mkdir -p "$LOG_DIR"

# Verbatim Hydra overrides (everything except dt and save_rate, which are per-run)
COMMON_OVERRIDES=(
    "qg.grid.Nx=128"
    "qg.grid.Ny=128"
    "qg.time.T=1"
    "qg.pde.nu=1.025e-2"
    "qg.ic.n_batch=1"
    "qg.ic.energy=0.1"
    "qg.ic.wavenumbers=[3.0,5.0]"
    "qg.seed=42"
)

echo "==================================================================="
echo " Verbatim decay convergence sweep with ALIGNED save times          "
echo "==================================================================="
echo "  out_root        : $OUT_ROOT"
echo "  run_py          : $RUN_PY"
echo "  target save dt  : 0.01 (all runs save at t = 0.01, 0.02, ..., 4.00)"
echo "  expected n_saves: 400"
echo
echo "  dt          save_rate  ->  dt*save_rate"
for i in "${!DT_VALUES[@]}"; do
    actual=$(awk "BEGIN{print ${DT_VALUES[$i]} * ${SAVE_RATES[$i]}}")
    printf "    %-10s  %-9s     %s\n" "${DT_VALUES[$i]}" "${SAVE_RATES[$i]}" "$actual"
done
echo "==================================================================="
echo

for i in "${!DT_VALUES[@]}"; do
    DT_VAL="${DT_VALUES[$i]}"
    DT_TAG="${DT_TAGS[$i]}"
    SAVE_RATE="${SAVE_RATES[$i]}"
    OUT_DIR="$OUT_ROOT/$DT_TAG"
    mkdir -p "$OUT_DIR"

    JOBNAME="decay_verb_${DT_TAG}"
    JOB_LOG="$LOG_DIR/${JOBNAME}.log"

    PY_ARGS=(
        "scenario=decaying_turbulence"
        "qg.time.dt=$DT_VAL"
        "qg.time.save_rate=$SAVE_RATE"
        "${COMMON_OVERRIDES[@]}"
        "hydra.run.dir=$OUT_DIR"
    )

    echo "----- $DT_TAG (dt=$DT_VAL, save_rate=$SAVE_RATE) -----"

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  [dry-run] python $RUN_PY ${PY_ARGS[*]}"
        continue
    fi

    if [ "$INTERACTIVE" -eq 1 ]; then
        source "$QG_ROOT/qg-env/bin/activate"
        export MPLCONFIGDIR="$QG_ROOT/.mplcache"
        mkdir -p "$MPLCONFIGDIR"
        export MPLBACKEND=Agg
        cd "$QG_DIR"
        python -u "$RUN_PY" "${PY_ARGS[@]}"
        if [ -f "$OUT_DIR/DNS.npy" ]; then
            mv "$OUT_DIR/DNS.npy" "$OUT_DIR/qg_data.npy"
            echo "  renamed $OUT_DIR/DNS.npy -> $OUT_DIR/qg_data.npy"
        fi
        echo "  done -> $OUT_DIR"
        continue
    fi

    TMP_JOB="$LOG_DIR/${JOBNAME}.run.sh"
    cat > "$TMP_JOB" <<EOF
#!/bin/bash
set -e
QG_ROOT=$QG_ROOT
source "\$QG_ROOT/qg-env/bin/activate"
export MPLCONFIGDIR="\$QG_ROOT/.mplcache"
mkdir -p "\$MPLCONFIGDIR"
export MPLBACKEND=Agg
echo "[$JOBNAME] hostname: \$HOSTNAME"
echo "[$JOBNAME] date:     \$(date -u +%FT%TZ)"
echo "[$JOBNAME] out_dir:  $OUT_DIR"
echo "[$JOBNAME] dt:       $DT_VAL"
echo "[$JOBNAME] save_rate:$SAVE_RATE"
echo "------------------------------------------------------------"
cd "$QG_DIR"
$(printf '%q ' python -u "$RUN_PY" "${PY_ARGS[@]}")
if [ -f "$OUT_DIR/DNS.npy" ]; then
    mv "$OUT_DIR/DNS.npy" "$OUT_DIR/qg_data.npy"
    echo "renamed $OUT_DIR/DNS.npy -> $OUT_DIR/qg_data.npy"
fi
echo "------------------------------------------------------------"
echo "[$JOBNAME] done at \$(date -u +%FT%TZ)"
EOF
    chmod +x "$TMP_JOB"

    qsub -N "$JOBNAME" \
         -o "$JOB_LOG" -e "$JOB_LOG" -j y -V \
         -wd "$QG_DIR" \
         -q "ibgpu.q" -l "gpu=1" \
         "$TMP_JOB"
    echo "  submitted -> $JOB_LOG"
done

echo
if [ "$DRY_RUN" -eq 1 ]; then
    echo "Dry-run complete."
elif [ "$INTERACTIVE" -eq 1 ]; then
    echo "All runs done. Outputs at $OUT_ROOT."
else
    echo "All jobs submitted. Watch with:  qstat -u \$USER"
    echo "Outputs will appear under: $OUT_ROOT"
fi