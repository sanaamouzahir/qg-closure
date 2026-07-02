#!/bin/bash
# run_colleague_t10_sweep.sh -- reproduce the colleague's T=10 sweep to
# verify slope 2 of AB2-CN2 in decay turbulence (after his fix to qg.py).
#
# YAML params (verbatim):
#   Nx=Ny=256, precision=float64, T=10, save_rate=1000,
#   nu=1.025e-2, ic.energy=0.1, ic.wavenumbers=[3,5], n_batch=1,
#   no forcing, no mask
#
# dts: [0.0001, 0.001, 0.002, 0.005, 0.01]   (his list, sorted)
# Reference: dt = 0.0001
#
# REQUIRES: qg.py must have his FIXED loop:
#   for it in tqdm(range(steps)):       # NOT range(steps - 1)
#       self.step(state)
#       if (it+1) % save_rate == 0:
#           solution[:, (it+1)//save_rate, ...] = state.out()
#   solution[:, -1, ...] = state.out()  # post-loop overwrite (idempotent now)

set -e

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR="$QG_ROOT/qg-simple-package-stable/src/qg"
RUN_PY="$QG_DIR/run_qg.py"
OUT_ROOT="$QG_DIR/outputs/akhil_sweep_Re1e5"

DT_VALUES=( 0.0001    0.001    0.002    0.005    0.01    )
DT_TAGS=(   dt_0001   dt_0010  dt_0020  dt_0050  dt_0100 )

INTERACTIVE=0
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --interactive)  INTERACTIVE=1; shift ;;
        --dry-run)      DRY_RUN=1; shift ;;
        --out-root)     OUT_ROOT="$2"; shift 2 ;;
        -h|--help)      sed -n '2,30p' "$0"; exit 0 ;;
        *)              echo "unknown arg: $1"; exit 1 ;;
    esac
done

if [ ! -f "$RUN_PY" ]; then
    echo "ERROR: $RUN_PY not found"
    exit 1
fi

# Sanity-check qg.py has the FIXED loop
QG_PY="$QG_DIR/qg.py"
if [ -f "$QG_PY" ]; then
    if grep -q "for it in tqdm(range(steps - 1))" "$QG_PY"; then
        echo "[WARN] qg.py STILL has 'range(steps - 1)' -- this is the buggy loop."
        echo "       Change it to 'range(steps)' before running this experiment."
        echo "       (continuing in 5 sec; Ctrl-C to abort)"
        sleep 5
    elif grep -q "for it in tqdm(range(steps))" "$QG_PY"; then
        echo "[ok] qg.py has the fixed 'range(steps)' loop"
    else
        echo "[WARN] qg.py has neither 'range(steps - 1)' nor 'range(steps)'."
        echo "       Verify the loop manually."
    fi
fi

mkdir -p "$OUT_ROOT"
LOG_DIR="$QG_DIR/logs"
mkdir -p "$LOG_DIR"

COMMON_OVERRIDES=(
    "qg.grid.Nx=256"
    "qg.grid.Ny=256"
    "qg.grid.precision=float64"
    "qg.time.T=10"
    "qg.time.save_rate=1000"
    "qg.pde.nu=1.025e-5"
    "qg.ic.n_batch=1"
    "qg.ic.energy=0.1"
    "qg.ic.wavenumbers=[3.0,5.0]"
    "qg.seed=42"
)

echo "==================================================================="
echo " Colleague T=10 verbatim reproduction sweep                        "
echo "==================================================================="
echo "  out_root  : $OUT_ROOT"
echo "  T=10, save_rate=1000, nu=1.025e-2, float64"
for i in "${!DT_VALUES[@]}"; do
    printf "    %-10s -> dt=%s\n" "${DT_TAGS[$i]}" "${DT_VALUES[$i]}"
done
echo "==================================================================="
echo

for i in "${!DT_VALUES[@]}"; do
    DT_VAL="${DT_VALUES[$i]}"
    DT_TAG="${DT_TAGS[$i]}"
    OUT_DIR="$OUT_ROOT/$DT_TAG"
    mkdir -p "$OUT_DIR"

    JOBNAME="t10v_${DT_TAG}"
    JOB_LOG="$LOG_DIR/${JOBNAME}.log"

    PY_ARGS=(
        "scenario=decaying_turbulence"
        "qg.time.dt=$DT_VAL"
        "${COMMON_OVERRIDES[@]}"
        "hydra.run.dir=$OUT_DIR"
    )

    echo "----- $DT_TAG (dt=$DT_VAL) -----"

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
echo "------------------------------------------------------------"
cd "$QG_DIR"
$(printf '%q ' python -u "$RUN_PY" "${PY_ARGS[@]}")
if [ -f "$OUT_DIR/DNS.npy" ]; then
    mv "$OUT_DIR/DNS.npy" "$OUT_DIR/qg_data.npy"
    echo "renamed DNS.npy -> qg_data.npy"
fi
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
    echo
    echo "Once all 5 finish, plot with:"
    echo "  cd $OUT_ROOT && python convergence_t10_plot.py"
fi
