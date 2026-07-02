#!/bin/bash
# submit_ft_reference.sh
#
# PHASE A of the forced-turbulence pipeline: run a single fine-resolution
# reference simulation from t=0 to t=10 with the MIT-recommended params.
#
# After this finishes, run submit_ft_sweep_v2.sh which restarts each dt of
# the convergence sweep from omega(t=10) of this reference.
#
# MIT-recommended params (per direct conversation with paper authors;
# slightly different from the published paper but these are what they say
# we should actually use):
#
#   IC:   randn with energy=0.01, wavenumbers=[3,5], seed=495
#   forcing:  A=-1/10, B=2, C=0, D=1/10, E=2, F=0   (cos forcing)
#
# Other physics inherits from the existing forced_turbulence.yaml:
#   nu=1.025e-4, mu=0.02, beta=1.0, Lx=Ly=4*pi (12.566...), 512x512
#
# Usage:
#   ./submit_ft_reference.sh                     # default 1024x1024, dt=1e-5, T=10
#   ./submit_ft_reference.sh --dry-run

set -e

# ---------------------------------------------------------------------- #
# Defaults                                                               #
# ---------------------------------------------------------------------- #

GRID_NX=1024
GRID_NY=1024
DT=1.0e-5
T_FINAL=10
SAVE_RATE=5000     # save_rate * dt = 0.05 -> 200 saved snapshots over [0, 10]
SCENARIO=forced_turbulence

# MIT-recommended IC + forcing
IC_ENERGY="0.01"
IC_SEED=495
FORCING_A="-0.1"
FORCING_B=2
FORCING_C=0
FORCING_D="0.1"
FORCING_E=2
FORCING_F=0

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR="$QG_ROOT/qg-simple-package-stable/src/qg"
SUBMIT_SCRIPT="$QG_DIR/submit_qg.sh"
OUT_DIR="$QG_DIR/outputs/forced_turb_reference_T${T_FINAL}_dt${DT}_${GRID_NX}"

DRY_RUN=0
while [ "$#" -ge 1 ]; do
    case "$1" in
        --dry-run)   DRY_RUN=1; shift ;;
        --grid-nx)   GRID_NX="$2"; GRID_NY="$2"; OUT_DIR="$QG_DIR/outputs/forced_turb_reference_T${T_FINAL}_dt${DT}_${GRID_NX}"; shift 2 ;;
        --t-final)   T_FINAL="$2"; shift 2 ;;
        --dt)        DT="$2"; shift 2 ;;
        --save-rate) SAVE_RATE="$2"; shift 2 ;;
        --out-dir)   OUT_DIR="$2"; shift 2 ;;
        -h|--help)   sed -n '2,30p' "$0"; exit 0 ;;
        *)           echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUT_DIR"

JOBNAME="ft_ref_T${T_FINAL}_dt${DT}_${GRID_NX}_gpu"

echo "==================================================================="
echo "Forced turbulence: PHASE A reference run"
echo "  scenario      : $SCENARIO"
echo "  grid          : ${GRID_NX} x ${GRID_NY}"
echo "  dt            : $DT"
echo "  T             : $T_FINAL"
echo "  save_rate     : $SAVE_RATE  (-> dt*save_rate = $(awk -v d=$DT -v s=$SAVE_RATE 'BEGIN{print d*s}'))"
echo "  IC            : randn, energy=$IC_ENERGY, seed=$IC_SEED"
echo "  forcing       : cos(2x)*(${FORCING_A}) + cos(2y)*(${FORCING_D})"
echo "  out_dir       : $OUT_DIR"
echo "  jobname       : $JOBNAME"
echo "  dry_run       : $DRY_RUN"
echo "==================================================================="
echo

CMD=(
    "$SUBMIT_SCRIPT" "$JOBNAME" --gpu --
    scenario="$SCENARIO"
    qg.grid.Nx="$GRID_NX"
    qg.grid.Ny="$GRID_NY"
    qg.time.T="$T_FINAL"
    qg.time.dt="$DT"
    qg.time.save_rate="$SAVE_RATE"
    qg.ic.energy="$IC_ENERGY"
    qg.ic.seed="$IC_SEED"
    qg.ic.wavenumbers="[3.0, 5.0]"
    qg.forcing.A="$FORCING_A"
    qg.forcing.B="$FORCING_B"
    qg.forcing.C="$FORCING_C"
    qg.forcing.D="$FORCING_D"
    qg.forcing.E="$FORCING_E"
    qg.forcing.F="$FORCING_F"
    hydra.run.dir="$OUT_DIR"
)

if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] would submit:"
    printf "    %s\n" "${CMD[@]}"
else
    "${CMD[@]}"
fi

echo
echo "==================================================================="
echo "After this finishes (~hours), do:"
echo "  1. Convert DNS_FR.npz -> DNS_FR_omega.npy (if needed for extract step)"
echo "  2. Run: ./submit_ft_sweep_v2.sh --reference-dir $OUT_DIR"
echo "==================================================================="
