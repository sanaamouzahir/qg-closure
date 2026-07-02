#!/bin/bash
# submit_ft_dt1em3_energy0p01.sh
#
# Single-run test for forced turbulence at dt=1e-3, MIT-paper-faithful but
# with reduced IC energy (0.01 instead of 0.1) to gentle the transient.
#
# Hypothesis: the failed dt=1e-3 (and 5e-4, 2.5e-4) runs at energy=0.1 hit
# CFL during the transient peak. A 10x smaller IC magnitude should reduce
# the transient peak by ~sqrt(10)~3x, bringing |omega|_max during the
# transient from estimated 40 down to ~13, which gives dt_CFL ~ 2e-3.
# That makes dt=1e-3 comfortably stable.
#
# This is a SINGLE run, not a sweep. If it succeeds, we'll know energy=0.01
# is the right setting and can re-launch the full sweep.

set -e

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
SUBMIT_SCRIPT="$QG_ROOT/qg-simple-package-stable/src/qg/submit_qg.sh"

# Output directory (separate from the energy=0.1 sweep so we don't mix data)
OUT_DIR="$QG_ROOT/qg-simple-package-stable/src/qg/outputs/forced_turbulence_dt1em3_test_energy0p01"
mkdir -p "$OUT_DIR"

JOBNAME="ft_dt1em3_e0p01_test"

echo "=========================================================="
echo "  FT test: dt=1e-3, energy=0.01, MIT-paper-faithful"
echo "  out_dir: $OUT_DIR"
echo "=========================================================="

"$SUBMIT_SCRIPT" "$JOBNAME" --gpu -- \
    scenario=forced_turbulence_paper \
    qg.grid.Nx=512 \
    qg.grid.Ny=512 \
    qg.time.T=50 \
    qg.time.dt=2.5e-4 \
    qg.time.save_rate=50 \
    qg.pde.nu=1.0e-3 \
    qg.pde.mu=0.02 \
    qg.pde.B=0.0 \
    qg.forcing.A=2.0 \
    qg.forcing.B=2.0 \
    qg.forcing.C=0.0 \
    qg.forcing.D=2.0 \
    qg.forcing.E=2.0 \
    qg.forcing.F=0.0 \
    qg.ic.function=randn \
    qg.ic.energy=0.01 \
    qg.ic.wavenumbers="[3.0,10.0]" \
    hydra.run.dir="$OUT_DIR"

echo
echo "Submitted. Monitor early with:"
echo "  qstat -u \$USER"
echo "  tail -f $QG_ROOT/qg-simple-package-stable/src/qg/logs/${JOBNAME}_*.log"
echo
echo "If it survives past t~17 (the transient peak), the IC energy fix worked."
echo "Then re-launch the full sweep with energy=0.01."
