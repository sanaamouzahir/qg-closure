#!/bin/bash
# submit_dt_sweep_ft_re1k.sh
#
# Submit a dt-sensitivity sweep for FORCED TURBULENCE simulations at
# Re = 10^3 (eddy regime, beta = 0), MIT-paper-faithful:
#   Suresh Babu, Sadam, Lermusiaux, JAMES 2026 (arXiv:2508.06678).
#
# Usage:
#   ./submit_dt_sweep_ft_re1k.sh
#
# Paper-faithful parameters (their Table 2 + IC/forcing description):
#   N x N        = 512 x 512
#   L_x = L_y    = 2*pi
#   Re           = 10^3 (so nu = 1/Re = 1.0e-3)
#   mu           = 0.02
#   beta         = 0.0  (eddy regime)
#   k_f          = 2  (forcing wavenumber, MIT paper Eq. 24)
#   F            = k_f * (cos(k_f x) + cos(k_f y))   = 2*[cos(2x)+cos(2y)]
#   IC           = randn filtered to k in [3, 10], using qg.ic._init_randn
#                  exactly as MIT paper Section 4.2. Energy parameter set
#                  to a modest non-zero value (the paper doesn't publish
#                  the exact value; the post-transient regime is forcing-
#                  dominated and insensitive to IC magnitude anyway).
#   T            = 50  (statistical regime starts t > 33 from prior diagnosis;
#                       MIT paper uses T=100 with t<50 discarded; we use T=50
#                       to keep wall-clock manageable while capturing the
#                       statistical regime t in [33,50])
#
# Differences from previous (failed) sweep at this Re:
#   - k_f = 2 instead of 4 (YAML defaulted to Srinivasan 2024's k_f=4)
#   - IC is randn (not all-zeros / "from rest")
#   - IC filter band [3,10] instead of [3,5]
#   - Forcing prefactors 2.0 instead of 4.0 (k_f = 2 is 2x weaker per mode)
#
# Re-estimated CFL at k_f = 2, expecting transient peak |omega|_max ~ 35
# (vs ~99 in the failed runs at k_f=4):
#   dt_CFL ~ dx * k_dom / |omega|_max ~ (0.0123) * 2 / 35 ~ 7e-4
# So dt = 1e-3 may still be marginal during the transient; kept anyway as
# the paper anchor and as a diagnostic stress point.
#
# dt sweep (7 values, factor-of-2 ratios for slope-2 fits where it matters):
#   dt = {1e-3, 5e-4, 2.5e-4, 1.25e-4, 5e-5, 2e-5, 1e-5}
#   save_rate chosen so dt * save_rate = 0.05 (uniform 0.05 t.u. snapshot
#   cadence for fair direct comparison across runs).
#   Total snapshots per run = T / 0.05 = 1000.
#
# Note: dt=2e-3 was DROPPED (CFL-violating even in the gentler config).
# The reference dt=1e-5 is 100x finer than the paper's anchor 1e-3.

set -e

# ---------------------------------------------------------------------- #
# Sweep configuration                                                    #
# ---------------------------------------------------------------------- #

# Format: "dt_value  save_rate  label  role"
# (dt * save_rate = 0.05 in every entry)
DT_CONFIGS=(
    "1.0e-3      50  1em3       anchor"      # MIT Table 2 paper dt; CFL-marginal
    "5.0e-4     100  5em4       convergent"  # 2x finer
    "2.5e-4     200  2p5em4     convergent"
    "1.25e-4    400  1p25em4    convergent"
    "5.0e-5    1000  5em5       fine"
    "2.0e-5    2500  2em5       fine"
    "1.0e-5    5000  1em5       reference"   # 100x finer than paper anchor
)

# ---------------------------------------------------------------------- #
# Common (paper-faithful) run parameters                                 #
# ---------------------------------------------------------------------- #

GRID_NX=512
GRID_NY=512
T_FINAL=50
SCENARIO=forced_turbulence_paper

# PDE coefficients (override scenario YAML defaults to be MIT-paper-faithful)
NU=1.0e-3       # nu = 1/Re for Re = 10^3
MU=0.02         # bottom drag (MIT Table 2)
BETA=0.0        # eddy regime

# Forcing: F = k_f * (cos(k_f x) + cos(k_f y)) with k_f = 2
# In the YAML's unscaled_cosine, F = A*cos(B*x) + D*cos(E*y).
# So set A = D = 2 (= k_f), B = E = 2 (= k_f), C = F = 0.
FORCE_A=2.0
FORCE_B=2.0
FORCE_C=0.0
FORCE_D=2.0
FORCE_E=2.0
FORCE_F=0.0

# IC: randn filtered to k in [3, 10], using the solver's built-in
# qg.ic._init_randn exactly as MIT paper Section 4.2 describes.
IC_FUNCTION=randn
IC_ENERGY=0.5
IC_K_MIN=3.0
IC_K_MAX=10.0

# ---------------------------------------------------------------------- #
# Paths                                                                  #
# ---------------------------------------------------------------------- #

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
SWEEP_ROOT="$QG_ROOT/qg-simple-package-stable/src/qg/outputs/forced_turbulence_dt_sweep_re1k"
SUBMIT_SCRIPT="$QG_ROOT/qg-simple-package-stable/src/qg/submit_qg.sh"

BETA_TAG=$(echo "$BETA" | tr '.' 'p')
BETA_DIR="$SWEEP_ROOT/beta_${BETA_TAG}"

mkdir -p "$BETA_DIR"

# ---------------------------------------------------------------------- #
# Banner                                                                 #
# ---------------------------------------------------------------------- #

echo "================================================================"
echo " Forced turbulence dt sweep, MIT-paper-faithful (Re = 10^3)     "
echo "================================================================"
echo "  scenario : $SCENARIO"
echo "  grid     : ${GRID_NX} x ${GRID_NY}"
echo "  T        : $T_FINAL"
echo "  Re       : 1e3   (nu = $NU, mu = $MU, beta = $BETA)"
echo "  forcing  : F = 2*[cos(2x) + cos(2y)]   (k_f = 2)"
echo "  IC       : omega ~ N(0,1) filtered to k in [$IC_K_MIN, $IC_K_MAX]"
echo "             (qg.ic.energy = $IC_ENERGY, calibrated)"
echo "  dt set   : 1e-3, 5e-4, 2.5e-4, 1.25e-4, 5e-5, 2e-5, 1e-5"
echo "  ref dt   : 1e-5  (100x finer than paper anchor)"
echo "  outputs  : $BETA_DIR/dt_<tag>/"
echo "================================================================"
echo

# ---------------------------------------------------------------------- #
# Manifest CSV                                                           #
# ---------------------------------------------------------------------- #

MANIFEST="$BETA_DIR/sweep_manifest.csv"
{
    echo "# Sweep manifest for $SWEEP_ROOT/beta_${BETA_TAG}/"
    echo "# Generated by submit_dt_sweep_ft_re1k.sh on $(date -Iseconds)"
    echo "# scenario=$SCENARIO  grid=${GRID_NX}x${GRID_NY}  T=$T_FINAL"
    echo "# nu=$NU  mu=$MU  beta=$BETA  k_f=2"
    echo "# IC: omega ~ N(0,1) filtered to k in [$IC_K_MIN, $IC_K_MAX], qg.ic.energy=$IC_ENERGY"
    echo "label,dt,save_rate,role,out_dir"
    for CONFIG in "${DT_CONFIGS[@]}"; do
        read -r DT SAVE_RATE DT_LABEL ROLE <<< "$CONFIG"
        echo "$DT_LABEL,$DT,$SAVE_RATE,$ROLE,$BETA_DIR/dt_${DT_LABEL}"
    done
} > "$MANIFEST"
echo "Wrote sweep manifest: $MANIFEST"
echo

# ---------------------------------------------------------------------- #
# Submit jobs                                                            #
# ---------------------------------------------------------------------- #

TOTAL=0
for CONFIG in "${DT_CONFIGS[@]}"; do
    read -r DT SAVE_RATE DT_LABEL ROLE <<< "$CONFIG"

    OUT_DIR="$BETA_DIR/dt_${DT_LABEL}"
    JOBNAME="ftR1k_b${BETA_TAG}_dt${DT_LABEL}_T${T_FINAL}_${GRID_NX}_gpu"

    mkdir -p "$OUT_DIR"

    echo "-----  dt=$DT  ($ROLE)  save_rate=$SAVE_RATE  -----"
    echo "  jobname : $JOBNAME"
    echo "  out_dir : $OUT_DIR"

    "$SUBMIT_SCRIPT" "$JOBNAME" --gpu -- \
        scenario="$SCENARIO" \
        qg.grid.Nx="$GRID_NX" \
        qg.grid.Ny="$GRID_NY" \
        qg.time.T="$T_FINAL" \
        qg.time.dt="$DT" \
        qg.time.save_rate="$SAVE_RATE" \
        qg.pde.nu="$NU" \
        qg.pde.mu="$MU" \
        qg.pde.B="$BETA" \
        qg.forcing.A="$FORCE_A" \
        qg.forcing.B="$FORCE_B" \
        qg.forcing.C="$FORCE_C" \
        qg.forcing.D="$FORCE_D" \
        qg.forcing.E="$FORCE_E" \
        qg.forcing.F="$FORCE_F" \
        qg.ic.function="$IC_FUNCTION" \
        qg.ic.energy="$IC_ENERGY" \
        qg.ic.wavenumbers="[$IC_K_MIN,$IC_K_MAX]" \
        hydra.run.dir="$OUT_DIR"

    TOTAL=$((TOTAL + 1))
    echo
done

# ---------------------------------------------------------------------- #
# Summary                                                                #
# ---------------------------------------------------------------------- #

echo "================================================================"
echo "Submitted $TOTAL forced-turbulence jobs at Re=10^3 (MIT-faithful)."
echo
echo "Monitor with:"
echo "  qstat -u \$USER"
echo
echo "Reference run (for step 1 convergence analysis):"
echo "  $BETA_DIR/dt_1em5/"
echo
echo "Statistical-regime analysis: t > 33 (after transient decay)."
echo
echo "Note: dt_1em3 may CFL-fail during the transient peak (t in [4,17])."
echo "If so, this is informative; the paper's dt=1e-3 anchor was at a"
echo "different transient overshoot; the post-transient steady state is"
echo "what counts."
echo "================================================================"