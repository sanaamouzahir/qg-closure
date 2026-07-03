#!/usr/bin/env bash
# Run ALL theoretical-guarantee checks on the 3 deep survivors, archived via
# run_guarantee.py into ./Results/<diagnostic>/<case>/<run>/.
#
# Run FROM Theoretical_guarantees/ (where the diagnostic scripts + run_guarantee.py live):
#     cd $QG_DIR/training/Theoretical_guarantees
#     bash run_all_guarantees.sh
#
# Data lives one level up in training/data, so paths use ../data/...
# Local-module imports (tests 4,5) must resolve from here -> we add training/ to
# PYTHONPATH so deriv_dataset / model_deriv_closure / build_training_data_fixD_v2 /
# closure_operators (which live in training/) are importable.
set -u
export PYTHONPATH="..:${PYTHONPATH:-}"      # training/ on the path for local imports
RG="python run_guarantee.py"
DEV=cuda
D=../data/ensemble_N5
DEEP="$D/FRC-Re25k/forced_turbulence_dT_5em3 \
      $D/FRC-combo/forced_turbulence_dT_5em3 \
      $D/FRC-kf4/forced_turbulence_dT_5em3"

echo "########## 1/5  convergence_radius (deep sources) ##########"
$RG --diagnostic convergence_radius --note "3 deep survivors; outer dT* + inner k=7 wall" -- \
  python convergence_radius.py --sources $DEEP \
    --max-order 5 --n-samples 32 --wall-k 7 --device $DEV --dtype float64

echo "########## 2/5  fd_depth_check (deep sources) ##########"
$RG --diagnostic fd_depth_check --note "4 vs 7 lag N_ddot floor" -- \
  python fd_depth_check.py --sources $DEEP \
    --target-dts 5e-3 1e-2 1.5e-2 --depths 4 7 --n-samples 48 --device $DEV --dtype float64

echo "########## 3/5  fd_floor (deep sources, n=3..7 all orders) ##########"
$RG --diagnostic fd_floor --note "per-order temporal-FD floor, n_time sweep" -- \
  python temporal_fd_floor_deep.py --sources $DEEP \
    --target-dts 5e-3 1e-2 1.5e-2 --n-list 3 4 5 6 7 --n-samples 48 --device $DEV --dtype float64

echo "########## 4/5  epoch0_faithfulness (4-lag sweep dirs) ##########"
$RG --diagnostic epoch0_faithfulness --note "physics-init, per-order error rises with dT" -- \
  python diagnose_ensemble_epoch0.py \
    $D/FRC-Re25k/sweep_dT_5em3 $D/FRC-Re25k/sweep_dT_1em2 $D/FRC-Re25k/sweep_dT_1p5em2 \
    $D/FRC-combo/sweep_dT_5em3 $D/FRC-combo/sweep_dT_1em2 $D/FRC-combo/sweep_dT_1p5em2 \
    $D/FRC-kf4/sweep_dT_5em3 $D/FRC-kf4/sweep_dT_1em2 $D/FRC-kf4/sweep_dT_1p5em2 \
    --device $DEV

echo "########## 5/5  error_propagation (per member; needs closure_operators.py) ##########"
for M in Re25k combo kf4; do
  $RG --diagnostic error_propagation --tag "$M" --note "per-op eps -> delta; default eps" -- \
    python closure_error_propagation.py $D/FRC-$M/sweep_dT_5em3 \
      --scheme ab2cn2 --orders 3 4 --n-samples 32 --device $DEV
done

echo ""
echo "All done. Browse Results/<diagnostic>/forced_turbulence/latest/report.md"