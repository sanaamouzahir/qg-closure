#!/bin/bash
# run_error_prop_3dt.sh
# Sweep closure_error_propagation.py over all sliced FRC members x {5e-3,1e-2,1.5e-2},
# WITH each member's real static forcing (rebuilt from its manifest). Run from
# $QG_DIR/training.
#
# Two passes, on purpose:
#   PASS 1  --eps 1 1 1  -> PURE AMPLIFICATION GEOMETRY (independent of training error).
#           Each row's /||delta|| is then the raw amplification factor of that term;
#           this is what shows how R4's L-amplification grows with dt, and it needs
#           NO trained model. Read this first.
#   PASS 2  --eps <real> -> ACTUAL closure error, once the S=7 run gives per-order
#           val rel-L2. Edit EPS below and rerun.
#
# Uses ONE sweep dir per member (sweep_dT_5em3) and varies --dt, so the FLOW is held
# fixed and only the dt-prefactor changes -- the clean "same fields, vary dt" test.

set -e
ROOT=data/ensemble_N5_7lag         # <-- match your slicer --out-root
ANCHOR_TAG=sweep_dT_5em3                    # sweep dir whose omega_0 anchors we reuse
DTS="5e-3 1e-2 1.5e-2"
NSAMP=32
EPS="1 1 1"                                 # PASS 1: pure geometry. PASS 2: real val rel-L2.

# auto-discover sliced members that actually have the anchor dir
MEMBERS=$(ls -d ${ROOT}/*/${ANCHOR_TAG} 2>/dev/null | sed -E "s#${ROOT}/([^/]+)/.*#\1#" | sort -u)
if [ -z "$MEMBERS" ]; then
  echo "no sliced members with ${ANCHOR_TAG} under ${ROOT}/ -- check --out-root / tag"; exit 1
fi
echo "members: $MEMBERS"
echo "dts:     $DTS      eps: $EPS"
echo "========================================================================"

for m in $MEMBERS; do
  MDIR=${ROOT}/${m}/${ANCHOR_TAG}
  python build_forcing_npy.py "$MDIR"
  for dt in $DTS; do
    echo ""
    echo ">>> member=$m  dt=$dt  (eps=$EPS) <<<"
    python closure_error_propagation.py "$MDIR" \
      --eps $EPS --scheme ab2cn2 --orders 3 4 \
      --dt "$dt" --forcing "$MDIR/forcing.npy" --n-samples $NSAMP
  done
  echo "------------------------------------------------------------------------"
done
