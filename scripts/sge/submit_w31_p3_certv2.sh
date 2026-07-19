#!/bin/bash
# submit_w31_p3_certv2.sh -- w31 p3: certificate-v2 stability arm
# (Sanaa order 2026-07-19: "implement, push and submit all the changes",
# EXCLUDING the 64-step-horizon fine-tune item -- horizons stay 48/16).
#
# Diagnosis this attacks (G4 double review 2026-07-19, no bugs found):
#   - p2 arms trained with free_mode=analytic = zero stability pressure;
#     the annulus growth hinge (the one anti-growth term) was OFF
#   - the vn certificate was IC-frozen, annulus-blind (validity mask +
#     shell placement) and mis-split the closure across AB2's 1.5/-0.5
#     slots (biased-benign for growing modes)
# Changes vs p2 vn05 (all G4-reviewed, defaults-off flags):
#   --free-mode hinge            annulus log-growth hinge ON (anti-growth)
#   --vn-single-slot             closure enters the companion once at slot n
#   --vn-annulus-shells 0.52,0.58,0.64   modes ~188/210/232 at 512^2 --
#                                inside the aliasing annulus, below the cut
#                                (startup prints their validity fraction)
#   --vn-developed-steps 24      certificate ALSO evaluated on a no-grad
#                                24-step developed state (mid blowup window)
# WARM from the ACCEPTED vn05 ep20 ckpt (Sanaa 2026-07-18; "always building
# on top"). Trust anchor unchanged (3e-2, tol 10%, Ndot+Nddot, w31 baseline).
# GATE NOTE: the finalize auto-gate stale-ep0-marker bug is still open; the
# ep20 acceptance gate for this arm will be run via the established
# p2_final_eval_job path (manual chain), NOT the finalize watcher.
# Usage: submit_w31_p3_certv2.sh [--go]   (dry-run default)

set -euo pipefail

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
W="$QG_ROOT/qg-wiener-conditioning"
LOGS="$W/logs"
CARD="$W/diagnostics/baseline_cards/T2_rollout.json"

D=data/ensemble_N5_7lag
WARM="$D/training_runs/rollout_ft_w31_p2_trust_vn05/best_final_ep20.pt"
BASECSV="$D/training_runs/deriv7_cond_local_w31/eval_by_root_val.csv"
ROOTS="data/ensemble_N5/FRC-kf4/forced_turbulence_dT_5em3 $D/FRC-256/forced_turbulence_dT_5em3 $D/FRC-b0/forced_turbulence_dT_5em3 $D/FRC-b05/forced_turbulence_dT_5em3 $D/FRC-b075/forced_turbulence_dT_5em3 $D/FRC-b1/forced_turbulence_dT_5em3 $D/FRC-b2/forced_turbulence_dT_5em3"
RN="rollout_ft_w31p3_certv2"

GO=0
while [ $# -gt 0 ]; do
    case "$1" in
        --go) GO=1 ;;
        *) echo "ERROR: unknown arg '$1' (only --go accepted)" >&2; exit 2 ;;
    esac
    shift
done
cd "$W"
[ -e "training/$WARM" ] || { echo "MISSING warm ckpt training/$WARM" >&2; exit 1; }
[ -e "training/$BASECSV" ] || { echo "MISSING baseline table training/$BASECSV (trust anchor source)" >&2; exit 1; }
[ -e "training/$D/training_runs/deriv7_cond_local_w31/config.json" ] || \
    { echo "MISSING w31 config.json (anchor-roots source)" >&2; exit 1; }
for r in $ROOTS; do [ -e "training/$r/split.npz" ] || { echo "MISSING split: $r" >&2; exit 1; }; done
[ -d "training/$D/training_runs/$RN" ] && { echo "EXISTS: $RN" >&2; exit 1; }
echo "[preflight] warm=vn05 ep20 + baseline CSV + 7 rollout roots OK; combo+b25 HELD OUT"
echo "[preflight] hinge free-mode; vn 0.5 single-slot + annulus shells + developed-24; trust 3e-2/10%"
if [ "$GO" -ne 1 ]; then echo "DRY RUN (1 unit, est 40-60 GPU-h; hinge tail ~ p2 cost + 24 no-grad steps/window)"; exit 0; fi
mkdir -p "$LOGS"

TRAIN=$(qsub -terse -q ibgpu.q -l gpu=1 -N "w31p3_TRN" -j y \
        -m ea -M "${QG_NOTIFY_EMAIL:-sanaamz@mit.edu}" \
        -o "$LOGS/w31p3_TRN.\$JOB_ID.log" \
        scripts/sge/train_deriv_rollout_job.sh \
        --deep-roots $ROOTS --init-ckpt "$WARM" \
        --strides 1,2,3 --grad-mode trunc:4 --free-horizon 48 \
        --free-mode hinge --free-weight 1.0e-2 --free-cap 10.0 \
        --vn-lambda 0.5 --vn-single-slot \
        --vn-annulus-shells 0.52,0.58,0.64 --vn-developed-steps 24 \
        --lr 5.0e-5 --compute-dtype float64 \
        --anchor-lambda 3.0e-2 --anchor-batch 4 \
        --anchor-mode trust --anchor-trust-tol 0.10 \
        --model auto --out-root "$D" \
        --unroll-schedule 16:6,21:14 --epochs 20 --run-name "$RN")
LIVE=$(qsub -terse -q all.q -N "w31p3_L" -j y \
       -o "$LOGS/w31p3_L.\$JOB_ID.log" \
       scripts/sge/monitor_training_job.sh \
       "training/$D/training_runs/$RN" wiener "$TRAIN" \
       "$CARD" "$LOGS/w31p3_TRN.$TRAIN.log")
FINAL=$(qsub -terse -q all.q -N "w31p3_F" -hold_jid "$TRAIN" \
        -v QG_MONITOR_FINALIZE=1 -j y \
        -o "$LOGS/w31p3_F.\$JOB_ID.log" \
        scripts/sge/monitor_training_job.sh \
        "training/$D/training_runs/$RN" wiener "$TRAIN" \
        "$CARD" "$LOGS/w31p3_TRN.$TRAIN.log")
echo "I18 unit $RN (certv2: hinge + single-slot + annulus + developed-24, warm vn05 ep20): trainer $TRAIN live $LIVE final $FINAL"
