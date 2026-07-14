#!/bin/bash
# submit_w31_p1.sh - P1 rollout + von Neumann FT of the WIDTH-31 conditioned
# model (Sanaa chat ruling 2026-07-14 ~16:00: "as soon as you see the val
# plateau, if it's less than cond_v2, fire the rollout+Von Neumann training
# -- don't wait for 150 epochs"). Exactly the p1_prod recipe ("rest doesn't
# change"): widened 7-root pool (combo + b25 HELD OUT for OOD), vn_lambda
# 0.1 (the sweep winner), free analytic 1e-2, trunc:4, 20 epochs -- only the
# warm start differs: deriv7_cond_local_w31/best.pt (grad_kernel 31 flows
# from the ckpt config via --model auto). NO anchor.
# Usage: submit_w31_p1.sh [--go]   (dry-run default; ~6-9 GPU-h)

set -euo pipefail

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
W="$QG_ROOT/qg-wiener-conditioning"
LOGS="$W/logs"
CARD="$W/diagnostics/baseline_cards/T2_rollout.json"

D=data/ensemble_N5_7lag
WARM="$D/training_runs/deriv7_cond_local_w31/best.pt"
ROOTS="data/ensemble_N5/FRC-kf4/forced_turbulence_dT_5em3 $D/FRC-256/forced_turbulence_dT_5em3 $D/FRC-b0/forced_turbulence_dT_5em3 $D/FRC-b05/forced_turbulence_dT_5em3 $D/FRC-b075/forced_turbulence_dT_5em3 $D/FRC-b1/forced_turbulence_dT_5em3 $D/FRC-b2/forced_turbulence_dT_5em3"
RN=rollout_ft_w31_p1

GO=0
[ "${1:-}" = "--go" ] && GO=1
cd "$W"
[ -e "training/$WARM" ] || { echo "MISSING warm ckpt training/$WARM" >&2; exit 1; }
[ -e "training/$D/training_runs/$RN/best.pt" ] && { echo "EXISTS: $RN" >&2; exit 1; }
for r in $ROOTS; do [ -e "training/$r/split.npz" ] || { echo "MISSING split: $r" >&2; exit 1; }; done
echo "[preflight] warm=w31 best.pt + 7 roots OK; combo+b25 HELD OUT; vn 0.1; no anchor"
if [ "$GO" -ne 1 ]; then echo "DRY RUN (~6-9 GPU-h)"; exit 0; fi
mkdir -p "$LOGS"

TRAIN=$(qsub -terse -q ibgpu.q -l gpu=1 -N w31p1_TRN -j y \
        -m ea -M "${QG_NOTIFY_EMAIL:-sanaamz@mit.edu}" \
        -o "$LOGS/w31p1_TRN.\$JOB_ID.log" \
        scripts/sge/train_deriv_rollout_job.sh \
        --deep-roots $ROOTS --init-ckpt "$WARM" \
        --strides 1,2,3 --grad-mode trunc:4 --free-horizon 16 \
        --free-mode analytic --free-weight 1.0e-2 --free-cap 10.0 \
        --vn-lambda 0.1 --lr 5.0e-5 --compute-dtype float64 \
        --model auto --out-root "$D" \
        --unroll-schedule 16:6,21:14 --epochs 20 --run-name "$RN")
LIVE=$(qsub -terse -q all.q -N w31p1_L -j y -o "$LOGS/w31p1_L.\$JOB_ID.log" \
       scripts/sge/monitor_training_job.sh \
       "training/$D/training_runs/$RN" wiener "$TRAIN" \
       "$CARD" "$LOGS/w31p1_TRN.$TRAIN.log")
FINAL=$(qsub -terse -q all.q -N w31p1_F -hold_jid "$TRAIN" \
        -v QG_MONITOR_FINALIZE=1 -j y -o "$LOGS/w31p1_F.\$JOB_ID.log" \
        scripts/sge/monitor_training_job.sh \
        "training/$D/training_runs/$RN" wiener "$TRAIN" \
        "$CARD" "$LOGS/w31p1_TRN.$TRAIN.log")
echo "I18 unit $RN: trainer $TRAIN live $LIVE final $FINAL"
