#!/bin/bash
# submit_w31_p1.sh - ANCHORED P1 rollout + von Neumann FT of the WIDTH-31
# conditioned model. History: prepared 2026-07-14 ~16:1x as the bare p1_prod
# recipe (NO anchor); REVISED ~18:1x per Sanaa chat GO ("1, 6 and 5 in
# parallel, 4 as acceptance ... on the full ensemble"): the FT now carries the
# a-priori accuracy ANCHOR (shelved impl 1180ad8, flags live in
# train_deriv_rollout.py) in TWO arms that bracket the stability<->accuracy
# trade the p1lam01 postmortem exposed:
#   w31p1a  --anchor-lambda 3e-2   anchor ~ rollout val scale (balanced)
#   w31p1b  --anchor-lambda 3e-1   anchor-dominant (hard anchor)
# Anchor pool = the warm ckpt's OWN config.json roots = all 41 filtered sweep
# roots = the FULL ENSEMBLE (Sanaa's "on the full ensemble"); rollout pool
# stays the widened 7-root p1_prod pool (combo + b25 HELD OUT for OOD),
# vn_lambda 0.1, free analytic 1e-2, trunc:4, 20 epochs. grad_kernel 31 flows
# from the ckpt config via --model auto.
# Fire condition (Sanaa ~16:00 + ~15:1x clarification): at the w31 TRUE VAL
# PLATEAU (>=6 epochs no new best), with below-cond_v2 as go/no-go gate only.
# Acceptance (option 4, applies to BOTH arms): per-member a-priori Nddot
# before/after via eval_deriv_by_root -- degradation beyond tolerance = arm
# rejected regardless of stability.
# Usage: submit_w31_p1.sh [--go]   (dry-run default; 2 arms x ~6-9 GPU-h)

set -euo pipefail

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
W="$QG_ROOT/qg-wiener-conditioning"
LOGS="$W/logs"
CARD="$W/diagnostics/baseline_cards/T2_rollout.json"

D=data/ensemble_N5_7lag
WARM="$D/training_runs/deriv7_cond_local_w31/best.pt"
ROOTS="data/ensemble_N5/FRC-kf4/forced_turbulence_dT_5em3 $D/FRC-256/forced_turbulence_dT_5em3 $D/FRC-b0/forced_turbulence_dT_5em3 $D/FRC-b05/forced_turbulence_dT_5em3 $D/FRC-b075/forced_turbulence_dT_5em3 $D/FRC-b1/forced_turbulence_dT_5em3 $D/FRC-b2/forced_turbulence_dT_5em3"

GO=0
[ "${1:-}" = "--go" ] && GO=1
cd "$W"
[ -e "training/$WARM" ] || { echo "MISSING warm ckpt training/$WARM" >&2; exit 1; }
[ -e "training/${WARM%best.pt}config.json" ] || \
    { echo "MISSING w31 config.json (anchor-roots source)" >&2; exit 1; }
for r in $ROOTS; do [ -e "training/$r/split.npz" ] || { echo "MISSING split: $r" >&2; exit 1; }; done
for tag in a b; do
    [ -e "training/$D/training_runs/rollout_ft_w31_p1$tag/best.pt" ] && \
        { echo "EXISTS: rollout_ft_w31_p1$tag" >&2; exit 1; }
done
echo "[preflight] warm=w31 best.pt + 7 rollout roots OK; combo+b25 HELD OUT; vn 0.1;"
echo "[preflight] ANCHOR arms lambda {3e-2, 3e-1}, pool = w31 config.json (41 roots)"
if [ "$GO" -ne 1 ]; then echo "DRY RUN (2 arms x ~6-9 GPU-h)"; exit 0; fi
mkdir -p "$LOGS"

fire_unit () {  # $1=tag(a|b)  $2=anchor_lambda
    local TAG="$1" LAM="$2"
    local RN="rollout_ft_w31_p1$TAG"
    local TRAIN LIVE FINAL
    TRAIN=$(qsub -terse -q ibgpu.q -l gpu=1 -N "w31p1${TAG}_TRN" -j y \
            -m ea -M "${QG_NOTIFY_EMAIL:-sanaamz@mit.edu}" \
            -o "$LOGS/w31p1${TAG}_TRN.\$JOB_ID.log" \
            scripts/sge/train_deriv_rollout_job.sh \
            --deep-roots $ROOTS --init-ckpt "$WARM" \
            --strides 1,2,3 --grad-mode trunc:4 --free-horizon 16 \
            --free-mode analytic --free-weight 1.0e-2 --free-cap 10.0 \
            --vn-lambda 0.1 --lr 5.0e-5 --compute-dtype float64 \
            --anchor-lambda "$LAM" --anchor-batch 4 --anchor-rel-floor 0.1 \
            --model auto --out-root "$D" \
            --unroll-schedule 16:6,21:14 --epochs 20 --run-name "$RN")
    LIVE=$(qsub -terse -q all.q -N "w31p1${TAG}_L" -j y \
           -o "$LOGS/w31p1${TAG}_L.\$JOB_ID.log" \
           scripts/sge/monitor_training_job.sh \
           "training/$D/training_runs/$RN" wiener "$TRAIN" \
           "$CARD" "$LOGS/w31p1${TAG}_TRN.$TRAIN.log")
    FINAL=$(qsub -terse -q all.q -N "w31p1${TAG}_F" -hold_jid "$TRAIN" \
            -v QG_MONITOR_FINALIZE=1 -j y \
            -o "$LOGS/w31p1${TAG}_F.\$JOB_ID.log" \
            scripts/sge/monitor_training_job.sh \
            "training/$D/training_runs/$RN" wiener "$TRAIN" \
            "$CARD" "$LOGS/w31p1${TAG}_TRN.$TRAIN.log")
    echo "I18 unit $RN (anchor_lambda=$LAM): trainer $TRAIN live $LIVE final $FINAL"
}

fire_unit a 3.0e-2
fire_unit b 3.0e-1
