#!/bin/bash
# submit_anchor_sweep.sh - accuracy-anchored multi-objective rollout FT sweep
# (four-way verdict 2026-07-14, case (b): pure cond_v2 blows up 12/18 rungs at
# dT>=1e-2, but is +84% more accurate than p1-NN where it survives. Fix:
# rollout-stability objective + a-priori derivative loss as hard anchor,
# warm-started from cond_v2 -- NOT from the p1 lineage.)
#
# Three I18 units on the lam01 recipe (kf4+256, trunc:4, free analytic 1e-2,
# vn 0.1 = the P1 sweep winner), differing ONLY in --anchor-lambda:
#   ancZ  lambda 0     control: pure-stability FT from cond_v2 (isolates the
#                      anchor's effect from the warm-start change)
#   ancA  lambda 3e-2  anchor term ~ rollout val scale at convergence
#   ancB  lambda 3e-1  anchor-dominant (hard anchor)
# Anchor pool = cond_v2's own 42 sweep roots (read from its config.json).
# Verdict metric: anc_med_Nddot must hold ~cond_v2's 0.057-0.075 (the rollout
# floor) while fb_s2/fb_s3 -> 0 (free-roll blow-up fraction at 1e-2/1.5e-2).
# Usage: submit_anchor_sweep.sh [--go]   (dry-run default; ~6-9 GPU-h/arm)

set -euo pipefail

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
W="$QG_ROOT/qg-wiener-conditioning"
LOGS="$W/logs"
CARD="$W/diagnostics/baseline_cards/T2_rollout.json"

D=data/ensemble_N5_7lag
WARM="$D/training_runs/deriv7_cond_local_v2/best.pt"
ROOTS="data/ensemble_N5/FRC-kf4/forced_turbulence_dT_5em3 $D/FRC-256/forced_turbulence_dT_5em3"

GO=0
[ "${1:-}" = "--go" ] && GO=1
cd "$W"
[ -e "training/$WARM" ] || { echo "MISSING warm ckpt training/$WARM" >&2; exit 1; }
[ -e "training/${WARM%best.pt}config.json" ] || \
    { echo "MISSING cond_v2 config.json (anchor roots source)" >&2; exit 1; }
for r in $ROOTS; do
    [ -e "training/$r/split.npz" ] || { echo "MISSING split: $r" >&2; exit 1; }
done
for tag in ancZ ancA ancB; do
    [ -e "training/$D/training_runs/rollout_ft_$tag/best.pt" ] && \
        { echo "EXISTS: rollout_ft_$tag" >&2; exit 1; }
done
echo "[preflight] warm=cond_v2(ep63) roots=kf4+FRC-256 arms=Z/A/B OK"
if [ "$GO" -ne 1 ]; then echo "DRY RUN (3 arms x ~6-9 GPU-h)"; exit 0; fi
mkdir -p "$LOGS"

fire_unit () {  # $1=tag  $2=anchor_lambda
    local TAG="$1" LAM="$2" RN="rollout_ft_$1"
    local ANC=()
    if [ "$LAM" != "0" ]; then
        ANC=(--anchor-lambda "$LAM" --anchor-batch 4 --anchor-rel-floor 0.1)
    fi
    local TRAIN LIVE FINAL
    TRAIN=$(qsub -terse -q ibgpu.q -l gpu=1 -N "${TAG}_TRN" -j y \
            -m ea -M "${QG_NOTIFY_EMAIL:-sanaamz@mit.edu}" \
            -o "$LOGS/${TAG}_TRN.\$JOB_ID.log" \
            scripts/sge/train_deriv_rollout_job.sh \
            --deep-roots $ROOTS --init-ckpt "$WARM" \
            --strides 1,2,3 --grad-mode trunc:4 --free-horizon 16 \
            --free-mode analytic --free-weight 1.0e-2 --free-cap 10.0 \
            --vn-lambda 0.1 --lr 5.0e-5 --compute-dtype float64 \
            --model auto --out-root "$D" \
            --unroll-schedule 12:4,16:6,21:10 --epochs 20 \
            ${ANC[@]+"${ANC[@]}"} --run-name "$RN")
    LIVE=$(qsub -terse -q all.q -N "${TAG}_L" -j y \
           -o "$LOGS/${TAG}_L.\$JOB_ID.log" \
           scripts/sge/monitor_training_job.sh \
           "training/$D/training_runs/$RN" wiener "$TRAIN" \
           "$CARD" "$LOGS/${TAG}_TRN.$TRAIN.log")
    FINAL=$(qsub -terse -q all.q -N "${TAG}_F" -hold_jid "$TRAIN" \
            -v QG_MONITOR_FINALIZE=1 -j y -o "$LOGS/${TAG}_F.\$JOB_ID.log" \
            scripts/sge/monitor_training_job.sh \
            "training/$D/training_runs/$RN" wiener "$TRAIN" \
            "$CARD" "$LOGS/${TAG}_TRN.$TRAIN.log")
    echo "I18 unit $RN (anchor_lambda=$LAM): trainer $TRAIN live $LIVE final $FINAL"
}

fire_unit ancZ 0
fire_unit ancA 3.0e-2
fire_unit ancB 3.0e-1
