#!/bin/bash
# submit_p1_prod.sh - P1 PRODUCTION widened-pool fine-tune (Sanaa overnight
# authorization 2026-07-13 ~21:15: "if the job finishes tonight with no
# blowing up and small error, have it run for all of the other ensemble
# members, leaving combo and another member of ur choice of OOD testing").
# Pool: kf4 + FRC-256 + b0,b05,b075,b1,b2 (7 roots). HELD OUT (OOD): combo +
# b25 (supervisor's pick: beta=2.5 extreme = strongest extrapolation probe).
# In-distribution testing: per-root val windows (trainer's val split) + the
# rep ladder on unseen ICs of training members. Warm from the LAMBDA WINNER
# of tonight's sweep (env: WINNER_RUN, WINNER_LAM — set at fire time from the
# verdict). Same losses as the sweep (free-mode analytic + vn certificate).
# Usage: WINNER_RUN=rollout_ft_p1_lam1 WINNER_LAM=1.0 submit_p1_prod.sh --go

set -euo pipefail

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
W="$QG_ROOT/qg-wiener-conditioning"
LOGS="$W/logs"
CARD="$W/diagnostics/baseline_cards/T2_rollout.json"

: "${WINNER_RUN:?set WINNER_RUN (e.g. rollout_ft_p1_lam1)}"
: "${WINNER_LAM:?set WINNER_LAM (e.g. 1.0)}"
WARM="data/ensemble_N5_7lag/training_runs/$WINNER_RUN/best.pt"
D=data/ensemble_N5_7lag
ROOTS="data/ensemble_N5/FRC-kf4/forced_turbulence_dT_5em3 $D/FRC-256/forced_turbulence_dT_5em3 $D/FRC-b0/forced_turbulence_dT_5em3 $D/FRC-b05/forced_turbulence_dT_5em3 $D/FRC-b075/forced_turbulence_dT_5em3 $D/FRC-b1/forced_turbulence_dT_5em3 $D/FRC-b2/forced_turbulence_dT_5em3"
RN=rollout_ft_p1_prod

GO=0
[ "${1:-}" = "--go" ] && GO=1
cd "$W"
[ -e "training/$WARM" ] || { echo "MISSING winner ckpt training/$WARM" >&2; exit 1; }
[ -e "training/data/ensemble_N5_7lag/training_runs/$RN/best.pt" ] && \
    { echo "EXISTS: $RN" >&2; exit 1; }
for r in $ROOTS; do [ -e "training/$r/split.npz" ] || { echo "MISSING split: $r" >&2; exit 1; }; done
echo "[preflight] winner $WINNER_RUN (lambda $WINNER_LAM) + 7 roots OK; combo+b25 HELD OUT"
if [ "$GO" -ne 1 ]; then echo "DRY RUN (cost ~4-6 GPU-h overnight)"; exit 0; fi
mkdir -p "$LOGS"

TRAIN=$(qsub -terse -q ibgpu.q -l gpu=1 -N p1prod -j y \
        -m ea -M "${QG_NOTIFY_EMAIL:-sanaamz@mit.edu}" \
        -o "$LOGS/p1prod.\$JOB_ID.log" \
        scripts/sge/train_deriv_rollout_job.sh \
        --deep-roots $ROOTS --init-ckpt "$WARM" \
        --strides 1,2,3 --grad-mode trunc:4 --free-horizon 16 \
        --free-mode analytic --free-weight 1.0e-2 --free-cap 10.0 \
        --vn-lambda "$WINNER_LAM" --lr 5.0e-5 --compute-dtype float64 \
        --model auto --out-root data/ensemble_N5_7lag \
        --unroll-schedule 16:6,21:14 --epochs 20 --run-name "$RN")
LIVE=$(qsub -terse -q all.q -N p1prodL -j y -o "$LOGS/p1prodL.\$JOB_ID.log" \
       scripts/sge/monitor_training_job.sh \
       "training/data/ensemble_N5_7lag/training_runs/$RN" wiener "$TRAIN" \
       "$CARD" "$LOGS/p1prod.$TRAIN.log")
FINAL=$(qsub -terse -q all.q -N p1prodF -hold_jid "$TRAIN" \
        -v QG_MONITOR_FINALIZE=1 -j y -o "$LOGS/p1prodF.\$JOB_ID.log" \
        scripts/sge/monitor_training_job.sh \
        "training/data/ensemble_N5_7lag/training_runs/$RN" wiener "$TRAIN" \
        "$CARD" "$LOGS/p1prod.$TRAIN.log")
echo "I18 unit $RN: trainer $TRAIN live $LIVE final $FINAL"
