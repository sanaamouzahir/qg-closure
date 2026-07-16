#!/bin/bash
# submit_w31_p2_trust.sh - P2 TRUST-REGION-ANCHORED rollout FT of the w31
# conditioned model (Sanaa direct GO 2026-07-16, in-session).
#
# Postmortem it answers: BOTH p1 arms (lambda 3e-2 / 3e-1, floored anchor)
# FAILed the 10% per-member Nddot acceptance gate (medians 2.36x / 1.70x).
# Root cause: anchor rel_floor=0.1 is POOLED -- members already below the
# floor (kf4 0.02, combo 0.039, b25 0.047) produced ZERO anchor gradient
# and degraded freely. Also arm A blew at step 29 of a 128-step CPU
# rollout: stability is horizon-limited at free_horizon 16.
#
# THE ONE NEW ARM (everything else = p1a machinery):
#   --anchor-mode trust : per-sample UNFLOORED rel-L2 / its own (member x dT)
#       baseline (deriv7_cond_local_w31/eval_by_root_val.csv), loss =
#       mean relu(err/baseline - 1.10), Ndot+Nddot ONLY (N3dot excluded,
#       Sanaa's order). lambda stays 3e-2 -- the trust-region SHAPE does
#       the work, not the weight.
#   --free-horizon 48   (was 16): the step-29 problem -- train the free
#       tail past the observed blowup horizon.
# Cost note: free tail 48 roughly triples per-epoch stepping vs p1
#   (~1 h/epoch observed there) -- lead estimate ~2-3 h/epoch, 20 epochs.
# Usage: submit_w31_p2_trust.sh [--go]   (dry-run default)

set -euo pipefail

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
W="$QG_ROOT/qg-wiener-conditioning"
LOGS="$W/logs"
CARD="$W/diagnostics/baseline_cards/T2_rollout.json"

D=data/ensemble_N5_7lag
WARM="$D/training_runs/deriv7_cond_local_w31/best.pt"
BASECSV="$D/training_runs/deriv7_cond_local_w31/eval_by_root_val.csv"
ROOTS="data/ensemble_N5/FRC-kf4/forced_turbulence_dT_5em3 $D/FRC-256/forced_turbulence_dT_5em3 $D/FRC-b0/forced_turbulence_dT_5em3 $D/FRC-b05/forced_turbulence_dT_5em3 $D/FRC-b075/forced_turbulence_dT_5em3 $D/FRC-b1/forced_turbulence_dT_5em3 $D/FRC-b2/forced_turbulence_dT_5em3"
RN="rollout_ft_w31_p2_trust"

GO=0
[ "${1:-}" = "--go" ] && GO=1
cd "$W"
[ -e "training/$WARM" ] || { echo "MISSING warm ckpt training/$WARM" >&2; exit 1; }
[ -e "training/$BASECSV" ] || { echo "MISSING baseline table training/$BASECSV (trust anchor source)" >&2; exit 1; }
[ -e "training/${WARM%best.pt}config.json" ] || \
    { echo "MISSING w31 config.json (anchor-roots source)" >&2; exit 1; }
for r in $ROOTS; do [ -e "training/$r/split.npz" ] || { echo "MISSING split: $r" >&2; exit 1; }; done
[ -e "training/$D/training_runs/$RN/best.pt" ] && { echo "EXISTS: $RN" >&2; exit 1; }
echo "[preflight] warm=w31 best.pt + baseline CSV + 7 rollout roots OK; combo+b25 HELD OUT"
echo "[preflight] TRUST anchor lambda 3e-2 tol 10% Ndot+Nddot only; free-horizon 48; vn 0.1"
if [ "$GO" -ne 1 ]; then echo "DRY RUN (1 unit, est 40-60 GPU-h -- free tail 3x p1)"; exit 0; fi
mkdir -p "$LOGS"

TRAIN=$(qsub -terse -q ibgpu.q -l gpu=1 -N "w31p2t_TRN" -j y \
        -m ea -M "${QG_NOTIFY_EMAIL:-sanaamz@mit.edu}" \
        -o "$LOGS/w31p2t_TRN.\$JOB_ID.log" \
        scripts/sge/train_deriv_rollout_job.sh \
        --deep-roots $ROOTS --init-ckpt "$WARM" \
        --strides 1,2,3 --grad-mode trunc:4 --free-horizon 48 \
        --free-mode analytic --free-weight 1.0e-2 --free-cap 10.0 \
        --vn-lambda 0.1 --lr 5.0e-5 --compute-dtype float64 \
        --anchor-lambda 3.0e-2 --anchor-batch 4 \
        --anchor-mode trust --anchor-trust-tol 0.10 \
        --model auto --out-root "$D" \
        --unroll-schedule 16:6,21:14 --epochs 20 --run-name "$RN")
LIVE=$(qsub -terse -q all.q -N "w31p2t_L" -j y \
       -o "$LOGS/w31p2t_L.\$JOB_ID.log" \
       scripts/sge/monitor_training_job.sh \
       "training/$D/training_runs/$RN" wiener "$TRAIN" \
       "$CARD" "$LOGS/w31p2t_TRN.$TRAIN.log")
FINAL=$(qsub -terse -q all.q -N "w31p2t_F" -hold_jid "$TRAIN" \
        -v QG_MONITOR_FINALIZE=1 -j y \
        -o "$LOGS/w31p2t_F.\$JOB_ID.log" \
        scripts/sge/monitor_training_job.sh \
        "training/$D/training_runs/$RN" wiener "$TRAIN" \
        "$CARD" "$LOGS/w31p2t_TRN.$TRAIN.log")
echo "I18 unit $RN (TRUST anchor 3e-2 tol 10%, free-horizon 48): trainer $TRAIN live $LIVE final $FINAL"
