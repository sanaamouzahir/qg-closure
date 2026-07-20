#!/bin/bash
# submit_w31_p4_chain.sh -- Sanaa order 2026-07-20: "when the training
# finishes immediately start it again for 20 more epochs ... regardless of
# whether the long rollouts are satisfactory" + the standing table
# convention. Everything HELD on the p3 trainer so it fires the moment
# ep20 lands, no human in the loop:
#   1. p4 trainer (GPU): 20 MORE epochs, warm from p3 last.pt (the true
#      continuation state; best.pt stays untouched for p3's own eval),
#      schedule 21:20 (M=21 throughout - p3 ended there), all certv2
#      flags identical to p3.
#   2. p4 LIVE + FINALIZE monitors (never optional - NaN policy).
#   3. p3 final eval + TIERED gate (p3_final_eval_job.sh; mails the table).
#   4. p3 64/128-step longroll on the 10 ref draws (CPU; informational
#      regardless of outcome, per the same order), + its table mail.
# Usage: submit_w31_p4_chain.sh <p3_trainer_jid> [--go]   (dry-run default)

set -euo pipefail

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
W="$QG_ROOT/qg-wiener-conditioning"
LOGS="$W/logs"
CARD="$W/diagnostics/baseline_cards/T2_rollout.json"
D=data/ensemble_N5_7lag
P3=rollout_ft_w31p3_certv2
RN=rollout_ft_w31p4_cont
ROOTS="data/ensemble_N5/FRC-kf4/forced_turbulence_dT_5em3 $D/FRC-256/forced_turbulence_dT_5em3 $D/FRC-b0/forced_turbulence_dT_5em3 $D/FRC-b05/forced_turbulence_dT_5em3 $D/FRC-b075/forced_turbulence_dT_5em3 $D/FRC-b1/forced_turbulence_dT_5em3 $D/FRC-b2/forced_turbulence_dT_5em3"
DRAWS='"FRC-kf4 532" "FRC-kf4 837" "FRC-kf4 912" "FRC-kf4 1356" "FRC-combo 527" "FRC-combo 884" "FRC-combo 1355" "FRC-256 549" "FRC-256 933" "FRC-256 1357"'

HOLD="${1:?p3 trainer job id}"
GO=0
[ "${2:-}" = "--go" ] && GO=1
cd "$W"
[ -d "training/$D/training_runs/$P3" ] || { echo "MISSING p3 run dir" >&2; exit 1; }
[ -d "training/$D/training_runs/$RN" ] && { echo "EXISTS: $RN" >&2; exit 1; }
echo "[preflight] p3 run dir present; p4 dir free; chain holds on job $HOLD"
echo "[preflight] NOTE: p3 last.pt is checked by the TRAINER at fire time (does not exist yet)"
if [ "$GO" -ne 1 ]; then echo "DRY RUN (p4 20-ep GPU continuation + p3 eval/gate + p3 longroll, all -hold_jid $HOLD)"; exit 0; fi
mkdir -p "$LOGS"

TRAIN=$(qsub -terse -q ibgpu.q -l gpu=1 -N "w31p4_TRN" -hold_jid "$HOLD" -j y \
        -m ea -M "${QG_NOTIFY_EMAIL:-sanaamz@mit.edu}" \
        -o "$LOGS/w31p4_TRN.\$JOB_ID.log" \
        scripts/sge/train_deriv_rollout_job.sh \
        --deep-roots $ROOTS --init-ckpt "$D/training_runs/$P3/last.pt" \
        --anchor-baseline-csv "$D/training_runs/deriv7_cond_local_w31/eval_by_root_val.csv" \
        --strides 1,2,3 --grad-mode trunc:4 --free-horizon 48 \
        --free-mode hinge --free-weight 1.0e-2 --free-cap 10.0 \
        --vn-lambda 0.5 --vn-single-slot \
        --vn-annulus-shells 0.52,0.58,0.64 --vn-developed-steps 24 \
        --lr 5.0e-5 --compute-dtype float64 \
        --anchor-lambda 3.0e-2 --anchor-batch 4 \
        --anchor-mode trust --anchor-trust-tol 0.10 \
        --model auto --out-root "$D" \
        --unroll-schedule 21:20 --epochs 20 --run-name "$RN")
LIVE=$(qsub -terse -q all.q -N "w31p4_L" -hold_jid "$HOLD" -j y \
       -o "$LOGS/w31p4_L.\$JOB_ID.log" \
       scripts/sge/monitor_training_job.sh \
       "training/$D/training_runs/$RN" wiener "$TRAIN" \
       "$CARD" "$LOGS/w31p4_TRN.$TRAIN.log")
FINAL=$(qsub -terse -q all.q -N "w31p4_F" -hold_jid "$TRAIN" \
        -v QG_MONITOR_FINALIZE=1 -j y \
        -o "$LOGS/w31p4_F.\$JOB_ID.log" \
        scripts/sge/monitor_training_job.sh \
        "training/$D/training_runs/$RN" wiener "$TRAIN" \
        "$CARD" "$LOGS/w31p4_TRN.$TRAIN.log")
EVAL=$(qsub -terse -N "w31p3_gate" -hold_jid "$HOLD" \
       -m ea -M "${QG_NOTIFY_EMAIL:-sanaamz@mit.edu}" \
       scripts/sge/p3_final_eval_job.sh)
ROLL=$(eval qsub -terse -q all.q -N "longroll_p3" -hold_jid "$HOLD" \
       -m ea -M "${QG_NOTIFY_EMAIL:-sanaamz@mit.edu}" \
       -v LONGROLL_OUT="$W/diagnostics/Results/apost_longroll_p3_20260720" \
       scripts/sge/postft_longroll_job.sh \
       "$W/training/$D/training_runs/$P3/best.pt" NONE $DRAWS)
echo "I18 chain: p4 trainer $TRAIN (live $LIVE final $FINAL)  p3 gate $EVAL  p3 longroll $ROLL  (all fire when $HOLD exits)"
