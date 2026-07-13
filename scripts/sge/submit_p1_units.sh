#!/bin/bash
# submit_p1_units.sh - P1 fine-tune v2 (Sanaa 3-phase plan, 2026-07-13):
# on-the-fly analytic free-tail targets (--free-mode analytic) + von Neumann
# certificate penalty (--vn-lambda), lambda swept {0.1, 1.0, 10.0} — the ONLY
# new hyper. WARM from rollout_ft_opt2_cond ep33 (always building on top).
# One GPU smoke first (compile+finiteness gate), then three I18 units.
# All else mirrors the landed opt2/psw3 continuation flags exactly.
# P2 (deep builds) SEALED without Sanaa's explicit consent.
# Dry-run default; --go to submit.

set -euo pipefail

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
W="$QG_ROOT/qg-wiener-conditioning"
LOGS="$W/logs"
CARD="$W/diagnostics/baseline_cards/T2_rollout.json"

ROOTS="data/ensemble_N5/FRC-kf4/forced_turbulence_dT_5em3 data/ensemble_N5_7lag/FRC-256/forced_turbulence_dT_5em3"
WARM="data/ensemble_N5_7lag/training_runs/rollout_ft_opt2_cond/best.pt"
COMMON="--strides 1,2,3 --grad-mode trunc:4 --free-horizon 16
        --free-mode analytic --free-weight 1.0e-2 --free-cap 10.0
        --lr 5.0e-5 --compute-dtype float64 --model auto
        --out-root data/ensemble_N5_7lag"

GO=0
[ "${1:-}" = "--go" ] && GO=1

cd "$W"
[ -e "training/$WARM" ] || { echo "MISSING warm ckpt: training/$WARM" >&2; exit 1; }
[ -e "$CARD" ] || { echo "MISSING card: $CARD" >&2; exit 1; }
for rn in rollout_ft_p1_lam01 rollout_ft_p1_lam1 rollout_ft_p1_lam10; do
    [ -e "training/data/ensemble_N5_7lag/training_runs/$rn/best.pt" ] && \
        { echo "EXISTS: $rn — refusing to clobber" >&2; exit 1; }
done
echo "[preflight] warm ckpt + card present; run names free"

if [ "$GO" -ne 1 ]; then
    echo "DRY RUN — would submit 1 GPU smoke + 3 I18 units (trainer+LIVE+FINAL x3)."
    echo "Cost: smoke ~10 min + 3 x ~1.5 h GPU = ~5 GPU-h."
    exit 0
fi
mkdir -p "$LOGS"

# SMOKE GATING (process change after the 1832637/47/57 sequence): the smoke
# is run STANDALONE and READ by the supervisor before this submitter is
# invoked — SGE hold_jid releases on completion regardless of exit status,
# which twice let broken-code trainers start. PASSED smoke on record:
# p1smk2 1832677 (|G_eff| valid-shell mean 0.9992, stab finite, 0 blown).

for spec in "lam01 0.1" "lam1 1.0" "lam10 10.0"; do
    set -- $spec
    TAG=$1; LAM=$2
    RN="rollout_ft_p1_$TAG"
    TRAIN=$(qsub -terse -q ibgpu.q -l gpu=1 -N "p1_$TAG" -j y \
            -m ea -M "${QG_NOTIFY_EMAIL:-sanaamz@mit.edu}" \
            -o "$LOGS/p1_$TAG.\$JOB_ID.log" \
            scripts/sge/train_deriv_rollout_job.sh \
            --deep-roots $ROOTS --init-ckpt "$WARM" $COMMON \
            --unroll-schedule 12:4,16:6,21:10 --epochs 20 \
            --vn-lambda "$LAM" --run-name "$RN")
    LIVE=$(qsub -terse -q all.q -N "p1L_$TAG" -j y \
           -o "$LOGS/p1L_$TAG.\$JOB_ID.log" \
           scripts/sge/monitor_training_job.sh \
           "training/data/ensemble_N5_7lag/training_runs/$RN" wiener "$TRAIN" \
           "$CARD" "$LOGS/p1_$TAG.$TRAIN.log")
    FINAL=$(qsub -terse -q all.q -N "p1F_$TAG" -hold_jid "$TRAIN" \
            -v QG_MONITOR_FINALIZE=1 -j y \
            -o "$LOGS/p1F_$TAG.\$JOB_ID.log" \
            scripts/sge/monitor_training_job.sh \
            "training/data/ensemble_N5_7lag/training_runs/$RN" wiener "$TRAIN" \
            "$CARD" "$LOGS/p1_$TAG.$TRAIN.log")
    echo "I18 unit $RN (lambda $LAM): trainer $TRAIN live $LIVE final $FINAL"
done
echo "[submit_p1_units] smoke + 3 units submitted."
