#!/bin/bash
# submit_w31_retrain.sh - "make cond_local_v2 suuuper accurate" (Sanaa chat
# ruling 2026-07-14 ~15:15): retrain the conditioned model with grad_kernel 31
# (vs 15), warm-started EXACTLY from deriv7_cond_local_v2 best.pt (bit-exact
# widening: center-embedded stencils + cond head, outer taps start at 0 =
# pure added capacity), N3dot DITCHED from the loss (--order-weights 1,1,0),
# same 42-root pool, same recipe otherwise.
#
# This run doubles as the width-31 ACCURACY instrument: cond_v2's plateau is
# pooled val_med_Nddot ~0.087 (kf4 a-priori 0.057) -- if W31 drops clearly
# below by ep ~30 the width wins; per-dT verdict via eval_deriv_by_root on
# best.pt (Sanaa gate: MUCH more accurate FOR ALL THREE DTS + cost OK from
# bench_grad31_cost). On PASS -> P1 rollout FT (vn_lambda 0.1, widened pool,
# warm from THIS run's best.pt). Cost note: cond_v2 was ~950 s/epoch; the
# 31x31 convs inflate that (bench measures how much) -- decision reads off
# the early epochs, convergence continues behind it.
#
# Also fires the one-shot GPU cost benchmark (bench_grad31_cost.py) on all.q?
# NO -- it needs the GPU: ibgpu.q gpu=1, ~10 min.
# Usage: submit_w31_retrain.sh [--go]   (dry-run default)

set -euo pipefail

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
W="$QG_ROOT/qg-wiener-conditioning"
LOGS="$W/logs"
CARD="$W/diagnostics/baseline_cards/T1_deriv7.json"

D=data/ensemble_N5_7lag
WARM="$D/training_runs/deriv7_cond_local_v2/best.pt"
RN=deriv7_cond_local_w31
# pool = cond_v2's roots VERBATIM from its config.json ("rest doesn't change"):
# already excludes Re25k@1.5e-2 (past its convergence radius, approved drop)
# and contains only filtered roots (rule 15). A fresh glob could sneak in
# unfiltered or unlearnable roots -- do not glob.
ROOTS=$(python -c "
import json
cfg = json.load(open('$W/training/$D/training_runs/deriv7_cond_local_v2/config.json'))
print(' '.join(cfg['roots']))")

GO=0
[ "${1:-}" = "--go" ] && GO=1
cd "$W"
[ -e "training/$WARM" ] || { echo "MISSING warm ckpt training/$WARM" >&2; exit 1; }
[ -e "training/$D/training_runs/$RN/best.pt" ] && { echo "EXISTS: $RN" >&2; exit 1; }
NR=$(echo "$ROOTS" | wc -w)
# 42 listed in cond_v2's config (41 valid at load; trainer re-validates
# manifest+packed and refuses anisotropic members)
[ "$NR" -ge 40 ] || { echo "root list suspicious: only $NR roots" >&2; exit 1; }
case " $ROOTS " in *Re25k/sweep_dT_1p5em2*) echo "Re25k@1.5e-2 must stay excluded" >&2; exit 1;; esac
echo "[preflight] warm=cond_v2 widened 15->31 (bit-exact, verified)  roots=$NR  order_weights=1,1,0"
if [ "$GO" -ne 1 ]; then echo "DRY RUN (bench ~10 min + trainer ~1-2 d, verdict readable at ep~30 ~overnight)"; exit 0; fi
mkdir -p "$LOGS"

BENCH=$(qsub -terse -q ibgpu.q -l gpu=1 -N w31bench -j y \
        -o "$LOGS/w31bench.\$JOB_ID.log" \
        scripts/sge/bench_grad31_job.sh)

TRAIN=$(qsub -terse -q ibgpu.q -l gpu=1 -N w31_TRN -j y \
        -m ea -M "${QG_NOTIFY_EMAIL:-sanaamz@mit.edu}" \
        -o "$LOGS/w31_TRN.\$JOB_ID.log" \
        scripts/sge/train_deriv_condlocal_job.sh \
        --model cond_local --sweep-roots $ROOTS \
        --n-snapshots 7 --out-orders 3 --grad-kernel 31 \
        --init-ckpt "$WARM" --order-weights 1,1,0 \
        --epochs 150 --lr 5.0e-5 --batch-size 4 --compute-dtype float64 \
        --rel-floor 0.1 --run-name "$RN")
LIVE=$(qsub -terse -q all.q -N w31_L -j y -o "$LOGS/w31_L.\$JOB_ID.log" \
       scripts/sge/monitor_training_job.sh \
       "training/$D/training_runs/$RN" wiener "$TRAIN" \
       "$CARD" "$LOGS/w31_TRN.$TRAIN.log")
FINAL=$(qsub -terse -q all.q -N w31_F -hold_jid "$TRAIN" \
        -v QG_MONITOR_FINALIZE=1 -j y -o "$LOGS/w31_F.\$JOB_ID.log" \
        scripts/sge/monitor_training_job.sh \
        "training/$D/training_runs/$RN" wiener "$TRAIN" \
        "$CARD" "$LOGS/w31_TRN.$TRAIN.log")
echo "bench $BENCH | I18 unit $RN: trainer $TRAIN live $LIVE final $FINAL"
