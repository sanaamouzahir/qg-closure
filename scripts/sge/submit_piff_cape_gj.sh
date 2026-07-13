#!/bin/bash
# submit_piff_cape_gj.sh - Cape ensemble training (FPC twin) on JACOBIAN-ONLY gaussian
# targets with signal-biased sampling (Sanaa ruling 2026-07-13 evening:
# "jacobian training only... adapt the sampling to the fact that there is
# very little energy anywhere"). Template T4. ONE unit (FPC only — cape
# J-only waits for FPC signs of life per Sanaa; its data builds regardless).
# Trainer holds on a gwait gate for the 5 FPC _gaussian_jonly files
# (rebuild fleet gJo_* 1832512-21).
# Dry-run default; --go to submit.

set -euo pipefail

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
BRANCH="$QG_ROOT/qg-sgs-closure"
LOGS="$BRANCH/logs"
CARD="$BRANCH/diagnostics/baseline_cards/SGS_piff_ens.json"
ML="$BRANCH/ml_closure"
ENS="$QG_ROOT/qg-simple-package-stable/src/qg/outputs/SGS_closure_ensemble"

GO=0
[ "${1:-}" = "--go" ] && GO=1

for f in "$ML/conf_piff_cape_gj.yaml" "$CARD"; do
    [ -e "$f" ] || { echo "MISSING: $f" >&2; exit 1; }
done
[ -e "$ML/runs_piff/piff_cape_gj/best.pt" ] && { echo "EXISTS: runs_piff/piff_cape_gj" >&2; exit 1; }
echo "[preflight] conf + card present; run dir free"

if [ "$GO" -ne 1 ]; then
    echo "DRY RUN - would submit gwait gate + I18 unit (trainer+2 monitors) + eval/recal chain."
    echo "Cost: ~8.3 h GPU trainer + ~30 min evals."
    exit 0
fi

cd "$BRANCH"
mkdir -p "$LOGS"

GW=$(qsub -terse -q all.q -N gwait_joc -j y -o "$LOGS/gwait_jo.\$JOB_ID.log" \
     -v WAIT_SUFFIX=_gaussian_jonly \
     scripts/sge/gaussian_wait_job.sh \
     "$ENS/FPCape-const" "$ENS/FPCape-sine" "$ENS/FPCape-ramp" "$ENS/FPCape-ou" "$ENS/FPCape-tel")
echo "gwait_joc: $GW"

TRAIN=$(qsub -terse -q ibgpu.q -l gpu=1 -N pJoC_cape -j y -hold_jid "$GW" \
        -m ea -M "${QG_NOTIFY_EMAIL:-sanaamz@mit.edu}" \
        -o "$LOGS/pJoC_cape.\$JOB_ID.log" \
        scripts/sge/piff_train_job.sh --config conf_piff_cape_gj.yaml --run-name piff_cape_gj)
LIVE=$(qsub -terse -q all.q -N pJoCL_cape -j y -o "$LOGS/pJoCL_cape.\$JOB_ID.log" \
       scripts/sge/piff_monitor_job.sh "$LOGS/pJoC_cape.$TRAIN.log" piff_cape_gj "$TRAIN" "$CARD")
FINAL=$(qsub -terse -q all.q -N pJoCF_cape -hold_jid "$TRAIN" -v QG_MONITOR_FINALIZE=1 -j y \
        -o "$LOGS/pJoCF_cape.\$JOB_ID.log" \
        scripts/sge/piff_monitor_job.sh "$LOGS/pJoC_cape.$TRAIN.log" piff_cape_gj "$TRAIN" "$CARD")
echo "I18 unit piff_cape_gj: trainer $TRAIN live $LIVE final $FINAL"

qsub -q ibgpu.q -l gpu=1 -N pJoCEv_cap -hold_jid "$TRAIN" -j y \
     -o "$LOGS/\$JOB_NAME.\$JOB_ID.log" \
     scripts/sge/piff_eval_job.sh runs_piff/piff_cape_gj/best.pt \
     --config conf_piff_cape_gj.yaml
qsub -q ibgpu.q -l gpu=1 -N pJoCCal_ca -hold_jid "$TRAIN" -j y \
     -o "$LOGS/\$JOB_NAME.\$JOB_ID.log" \
     scripts/sge/piff_tool_job.sh calibrate_piff.py \
     --ckpt runs_piff/piff_cape_gj/best.pt --config conf_piff_cape_gj.yaml

echo "[submit_piff_cape_gj] unit submitted."
