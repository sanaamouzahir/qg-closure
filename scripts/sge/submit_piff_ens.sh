#!/bin/bash
# submit_piff_ens.sh - ORDER-3 ensemble trainings (template T4, 2026-07-13).
# Two three-job I18 units (trainer + LIVE monitor + FINALIZE monitor) plus the
# chained post-processing per trainer:
#
#   piff_fpc_ens   train_piff.py --config conf_piff_fpc_ens.yaml  (5 FPC members,
#                  conditioned: zeta_dot FiLM+ARD, |grad omega_bar| ARD, 150 ep;
#                  re-gated T6 acceptance read at ep 100)
#     -> pEvE_fpc  eval_piff.py on best.pt (pooled S4 package)
#     -> pXf_<m>   per-member eval x5 (P3 spread, vs the prod_ext150 xeval rows)
#     -> pCalE_fpc calibrate_piff.py (scalar recalibration sidecar)
#   piff_cape_cond same, conf_piff_cape_cond.yaml (cape 5-member conditioned
#                  rerun; baseline = cape_base_100ep) -> pooled eval + recal
#
# Gates before --go: T1-T5+T8 PASS (jobs 1832194-96), G4 reviewer verdict,
# G5 sge-checker on this file. Predictions recorded BEFORE submission in
# ml_closure/PREDICTIONS_ensemble_2026-07-13.md (P1-P3, G-a..G-c).
# Dry-run by default; pass --go to submit.

set -euo pipefail

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
BRANCH="$QG_ROOT/qg-sgs-closure"
LOGS="$BRANCH/logs"
CARD="$BRANCH/diagnostics/baseline_cards/SGS_piff_ens.json"
ML="$BRANCH/ml_closure"

GO=0
[ "${1:-}" = "--go" ] && GO=1

# ---- preflight ----------------------------------------------------------- #
for f in "$ML/conf_piff_fpc_ens.yaml" "$ML/conf_piff_cape_cond.yaml" "$CARD" \
         "$ML/PREDICTIONS_ensemble_2026-07-13.md"; do
    [ -e "$f" ] || { echo "MISSING: $f" >&2; exit 1; }
done
for m in FPC-const FPC-sine FPC-ramp FPC-ou FPC-telS-A; do
    RUN="$QG_ROOT/qg-simple-package-stable/src/qg/outputs/SGS_closure_ensemble/$m"
    [ -e "$RUN/DATASET_MANIFEST.md" ] && [ -e "$RUN/DNS_LES_s4.npz" ] || \
        { echo "MISSING Step-0: $RUN" >&2; exit 1; }
done
for d in piff_fpc_ens piff_cape_cond; do
    [ -e "$ML/runs_piff/$d/best.pt" ] && { echo "EXISTS: runs_piff/$d — refusing to clobber" >&2; exit 1; }
done
echo "[preflight] confs + card + predictions + Step-0 x5 present; run dirs free"

if [ "$GO" -ne 1 ]; then
    echo "DRY RUN - would submit 2 trainer units (3 jobs each) + 7 chained evals + 2 recals."
    echo "Cost: ~2.8 h/trainer GPU (64 s/ep x 150) x2 + ~25 min evals = ~6.5 GPU-h."
    exit 0
fi

cd "$BRANCH"
mkdir -p "$LOGS"

submit_unit () {  # $1 run-name  $2 conf  $3 short-tag
    local RN=$1 CONF=$2 TAG=$3
    local TRAIN LIVE FINAL
    # piff_train_job.sh cds into ml_closure; --config is relative to it
    TRAIN=$(qsub -terse -q ibgpu.q -l gpu=1 -N "pEns_$TAG" -j y \
            -m ea -M "${QG_NOTIFY_EMAIL:-sanaamz@mit.edu}" \
            -o "$LOGS/pEns_$TAG.\$JOB_ID.log" \
            scripts/sge/piff_train_job.sh --config "$CONF" --run-name "$RN")
    echo "trainer $RN: $TRAIN"
    LIVE=$(qsub -terse -q all.q -N "pMonL_$TAG" -j y \
           -o "$LOGS/pMonL_$TAG.\$JOB_ID.log" \
           scripts/sge/piff_monitor_job.sh "$LOGS/pEns_$TAG.$TRAIN.log" "$RN" "$TRAIN" "$CARD")
    FINAL=$(qsub -terse -q all.q -N "pMonF_$TAG" -hold_jid "$TRAIN" -v QG_MONITOR_FINALIZE=1 -j y \
            -o "$LOGS/pMonF_$TAG.\$JOB_ID.log" \
            scripts/sge/piff_monitor_job.sh "$LOGS/pEns_$TAG.$TRAIN.log" "$RN" "$TRAIN" "$CARD")
    echo "I18 unit $RN: trainer $TRAIN live $LIVE final $FINAL"
    UNIT_TRAIN=$TRAIN
}

# ---- FPC ensemble unit ---------------------------------------------------- #
submit_unit piff_fpc_ens conf_piff_fpc_ens.yaml fpc
FPC_TRAIN=$UNIT_TRAIN
qsub -q ibgpu.q -l gpu=1 -N pEvE_fpc -hold_jid "$FPC_TRAIN" -j y \
     -o "$LOGS/\$JOB_NAME.\$JOB_ID.log" \
     scripts/sge/piff_eval_job.sh runs_piff/piff_fpc_ens/best.pt \
     --config conf_piff_fpc_ens.yaml
for m in FPC-const FPC-sine FPC-ramp FPC-ou FPC-telS-A; do
    qsub -q ibgpu.q -l gpu=1 -N "pXf_${m#FPC-}" -hold_jid "$FPC_TRAIN" -j y \
         -o "$LOGS/\$JOB_NAME.\$JOB_ID.log" \
         scripts/sge/piff_eval_job.sh runs_piff/piff_fpc_ens/best.pt \
         --config "conf_xeval/conf_xeval_$m.yaml" \
         --outdir "runs_piff/piff_fpc_ens/xeval/$m"
done
qsub -q ibgpu.q -l gpu=1 -N pCalE_fpc -hold_jid "$FPC_TRAIN" -j y \
     -o "$LOGS/\$JOB_NAME.\$JOB_ID.log" \
     scripts/sge/piff_tool_job.sh calibrate_piff.py \
     --ckpt runs_piff/piff_fpc_ens/best.pt --config conf_piff_fpc_ens.yaml

# ---- cape conditioned unit ------------------------------------------------ #
submit_unit piff_cape_cond conf_piff_cape_cond.yaml cape
CAPE_TRAIN=$UNIT_TRAIN
qsub -q ibgpu.q -l gpu=1 -N pEvE_cape -hold_jid "$CAPE_TRAIN" -j y \
     -o "$LOGS/\$JOB_NAME.\$JOB_ID.log" \
     scripts/sge/piff_eval_job.sh runs_piff/piff_cape_cond/best.pt \
     --config conf_piff_cape_cond.yaml
qsub -q ibgpu.q -l gpu=1 -N pCalE_cape -hold_jid "$CAPE_TRAIN" -j y \
     -o "$LOGS/\$JOB_NAME.\$JOB_ID.log" \
     scripts/sge/piff_tool_job.sh calibrate_piff.py \
     --ckpt runs_piff/piff_cape_cond/best.pt --config conf_piff_cape_cond.yaml

echo "[submit_piff_ens] both units + post-chains submitted."
