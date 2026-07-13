#!/bin/bash
# submit_piff_ens_gaussian.sh - GAUSSIAN-TARGET redo of the ORDER-3 ensemble
# trainings (Sanaa filter ruling 2026-07-13: sharp filter abandoned; jobs
# 1832221/1832231 qdel-ed). Template T4; trainers HOLD on gwait gate jobs that
# release the moment every member has DNS_LES_s4_gaussian.npz.
# Two three-job I18 units (trainer + LIVE monitor + FINALIZE monitor) plus the
# chained post-processing per trainer:
#
#   piff_fpc_ens_gauss   train_piff.py --config conf_piff_fpc_ens_gaussian.yaml  (5 FPC members,
#                  conditioned: zeta_dot FiLM+ARD, |grad omega_bar| ARD, 150 ep;
#                  re-gated T6 acceptance read at ep 100)
#     -> pGvE_fpc  eval_piff.py on best.pt (pooled S4 package)
#     -> pGXf_<m>   per-member eval x5 (P3 spread, vs the prod_ext150 xeval rows)
#     -> pGCal_fpc calibrate_piff.py (scalar recalibration sidecar)
#   piff_cape_gauss same, conf_piff_cape_cond_gaussian.yaml (cape 5-member conditioned
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
for f in "$ML/conf_piff_fpc_ens_gaussian.yaml" "$ML/conf_piff_cape_cond_gaussian.yaml" "$CARD" \
         "$ML/PREDICTIONS_ensemble_2026-07-13.md"; do
    [ -e "$f" ] || { echo "MISSING: $f" >&2; exit 1; }
done
for m in FPC-const FPC-sine FPC-ramp FPC-ou FPC-telS-A; do
    RUN="$QG_ROOT/qg-simple-package-stable/src/qg/outputs/SGS_closure_ensemble/$m"
    [ -e "$RUN/DATASET_MANIFEST.md" ] && [ -e "$RUN/DNS_LES_s4.npz" ] || \
        { echo "MISSING Step-0: $RUN" >&2; exit 1; }
done
for d in piff_fpc_ens_gauss piff_cape_gauss; do
    [ -e "$ML/runs_piff/$d/best.pt" ] && { echo "EXISTS: runs_piff/$d — refusing to clobber" >&2; exit 1; }
done
echo "[preflight] confs + card + predictions + Step-0 x5 present; run dirs free"

if [ "$GO" -ne 1 ]; then
    echo "DRY RUN - would submit 2 trainer units (3 jobs each) + 7 chained evals + 2 recals."
    echo "Cost: ~8.3 h/trainer GPU (observed ~200 s/ep x 150) x2 + ~30 min evals = ~17.5 GPU-h."
    exit 0
fi

cd "$BRANCH"
mkdir -p "$LOGS"

submit_unit () {  # $1 run-name  $2 conf  $3 short-tag  $4 hold-jid (gwait gate)
    local RN=$1 CONF=$2 TAG=$3 HOLD=$4
    local TRAIN LIVE FINAL
    # piff_train_job.sh cds into ml_closure; --config is relative to it.
    # -hold_jid on the gwait gate: trainer starts the moment the s4 gaussian
    # files exist (gate exits 1 on rebuild timeout -> trainer dies visibly).
    TRAIN=$(qsub -terse -q ibgpu.q -l gpu=1 -N "pGns_$TAG" -j y \
            -hold_jid "$HOLD" \
            -m ea -M "${QG_NOTIFY_EMAIL:-sanaamz@mit.edu}" \
            -o "$LOGS/pGns_$TAG.\$JOB_ID.log" \
            scripts/sge/piff_train_job.sh --config "$CONF" --run-name "$RN")
    echo "trainer $RN: $TRAIN"
    LIVE=$(qsub -terse -q all.q -N "pGnsL_$TAG" -j y \
           -o "$LOGS/pGnsL_$TAG.\$JOB_ID.log" \
           scripts/sge/piff_monitor_job.sh "$LOGS/pGns_$TAG.$TRAIN.log" "$RN" "$TRAIN" "$CARD")
    FINAL=$(qsub -terse -q all.q -N "pGnsF_$TAG" -hold_jid "$TRAIN" -v QG_MONITOR_FINALIZE=1 -j y \
            -o "$LOGS/pGnsF_$TAG.\$JOB_ID.log" \
            scripts/sge/piff_monitor_job.sh "$LOGS/pGns_$TAG.$TRAIN.log" "$RN" "$TRAIN" "$CARD")
    echo "I18 unit $RN: trainer $TRAIN live $LIVE final $FINAL"
    UNIT_TRAIN=$TRAIN
}

# ---- gwait gates (release when every member's s4 gaussian file exists) ---- #
ENS="$QG_ROOT/qg-simple-package-stable/src/qg/outputs/SGS_closure_ensemble"
GW_FPC=$(qsub -terse -q all.q -N gwait_fpc -j y -o "$LOGS/gwait_fpc.\$JOB_ID.log" \
         scripts/sge/gaussian_wait_job.sh \
         "$ENS/FPC-const" "$ENS/FPC-sine" "$ENS/FPC-ramp" "$ENS/FPC-ou" "$ENS/FPC-telS-A")
GW_CAPE=$(qsub -terse -q all.q -N gwait_cape -j y -o "$LOGS/gwait_cape.\$JOB_ID.log" \
          scripts/sge/gaussian_wait_job.sh \
          "$ENS/FPCape-const" "$ENS/FPCape-sine" "$ENS/FPCape-ramp" "$ENS/FPCape-ou" "$ENS/FPCape-tel")
echo "gwait gates: fpc $GW_FPC cape $GW_CAPE"

# ---- FPC ensemble unit ---------------------------------------------------- #
submit_unit piff_fpc_ens_gauss conf_piff_fpc_ens_gaussian.yaml fpc "$GW_FPC"
FPC_TRAIN=$UNIT_TRAIN
qsub -q ibgpu.q -l gpu=1 -N pGvE_fpc -hold_jid "$FPC_TRAIN" -j y \
     -o "$LOGS/\$JOB_NAME.\$JOB_ID.log" \
     scripts/sge/piff_eval_job.sh runs_piff/piff_fpc_ens_gauss/best.pt \
     --config conf_piff_fpc_ens_gaussian.yaml
for m in FPC-const FPC-sine FPC-ramp FPC-ou FPC-telS-A; do
    qsub -q ibgpu.q -l gpu=1 -N "pGXf_${m#FPC-}" -hold_jid "$FPC_TRAIN" -j y \
         -o "$LOGS/\$JOB_NAME.\$JOB_ID.log" \
         scripts/sge/piff_eval_job.sh runs_piff/piff_fpc_ens_gauss/best.pt \
         --config "conf_xeval/conf_xeval_$m.yaml" \
         --outdir "runs_piff/piff_fpc_ens_gauss/xeval/$m"
done
qsub -q ibgpu.q -l gpu=1 -N pGCal_fpc -hold_jid "$FPC_TRAIN" -j y \
     -o "$LOGS/\$JOB_NAME.\$JOB_ID.log" \
     scripts/sge/piff_tool_job.sh calibrate_piff.py \
     --ckpt runs_piff/piff_fpc_ens_gauss/best.pt --config conf_piff_fpc_ens_gaussian.yaml

# ---- cape conditioned unit ------------------------------------------------ #
submit_unit piff_cape_gauss conf_piff_cape_cond_gaussian.yaml cape "$GW_CAPE"
CAPE_TRAIN=$UNIT_TRAIN
qsub -q ibgpu.q -l gpu=1 -N pGvE_cape -hold_jid "$CAPE_TRAIN" -j y \
     -o "$LOGS/\$JOB_NAME.\$JOB_ID.log" \
     scripts/sge/piff_eval_job.sh runs_piff/piff_cape_gauss/best.pt \
     --config conf_piff_cape_cond_gaussian.yaml
qsub -q ibgpu.q -l gpu=1 -N pGCal_cape -hold_jid "$CAPE_TRAIN" -j y \
     -o "$LOGS/\$JOB_NAME.\$JOB_ID.log" \
     scripts/sge/piff_tool_job.sh calibrate_piff.py \
     --ckpt runs_piff/piff_cape_gauss/best.pt --config conf_piff_cape_cond_gaussian.yaml

echo "[submit_piff_ens] both units + post-chains submitted."
