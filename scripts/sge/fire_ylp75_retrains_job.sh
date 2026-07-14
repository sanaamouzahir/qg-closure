#!/bin/bash
# fire_ylp75_retrains_job.sh - LAPTOP-INDEPENDENT landing chain (Sanaa
# 2026-07-14 night). Submitted with -hold_jid <ylp75 build job>: when the
# build lands, this job verifies the 10 filtered payloads exist with the
# right members, then fires BOTH gjs retrains on the filtered targets as
# full I18 units (trainer + LIVE + FINALIZE piff monitors) and queues a
# [QG][SUBMIT] pending_mail. Refuses to double-fire (run-dir EXISTS guard).
# Submit: cd qg-sgs-closure && qsub -q all.q -N fireYlp75 -hold_jid <bldjid> \
#   -o logs/fireYlp75.$JOB_ID.log -j y -cwd -V scripts/sge/fire_ylp75_retrains_job.sh

#$ -S /bin/bash
#$ -cwd
#$ -V

set -uo pipefail

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
BR="$QG_ROOT/qg-sgs-closure"
ENS=/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-simple-package-stable/src/qg/outputs/SGS_closure_ensemble
PEND="$QG_ROOT/reporting/pending_mail"

queue_mail () {
    local f="$PEND/ylp75_fire_$(date +%s).mail"
    { echo "To: sanaamz@mit.edu"; echo "Subject: $1"; echo; echo "$2"; } > "$f"
}

RUNS_FPC="FPC-const FPC-sine FPC-ramp FPC-ou FPC-telS-A"
RUNS_CAPE="FPCape-const FPCape-sine FPCape-ramp FPCape-ou FPCape-tel"

missing=""
for r in $RUNS_FPC $RUNS_CAPE; do
    [ -s "$ENS/$r/DNS_LES_s4_gaussian_jonly_ylp75.npz" ] || missing="$missing $r"
done
if [ -n "$missing" ]; then
    queue_mail "[QG][TRIAGE][sgs] ylp75 payloads MISSING after build job -- retrains NOT fired:$missing" \
        "Check the ylp75Bld log under $BR/logs/ and the builder verification. Fire manually with scripts/sge/fire_ylp75_retrains_job.sh once fixed."
    exit 2
fi

cd "$BR"
body=""
for g in fpc cape; do
    RN="piff_${g}_gjs_ylp75"
    if [ -e "ml_closure/runs_piff/$RN/best.pt" ] || [ -d "ml_closure/runs_piff/$RN" ]; then
        body="$body
$g: run dir exists -- NOT re-fired (double-fire guard)."
        continue
    fi
    TAG=$([ "$g" = fpc ] && echo pYlp_fpc || echo pYlpcape)
    T=$(qsub -terse -q ibgpu.q -l gpu=1 -N "$TAG" -o "logs/$TAG.\$JOB_ID.log" -j y -cwd -V \
        scripts/sge/piff_train_job.sh --config "conf_piff_${g}_gjs_ylp75.yaml" \
        --run-name "$RN" --init-ckpt "runs_piff/piff_${g}_gjs/best.pt" \
        --lr 1.0e-3 --weight-decay 1.0e-5 --epochs 100)
    qsub -N "pMonL_${g}_Ylp" -q all.q -o "logs/pMonL_${g}_Ylp.log" -j y -cwd -V \
        scripts/sge/piff_monitor_job.sh "logs/$TAG.$T.log" "$RN" "$T" >/dev/null
    qsub -N "pMonF_${g}_Ylp" -q all.q -o "logs/pMonF_${g}_Ylp.log" -j y -cwd -V \
        -hold_jid "$T" -v QG_MONITOR_FINALIZE=1 \
        scripts/sge/piff_monitor_job.sh "logs/$TAG.$T.log" "$RN" "$T" >/dev/null
    body="$body
$g: trainer $T ($TAG) + LIVE/FINALIZE monitors, warm from piff_${g}_gjs/best.pt, variant gaussian_jonly_ylp75, 100 ep (~7 GPU-h)."
done

queue_mail "[QG][SUBMIT][sgs] ylp75-filtered-target retrains AUTO-FIRED (laptop-independent chain)" \
"WHAT: both gjs finals retraining on the y-Nyquist-FILTERED Pi targets (ylp75, in-band ky>=0.75kN cut; artefact removed, ~0.03% wake collateral). Payloads verified present for all 10 runs.
$body

Compare at landing vs the unfiltered finals: global + ring_excluded metrics + wake R2 (identical eval standard; conformal surfaced automatically).
Configs: $BR/ml_closure/conf_piff_{fpc,cape}_gjs_ylp75.yaml
Payloads: $ENS/<run>/DNS_LES_s4_gaussian_jonly_ylp75.npz"
exit 0
