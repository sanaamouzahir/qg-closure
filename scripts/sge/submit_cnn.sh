#!/bin/bash
# submit_cnn.sh - CNN-only Pi_FF unit (Sanaa order 2026-07-22), FPC ensemble:
#
#   trainer  cnn_train_job.sh --config conf_piff_fpc_cnn.yaml
#            --run-name piff_fpc_cnn_v1            (ibgpu.q gpu=1)
#   + LIVE and FINALIZE monitors AUTO-WIRED AT FIRE TIME (never-again NaN
#     policy 2026-07-19: "the supervisor wires at fire time" convention is
#     DEAD) via piff_monitor_job.sh with the CNN baseline card
#     diagnostics/baseline_cards/SGS_piff_cnn.json
#   -> eval_cnn.py on the best ckpt, -hold_jid trainer (GPU via
#      piff_tool_job.sh: full-frame forwards; spools the per-member table
#      mail itself — diagnostics-table convention).
#
# Dry-run by default; pass --go to submit. Day-mode via I21c ssh ONLY.

set -euo pipefail

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
BRANCH="$QG_ROOT/qg-sgs-closure"
ML="$BRANCH/ml_closure"
LOGS="$BRANCH/logs"
QG_NOTIFY_EMAIL="${QG_NOTIFY_EMAIL:-sanaamz@mit.edu}"

RN="piff_fpc_cnn_v1"
CONF="conf_piff_fpc_cnn.yaml"
CARD="$BRANCH/diagnostics/baseline_cards/SGS_piff_cnn.json"

GO=0
while [ $# -gt 0 ]; do
    case "$1" in
        --go) GO=1 ;;
        *) echo "ERROR: unknown arg '$1' (only --go accepted)" >&2; exit 2 ;;
    esac
    shift
done

# ---- preflight ----------------------------------------------------------- #
[ -f "$QG_ROOT/qg-env-piff/bin/activate" ] || { echo "ERROR: qg-env-piff venv missing" >&2; exit 1; }
for f in "$ML/$CONF" "$ML/train_cnn.py" "$ML/model_cnn.py" "$ML/eval_cnn.py" \
         "$CARD" "$BRANCH/scripts/sge/cnn_train_job.sh" \
         "$BRANCH/scripts/sge/piff_monitor_job.sh" \
         "$BRANCH/scripts/sge/piff_tool_job.sh"; do
    [ -e "$f" ] || { echo "MISSING: $f" >&2; exit 1; }
done
# guard on the run DIR, not best.pt (G5 audit 2026-07-19)
[ -d "$ML/runs_piff/$RN" ] && \
    { echo "EXISTS: runs_piff/$RN — refusing to clobber" >&2; exit 1; }
echo "[preflight] conf + trainer + eval + card present; run dir free"

if [ "$GO" -ne 1 ]; then
    echo "DRY RUN - 1 GPU trainer (ibgpu.q gpu=1) + LIVE/FINALIZE monitors"
    echo "(all.q) + 1 held GPU eval. Re-run with --go to submit."
fi

qsub_go() {
    if [ "$GO" -eq 1 ]; then qsub "$@"; else
        echo "[DRY-RUN] qsub $*" >&2; echo "DRY"; fi
}

cd "$BRANCH"
mkdir -p "$LOGS"

TLOG="$LOGS/pCNN_fpc.\$JOB_ID.log"
TRAIN=$(qsub_go -terse -q ibgpu.q -l gpu=1 -N "pCNN_fpc" -j y -cwd -V \
        -m ea -M "$QG_NOTIFY_EMAIL" \
        -o "$TLOG" \
        scripts/sge/cnn_train_job.sh --config "$CONF" --run-name "$RN")
echo "trainer $RN: $TRAIN"
# the trainer's actual log path (JOB_ID resolved)
TLOG_REAL="$LOGS/pCNN_fpc.$TRAIN.log"

# monitors wired HERE, at fire time (2026-07-19 policy)
MONL=$(qsub_go -terse -q all.q -N "pCNNL_f" -j y -cwd -V \
       -o "$LOGS/pCNNL_f.\$JOB_ID.log" \
       scripts/sge/piff_monitor_job.sh "$TLOG_REAL" "$RN" "$TRAIN" "$CARD")
MONF=$(qsub_go -terse -q all.q -N "pCNNF_f" -hold_jid "$TRAIN" \
       -v QG_MONITOR_FINALIZE=1 -j y -cwd -V \
       -o "$LOGS/pCNNF_f.\$JOB_ID.log" \
       scripts/sge/piff_monitor_job.sh "$TLOG_REAL" "$RN" "$TRAIN" "$CARD")
echo "monitors: live $MONL finalize $MONF"

# per-member eval + table mail, held on the trainer
EVAL=$(qsub_go -terse -q ibgpu.q -l gpu=1 -N "pCNNev_f" -hold_jid "$TRAIN" \
       -j y -cwd -V -m ea -M "$QG_NOTIFY_EMAIL" \
       -o "$LOGS/pCNNev_f.\$JOB_ID.log" \
       scripts/sge/piff_tool_job.sh eval_cnn.py \
       --ckpt "runs_piff/$RN/best.pt")
echo "eval: $EVAL"
echo "[submit_cnn] unit complete: trainer $TRAIN, monitors $MONL/$MONF, eval $EVAL"
