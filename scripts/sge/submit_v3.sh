#!/bin/bash
# submit_v3.sh - V3CNN full chains (Sanaa order 2026-07-23), both geometries:
#   trainer (2400 ep, warm-grown from v2 best, EnsCon beta 0.1)
#   -> residual-GP (hold) -> GP eval (hold) ; CNN eval (hold trainer)
#   + LIVE/FINALIZE monitors wired at fire time.
# Dry-run by default; --go to submit. Run from the branch root on mseas.
set -euo pipefail
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
B="$QG_ROOT/qg-sgs-closure"
CARD=$B/diagnostics/baseline_cards/SGS_piff_cnn.json
GO=0; [ "${1:-}" = "--go" ] && GO=1
q() { if [ "$GO" -eq 1 ]; then qsub "$@"; else echo "[DRY] qsub $*" >&2; echo DRY; fi; }
cd "$B"; mkdir -p logs
for G in fpc cape; do
  RN=piff_${G}_cnn_v3; WARM=runs_piff/piff_${G}_cnn_v2_ext1800/best.pt
  [ -f ml_closure/$WARM ] || { echo "$G warm MISSING: $WARM" >&2; exit 1; }
  [ -f ml_closure/conf_piff_${G}_cnn_v3.yaml ] || { echo "$G v3 conf MISSING" >&2; exit 1; }
  [ -d ml_closure/runs_piff/$RN ] && { echo "$RN EXISTS - skip"; continue; }
  T=$(q -terse -q ibgpu.q -l gpu=1 -N pW3${G:0:1}_tr -j y -cwd -V -m ea -M sanaamz@mit.edu \
      -o "$B/logs/pW3${G:0:1}_tr.\$JOB_ID.log" scripts/sge/cnn_train_job.sh \
      --config conf_piff_${G}_cnn_v3.yaml --run-name $RN --epochs 2400 \
      --enscon-beta 0.1 --init-ckpt-grow $WARM)
  TL="$B/logs/pW3${G:0:1}_tr.$T.log"
  ML=$(q -terse -q all.q -N pW3${G:0:1}L -j y -cwd -V -o "$B/logs/pW3${G:0:1}L.\$JOB_ID.log" \
      scripts/sge/piff_monitor_job.sh "$TL" $RN "$T" "$CARD")
  MF=$(q -terse -q all.q -N pW3${G:0:1}F -hold_jid "$T" -v QG_MONITOR_FINALIZE=1 -j y -cwd -V \
      -o "$B/logs/pW3${G:0:1}F.\$JOB_ID.log" scripts/sge/piff_monitor_job.sh "$TL" $RN "$T" "$CARD")
  GP=$(q -terse -q ibgpu.q -l gpu=1 -N pW3${G:0:1}gp -hold_jid "$T" -j y -cwd -V -m ea -M sanaamz@mit.edu \
      -o "$B/logs/pW3${G:0:1}gp.\$JOB_ID.log" scripts/sge/gp_residual_job.sh \
      --cnn-ckpt runs_piff/$RN/best.pt --run-name piff_${G}_gpres_v3)
  EV=$(q -terse -q ibgpu.q -l gpu=1 -N pW3${G:0:1}ev -hold_jid "$T" -j y -cwd -V -m ea -M sanaamz@mit.edu \
      -o "$B/logs/pW3${G:0:1}ev.\$JOB_ID.log" scripts/sge/piff_tool_job.sh eval_cnn.py \
      --ckpt runs_piff/$RN/best.pt --pred-filter ylp75)
  GE=$(q -terse -q ibgpu.q -l gpu=1 -N pW3${G:0:1}ge -hold_jid "$GP" -j y -cwd -V -m ea -M sanaamz@mit.edu \
      -o "$B/logs/pW3${G:0:1}ge.\$JOB_ID.log" scripts/sge/piff_tool_job.sh eval_residual_gp.py \
      --ckpt runs_piff/piff_${G}_gpres_v3/best.pt)
  echo "$G V3: trainer $T mon $ML/$MF -> GP $GP -> gp-eval $GE; cnn-eval $EV"
done
