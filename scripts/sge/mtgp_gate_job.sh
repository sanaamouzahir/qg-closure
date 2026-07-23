#!/bin/bash
# mtgp_gate_job.sh - final gate + summary for ONE geometry of the MULTITASK
# (coregionalized) SGS closure (Sanaa GO 2026-07-21). Held on the whole
# diagnostics ladder by submit_mtgp.sh. Mirrors bandmodels_gate_job.sh, but the
# primary model is the single multitask checkpoint (no blended handle) and the
# table adds a `mtgp` column beside ylp75/lap/wallv2/blended.
#
# Does three things, in order, and ALWAYS mails before exiting:
#   1. gate_piff_events.py  mtgp vs the ylp75 baseline (--expect-members pins
#      the ensemble so a partially-failed diagnostic cannot PASS on a shrunken
#      universe).
#   2. make_band_table.py   NEAR/FAR/ALL r2 + err% for mtgp vs blended vs wallv2
#      vs lap vs ylp75, and the pre-registered bar NEAR r2 >= 0.954 AND
#      FAR r2 >= 0.934 (applied to mtgp).
#   3. spools a [QG][REPORT] mail CONTAINING THE TABLE (diagnostics-table
#      convention 2026-07-19 - never a bare "done").
# Exit rc: 0 both clean; 4 bar not met; 3 event gate non-PASS; 5 both. The
# non-zero rc makes a non-PASS show up as a failed job + fail digest (loud).
#
# Usage (from submit_mtgp.sh):
#   mtgp_gate_job.sh <geom:fpc|cape> <geom_dir> <model> <baseline_run> <members_csv>

#$ -S /bin/bash
#$ -cwd
#$ -V

set -uo pipefail

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
BRANCH="$QG_ROOT/qg-sgs-closure"
PEND="$QG_ROOT/reporting/pending_mail"

G="${1:?geometry (fpc|cape)}"
GEOM_DIR="${2:?geometry dir (flow_past_cylinder|flow_past_cape)}"
MODEL="${3:?multitask model run name}"
BASE="${4:?baseline run name}"
MEMBERS="${5:?comma-separated member list}"

[[ -d /opt/rocks/bin ]] && export PATH=/opt/rocks/bin:$PATH
source "$QG_ROOT/qg-env-piff/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$BRANCH/ml_closure"

RES="results/$GEOM_DIR/$MODEL"
BM="$RES/band_metrics"
GATE_TXT="runs_piff/$MODEL/gate_mtgp.txt"
TABLE_TXT="$BM/band_table.txt"
mkdir -p "$BM"

echo "[mtgpgate] host $HOSTNAME date $(date -u +%FT%TZ)"
echo "[mtgpgate] geometry $G  model $MODEL  baseline $BASE"
echo "----------------------------------------------------------------------"

# ---- 1. event gate vs the ylp75 baseline ------------------------------- #
python -u gate_piff_events.py \
    --new-run "runs_piff/$MODEL" \
    --baseline-run "runs_piff/$BASE" \
    --expect-members "$MEMBERS" \
    --out-name gate_mtgp.txt \
    --report-run "gate_mtgp_$G"
GATE_RC=$?
echo "[mtgpgate] gate_piff_events rc=$GATE_RC"

# ---- 2. band table: mtgp vs the baselines (mtgp is the primary) -------- #
# Every baseline keeps its COLUMN even when its CSV is absent (prints n/a).
python -u make_band_table.py --geometry "$G" --primary mtgp \
    --csv "ylp75=$BM/ylp75.csv" \
    --csv "lap=$BM/lap.csv" \
    --csv "wallv2=$BM/wallv2.csv" \
    --csv "blended=$BM/blended.csv" \
    --csv "mtgp=$BM/mtgp.csv" \
    --out "$TABLE_TXT"
TABLE_RC=$?
echo "[mtgpgate] make_band_table rc=$TABLE_RC"

# ---- 3. [QG][REPORT] mail -- FULL REPORT CONTRACT ---------------------- #
ABS="$BRANCH/ml_closure"
mkdir -p "$PEND"
MAILF="$PEND/mtgp_${G}_$(date +%s)_$$.mail"

list_dir() {   # $1 = label, $2 = absolute path
  echo "  $1"
  if [ -e "$2" ]; then
    echo "    $2"
    find "$2" -maxdepth 2 -mindepth 1 \( -type d -o -type f \) 2>/dev/null \
      | sort | sed 's|^|      |' | head -40
    N=$(find "$2" -type f 2>/dev/null | wc -l)
    echo "      ... ($N files total)"
  else
    echo "    $2   (NOT PRODUCED)"
  fi
}

{
  echo "To: sanaamz@mit.edu"
  echo "Subject: [QG][REPORT] multitask (coregionalized) SGS closure - $G - verdict, table, artifacts"
  echo
  echo "MULTITASK (COREGIONALIZED) SGS CLOSURE - ONE GP, TWO CORRELATED TASKS"
  echo "geometry : $G      multitask model: $MODEL"
  echo "baseline : $BASE   members: $MEMBERS"
  echo "job      : ${JOB_ID:-?} on $HOSTNAME at $(date -u +%FT%TZ)"
  echo
  echo "ONE GP head with an IndexKernel (ICM) coregionalization matrix over two"
  echo "tasks: task 0 = near-wall closure (pixels sdf <= 1.25 D), task 1 ="
  echo "far-field closure (sdf > 1.25 D). Trained on the WHOLE field (no band"
  echo "restriction). Each task gets its own y-standardization AND its own output"
  echo "scale via the coreg matrix B; B's off-diagonal is a LEARNED cross-task"
  echo "correlation, so far predictions borrow strength from near-wall data -- the"
  echo "cross-region context the two-band specialists lost."
  echo
  # ---- (1) verdict + (2) table -------------------------------------- #
  if [ -f "$TABLE_TXT" ]; then cat "$TABLE_TXT"; else
    echo "VERDICT: NOT AVAILABLE -- the band table was not produced, so the"
    echo "multitask model could not be compared against the baselines."
  fi
  echo
  echo "=============================================================================="
  echo "EVENT GATE vs $BASE (extreme-event regression check)"
  echo "=============================================================================="
  if [ -f "$GATE_TXT" ]; then cat "$GATE_TXT"; else echo "(not produced)"; fi
  echo
  echo "  exit codes: gate_piff_events rc=$GATE_RC (0 PASS / 2 PASS-conditional /"
  echo "  3 REGRESSED);  make_band_table rc=$TABLE_RC (0 bar met / 4 bar NOT met)"
  echo
  echo "=============================================================================="
  echo "LEARNED COREGIONALIZATION + PER-TASK STANDARDIZATION (from final.yaml)"
  echo "=============================================================================="
  echo "  How the two tasks ended up scaled and correlated -- read this WITH the"
  echo "  table: a near-zero cross_corr means the tasks decoupled (the model chose"
  echo "  two independent GPs); a large |cross_corr| means the far task is leaning"
  echo "  on near-wall structure (the intended coupling)."
  FY="$ABS/runs_piff/$MODEL/final.yaml"
  if [ -f "$FY" ]; then
    python - "$FY" <<'PYEOF' 2>/dev/null || echo "  (no multitask block in final.yaml)"
import sys, yaml
d = yaml.safe_load(open(sys.argv[1]))
cr = d.get('multitask_coregionalization')
ys = d.get('y_standardization_per_task')
if not cr and not ys:
    raise SystemExit(1)
if ys:
    print(f"  per-task y-standardization: task0(near) mean={ys['mean'][0]:.4e} "
          f"std={ys['std'][0]:.4e}  task1(far) mean={ys['mean'][1]:.4e} "
          f"std={ys['std'][1]:.4e}")
    print(f"  near/far output-scale ratio (std): {ys['std'][0]/max(ys['std'][1],1e-30):.1f}x")
if cr:
    print(f"  coreg matrix B diagonal (per-task output var): "
          f"B[0,0]={cr['task_var'][0]:.4e}  B[1,1]={cr['task_var'][1]:.4e}")
    print(f"  learned cross-task correlation B[0,1]/sqrt(B00 B11): {cr['cross_corr']:+.4f}")
PYEOF
  else
    echo "  (final.yaml missing)"
  fi
  echo
  # ---- (3) artifact directory list, absolute paths, by suite --------- #
  echo "=============================================================================="
  echo "ARTIFACTS PRODUCED BY THIS CHAIN (absolute cluster paths)"
  echo "=============================================================================="
  echo "  CHECKPOINT (single multitask model)"
  echo "    $ABS/runs_piff/$MODEL/"
  for f in best.pt last.pt curves.png final.yaml run_info.yaml metrics.npz \
           gate_mtgp.txt NAN_ABORT.txt; do
    [ -f "$ABS/runs_piff/$MODEL/$f" ] && echo "      $ABS/runs_piff/$MODEL/$f"
  done
  echo
  list_dir "MEAN PREDICTION (per-member r2)"        "$ABS/$RES/mean_prediction"
  echo
  list_dir "ERROR TAILS (+ extreme_events.csv)"     "$ABS/$RES/error_tails"
  echo
  list_dir "SIGMA AT EVENTS (z_true per member)"    "$ABS/$RES/sigma_at_events"
  echo
  list_dir "ASSESS PANELS (linear shared-scale)"    "$ABS/$RES/eval_assess"
  echo
  list_dir "FIELD6 EVAL PLOTS (symlog)"             "$ABS/$RES/eval"
  echo
  list_dir "BAND METRIC CSVs (table inputs)"        "$ABS/$RES/band_metrics"
  echo
  echo "  LOGS (raw, never committed)"
  echo "    $BRANCH/logs/pMt_$G.*.log         trainer"
  echo "    $BRANCH/logs/pMtmL_$G.*.log       LIVE monitor (nan/stall watch)"
  echo "    $BRANCH/logs/pMtmF_$G.*.log       FINALIZE monitor (postmortem)"
  echo "    $BRANCH/logs/pMt*_$G.*.log        mtgp diagnostics"
  echo "    $BRANCH/logs/pMts$G*.*.log        sigma-at-events, one per member"
  echo "    $BRANCH/logs/pMtb$G*.*.log        baseline band metrics (ylp75/lap/wallv2/blended)"
  echo "    $BRANCH/logs/pMtgt_$G.*.log       this gate job"
  echo
  # ---- (4) results-standard confirmation ----------------------------- #
  echo "=============================================================================="
  echo "RESULTS STANDARD"
  echo "=============================================================================="
  echo "  Everything above is under ONE place, one subdirectory per case/model/member:"
  echo
  echo "      ml_closure/results/<geometry>/<model>/<suite>/<member_modulation>/"
  echo
  echo "  For this geometry that resolves to:"
  echo "      $ABS/results/$GEOM_DIR/$MODEL/<suite>/<member_modulation>/"
  echo
  echo "  geometry : $GEOM_DIR"
  echo "  model    : $MODEL   (multitask coregionalized, single checkpoint)"
  echo "  suites   : mean_prediction, error_tails, sigma_at_events, eval_assess,"
  echo "             eval, band_metrics"
  echo "  members  : $MEMBERS"
  echo
  echo "  migrate_results_tree.sh runs after this job as the sweep-up."
} > "$MAILF"
echo "[mtgpgate] queued $MAILF"

RC=0
[ "$GATE_RC" -ne 0 ] && RC=3
[ "$TABLE_RC" -ne 0 ] && RC=$((RC == 3 ? 5 : 4))
echo "----------------------------------------------------------------------"
echo "[mtgpgate] done at $(date -u +%FT%TZ) rc=$RC"
exit $RC
