#!/bin/bash
# bandmodels_gate_job.sh - final gate + summary for ONE geometry of the
# TWO-BAND SGS closure (Sanaa GO 2026-07-20). Held on the whole diagnostics
# ladder by submit_bandmodels.sh.
#
# Does three things, in order, and ALWAYS mails before exiting:
#   1. gate_piff_events.py  blended vs the ylp75 baseline (--expect-members
#      pins the ensemble so a partially-failed diagnostic cannot PASS on a
#      shrunken universe).
#   2. make_band_table.py   the three-band table (NEAR/FAR/ALL r2 + err%) for
#      blended vs wallv2 vs lap, and the pre-registered bar
#      NEAR r2 >= 0.954 AND FAR r2 >= 0.934.
#   3. spools a [QG][REPORT] mail CONTAINING THE TABLE (diagnostics-table
#      convention 2026-07-19 - never a bare "done").
# Exit rc: 0 both clean; 4 bar not met; 3 event gate non-PASS; 5 both. The
# non-zero rc makes a non-PASS show up as a failed job + fail digest, which
# is intended: it must be loud.
#
# Usage (from submit_bandmodels.sh):
#   bandmodels_gate_job.sh <geom:fpc|cape> <geom_dir> <model> <baseline_run> <members_csv>

#$ -S /bin/bash
#$ -cwd
#$ -V

set -uo pipefail

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
BRANCH="$QG_ROOT/qg-sgs-closure"
PEND="$QG_ROOT/reporting/pending_mail"

G="${1:?geometry (fpc|cape)}"
GEOM_DIR="${2:?geometry dir (flow_past_cylinder|flow_past_cape)}"
MODEL="${3:?blended model run name}"
BASE="${4:?baseline run name}"
MEMBERS="${5:?comma-separated member list}"

[[ -d /opt/rocks/bin ]] && export PATH=/opt/rocks/bin:$PATH
source "$QG_ROOT/qg-env-piff/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$BRANCH/ml_closure"

RES="results/$GEOM_DIR/$MODEL"
BM="$RES/band_metrics"
GATE_TXT="runs_piff/$MODEL/gate_bandmodels.txt"
TABLE_TXT="$BM/band_table.txt"
mkdir -p "$BM"

echo "[bandgate] host $HOSTNAME date $(date -u +%FT%TZ)"
echo "[bandgate] geometry $G  model $MODEL  baseline $BASE"
echo "----------------------------------------------------------------------"

# ---- 1. event gate vs the ylp75 baseline ------------------------------- #
python -u gate_piff_events.py \
    --new-run "runs_piff/$MODEL" \
    --baseline-run "runs_piff/$BASE" \
    --expect-members "$MEMBERS" \
    --out-name gate_bandmodels.txt \
    --report-run "gate_bandmodels_$G"
GATE_RC=$?
echo "[bandgate] gate_piff_events rc=$GATE_RC"

# ---- 2. three-band table: blended vs the three baselines --------------- #
# Every baseline keeps its COLUMN even when its CSV is absent (prints n/a) --
# a missing baseline must never silently vanish from the comparison.
python -u make_band_table.py --geometry "$G" --primary blended \
    --csv "ylp75=$BM/ylp75.csv" \
    --csv "lap=$BM/lap.csv" \
    --csv "wallv2=$BM/wallv2.csv" \
    --csv "blended=$BM/blended.csv" \
    --out "$TABLE_TXT"
TABLE_RC=$?
echo "[bandgate] make_band_table rc=$TABLE_RC"

# ---- 3. [QG][REPORT] mail -- FULL REPORT CONTRACT (Sanaa 2026-07-20) --- #
# (1) plain-English verdict first, (2) the TABLE inline (numbers, never a
# path to them), (3) a directory list of every artifact with absolute cluster
# paths grouped by suite, (4) the explicit results-standard confirmation.
ABS="$BRANCH/ml_closure"
mkdir -p "$PEND"
MAILF="$PEND/bandmodels_${G}_$(date +%s)_$$.mail"

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
  echo "Subject: [QG][REPORT] two-band SGS closure - $G - verdict, table, artifacts"
  echo
  echo "TWO-BAND SGS CLOSURE (near + far specialists, blended inference)"
  echo "geometry : $G      blended model: $MODEL"
  echo "baseline : $BASE   members: $MEMBERS"
  echo "job      : ${JOB_ID:-?} on $HOSTNAME at $(date -u +%FT%TZ)"
  echo
  echo "The two specialists differ ONLY by data.band (near sdf <= 1.5 D, far"
  echo "sdf >= 1.0 D); both use the plain lap recipe, so band-vs-pooled is the"
  echo "single variable under test. Inference blends them with a smoothstep"
  echo "partition of unity over the 1.0-1.5 D overlap."
  echo
  # ---- (1) verdict + (2) table, both inside the table artifact --------- #
  if [ -f "$TABLE_TXT" ]; then cat "$TABLE_TXT"; else
    echo "VERDICT: NOT AVAILABLE -- the band table was not produced, so the"
    echo "two-band model could not be compared against the specialists."
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
  echo "EXPERT SHARE OF SCORED PIXELS (how much of the result is the hand-over)"
  echo "=============================================================================="
  if [ -f "$BM/band_fractions.yaml" ]; then
    cat "$BM/band_fractions.yaml"
    echo
    echo "  Read this WITH the table: a large 'overlap' share means the numbers"
    echo "  above are a property of the blend COLLAR, not of either specialist,"
    echo "  and a run landing near a bar cannot be interpreted without it."
  else
    echo "(not produced -- the blended band-metrics job did not run)"
  fi
  echo
  echo "=============================================================================="
  echo "CROP-REDRAW CENSUS (what the specialists were actually trained on)"
  echo "=============================================================================="
  echo "  Under data.band the sampler redraws out-of-band crop centers, so the"
  echo "  ACTUAL crop distribution differs from the conf's nominal shares."
  for RN in "piff_${G}_gjs_bandnear" "piff_${G}_gjs_bandfar"; do
    RI="$ABS/runs_piff/$RN/run_info.yaml"
    echo "  $RN:"
    if [ -f "$RI" ]; then
      python - "$RI" <<'PYEOF' 2>/dev/null || echo "    (no band_crop_redraw block)"
import sys, yaml
d = yaml.safe_load(open(sys.argv[1])).get('band_crop_redraw')
if not d:
    raise SystemExit(1)
for split, s in d.items():
    if not s:
        continue
    print(f"    {split:5s} ep{s['epoch']}: {s['n_redrawn']}/{s['n_band_crops']}"
          f" redrawn ({100*s['redraw_frac']:.1f}%), of which"
          f" {100*s['signal_frac_of_redraws']:.1f}% from the high-|Pi| in-band"
          f" tail (conf signal_frac={s['conf_signal_frac']},"
          f" q={s['band_signal_quantile']})")
PYEOF
    else
      echo "    (run_info.yaml missing)"
    fi
  done
  echo
  # ---- (3) artifact directory list, absolute paths, by suite ----------- #
  echo "=============================================================================="
  echo "ARTIFACTS PRODUCED BY THIS CHAIN (absolute cluster paths)"
  echo "=============================================================================="
  echo "  CHECKPOINTS (the two specialists + the blended handle)"
  for RN in "piff_${G}_gjs_bandnear" "piff_${G}_gjs_bandfar" "$MODEL"; do
    echo "    $ABS/runs_piff/$RN/"
    for f in best.pt last.pt curves.png final.yaml run_info.yaml metrics.npz \
             blended_manifest.yaml blended.pt gate_bandmodels.txt NAN_ABORT.txt; do
      [ -f "$ABS/runs_piff/$RN/$f" ] && echo "      $ABS/runs_piff/$RN/$f"
    done
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
  echo "    $BRANCH/logs/pBnd*_$G.*.log      trainers + blended diagnostics"
  echo "    $BRANCH/logs/pBmL*_$G.*.log      LIVE monitors (nan/stall watch)"
  echo "    $BRANCH/logs/pBmF*_$G.*.log      FINALIZE monitors (postmortem)"
  echo "    $BRANCH/logs/pBs$G*.*.log        sigma-at-events, one per member"
  echo "    $BRANCH/logs/pBb$G*.*.log        baseline band metrics (ylp75/lap/wallv2)"
  echo "    $BRANCH/logs/pBndgt_$G.*.log     this gate job"
  echo
  # ---- (4) results-standard confirmation ------------------------------- #
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
  echo "  model    : $MODEL   (blended two-band)"
  echo "  suites   : mean_prediction, error_tails, sigma_at_events, eval_assess,"
  echo "             eval, band_metrics"
  echo "  members  : $MEMBERS"
  echo
  echo "  The two specialist RUN NAMES that produced it:"
  echo "      piff_${G}_gjs_bandnear   (near-wall specialist, data.band near)"
  echo "      piff_${G}_gjs_bandfar    (far-field specialist, data.band far)"
  echo
  echo "  migrate_results_tree.sh runs after this job as the sweep-up: it moves"
  echo "  any artifact still sitting next to a checkpoint into the tree above and"
  echo "  renames member subdirs to their plain modulation names, leaving relative"
  echo "  symlinks at the old paths so nothing existing breaks."
} > "$MAILF"
echo "[bandgate] queued $MAILF"

RC=0
[ "$GATE_RC" -ne 0 ] && RC=3
[ "$TABLE_RC" -ne 0 ] && RC=$((RC == 3 ? 5 : 4))
echo "----------------------------------------------------------------------"
echo "[bandgate] done at $(date -u +%FT%TZ) rc=$RC"
exit $RC
