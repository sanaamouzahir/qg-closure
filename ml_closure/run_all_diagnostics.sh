#!/bin/bash
# run_all_diagnostics.sh -- NEW MODEL => FULL RE-EVAL (Sanaa direct order,
# 2026-07-17, results-organization STANDARD).
#
# For one trained Pi_FF model run, submits EVERY diagnostic suite as an
# all.q CPU job (diagnostics never run on the GPU queue -- Sanaa ruling
# 2026-07-15) via piff_tool_job.sh, with --outdir/--fig-dir pointed INTO the
# standard results tree:
#
#     ml_closure/results/<geometry>/<model>/<suite>/<member_modulation>/
#
# Suites: evaluation, mean_prediction, error_tails, sigma_at_events,
#         reflection_probe, nearwall_variants, feature_screen.
# sigma_at_events / nearwall_variants / reflection_probe consume
# error_tails' extreme_events.csv, so they are chained with -hold_jid.
# Ends with a chained mail job (queue_landing_mail_job.sh) that emails the
# full directory map once everything has landed.
#
# RUN FROM THE CLUSTER FRONT END (submission only -- no compute here):
#   cd $BRANCH/ml_closure && ./run_all_diagnostics.sh <model_run_name>
#   e.g. ./run_all_diagnostics.sh piff_fpc_gjs_ylp75
#
# Geometry is resolved from the run name (piff_cape_* / *cape* -> cape, else
# cylinder -- every non-cape Pi_FF run in this branch is the cylinder);
# members come from conf_<model>.yaml (the same config the run trained on).
# SGE per I1/G5: -q all.q (CPU), thread caps 8, -m ea -M, logs to ../logs/.

set -uo pipefail

MODEL="${1:?usage: run_all_diagnostics.sh <model_run_name> (e.g. piff_fpc_gjs_ylp75)}"

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
BRANCH="$QG_ROOT/qg-sgs-closure"
[[ -d /opt/rocks/bin ]] && export PATH=/opt/rocks/bin:$PATH
source "$QG_ROOT/qg-env-piff/bin/activate"
cd "$BRANCH/ml_closure"

CKPT="runs_piff/$MODEL/best.pt"
CONF="conf_${MODEL}.yaml"
[[ -f "$CKPT" ]] || { echo "[rad] FATAL: $CKPT missing"; exit 1; }
[[ -f "$CONF" ]] || { echo "[rad] FATAL: $CONF missing (config must be conf_<model>.yaml)"; exit 1; }

case "$MODEL" in
    *cape*) GEOM=flow_past_cape ;;
    *)      GEOM=flow_past_cylinder ;;
esac
RES="results/$GEOM/$MODEL"
mkdir -p "$RES"

LOGS="$BRANCH/logs"
mkdir -p "$LOGS"
JOBSH="$BRANCH/scripts/sge/piff_tool_job.sh"
MAILSH="$BRANCH/scripts/sge/queue_landing_mail_job.sh"
NOTIFY="${QG_NOTIFY_EMAIL:-sanaamz@mit.edu}"
THREADS="OMP_NUM_THREADS=8,MKL_NUM_THREADS=8,OPENBLAS_NUM_THREADS=8"

# members of this model's pool (basenames of data.runs)
MEMBERS=$(python - "$CONF" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
print(' '.join(p.rstrip('/').split('/')[-1] for p in c['data']['runs']))
PY
) || { echo "[rad] FATAL: could not parse members from $CONF"; exit 1; }
echo "[rad] model=$MODEL geometry=$GEOM members: $MEMBERS"

mod() {  # member codename -> plain modulation dirname (sibling-aware telS-A)
    python -c "import member_naming as m; import sys; \
print(m.modulation_name(sys.argv[1], sys.argv[2].split()))" "$1" "$MEMBERS"
}

# submit <name> <hold_jid|-> <digest-run|-> <tool.py> [args...]
# Sets $JID (global) -- NOT command substitution, so a qsub failure aborts the
# whole script (G4 2026-07-17 findings 1+2: `$(...)` swallowed exit 1, and
# "${empty[@]}" dies under set -u on bash 4.2).
JID=''
submit() {
    local name=$1 hold=$2 digest=$3; shift 3
    local extra=()
    [[ "$hold" != "-" ]] && extra+=(-hold_jid "$hold")
    local v="$THREADS"
    [[ "$digest" != "-" ]] && v="$v,QG_DIGEST_RUN=$digest"
    local out
    out=$(qsub -q all.q -N "$name" -m ea -M "$NOTIFY" -j y -cwd -V \
              -v "$v" -o "$LOGS/$name.\$JOB_ID.log" \
              ${extra[@]+"${extra[@]}"} \
              "$JOBSH" "$@") \
        || { echo "[rad] FATAL: qsub failed for $name"; exit 1; }
    JID=$(awk '{print $3}' <<<"$out")
    [[ -n "$JID" ]] || { echo "[rad] FATAL: no job id parsed for $name: $out"; exit 1; }
    echo "[rad] $name -> job $JID"
}

# ---- independent suites -------------------------------------------------- #
submit "rE_${MODEL}" - "rad_${MODEL}_evaluation" \
    eval_piff.py --ckpt "$CKPT" --config "$CONF" --device cpu \
    --outdir "$RES/evaluation" --fig-dir "$RES/evaluation"
J_EVAL=$JID

submit "rM_${MODEL}" - "rad_${MODEL}_mean_prediction" \
    diagnose_mean_prediction.py --ckpt "$CKPT" --config "$CONF" --device cpu \
    --plain-member-names --outdir "$RES/mean_prediction" \
    --fig-dir "$RES/mean_prediction" --report-run "rad_${MODEL}_mean_prediction"
J_MEAN=$JID

submit "rT_${MODEL}" - "rad_${MODEL}_error_tails" \
    diagnose_error_tails.py --ckpt "$CKPT" --config "$CONF" --device cpu \
    --plain-member-names --outdir "$RES/error_tails" \
    --fig-dir "$RES/error_tails" --report-run "rad_${MODEL}_error_tails"
J_TAILS=$JID

submit "rF_${MODEL}" - "rad_${MODEL}_feature_screen" \
    diagnose_feature_candidates.py --config "$CONF" \
    --outdir "$RES/feature_screen" --report-run "rad_${MODEL}_feature_screen"
J_FEAT=$JID

ALL="$J_EVAL,$J_MEAN,$J_TAILS,$J_FEAT"

# ---- per-member suites that need error_tails' extreme_events.csv --------- #
i=0
for M in $MEMBERS; do
    i=$((i + 1))
    MODN=$(mod "$M")
    [[ -n "$MODN" ]] || { echo "[rad] FATAL: modulation name failed for $M"; exit 1; }
    EVENTS="$RES/error_tails/$MODN/extreme_events.csv"
    submit "rS${i}_${MODEL}" "$J_TAILS" "rad_${MODEL}_sigma_at_events" \
        diagnose_sigma_at_events.py --ckpt "$CKPT" --config "$CONF" \
        --member "$M" --events "$EVENTS" --plain-member-names \
        --pool-members "$MEMBERS" --outdir "$RES/sigma_at_events" \
        --report-run "rad_${MODEL}_sigma_at_events"
    J_SIG=$JID
    submit "rN${i}_${MODEL}" "$J_TAILS" "rad_${MODEL}_nearwall_variants" \
        diagnose_nearwall_variants.py --config "$CONF" --member "$M" \
        --events "$EVENTS" --fig-dir "$RES/nearwall_variants/$MODN" \
        --outdir "$RES/nearwall_variants/$MODN" \
        --report-run "rad_${MODEL}_nearwall_variants"
    J_NW=$JID
    ALL="$ALL,$J_SIG,$J_NW"
done

# ---- reflection probe: the geometry's worst member (probe convention) ---- #
case "$GEOM" in
    flow_past_cape) WORST=FPCape-sine ;;
    *)              WORST=FPC-const ;;
esac
grep -qw "$WORST" <<<"$MEMBERS" || WORST=$(awk '{print $1}' <<<"$MEMBERS")
WMOD=$(mod "$WORST")
[[ -n "$WMOD" ]] || { echo "[rad] FATAL: modulation name failed for $WORST"; exit 1; }
submit "rR_${MODEL}" "$J_TAILS" "rad_${MODEL}_reflection_probe" \
    probe_reflection_hypothesis.py --ckpt "$CKPT" --config "$CONF" \
    --member "$WORST" --events "$RES/error_tails/$WMOD/extreme_events.csv" \
    --outdir "$RES/reflection_probe/$WMOD" \
    --fig-dir "$RES/reflection_probe/$WMOD" \
    --report-run "rad_${MODEL}_reflection_probe"
J_REFL=$JID
ALL="$ALL,$J_REFL"

# ---- directory map + landing mail ---------------------------------------- #
MAP="$RES/DIRECTORY_MAP.txt"
{
    echo "FULL RE-EVAL of model $MODEL ($(date -u +%FT%TZ))"
    echo "PARAMETERS: ckpt $BRANCH/ml_closure/$CKPT, config $CONF,"
    echo "  val split, filtered targets per the ckpt's own variant, CPU (all.q,"
    echo "  8 threads), members: $MEMBERS."
    echo ""
    echo "RESULTS TREE (geometry > model > suite > member_modulation):"
    echo "  $BRANCH/ml_closure/$RES/"
    echo "    evaluation/           a-priori eval package (calibration, field panels, Re trace)"
    echo "    mean_prediction/      per-modulation mean-error suite + summary_all_members.csv"
    echo "    error_tails/          per-modulation tails/exceedance suite + extreme_events.csv"
    echo "    sigma_at_events/      per-modulation sigma-vs-truth at the extreme events"
    echo "    nearwall_variants/    per-modulation near-wall filter-variant check"
    echo "    reflection_probe/$WMOD/  effective-Re conditioning probe (worst member)"
    echo "    feature_screen/       near-wall feature-candidate ranking"
    echo "  member_modulation dirs: $(for M in $MEMBERS; do printf '%s ' "$(mod "$M")"; done)"
    echo ""
    echo "git summaries: reports/rad_${MODEL}_*/ (pushed via digest_writer)"
    echo "logs: $LOGS/r[EMTFSNR]*_${MODEL}.*.log"
    echo "jobs: $ALL"
    echo ""
    echo "NEXT: read $RES/mean_prediction/summary.md and"
    echo "$RES/error_tails/summary.md first (headline R2 / tail shares);"
    echo "then the per-modulation panels in evaluation/."
} > "$MAP"

qsub -q all.q -N "rMail_${MODEL}" -m ea -M "$NOTIFY" -j y -cwd -V \
     -hold_jid "$ALL" -o "$LOGS/rMail_${MODEL}.\$JOB_ID.log" \
     "$MAILSH" "[QG][LANDED][sgs-closure] full re-eval of $MODEL landed -- results tree map inside" \
     "$MAP" \
    || echo "[rad] WARNING: landing-mail job failed to submit"

echo "[rad] all suites submitted for $MODEL -- jobs: $ALL"
echo "[rad] directory map: $BRANCH/ml_closure/$MAP"
