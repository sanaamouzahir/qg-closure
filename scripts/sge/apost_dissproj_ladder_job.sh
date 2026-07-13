#!/bin/bash
# apost_dissproj_ladder_job.sh - P0 dissipative-projection A/B ladder
# (Sanaa 2026-07-13 WIENER STABILITY+ACCURACY PLAN).
#
# Per draw (member, IC) passed as "MEMBER IC" args:
#   1. 16-step ladder at dT {1.5e-2, 1e-2, 5e-3}, A/B proj OFF/ON,
#      TRUTH REFS REUSED (hard-fail if missing -- never recompute):
#      rep refs in Results/apost_opt2_rep_20260711/<member>_ic<IC>/;
#      IC837 refs in apost_smoke3 (1.5e-2) + apost_ladder_20260709 (1e-2,5e-3).
#   2. 64- and 128-step horizons at 5e-3 (K=500, h_fine=1e-5), A/B OFF/ON.
#      Refs EXTENDED: OFF pass runs --save-refs, refs then MOVED to the rep
#      case dir (next to the existing 16-step refs, for future reuse); the
#      ON pass loads them. Idempotent: an existing extended ref is reused.
#   3. Consolidation: one npz per case (consolidate_apost_cases.py),
#      per-horizon summary CSVs (merged by the finalize job).
#
# Ckpt: rollout_ft_opt2_cond/best.pt (ep33) -- same as the 07-11 rep ladder.
# Flags otherwise IDENTICAL to apost_opt2_rep_job.sh: --n-steps 16
# --n-checkpoints 24 --arms bare,closure --log-sigma, K = 1500/1000/500.
#
# Refuses to run without $NEW/GATE_PASS (G3 gate job must pass first).
#
# Submit (two balanced halves, one GPU each; gate held via -hold_jid):
#   qsub -q ibgpu.q -l gpu=1 -N dissproj_A -hold_jid <gate_id> \
#     scripts/sge/apost_dissproj_ladder_job.sh \
#     "FRC-kf4 532" "FRC-kf4 912" "FRC-kf4 1356" "FRC-kf4 837" "FRC-256 549"
#   qsub -q ibgpu.q -l gpu=1 -N dissproj_B -hold_jid <gate_id> \
#     scripts/sge/apost_dissproj_ladder_job.sh \
#     "FRC-combo 527" "FRC-combo 884" "FRC-combo 1355" "FRC-256 933" "FRC-256 1357"
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -o /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/$JOB_NAME.o$JOB_ID
#$ -e /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/$JOB_NAME.e$JOB_ID
#$ -m ea
#$ -M sanaamz@mit.edu

set -e
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
WT=$QG_ROOT/qg-wiener-conditioning
NEW=$WT/diagnostics/Results/apost_dissproj_20260713
REP=$WT/diagnostics/Results/apost_opt2_rep_20260711
LADDER=$WT/diagnostics/Results/apost_ladder_20260709
SMOKE3=$WT/diagnostics/Results/apost_smoke3
CKPT=data/ensemble_N5_7lag/training_runs/rollout_ft_opt2_cond/best.pt
source $QG_ROOT/qg-env/bin/activate
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR=$QG_ROOT/.mplcache; mkdir -p "$MPLCONFIGDIR"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd $WT/training
echo "[dissproj_ladder] host=$HOSTNAME date=$(date -u +%FT%TZ) draws: $*"
echo "[dissproj_ladder] git HEAD: $(/opt/rocks/bin/git rev-parse --short HEAD 2>/dev/null || echo '?')"
[[ -f "$NEW/GATE_PASS" ]] || { echo "[dissproj_ladder] NO GATE_PASS -- refusing (G3 gate failed or not run)"; exit 4; }
python -u -c 'import sys, torch; ok = torch.cuda.is_available(); print("[gpu-check]", ok, flush=True); sys.exit(0 if ok else 2)'

roll () {   # roll <member> <ic> <sweep> <K> <nsteps> <tag> <outdir> <refflag...>
    local member=$1 ic=$2 sweep=$3 K=$4 nsteps=$5 tag=$6 outdir=$7; shift 7
    python -u rollout_aposteriori.py \
        --root-dir "data/ensemble_N5_7lag/$member/$sweep" \
        --ckpt "$CKPT" \
        --ic-index "$ic" --K "$K" --n-steps "$nsteps" --n-checkpoints 24 \
        --arms bare,closure --log-sigma \
        --device cuda --out-dir "$outdir" --tag "$tag" "$@"
}

for spec in "$@"; do
    set -- $spec; member=$1; ic=$2; short=${member#FRC-}
    CD=$NEW/${member}_ic${ic}
    REPD=$REP/${member}_ic${ic}
    mkdir -p "$CD" "$REPD"

    # ---- 16-step ladder, refs REUSED (hard-fail if missing) ---- #
    for row in "1p5em2 1500" "1em2 1000" "5em3 500"; do
        set -- $row; dtl=$1; K=$2
        if [[ $ic == 837 ]]; then
            case $dtl in
                1p5em2) refs=$SMOKE3/apost_refs_ladderrefs.npz ;;
                1em2)   refs=$LADDER/apost_refs_full_1em2.npz ;;
                5em3)   refs=$LADDER/apost_refs_full_5em3.npz ;;
            esac
        else
            refs=$REPD/apost_refs_ic${ic}_${dtl}.npz
        fi
        [[ -f "$refs" ]] || { echo "[dissproj_ladder] HARD-FAIL: missing truth refs $refs (never recompute silently)"; exit 3; }
        echo "==== $member ic=$ic dT=$dtl OFF (refs: $refs) ===="
        roll "$member" "$ic" "sweep_dT_$dtl" "$K" 16 "ic${ic}_${dtl}_off" "$CD" --load-refs "$refs"
        echo "==== $member ic=$ic dT=$dtl ON ===="
        roll "$member" "$ic" "sweep_dT_$dtl" "$K" 16 "ic${ic}_${dtl}_on" "$CD" --load-refs "$refs" --nn-dissipative-proj
    done

    # ---- 64/128-step horizons at 5e-3, refs EXTENDED then reused ---- #
    for H in 64 128; do
        RH=$REPD/apost_refs_ic${ic}_5em3_h${H}.npz
        if [[ -f "$RH" ]]; then
            echo "==== $member ic=$ic 5e-3 h=$H OFF (extended refs reused) ===="
            roll "$member" "$ic" sweep_dT_5em3 500 "$H" "ic${ic}_5em3_h${H}_off" "$CD" --load-refs "$RH"
        else
            echo "==== $member ic=$ic 5e-3 h=$H OFF (extending refs: ${H}x500 RK4 steps) ===="
            roll "$member" "$ic" sweep_dT_5em3 500 "$H" "ic${ic}_5em3_h${H}_off" "$CD" --save-refs
            mv "$CD/apost_refs_ic${ic}_5em3_h${H}_off.npz" "$RH"
            echo "[dissproj_ladder] extended refs -> $RH (next to the 16-step refs)"
        fi
        echo "==== $member ic=$ic 5e-3 h=$H ON ===="
        roll "$member" "$ic" sweep_dT_5em3 500 "$H" "ic${ic}_5em3_h${H}_on" "$CD" --load-refs "$RH" --nn-dissipative-proj
    done

    # ---- consolidate: one npz per case, per-horizon summaries ---- #
    echo "==== CONSOLIDATE $member ic=$ic ===="
    python -u $WT/diagnostics/consolidate_apost_cases.py --dir "$CD" \
        --tags "ic${ic}_1p5em2_off,ic${ic}_1p5em2_on,ic${ic}_1em2_off,ic${ic}_1em2_on,ic${ic}_5em3_off,ic${ic}_5em3_on" \
        --arm-labels "closure=opt2_${short}_ic${ic}" \
        --summary-csv ladder_matrix_summary.csv --delete-intermediates
    python -u $WT/diagnostics/consolidate_apost_cases.py --dir "$CD" \
        --tags "ic${ic}_5em3_h64_off,ic${ic}_5em3_h64_on" \
        --arm-labels "closure=opt2_${short}_ic${ic}_h64" \
        --summary-csv ladder_matrix_summary_h64.csv --delete-intermediates
    python -u $WT/diagnostics/consolidate_apost_cases.py --dir "$CD" \
        --tags "ic${ic}_5em3_h128_off,ic${ic}_5em3_h128_on" \
        --arm-labels "closure=opt2_${short}_ic${ic}_h128" \
        --summary-csv ladder_matrix_summary_h128.csv --delete-intermediates
done
echo "[dissproj_ladder] done $(date -u +%FT%TZ)"
