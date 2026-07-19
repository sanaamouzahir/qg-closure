#!/bin/bash
# postft_longroll_job.sh -- LONG a-posteriori rollouts (past the 21-step
# training horizon) of the accepted rollout-FT arm ckpt, CPU ONLY
# (Sanaa standing order 2026-07-16: no diagnostics/rollouts on GPU).
# [fable-authored 2026-07-16]
#
# Per draw ("MEMBER IC" args): 64- and 128-step horizons at dT=5e-3, K=500
# (h_fine=1e-5, full-RESULT tier per I15), arms bare+closure, truth refs
# REUSED from the rep archive (hard-fail if missing -- NEVER recompute truth
# on CPU). cp grids match the saved refs (--n-steps H --n-checkpoints 24).
# Refs exist only for FRC-kf4 / FRC-combo / FRC-256 (10 ICs total); other
# members are reported as no-refs, not computed.
#
# Usage:
#   qsub -q all.q -N longroll_<x> scripts/sge/postft_longroll_job.sh \
#       <CKPT> <CKPT2|NONE> "FRC-kf4 532" "FRC-kf4 912" ...
# CKPT2 != NONE adds arm 'closure2' (--ckpt2, identical code path) -- used
# 2026-07-16 to roll BOTH FT arms in one pass after both gates FAILed.
#$ -S /bin/bash
#$ -q all.q
#$ -cwd
#$ -j y
#$ -o /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/$JOB_NAME.o$JOB_ID
#$ -m ea
#$ -M sanaamz@mit.edu
set -uo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
WT=$QG_ROOT/qg-wiener-conditioning
REP=$WT/diagnostics/Results/apost_opt2_rep_20260711
# LONGROLL_OUT (qsub -v) overrides the output tree; default = the 2026-07-16
# p1-arm rolls dir, byte-identical behavior when unset (tags collide across
# ckpts -- ALWAYS override for a new ckpt).
OUT=${LONGROLL_OUT:-$WT/diagnostics/Results/apost_longroll_postft_20260716}
export MPLCONFIGDIR=$QG_ROOT/.mplcache; mkdir -p "$MPLCONFIGDIR"
source "$QG_ROOT/qg-env/bin/activate"
cd "$WT/training"

CKPT=$1; CKPT2=$2; shift 2
echo "[longroll] host=$HOSTNAME date=$(date -u +%FT%TZ) ckpt=$CKPT ckpt2=$CKPT2 draws: $*"
[ -f "$CKPT" ] || { echo "[longroll] HARD-FAIL: ckpt $CKPT not found"; exit 4; }
ARMS=bare,closure
EXTRA=""
if [ "$CKPT2" != "NONE" ]; then
    [ -f "$CKPT2" ] || { echo "[longroll] HARD-FAIL: ckpt2 $CKPT2 not found"; exit 4; }
    ARMS=bare,closure,closure2
    EXTRA="--ckpt2 $CKPT2"   # no spaces in repo paths; old-bash set -u safe
fi

for spec in "$@"; do
    set -- $spec; member=$1; ic=$2
    CD=$OUT/${member}_ic${ic}; mkdir -p "$CD"
    for H in 64 128; do
        RH=$REP/${member}_ic${ic}/apost_refs_ic${ic}_5em3_h${H}.npz
        if [ ! -f "$RH" ]; then
            echo "[longroll] MISSING_REFS $member ic=$ic h=$H ($RH) -- skipped, never recomputed"
            continue
        fi
        echo "==== $member ic=$ic dT=5e-3 h=$H (refs: $RH) ===="
        python -u rollout_aposteriori.py \
            --root-dir "data/ensemble_N5_7lag/$member/sweep_dT_5em3" \
            --ckpt "$CKPT" \
            --ic-index "$ic" --K 500 --n-steps "$H" --n-checkpoints 24 \
            --arms "$ARMS" --log-sigma $EXTRA \
            --device cpu --out-dir "$CD" \
            --tag "ic${ic}_5em3_h${H}_postft" \
            --load-refs "$RH" || echo "ROLL_FAIL $member $ic h$H"
    done
done
echo "[longroll] done $(date -u +%FT%TZ)"
