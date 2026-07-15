#!/bin/bash
#$ -N apost_ladder
#$ -q ibgpu.q
#$ -l gpu=1
#$ -j y
#$ -o /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/$JOB_NAME.$JOB_ID.log
#$ -e /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/$JOB_NAME.$JOB_ID.err
#$ -cwd
#$ -m ea
#$ -M sanaamz@mit.edu
# apost_ladder_job.sh -- I16 remediation ladder for the NN-feedback blowup
# (smoke3a_val config: kf4 @ 1.5e-2, IC 837, RK4 truth reused via --load-refs).
# Rungs in order, ONE variable each; stop at the first rung that is STABLE
# (no blowup) AND beats bare >= 5x at the horizon (t=0.24):
#   R1a --nn-kcut 0.5   R1b --nn-kcut 0.75
#   R2a --nn-gamma 0.5  R2b --nn-gamma 0.75  R2c --nn-gamma 0.9
#   R3  --nn-clip 3
set -e
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
WT=$QG_ROOT/qg-wiener-conditioning
OUT=$WT/diagnostics/Results/apost_smoke3/ladder
# refs regenerated under a DISTINCT tag (the original smoke3a_val npz working
# copies were externally deleted 2026-07-08; committed csv/json restored from
# git -- regenerating under the same tag would shadow them)
REFS=$WT/diagnostics/Results/apost_smoke3/apost_refs_ladderrefs.npz
CKPT=data/ensemble_N5_7lag/training_runs/deriv7_filtered_lr5e-5/best.pt
source $QG_ROOT/qg-env/bin/activate
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR=$QG_ROOT/.mplcache; mkdir -p "$MPLCONFIGDIR"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd $WT/training
mkdir -p "$OUT"
echo "[ladder] host=$HOSTNAME date=$(date -u +%FT%TZ)"
python -u -c 'import sys, torch; ok = torch.cuda.is_available(); print("[gpu-check]", ok, flush=True); sys.exit(0 if ok else 2)'

if [ ! -f "$REFS" ]; then
    echo "[ladder] refs missing -- regenerating RK4 truth (K=1500, IC 837) ..."
    python -u rollout_aposteriori.py \
        --root-dir data/ensemble_N5_7lag/FRC-kf4/sweep_dT_1p5em2 \
        --ckpt $CKPT --ic-index 837 --K 1500 --n-steps 16 \
        --arms bare --save-refs --no-log-sigma \
        --device cuda --out-dir "$(dirname "$REFS")" --tag ladderrefs
fi

run_rung () {
    local label=$1; shift
    echo "===================== RUNG $label : $* ====================="
    # crash != scientific negative (review F1): abort the ladder on a
    # non-zero rollout exit instead of reclassifying it as 'insufficient'
    if ! python -u rollout_aposteriori.py \
        --root-dir data/ensemble_N5_7lag/FRC-kf4/sweep_dT_1p5em2 \
        --ckpt $CKPT --ic-index 837 --K 1500 --n-steps 16 \
        --load-refs "$REFS" --arms bare,closure --track-lte \
        --device cuda --out-dir "$OUT" --tag "ladder_${label}" "$@"; then
        echo "[ladder] rung $label CRASHED -- aborting ladder (fix before rerun)."
        exit 3
    fi
    python - "$OUT/rollout_apost_ladder_${label}.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
blow = d.get('closure_blowup_step')
imp = d.get('improvement_x_closure') or 0.0
ok = (blow is None) and imp >= 5.0
print(f"[ladder] verdict: blowup={blow}  improvement={imp:.2f}x  "
      f"{'SUCCESS' if ok else 'insufficient'}")
sys.exit(0 if ok else 1)
PYEOF
}

for spec in "R1a --nn-kcut 0.5" "R1b --nn-kcut 0.75" \
            "R2a --nn-gamma 0.5" "R2b --nn-gamma 0.75" "R2c --nn-gamma 0.9" \
            "R3 --nn-clip 3"; do
    set -- $spec
    if run_rung "$@"; then
        echo "[ladder] FIRST SUCCESS at rung $1 -- stopping per I16."
        echo "[ladder] done $(date -u +%FT%TZ)"
        exit 0
    fi
done
echo "[ladder] NO rung met (stable AND >=5x). Full table in $OUT; escalate R4."
echo "[ladder] done $(date -u +%FT%TZ)"
