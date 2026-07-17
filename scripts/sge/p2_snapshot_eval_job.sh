#!/bin/bash
# p2_snapshot_eval_job.sh -- Sanaa order 2026-07-17: mid-training eval of the
# two trust arms (per-root a-priori tables + 10% gates) on snapshot copies of
# their current best.pt. CPU node (diagnostics rule). [fable-authored]
#$ -S /bin/bash
#$ -q all.q
#$ -j y
#$ -cwd
set -uo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
D=$QG_ROOT/qg-simple-package-stable/src/qg/training/data/ensemble_N5_7lag/training_runs
SP=$QG_ROOT/reporting/pending_mail
source "$QG_ROOT/qg-env/bin/activate"
cd "$QG_ROOT/qg-wiener-conditioning/training"
TAG="snap$(date +%H%M)"
ROOTS=$(python -c "import json; print(' '.join(json.load(open('$D/deriv7_cond_local_w31/config.json'))['roots']))")
for a in rollout_ft_w31_p2_trust rollout_ft_w31_p2_trust_vn05; do
    [ -f "$D/$a/best.pt" ] || { echo "no best.pt for $a yet"; continue; }
    cp "$D/$a/best.pt" "$D/$a/best_${TAG}.pt"
    # shellcheck disable=SC2086
    python -u eval_deriv_by_root.py --ckpt "$D/$a/best_${TAG}.pt" \
        --sweep-roots $ROOTS --grad-kernel 31 --device cpu \
        || echo "EVAL_FAIL_$a"
    python -u accept_ft_gate.py \
        --before "$D/deriv7_cond_local_w31/eval_by_root_val.csv" \
        --after  "$D/$a/eval_by_root_val.csv" --tol-rel 0.10 \
        > "$D/$a/gate_${TAG}.txt" 2>&1 || true
done
mkdir -p "$SP"
{
echo "To: ${QG_NOTIFY_EMAIL:-sanaamz@mit.edu}"
echo "Subject: [QG][MONITOR][wiener] trust arms MID-TRAINING per-root tables + gates ($TAG)"
echo
echo "PARAMETERS: eval_deriv_by_root (41 roots, grad-kernel 31, CPU) + 10%"
echo "Nddot gate vs w31-ep32, on snapshot copies of both arms current best.pt"
echo "(trainers untouched, ~ep10-11 of 20; vn 0.1 vs 0.5, trust anchor both)."
echo
for a in rollout_ft_w31_p2_trust rollout_ft_w31_p2_trust_vn05; do
    echo "== $a"; tail -8 "$D/$a/gate_${TAG}.txt" 2>/dev/null; echo
done
echo "DIRECTORIES: per-root CSVs $D/<arm>/eval_by_root_val.csv; gate texts"
echo "$D/<arm>/gate_${TAG}.txt; training logs qg-wiener-conditioning/logs/."
echo
echo "NEXT: final gates fire automatically at ep20 (~tonight); ruling vn 0.1"
echo "vs 0.5 then, with these mid-training tables as the trend line."
} > "$SP/$(date +%Y%m%dT%H%M%S)_p2_snapshot_gates.mail"
echo "[p2snap] done"
