#!/bin/bash
# p2_final_eval_job.sh -- FINAL post-training eval of the two trust arms
# (per-root a-priori tables + 10% gates) on their FINISHED best.pt (both
# trainers exited 2026-07-18 ~11:51; ep20/20). The ep20 auto-gate never
# fired (finalize watcher exited on a stale ep0 delivery marker), so this
# job produces the promised final tables. CPU node (diagnostics rule).
# Derived from p2_snapshot_eval_job.sh; only TAG + email wording differ.
# [fable-authored]
#$ -S /bin/bash
#$ -q all.q
#$ -j y
#$ -cwd
#$ -o /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/
#$ -e /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/
set -uo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
BRANCH=$QG_ROOT/qg-wiener-conditioning
D=$QG_ROOT/qg-simple-package-stable/src/qg/training/data/ensemble_N5_7lag/training_runs
SP=$QG_ROOT/reporting/pending_mail
DIGEST="$BRANCH/diagnostics/digest_writer.py"
RUN_NAME=w31p2_finalgate
digest_event() {  # I23b; no-op if digest_writer not on this checkout
    [[ -f "$DIGEST" && -n "$RUN_NAME" ]] && \
        python "$DIGEST" --repo-dir "$BRANCH" --run-name "$RUN_NAME" \
            --event "$1" --job-id "${JOB_ID:-}" --note "$2" || true
}
trap 'digest_event fail "p2 final gate eval exited rc=$?"' ERR
source "$QG_ROOT/qg-env/bin/activate"
cd "$BRANCH/training"
digest_event start "final ep20 eval+gate, both trust arms, CPU"
TAG="final_ep20"
ROOTS=$(python -c "import json; print(' '.join(json.load(open('$D/deriv7_cond_local_w31/config.json'))['roots']))")
for a in rollout_ft_w31_p2_trust rollout_ft_w31_p2_trust_vn05; do
    [ -f "$D/$a/best.pt" ] || { echo "no best.pt for $a"; continue; }
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
echo "Subject: [QG][MONITOR][wiener] trust arms FINAL per-root tables + gates (${TAG})"
echo
echo "PARAMETERS: eval_deriv_by_root (41 roots, grad-kernel 31, CPU) + 10%"
echo "Nddot gate vs w31-ep32, on the FINAL best.pt of both finished arms"
echo "(trainers done 07-18 ~11:51, 20/20 epochs; vn 0.1 vs 0.5, trust"
echo "anchor both). The ep20 auto-gate had not fired (watcher bug) -- this"
echo "is the promised final table."
echo
for a in rollout_ft_w31_p2_trust rollout_ft_w31_p2_trust_vn05; do
    echo "== $a"; tail -8 "$D/$a/gate_${TAG}.txt" 2>/dev/null; echo
done
echo "DIRECTORIES: per-root CSVs $D/<arm>/eval_by_root_val.csv; gate texts"
echo "$D/<arm>/gate_${TAG}.txt; training logs qg-wiener-conditioning/logs/."
echo
echo "CONTEXT: mid-training gates FAILED both arms (snap1455) and the 07-18"
echo "long rollouts show both arms blowing up mid-horizon (see catch-up"
echo "report). This final table completes the record for your vn ruling."
} > "$SP/$(date +%Y%m%dT%H%M%S)_p2_final_gates.mail"
digest_event done "final gates written: gate_${TAG}.txt both arms; mail queued"
echo "[p2final] done"
