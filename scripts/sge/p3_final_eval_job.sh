#!/bin/bash
# p3_final_eval_job.sh -- FINAL post-training eval of the w31p3 certv2 arm
# (per-root a-priori table + TIERED gate) on its finished best.pt. Runs
# HELD on the p3 trainer (fired at submit time by submit_w31_p4_chain.sh);
# the finalize-watcher stale-marker bug is still routed around. CPU
# (diagnostics rule). Derived from p2_final_eval_job.sh (single arm; the
# gate is the tiered accept_ft_gate: PASS/PASS-conditional/REGRESSED).
# [fable-authored 2026-07-20]
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
RUN_NAME=w31p3_finalgate
digest_event() {  # I23b; no-op if digest_writer not on this checkout
    [[ -f "$DIGEST" && -n "$RUN_NAME" ]] && \
        python "$DIGEST" --repo-dir "$BRANCH" --run-name "$RUN_NAME" \
            --event "$1" --job-id "${JOB_ID:-}" --note "$2" || true
}
trap 'digest_event fail "p3 final gate eval exited rc=$?"' ERR
source "$QG_ROOT/qg-env/bin/activate"
cd "$BRANCH/training"
digest_event start "p3 final ep20 eval + tiered gate, CPU"
TAG="final_ep20"
A=rollout_ft_w31p3_certv2
ROOTS=$(python -c "import json; print(' '.join(json.load(open('$D/deriv7_cond_local_w31/config.json'))['roots']))")
[ -f "$D/$A/best.pt" ] || { echo "no best.pt for $A"; digest_event fail "no best.pt"; exit 4; }
cp "$D/$A/best.pt" "$D/$A/best_${TAG}.pt"
# shellcheck disable=SC2086
python -u eval_deriv_by_root.py --ckpt "$D/$A/best_${TAG}.pt" \
    --sweep-roots $ROOTS --grad-kernel 31 --device cpu \
    || echo "EVAL_FAIL_$A"
python -u accept_ft_gate.py \
    --before "$D/deriv7_cond_local_w31/eval_by_root_val.csv" \
    --after  "$D/$A/eval_by_root_val.csv" --tol-rel 0.10 \
    > "$D/$A/gate_${TAG}.txt" 2>&1 || true
mkdir -p "$SP"
{ echo "To: ${QG_NOTIFY_EMAIL:-sanaamz@mit.edu}"
  echo "Subject: [QG][REPORT][wiener-conditioning] w31p3 FINAL tiered gate table (${TAG})"
  echo
  cat "$D/$A/gate_${TAG}.txt"
} > "$SP/$(date +%Y%m%dT%H%M%S)_w31p3_finalgate.mail"
digest_event done "p3 final gate written: gate_${TAG}.txt; table mail spooled"
