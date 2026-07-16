#!/bin/bash
# postft_eval_job.sh -- post-landing eval + acceptance gates for the FT arms
# (Sanaa standing order 2026-07-15). CPU per the diagnostics-never-GPU ruling.
# [fable-authored 2026-07-16; replaces the quoting-mangled inline 1835423]
#$ -S /bin/bash
#$ -q all.q
#$ -j y
#$ -cwd
set -uo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
D=$QG_ROOT/qg-simple-package-stable/src/qg/training/data/ensemble_N5_7lag/training_runs
SPOOL=$QG_ROOT/reporting/pending_mail
source "$QG_ROOT/qg-env/bin/activate"
cd "$QG_ROOT/qg-wiener-conditioning/training"

# the a-priori pool = the warm ckpt's own 41 roots (config.json, relative to
# this training dir via the data symlink)
ROOTS=$(python -c "import json; print(' '.join(json.load(open('$D/deriv7_cond_local_w31/config.json'))['roots']))")
for a in rollout_ft_w31_p1a rollout_ft_w31_p1b; do
    ck=$D/$a/best.pt
    [ -f "$ck" ] || ck=$D/$a/last.pt
    echo "== eval $a ($ck)"
    # shellcheck disable=SC2086
    python -u eval_deriv_by_root.py --ckpt "$ck" --sweep-roots $ROOTS \
        --grad-kernel 31 --device cpu || echo "EVAL_FAIL_$a"
done
for a in rollout_ft_w31_p1a rollout_ft_w31_p1b; do
    python -u accept_ft_gate.py \
        --before "$D/deriv7_cond_local_w31/eval_by_root_val.csv" \
        --after  "$D/$a/eval_by_root_val.csv" --tol-rel 0.10 \
        > "$D/$a/gate_verdict.txt" 2>&1
    echo "$a gate rc=$? (0 PASS / 3 FAIL)"
done

mkdir -p "$SPOOL"
STAMP=$(date +%Y%m%dT%H%M%S)
{
  echo "To: ${QG_NOTIFY_EMAIL:-sanaamz@mit.edu}"
  echo "Subject: [QG][MONITOR][wiener] FT arms: per-root eval + 10% Nddot gates"
  echo
  for a in rollout_ft_w31_p1a rollout_ft_w31_p1b; do
      echo "== $a"; tail -12 "$D/$a/gate_verdict.txt"; echo
  done
  echo "NEXT: pick the passing arm (A preferred if both pass); OOD read on"
  echo "combo/b25 in the eval CSVs; then promotion ruling."
} > "$SPOOL/${STAMP}_postft_gates.mail"
echo "[postft] done"
