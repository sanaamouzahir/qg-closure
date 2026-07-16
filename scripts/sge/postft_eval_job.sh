#!/bin/bash
# postft_eval_job.sh -- post-landing eval + acceptance gates for the FT arms
# (Sanaa standing order 2026-07-15). CPU per the diagnostics-never-GPU ruling.
# [fable-authored 2026-07-16; replaces the quoting-mangled inline 1835423]
#
# Modes (first arg):
#   eval <arm-dir-name>  : per-root eval of that arm's best.pt (own CSV dir,
#                          so the two arms run as CONCURRENT jobs -- the
#                          serial both-arms job was ~11 h on one CPU slot)
#   gate                 : both 10%-Nddot acceptance gates + spooled email
#                          (submit with -hold_jid on both eval jobs)
#$ -S /bin/bash
#$ -q all.q
#$ -j y
#$ -cwd
#$ -o /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/$JOB_NAME.o$JOB_ID
#$ -m ea
#$ -M sanaamz@mit.edu
set -uo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONUNBUFFERED=1

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
D=$QG_ROOT/qg-simple-package-stable/src/qg/training/data/ensemble_N5_7lag/training_runs
SPOOL=$QG_ROOT/reporting/pending_mail
source "$QG_ROOT/qg-env/bin/activate"
cd "$QG_ROOT/qg-wiener-conditioning/training"

MODE=${1:-eval_both}

if [ "$MODE" = "eval" ]; then
    a=$2
    # the a-priori pool = the warm ckpt's own 41 roots (config.json, relative
    # to this training dir via the data symlink)
    ROOTS=$(python -c "import json; print(' '.join(json.load(open('$D/deriv7_cond_local_w31/config.json'))['roots']))")
    ck=$D/$a/best.pt
    [ -f "$ck" ] || ck=$D/$a/last.pt
    echo "== eval $a ($ck)"
    # NB the arms' config.json records grad_kernel 15 (stale CLI default from
    # the FT trainer); the ckpt state_dict is 31x31 -- pass it explicitly.
    # shellcheck disable=SC2086
    python -u eval_deriv_by_root.py --ckpt "$ck" --sweep-roots $ROOTS \
        --grad-kernel 31 --device cpu || echo "EVAL_FAIL_$a"
    exit 0
fi

if [ "$MODE" = "gate" ]; then
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
    echo "[postft] gates done"
    exit 0
fi

echo "[postft] unknown mode '$MODE' (use: eval <arm> | gate)"; exit 4
