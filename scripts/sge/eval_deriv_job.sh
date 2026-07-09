#!/bin/bash
# eval_deriv_job.sh - SGE worker for training/eval_deriv_by_root.py ON THE
# WORKTREE (per-(member,dt)xorder rel-L2 of a ckpt; CSV lands next to the
# ckpt). Same pattern as train_deriv_condlocal_job.sh: run FROM the worktree
# training/ so the patched eval code (--model auto: reads config.json next to
# the ckpt, supports cond_local) is the code that runs; data + ckpts flow
# through the training/data symlink.
#
# GPU rule: exactly -q ibgpu.q -l gpu=1; never -l h_vmem, never ibamd.q.
# Submit from the worktree root:
#   qsub -q ibgpu.q -l gpu=1 -N evC_<tag> \
#        scripts/sge/eval_deriv_job.sh \
#        --ckpt data/ensemble_N5_7lag/training_runs/<run>/best.pt \
#        --sweep-roots data/ensemble_N5_7lag/... [...] \
#        --n-snapshots 7 --out-orders 3 --grad-kernel 15 --compute-dtype float64

#$ -o /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/$JOB_NAME.$JOB_ID.log
#$ -e /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/$JOB_NAME.$JOB_ID.err
#$ -S /bin/bash
#$ -cwd
#$ -V
set -e
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
source "$QG_ROOT/qg-env/bin/activate"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$QG_ROOT/qg-wiener-conditioning/training"
cd "$SCRIPT_DIR"

echo "[eval_deriv_job] hostname: $HOSTNAME"
echo "[eval_deriv_job] date:     $(date -u +%FT%TZ)"
echo "[eval_deriv_job] cuda dev: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "[eval_deriv_job] cwd:      $PWD  (worktree)"
echo "[eval_deriv_job] git HEAD: $(/opt/rocks/bin/git rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "------------------------------------------------------------"

python -u eval_deriv_by_root.py "$@"

echo "------------------------------------------------------------"
echo "[eval_deriv_job] done at $(date -u +%FT%TZ)"
