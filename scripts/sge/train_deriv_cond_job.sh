#!/bin/bash
# train_deriv_cond_job.sh - SGE worker for train_deriv.py ON THE WORKTREE.
#
# WHY a branch-specific wrapper: the shared train_deriv_job.sh cd's into the
# package-stable checkout ($QG_ROOT/qg-simple-package-stable/src/qg/training),
# which does NOT carry this branch's cond_deriv code. cond_deriv lives only in
# the exp/wiener-conditioning worktree, so we run FROM the worktree training/.
# Data + outputs still flow through training/data -> the shared ensemble
# (symlink), so runs land in the same data/ensemble_N5_7lag/training_runs/.
#
# Submit (GPU rule: exactly -q ibgpu.q -l gpu=1; never -l h_vmem, never ibamd.q):
#   qsub -q ibgpu.q -l gpu=1 -N deriv7_cond \
#        scripts/sge/train_deriv_cond_job.sh --model cond_deriv <args...>
#$ -o /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/$JOB_NAME.$JOB_ID.log
#$ -e /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/$JOB_NAME.$JOB_ID.err
set -e
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
source "$QG_ROOT/qg-env/bin/activate"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# Run from the WORKTREE training/ so the cond_deriv code is the code that runs.
SCRIPT_DIR="$QG_ROOT/qg-wiener-conditioning/training"
cd "$SCRIPT_DIR"

echo "[train_deriv_cond_job] hostname: $HOSTNAME"
echo "[train_deriv_cond_job] date:     $(date -u +%FT%TZ)"
echo "[train_deriv_cond_job] cuda dev: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "[train_deriv_cond_job] cwd:      $PWD  (worktree)"
echo "[train_deriv_cond_job] git HEAD: $(/opt/rocks/bin/git rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "[train_deriv_cond_job] args:     $(printf '%s ' "$@" | grep -oE '\-\-[a-z-]+ [^ ]*' | grep -vE 'sweep-roots' )"
echo "------------------------------------------------------------"

python -u train_deriv.py "$@"

echo "------------------------------------------------------------"
echo "[train_deriv_cond_job] done at $(date -u +%FT%TZ)"
