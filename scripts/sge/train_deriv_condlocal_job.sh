#!/bin/bash
# train_deriv_condlocal_job.sh - SGE worker for deriv7_cond_local ON THE WORKTREE.
#
# The DELIVERABLE conditioned model: cheap_deriv control pipeline + sigma-hat
# tap modulation (model_cond_local.py; inference = control + 2 FFTs/step).
# cond_local lives only in the exp/wiener-conditioning worktree, so we run
# FROM the worktree training/ (same pattern as train_deriv_cond_job.sh).
# Data + outputs flow through the training/data symlink -> shared ensemble;
# runs land in data/ensemble_N5_7lag/training_runs/.
#
# PREPARED 2026-07-07 (Sanaa work order) -- DO NOT SUBMIT before she gives the
# go; acceptance gate diagnostics/diagnose_condlocal_init.py must PASS first.
#
# Submit (GPU rule: exactly -q ibgpu.q -l gpu=1; never -l h_vmem, never ibamd.q):
#   cd $QG_ROOT/qg-wiener-conditioning
#   qsub -q ibgpu.q -l gpu=1 -N deriv7_cond_local \
#        scripts/sge/train_deriv_condlocal_job.sh \
#        --model cond_local \
#        --sweep-roots \
#          data/ensemble_N5_7lag/FRC-256/sweep_dT_1em2 \
#          data/ensemble_N5_7lag/FRC-256/sweep_dT_1p5em2 \
#          data/ensemble_N5_7lag/FRC-256/sweep_dT_5em3 \
#          data/ensemble_N5_7lag/FRC-b25/sweep_dT_1em2 \
#          data/ensemble_N5_7lag/FRC-b25/sweep_dT_1p5em2 \
#          data/ensemble_N5_7lag/FRC-b25/sweep_dT_5em3 \
#          data/ensemble_N5_7lag/FRC-b2/sweep_dT_1em2 \
#          data/ensemble_N5_7lag/FRC-b2/sweep_dT_1p5em2 \
#          data/ensemble_N5_7lag/FRC-b2/sweep_dT_5em3 \
#          data/ensemble_N5_7lag/FRC-combo/sweep_dT_1em2 \
#          data/ensemble_N5_7lag/FRC-combo/sweep_dT_1p5em2 \
#          data/ensemble_N5_7lag/FRC-combo/sweep_dT_5em3 \
#          data/ensemble_N5_7lag/FRC-kf4/sweep_dT_1em2 \
#          data/ensemble_N5_7lag/FRC-kf4/sweep_dT_1p5em2 \
#          data/ensemble_N5_7lag/FRC-kf4/sweep_dT_5em3 \
#          data/ensemble_N5_7lag/FRC-Re25k/sweep_dT_1em2 \
#          data/ensemble_N5_7lag/FRC-Re25k/sweep_dT_5em3 \
#        --n-snapshots 7 --out-orders 3 --grad-kernel 15 \
#        --epochs 300 --lr 5e-5 --batch-size 4 --compute-dtype float64 \
#        --rel-floor 0.1 --run-name deriv7_cond_local
#
# Root list = control config: ALL filtered members MINUS the Re25k dT=1.5e-2
# sweep root (past the Re25k convergence radius Delta_T* = 0.066: the 6*dt
# stencil span 0.09 exceeds it -- unlearnable by construction; approved drop).
# The new8 members' sweeps (FRC-b0..b1, DEC-*) can be APPENDED once their
# slice->resplit->filter chain lands and filter_quiescent_windows has run
# (rule: never train on unfiltered roots).
#$ -o /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-simple-package-stable/src/qg/logs/$JOB_NAME.$JOB_ID.log
set -e
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
source "$QG_ROOT/qg-env/bin/activate"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# Run from the WORKTREE training/ so the cond_local code is the code that runs.
SCRIPT_DIR="$QG_ROOT/qg-wiener-conditioning/training"
cd "$SCRIPT_DIR"

echo "[train_deriv_condlocal_job] hostname: $HOSTNAME"
echo "[train_deriv_condlocal_job] date:     $(date -u +%FT%TZ)"
echo "[train_deriv_condlocal_job] cuda dev: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "[train_deriv_condlocal_job] cwd:      $PWD  (worktree)"
echo "[train_deriv_condlocal_job] git HEAD: $(/opt/rocks/bin/git rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "------------------------------------------------------------"

python -u train_deriv.py "$@"

echo "------------------------------------------------------------"
echo "[train_deriv_condlocal_job] done at $(date -u +%FT%TZ)"
