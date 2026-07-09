#!/bin/bash
# diag_job.sh - generic SGE worker for wiener-branch diagnostics/ scripts that
# import training/ siblings (deriv_dataset, model_deriv_closure, ...).
# Runs FROM the worktree training/ with the diagnostics dir handled by the
# script-dir sys.path rule + PYTHONPATH for the training siblings.
#
# Usage:
#   qsub -q ibgpu.q -l gpu=1 -N dg_<tag> scripts/sge/diag_job.sh \
#        <script-name-in-diagnostics/> [args...]
# CPU variants: -q all.q (only for scripts that don't need CUDA).

#$ -o /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/$JOB_NAME.$JOB_ID.log
#$ -e /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/$JOB_NAME.$JOB_ID.err
#$ -S /bin/bash
#$ -cwd
#$ -V
set -e
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
source "$QG_ROOT/qg-env/bin/activate"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

WT="$QG_ROOT/qg-wiener-conditioning"
SCRIPT="$1"; shift
cd "$WT/training"
export PYTHONPATH="$WT/training:${PYTHONPATH:-}"

echo "[diag_job] host $HOSTNAME  date $(date -u +%FT%TZ)"
echo "[diag_job] script: $SCRIPT  args: $*"
echo "[diag_job] git HEAD: $(/opt/rocks/bin/git rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "------------------------------------------------------------"

python -u "$WT/diagnostics/$SCRIPT" "$@"

echo "------------------------------------------------------------"
echo "[diag_job] done $(date -u +%FT%TZ)"
