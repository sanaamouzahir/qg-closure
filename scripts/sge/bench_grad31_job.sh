#!/bin/bash
# bench_grad31_job.sh - one-shot GPU cost benchmark for the width-31 decision
# (diagnostics/bench_grad31_cost.py). Runs FROM the worktree training/ (flat
# sibling imports + data symlink). GPU rule: exactly -q ibgpu.q -l gpu=1.
#$ -S /bin/bash
#$ -cwd
#$ -j y

set -euo pipefail
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
source "$QG_ROOT/qg-env/bin/activate"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
cd "$QG_ROOT/qg-wiener-conditioning/training"
echo "[bench_grad31_job] host $HOSTNAME  cuda ${CUDA_VISIBLE_DEVICES:-<not set>}  $(date -u +%FT%TZ)"
python -u ../diagnostics/bench_grad31_cost.py
echo "[bench_grad31_job] done $(date -u +%FT%TZ)"
