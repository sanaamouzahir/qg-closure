#!/bin/bash
# piff_tests_job.sh - ML SPEC 01 S5 gate tests, two-part worker.
#   arm "cpu": T1-T4 (loader hash, normalization identity, FiLM identity init,
#              periodic-pad equivariance) -> all.q (tests/sidecar only; Pi_FF
#              COMPUTE never runs on all.q)
#   arm "gpu": T5-T6 (SVGP synthetic noise/coverage, 500-crop overfit) -> ibgpu.q
# T7 is piff_smoke_job.sh, not pytest.
#
# Venv: qg-env-piff (the CP-ML-1 clone with gpytorch; NEVER the shared qg-env).
# All tests run as batch jobs, never on the frontend (spec S5).
#
# Usage (absolute -o path — do not rely on submit cwd):
#   LOGS=/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-sgs-closure/logs
#   qsub -N piffT_cpu -q all.q \
#        -o "$LOGS/\$JOB_NAME.\$JOB_ID.log" -j y -cwd -V piff_tests_job.sh cpu
#   qsub -N piffT_gpu -q ibgpu.q -l gpu=1 \
#        -o "$LOGS/\$JOB_NAME.\$JOB_ID.log" -j y -cwd -V piff_tests_job.sh gpu

#$ -S /bin/bash
#$ -cwd
#$ -V

set -e

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
BRANCH="$QG_ROOT/qg-sgs-closure"

source "$QG_ROOT/qg-env-piff/bin/activate"
export TMPDIR="$QG_ROOT/tmp"
export MPLCONFIGDIR="$QG_ROOT/mplcache"
export PYTHONUNBUFFERED=1

ARM="${1:?usage: piff_tests_job.sh cpu|gpu|t8}"
case "$ARM" in
    cpu) KEXPR="t1 or t2 or t3 or t4 or t8b" ;;   # t8b: zeta_dot const-member zero (data, no GPU)
    t8)  KEXPR="t8" ;;                             # ORDER-3 conditioning gates, CPU-safe (all.q)
    gpu) KEXPR="t5 or t6"
         if command -v nvidia-smi >/dev/null 2>&1; then
             IDLE_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
                 | sort -t',' -k2 -n | head -1 | awk -F',' '{gsub(/ /,""); print $1}')
             export CUDA_VISIBLE_DEVICES="$IDLE_GPU"
             echo "[piff_tests] selected GPU $IDLE_GPU on $HOSTNAME"
         fi ;;
    *) echo "ERROR: arm must be cpu or gpu, got '$ARM'"; exit 2 ;;
esac

cd "$BRANCH/ml_closure"
echo "[piff_tests] host $HOSTNAME arm $ARM date $(date -u +%FT%TZ)"
echo "[piff_tests] python $(which python); pytest -k \"$KEXPR\""
echo "----------------------------------------------------------------------"
python -m pytest tests_piff.py -k "$KEXPR" -v -rs
echo "----------------------------------------------------------------------"
echo "[piff_tests] arm $ARM done at $(date -u +%FT%TZ)"
