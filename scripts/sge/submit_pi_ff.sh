#!/bin/bash
# submit_pi_ff.sh - Convenience wrapper around qsub for compute_pi_ff.py.
#
# Usage:
#   ./submit_pi_ff.sh <jobname> [--gpu] -- <args-to-compute_pi_ff.py>
#
# Examples:
#   ./submit_pi_ff.sh pi_beta0p0 --gpu -- \
#       outputs/cape_sweep/beta_0p0 --scale 2 --alpha 1.5 --device cuda
#
#   ./submit_pi_ff.sh pi_beta1p0_cpu -- \
#       outputs/cape_sweep/beta_1p0 --scale 2 --alpha 1.5 --device cpu

set -e

if [ "$#" -lt 3 ]; then
    cat <<USAGE
Usage: $0 <jobname> [--gpu] -- <args-to-compute_pi_ff.py>

Examples:
  $0 pi_beta0p0 --gpu -- outputs/cape_sweep/beta_0p0 \\
      --scale 2 --alpha 1.5 --device cuda

  $0 pi_beta1p0_cpu -- outputs/cape_sweep/beta_1p0 \\
      --scale 2 --alpha 1.5 --device cpu
USAGE
    exit 1
fi

JOBNAME="$1"
shift

USE_GPU=0
if [ "$1" = "--gpu" ]; then
    USE_GPU=1
    shift
fi

if [ "$1" != "--" ]; then
    echo "Error: missing '--' separator before compute_pi_ff.py args"
    exit 1
fi
shift

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
LOG_DIR="$QG_ROOT/qg-wiener-conditioning/logs"
mkdir -p "$LOG_DIR"
export PYTHONDONTWRITEBYTECODE=1
LOG="$LOG_DIR/${JOBNAME}.log"
JOB_SCRIPT="$QG_ROOT/qg-simple-package-stable/src/qg/compute_pi_ff_job.sh"

QSUB_FLAGS=(
    -N "$JOBNAME"
    -o "$LOG"
    -e "$LOG"
    -j y
    -cwd
    -V
)

if [ "$USE_GPU" -eq 1 ]; then
    QSUB_FLAGS+=(-q ibgpu.q)
    QSUB_FLAGS+=(-l "gpu=1")
else
    QSUB_FLAGS+=(-q "all.q")
fi

echo "Submitting job '$JOBNAME'"
echo "  log: $LOG"
echo "  args: $*"
echo

qsub "${QSUB_FLAGS[@]}" "$JOB_SCRIPT" "$@"

echo
echo "Watch progress with:"
echo "  tail -f $LOG"
echo
echo "Check job status:"
echo "  qstat -u \$USER"
