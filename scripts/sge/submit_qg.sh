#!/bin/bash
# submit_qg.sh - Convenience wrapper around qsub for QG simulations.
#
# Usage:
#   ./submit_qg.sh <jobname> [--gpu] -- <args-to-run_qg.py>
#
# Examples:
#   ./submit_qg.sh cape_v3_512_T20 -- \
#       scenario=flow_past_cape qg.grid.Nx=512 qg.grid.Ny=512 qg.time.T=20 \
#       hydra.run.dir=outputs/cape_v3_512_T20
#
#   ./submit_qg.sh cape_v3_1024_gpu --gpu -- \
#       scenario=flow_past_cape qg.grid.Nx=1024 qg.grid.Ny=1024 +qg.grid.device=cuda \
#       qg.time.T=20 hydra.run.dir=outputs/cape_v3_1024_gpu

set -e

if [ "$#" -lt 3 ]; then
    cat <<USAGE
Usage: $0 <jobname> [--gpu] -- <args-to-run_qg.py>

Examples:
  $0 cape_v3_512_T20 -- scenario=flow_past_cape qg.grid.Nx=512 qg.grid.Ny=512 \\
      qg.time.T=20 hydra.run.dir=outputs/cape_v3_512_T20

  $0 cape_v3_1024_gpu --gpu -- scenario=flow_past_cape qg.grid.Nx=1024 \\
      qg.grid.Ny=1024 +qg.grid.device=cuda qg.time.T=20 \\
      hydra.run.dir=outputs/cape_v3_1024_gpu
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
    echo "Error: missing '--' separator before run_qg.py args"
    exit 1
fi
shift

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
LOG_DIR="$QG_ROOT/qg-wiener-conditioning/logs"
mkdir -p "$LOG_DIR"

LOG="$LOG_DIR/${JOBNAME}.log"
JOB_SCRIPT="$QG_ROOT/qg-simple-package-stable/src/qg/qg_job.sh"

QSUB_FLAGS=(
    -N "$JOBNAME"
    -o "$LOG"
    -e "$LOG"
    -j y                  # merge stderr into stdout
    -cwd
    -V
    -m ea
    -M "${QG_NOTIFY_EMAIL:-sanaamz@mit.edu}"
)

if [ "$USE_GPU" -eq 1 ]; then
    # Pick a queue that targets ibgpu nodes. Adjust if your cluster uses a
    # different queue name or GPU resource attribute.
    QSUB_FLAGS+=(-q ibgpu.q)
    QSUB_FLAGS+=(-l "gpu=1")
else
    # CPU job (hard rule: the amd queue and per-job vmem requests are forbidden;
    # all.q = the cluster's CPU queue, same convention as the monitor sidecars).
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
