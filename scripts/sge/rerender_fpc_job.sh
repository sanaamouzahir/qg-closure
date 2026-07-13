#!/bin/bash
# rerender_fpc_job.sh - CPU worker: re-render FR videos for one FPC Phase-B
# wave-2 run dir via rerender_videos.py (jpcm.draw + ffmpeg, CPU-only, no
# torch/GPU). Queue: all.q. Never the forbidden queue/memory-reservation
# flags.
#
# rerender_videos.py now mmap-loads {name}.npy (was: full np.load, which is
# what caused the FPC-sine in-sim MemoryError -- 33 GiB array already
# resident from the solver + another 8.34 GiB contiguous slice needed).
# mmap keeps peak RSS to ~the single materialized slice (~8-9 GiB).
#
# Usage (submit from qg-sgs-closure root so logs land in logs/):
#   qsub -q all.q -N rerender_<tag> \
#        -o "$PWD/logs/\$JOB_NAME.\$JOB_ID.log" -j y -cwd -V \
#        scripts/sge/rerender_fpc_job.sh <run-dir> [extra args to rerender_videos.py]

#$ -S /bin/bash
#$ -cwd
#$ -V

set -e

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR="$QG_ROOT/qg-simple-package-stable/src/qg"
RERENDER="$QG_DIR/rerender_videos.py"

source "$QG_ROOT/qg-env/bin/activate"
export TMPDIR="$QG_ROOT/tmp"
export MPLCONFIGDIR="$QG_ROOT/mplcache"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

echo "[rerender_fpc_job] hostname: $HOSTNAME"
echo "[rerender_fpc_job] date: $(date -u +%FT%TZ)"
echo "[rerender_fpc_job] args: $*"
echo "----------------------------------------------------------------------"

python -u "$RERENDER" "$@"

echo "----------------------------------------------------------------------"
echo "[rerender_fpc_job] done at $(date -u +%FT%TZ)"
