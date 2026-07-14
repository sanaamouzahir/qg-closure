#!/bin/bash
# wiener_tool_job.sh - run one wiener-worktree python tool on GPU:
#   $1 = script path RELATIVE TO training/ (e.g. eval_deriv_by_root.py or
#        ../diagnostics/spectral_error_profile.py), rest = its args.
# Mirrors qg-sgs-closure/scripts/sge/piff_tool_job.sh (idle-GPU picker).
#
# Submit: qsub -q ibgpu.q -l gpu=1 -N <name> -o logs/<name>.\$JOB_ID.log \
#              -j y -cwd -V scripts/sge/wiener_tool_job.sh <tool.py> [args...]

#$ -S /bin/bash
#$ -cwd
#$ -V

set -e
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
WT=$QG_ROOT/qg-wiener-conditioning
source $QG_ROOT/qg-env/bin/activate
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR=$QG_ROOT/.mplcache; mkdir -p "$MPLCONFIGDIR"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

if command -v nvidia-smi >/dev/null 2>&1; then
    IDLE_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t',' -k2 -n | head -1 | awk -F',' '{gsub(/ /,""); print $1}')
    export CUDA_VISIBLE_DEVICES="$IDLE_GPU"
    echo "[wiener_tool] selected GPU $IDLE_GPU on $HOSTNAME"
fi

TOOL="${1:?usage: wiener_tool_job.sh <script.py> [args...]}"
shift
cd "$WT/training"
echo "[wiener_tool] host $HOSTNAME date $(date -u +%FT%TZ)"
echo "[wiener_tool] cmd: python -u $TOOL $*"
echo "----------------------------------------------------------------------"
python -u "$TOOL" "$@"
echo "----------------------------------------------------------------------"
echo "[wiener_tool] done at $(date -u +%FT%TZ)"
