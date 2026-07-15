#!/bin/bash
# piff_tool_job.sh - run one ml_closure tool script on GPU: $1 = script name
# (e.g. calibrate_piff.py), rest = its args. Same env prep as piff_eval_job.sh.
#
# Usage:
#   LOGS=/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-sgs-closure/logs
#   qsub -N <name> -q ibgpu.q -l gpu=1 \
#        -o "$LOGS/\$JOB_NAME.\$JOB_ID.log" -j y -cwd -V \
#        piff_tool_job.sh calibrate_piff.py --ckpt runs_piff/<name>/best.pt

#$ -S /bin/bash
#$ -cwd
#$ -V

set -e

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
BRANCH="$QG_ROOT/qg-sgs-closure"

# worktree-capable git first (default 1.8.3.1 cannot parse linked worktrees;
# the digest push inside tools needs this on compute nodes too)
[[ -d /opt/rocks/bin ]] && export PATH=/opt/rocks/bin:$PATH

source "$QG_ROOT/qg-env-piff/bin/activate"
export TMPDIR="$QG_ROOT/tmp"
export MPLCONFIGDIR="$QG_ROOT/mplcache"
export PYTHONUNBUFFERED=1

mkdir -p "$QG_ROOT/cache/torch" "$QG_ROOT/cache/triton" "$QG_ROOT/cache/nvrtc"
export TORCH_EXTENSIONS_DIR="$QG_ROOT/cache/torch"
export TRITON_CACHE_DIR="$QG_ROOT/cache/triton"
export PYTORCH_KERNEL_CACHE_PATH="$QG_ROOT/cache/nvrtc"

if command -v nvidia-smi >/dev/null 2>&1; then
    IDLE_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t',' -k2 -n | head -1 | awk -F',' '{gsub(/ /,""); print $1}')
    export CUDA_VISIBLE_DEVICES="$IDLE_GPU"
    echo "[piff_tool] selected GPU $IDLE_GPU on $HOSTNAME"
fi

TOOL="${1:?usage: piff_tool_job.sh <script.py> [args...]}"
shift

# I23b opt-in (v1.4 retrofit): submitters that pass -v QG_DIGEST_RUN=<name>
# get start/fail digest events even if the tool crashes before its own digest
# emission (G5 advisory 2026-07-15). No-op for all existing callers.
digest_event() {
    [[ -n "${QG_DIGEST_RUN:-}" && -f "$BRANCH/diagnostics/digest_writer.py" ]] && \
        python "$BRANCH/diagnostics/digest_writer.py" --repo-dir "$BRANCH" \
            --run-name "$QG_DIGEST_RUN" --event "$1" --job-id "${JOB_ID:-}" \
            --note "$2" || true
}
trap 'digest_event fail "$TOOL exited rc=$? -- raw log in <branch>/logs/"' ERR

cd "$BRANCH/ml_closure"
echo "[piff_tool] host $HOSTNAME date $(date -u +%FT%TZ)"
echo "[piff_tool] cmd: python -u $TOOL $*"
echo "----------------------------------------------------------------------"
digest_event start "$TOOL $*"
python -u "$TOOL" "$@"
echo "----------------------------------------------------------------------"
echo "[piff_tool] done at $(date -u +%FT%TZ)"
