#!/bin/bash
# gp_residual_job.sh - residual-GP head training on a frozen plateaued CNN
# (Sanaa GO 2026-07-22): forwards args verbatim to train_gp_residual.py.
# Mirror of cnn_train_job.sh; GPU jobs -q ibgpu.q -l gpu=1 ONLY.
#
# Usage:
#   qsub -N pGPr_<tag> -q ibgpu.q -l gpu=1 -m ea -M $QG_NOTIFY_EMAIL \
#        -o logs/\$JOB_NAME.\$JOB_ID.log -j y -cwd -V \
#        gp_residual_job.sh --cnn-ckpt runs_piff/<plateau>/best.pt --run-name <name>

#$ -S /bin/bash
#$ -cwd
#$ -V

set -e

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
BRANCH="$QG_ROOT/qg-sgs-closure"
DIGEST="$BRANCH/diagnostics/digest_writer.py"
RUN_NAME=""
prev=""
for a in "$@"; do [[ "$prev" == "--run-name" ]] && RUN_NAME=$a; prev=$a; done
digest_event() {
    [[ -f "$DIGEST" && -n "$RUN_NAME" ]] && \
        python "$DIGEST" --repo-dir "$BRANCH" --run-name "$RUN_NAME" \
            --event "$1" --job-id "${JOB_ID:-}" --note "$2" || true
}
trap 'digest_event fail "gp-residual train exited rc=$?"' ERR

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
    echo "[gp_residual] selected GPU $IDLE_GPU on $HOSTNAME"
fi

cd "$BRANCH/ml_closure"
echo "[gp_residual] host $HOSTNAME date $(date -u +%FT%TZ)"
echo "[gp_residual] cmd: python -u train_gp_residual.py $*"
echo "----------------------------------------------------------------------"
digest_event start "gp-residual training launched: $*"
python -u train_gp_residual.py "$@"
digest_event done "gp-residual training complete"
echo "----------------------------------------------------------------------"
echo "[gp_residual] done at $(date -u +%FT%TZ)"
