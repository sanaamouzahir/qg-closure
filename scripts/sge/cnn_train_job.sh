#!/bin/bash
# cnn_train_job.sh - CNN-only Pi_FF training (Sanaa order 2026-07-22):
# forwards args verbatim to train_cnn.py. Mirror of piff_train_job.sh
# (same env, digest events, idle-GPU pick); GPU jobs -q ibgpu.q -l gpu=1 ONLY.
#
# Usage:
#   qsub -N pCNN_<tag> -q ibgpu.q -l gpu=1 -m ea -M $QG_NOTIFY_EMAIL \
#        -o logs/\$JOB_NAME.\$JOB_ID.log -j y -cwd -V \
#        cnn_train_job.sh --config conf_piff_fpc_cnn.yaml --run-name <name>

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
digest_event() {  # I23b; no-op if digest_writer not yet on this checkout
    [[ -f "$DIGEST" && -n "$RUN_NAME" ]] && \
        python "$DIGEST" --repo-dir "$BRANCH" --run-name "$RUN_NAME" \
            --event "$1" --job-id "${JOB_ID:-}" --note "$2" || true
}
trap 'digest_event fail "cnn train exited rc=$?"' ERR

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
    echo "[cnn_train] selected GPU $IDLE_GPU on $HOSTNAME"
fi

cd "$BRANCH/ml_closure"
echo "[cnn_train] host $HOSTNAME date $(date -u +%FT%TZ)"
echo "[cnn_train] cmd: python -u train_cnn.py $*"
echo "----------------------------------------------------------------------"
digest_event start "cnn-only training launched: $*"
python -u train_cnn.py "$@"
digest_event done "cnn-only training complete"
echo "----------------------------------------------------------------------"
echo "[cnn_train] done at $(date -u +%FT%TZ)"
