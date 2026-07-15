#!/bin/bash
# piff_train_job.sh - one Pi_FF grid point (ML SPEC 01 S3.2): forwards args
# verbatim to train_piff.py. Submitted 6x by submit_piff_grid.sh
# (lr x weight-decay grid), one GPU job each: -q ibgpu.q -l gpu=1 ONLY.
#
# CHARTER v1.4 retrofit (T4, 2026-07-15): raw logs -> <branch>/logs/ (I23a,
# never committed); start/done/fail digest events -> reports/<run-name>/
# (I23b, pushed via diagnostics/digest_writer.py); LIVE+FINALIZE monitors
# chained per I18a must carry the I24 reflex ladder (monitor v3 on main --
# supervisor confirms adoption in the next digest). Day-mode submissions via
# the I21c ssh sequence ONLY.
#
# Usage:
#   qsub -N piff_<tag> -q ibgpu.q -l gpu=1 -m ea -M $QG_NOTIFY_EMAIL \
#        -o logs/\$JOB_NAME.\$JOB_ID.log -j y -cwd -V \
#        piff_train_job.sh --run-name <name> --lr 3.0e-4 --weight-decay 1.0e-4

#$ -S /bin/bash
#$ -cwd
#$ -V

set -e

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
BRANCH="$QG_ROOT/qg-sgs-closure"
DIGEST="$BRANCH/diagnostics/digest_writer.py"
# run-name for the digest: value following --run-name in the forwarded args
RUN_NAME=""
prev=""
for a in "$@"; do [[ "$prev" == "--run-name" ]] && RUN_NAME=$a; prev=$a; done
digest_event() {  # I23b; no-op if digest_writer not yet on this checkout
    [[ -f "$DIGEST" && -n "$RUN_NAME" ]] && \
        python "$DIGEST" --repo-dir "$BRANCH" --run-name "$RUN_NAME" \
            --event "$1" --job-id "${JOB_ID:-}" --note "$2" || true
}
trap 'digest_event fail "piff train exited rc=$?"' ERR

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
    echo "[piff_train] selected GPU $IDLE_GPU on $HOSTNAME"
fi

cd "$BRANCH/ml_closure"
echo "[piff_train] host $HOSTNAME date $(date -u +%FT%TZ)"
echo "[piff_train] cmd: python -u train_piff.py $*"
echo "----------------------------------------------------------------------"
digest_event start "piff grid point launched: $*"
python -u train_piff.py "$@"
digest_event done "piff train complete"
echo "----------------------------------------------------------------------"
echo "[piff_train] done at $(date -u +%FT%TZ)"
