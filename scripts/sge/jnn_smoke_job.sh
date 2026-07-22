#!/bin/bash
#$ -S /bin/bash
#$ -q ibgpu.q
#$ -l gpu=1
#$ -j y
#$ -cwd
#$ -V
QG=/gdata/projects/ml_scope/Closure_modeling/QG-closure
[[ -d /opt/rocks/bin ]] && export PATH=/opt/rocks/bin:$PATH
source "$QG/qg-env/bin/activate"
if command -v nvidia-smi >/dev/null 2>&1; then
  sleep $((RANDOM % 5))
  GPUDIR="$QG/tmp/gpu_locks"; mkdir -p "$GPUDIR"
  for d in "$GPUDIR"/gpu_*.lock; do
    [ -d "$d" ] || continue
    p=$(cat "$d/pid" 2>/dev/null)
    { [ -n "$p" ] && kill -0 "$p" 2>/dev/null; } || rm -rf "$d"
  done
  CANDS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F',' '{u=$2+0; if(u<2000) print $1+0}' | shuf)
  for g in $CANDS; do
    if mkdir "$GPUDIR/gpu_$g.lock" 2>/dev/null; then
      echo $$ > "$GPUDIR/gpu_$g.lock/pid"
      export CUDA_VISIBLE_DEVICES=$g
      trap "rm -rf '$GPUDIR/gpu_$g.lock'" EXIT
      break
    fi
  done
  echo "[jnn_smoke] GPU $CUDA_VISIBLE_DEVICES on $HOSTNAME"
fi
CK="$QG/qg-simple-package-stable/src/qg/training/data/ensemble_N5_7lag/training_runs/rollout_ft_w31p3_certv2/best.pt"
cd "$QG/qg-wiener-conditioning/training"
python -u train_deriv_rollout.py \
  --deep-roots data/ensemble_N5/FRC-kf4/forced_turbulence_dT_5em3 \
  --strides 1 --init-ckpt "$CK" \
  --windows-per-epoch 2 --epochs 1 \
  --vn-mode jnn --vn-lambda 0.5 --vn-developed-steps 10 --run-name jnn_smoke
