#!/bin/bash
#$ -S /bin/bash
#$ -q ibgpu.q
#$ -l gpu=1
#$ -j y
#$ -cwd
#$ -V
QG=/gdata/projects/ml_scope/Closure_modeling/QG-closure
[[ -d /opt/rocks/bin ]] && export PATH=/opt/rocks/bin:$PATH
source $QG/qg-env/bin/activate
if command -v nvidia-smi >/dev/null 2>&1; then
  sleep $((RANDOM % 5))
  GPUDIR=$QG/tmp/gpu_locks; mkdir -p $GPUDIR
  for d in $GPUDIR/gpu_*.lock; do [ -d $d ] || continue; p=$(cat $d/pid 2>/dev/null); { [ -n "$p" ] && kill -0 $p 2>/dev/null; } || rm -rf $d; done
  CANDS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F',' '{u=$2+0; if(u<2000) print $1+0}' | shuf)
  [ -z "$CANDS" ] && CANDS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t',' -k2 -n | awk -F',' '{gsub(/ /,"");print $1}')
  for g in $CANDS; do if mkdir $GPUDIR/gpu_$g.lock 2>/dev/null; then echo $$ > $GPUDIR/gpu_$g.lock/pid; export CUDA_VISIBLE_DEVICES=$g; trap "rm -rf $GPUDIR/gpu_$g.lock" EXIT; break; fi; done
  echo "[testA] GPU $CUDA_VISIBLE_DEVICES on $HOSTNAME"
fi
cd $QG/qg-wiener-conditioning/training
python -u ../diagnostics/test_A.py
