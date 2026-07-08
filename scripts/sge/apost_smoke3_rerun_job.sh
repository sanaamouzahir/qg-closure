#!/bin/bash
#$ -N apost_smoke3
#$ -q ibgpu.q
#$ -l gpu=1
#$ -j y
#$ -o /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/training/logs/apost_smoke3.log
#$ -cwd
#$ -m ea
#$ -M sanaamz@mit.edu
# apost_smoke3_rerun_job.sh -- rerun of the 2026-07-08 control smokes with
#   (1) truth at h_fine = 1e-5 (K=1500/500/1000; the old K=20 truth carried
#       the RK4 LTE in the comparison -- we do NOT model tau_RK4 by design),
#   (2) new r3anal arm (full analytic R3, exact chain-rule Ndot/Nddot -- the
#       blowup discriminator vs NN-injected noise),
#   (3) --track-lte (full analytic LTE per checkpoint + NN-vs-analytic drift).
# Old artifacts kept in diagnostics/Results/apost_smoke2; new in apost_smoke3.
set -e
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
WT=$QG_ROOT/qg-wiener-conditioning
VENV=$QG_ROOT/qg-env
OUT=$WT/diagnostics/Results/apost_smoke3
CKPT=data/ensemble_N5_7lag/training_runs/deriv7_filtered_lr5e-5/best.pt
source $VENV/bin/activate
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR=$QG_ROOT/.mplcache; mkdir -p "$MPLCONFIGDIR"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd $WT/training
mkdir -p logs "$OUT"
echo "[smoke3] host=$HOSTNAME date=$(date -u +%FT%TZ)"
nvidia-smi -L || echo "[smoke3] nvidia-smi unavailable"
python -u -c 'import sys, torch; ok = torch.cuda.is_available(); print("[gpu-check] cuda_available =", ok, flush=True); sys.exit(0 if ok else 2)'

COMMON="--ckpt $CKPT --arms bare,r3only,r3anal,closure --track-lte --diag --save-refs --device cuda --out-dir $OUT"

# 3a: kf4 @ 1.5e-2, IC 820 (train row), h_fine = 1e-5
python -u rollout_aposteriori.py --root-dir data/ensemble_N5_7lag/FRC-kf4/sweep_dT_1p5em2 \
    --ic-index 820 --K 1500 --n-steps 16 --tag smoke3a $COMMON

# 3b: b2 @ 5e-3, IC 964 (train row), h_fine = 1e-5
python -u rollout_aposteriori.py --root-dir data/ensemble_N5_7lag/FRC-b2/sweep_dT_5em3 \
    --ic-index 964 --K 500 --n-steps 12 --tag smoke3b $COMMON

# 3c: kf4 @ 1e-2, IC 820, h_fine = 1e-5 (dt-sweep disambiguation point)
python -u rollout_aposteriori.py --root-dir data/ensemble_N5_7lag/FRC-kf4/sweep_dT_1em2 \
    --ic-index 820 --K 1000 --n-steps 16 --tag smoke3c $COMMON

# val-row repeats (leakage check under the clean truth)
python -u rollout_aposteriori.py --root-dir data/ensemble_N5_7lag/FRC-kf4/sweep_dT_1p5em2 \
    --ic-index 837 --K 1500 --n-steps 16 --tag smoke3a_val $COMMON

python -u rollout_aposteriori.py --root-dir data/ensemble_N5_7lag/FRC-b2/sweep_dT_5em3 \
    --ic-index 934 --K 500 --n-steps 12 --tag smoke3b_val $COMMON

# ---- quantify the OLD K=20 truth error: rel-L2(old truth, new truth) ---- #
python -u - <<'PYEOF'
import numpy as np
old_dir = '../diagnostics/Results/apost_smoke2/rollout_apost_smoke2%s.npz'
new_dir = '../diagnostics/Results/apost_smoke3/rollout_apost_smoke3%s.npz'
rel = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b ** 2)))
print('\n========== OLD K=20 TRUTH ERROR vs h_fine=1e-5 TRUTH ==========')
for s in ('a', 'b', 'c', 'a_val', 'b_val'):
    try:
        o, n = np.load(old_dir % s), np.load(new_dir % s)
    except FileNotFoundError as e:
        print(f'  smoke {s}: skipped ({e})'); continue
    co = {int(k): i for i, k in enumerate(o['cp_steps'])}
    cn = {int(k): i for i, k in enumerate(n['cp_steps'])}
    common = sorted(set(co) & set(cn))[1:]
    errs = [(k, rel(o['truth_stack'][co[k]].astype(np.float64),
                    n['truth_stack'][cn[k]].astype(np.float64))) for k in common]
    show = errs[:2] + errs[-2:] if len(errs) > 4 else errs
    print(f'  smoke {s}: ' + '  '.join(f'step{k}={e:.3e}' for k, e in show))
PYEOF
echo "[smoke3] done $(date -u +%FT%TZ)"
