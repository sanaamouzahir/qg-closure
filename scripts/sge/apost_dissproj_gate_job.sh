#!/bin/bash
# apost_dissproj_gate_job.sh - G3 driver gate for the --nn-dissipative-proj
# edit of rollout_aposteriori.py (Sanaa P0 2026-07-13).
#
#   GATE-1 (flag OFF, standing smoke): FRC-b2 5e-3, IC 0, K=20, 12 steps,
#          deriv7_filtered_lr5e-5 ckpt -- final rel-L2 per arm must match
#          the documented reference (apost_smoke, 2026-07-08) to <= 1%:
#            bare 2.1105790523239925e-08 | r3only 1.996186035829837e-08 |
#            closure 2.207535597526632e-05        [SMOKE numbers, K=20]
#   GATE-2 (flag OFF, in-distribution smoke): kf4 1.5e-2, IC 837, K=1500,
#          16 steps, refs REUSED (apost_smoke3/apost_refs_ladderrefs.npz);
#          must match rollout_apost_smoke3a_val.json to <= 1% per arm:
#            bare 0.0380264539661592 | r3only 0.03801253063298178 |
#            r3anal 0.0002867890699509995 | closure UNSTABLE step 12
#   GATE-3 (flag ON): both smokes rerun with --nn-dissipative-proj; the
#          kf4 arm must LOG projected shells (mean > 0 -- the NN injects
#          there, that is what blows it up); per-step counts printed.
#
# PASS => touch $NEW/GATE_PASS (the ladder jobs refuse to run without it).
#
# Submit: qsub -q ibgpu.q -l gpu=1 -N dissproj_gate \
#           scripts/sge/apost_dissproj_gate_job.sh
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -o /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/$JOB_NAME.o$JOB_ID
#$ -e /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/$JOB_NAME.e$JOB_ID
#$ -m ea
#$ -M sanaamz@mit.edu

set -e
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
WT=$QG_ROOT/qg-wiener-conditioning
NEW=$WT/diagnostics/Results/apost_dissproj_20260713
GATE=$NEW/gate
SMOKE3=$WT/diagnostics/Results/apost_smoke3
FILT=data/ensemble_N5_7lag/training_runs/deriv7_filtered_lr5e-5/best.pt
source $QG_ROOT/qg-env/bin/activate
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR=$QG_ROOT/.mplcache; mkdir -p "$MPLCONFIGDIR"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd $WT/training
mkdir -p "$GATE"
rm -f "$NEW/GATE_PASS"
echo "[dissproj_gate] host=$HOSTNAME date=$(date -u +%FT%TZ)"
echo "[dissproj_gate] git HEAD: $(/opt/rocks/bin/git rev-parse --short HEAD 2>/dev/null || echo '?')"
python -u -c 'import sys, torch; ok = torch.cuda.is_available(); print("[gpu-check]", ok, flush=True); sys.exit(0 if ok else 2)'

echo "==== GATE-1: standing smoke b2@5e-3 K=20 12 steps, flag OFF ===="
python -u rollout_aposteriori.py \
    --root-dir data/ensemble_N5_7lag/FRC-b2/sweep_dT_5em3 \
    --ckpt "$FILT" --ic-index 0 --K 20 --n-steps 12 \
    --arms bare,r3only,closure \
    --device cuda --out-dir "$GATE" --tag g1_b2_off

echo "==== GATE-2: kf4@1.5e-2 IC837 K=1500 16 steps, flag OFF, refs reused ===="
python -u rollout_aposteriori.py \
    --root-dir data/ensemble_N5_7lag/FRC-kf4/sweep_dT_1p5em2 \
    --ckpt "$FILT" --ic-index 837 --K 1500 --n-steps 16 \
    --arms bare,r3only,r3anal,closure \
    --load-refs "$SMOKE3/apost_refs_ladderrefs.npz" \
    --device cuda --out-dir "$GATE" --tag g2_kf4_off

echo "==== GATE-3a: b2 smoke, flag ON ===="
python -u rollout_aposteriori.py \
    --root-dir data/ensemble_N5_7lag/FRC-b2/sweep_dT_5em3 \
    --ckpt "$FILT" --ic-index 0 --K 20 --n-steps 12 \
    --arms bare,r3only,closure --nn-dissipative-proj \
    --device cuda --out-dir "$GATE" --tag g3_b2_on

echo "==== GATE-3b: kf4 smoke, flag ON ===="
python -u rollout_aposteriori.py \
    --root-dir data/ensemble_N5_7lag/FRC-kf4/sweep_dT_1p5em2 \
    --ckpt "$FILT" --ic-index 837 --K 1500 --n-steps 16 \
    --arms bare,r3only,r3anal,closure --nn-dissipative-proj \
    --load-refs "$SMOKE3/apost_refs_ladderrefs.npz" \
    --device cuda --out-dir "$GATE" --tag g3_kf4_on

echo "==== GATE VERDICT ===="
python -u - "$GATE" "$NEW" <<'PYEOF'
import json, sys
from pathlib import Path
gate, new = Path(sys.argv[1]), Path(sys.argv[2])
ok = True

def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)

# GATE-1: standing smoke reference (apost_smoke 2026-07-08, K=20 SMOKE)
r1 = json.loads((gate / 'rollout_apost_g1_b2_off.json').read_text())
ref1 = {'bare': 2.1105790523239925e-08, 'r3only': 1.996186035829837e-08,
        'closure': 2.207535597526632e-05}
for arm, v in ref1.items():
    got = r1['final_relL2'].get(arm)
    d = rel(got, v)
    st = 'PASS' if d <= 0.01 else 'FAIL'
    ok &= d <= 0.01
    print(f'[GATE-1] {arm:8s} got={got:.10e} ref={v:.10e} rel-diff={d:.2e} {st}')

# GATE-2: in-distribution smoke reference (smoke3a_val, h_fine=1e-5)
r2 = json.loads((gate / 'rollout_apost_g2_kf4_off.json').read_text())
ref2 = {'bare': 0.0380264539661592, 'r3only': 0.03801253063298178,
        'r3anal': 0.0002867890699509995}
for arm, v in ref2.items():
    got = r2['final_relL2'].get(arm)
    d = rel(got, v)
    st = 'PASS' if d <= 0.01 else 'FAIL'
    ok &= d <= 0.01
    print(f'[GATE-2] {arm:8s} got={got:.10e} ref={v:.10e} rel-diff={d:.2e} {st}')
blow = r2.get('closure_blowup_step')
st = 'PASS' if blow == 12 else 'FAIL'
ok &= blow == 12
print(f'[GATE-2] closure blowup step got={blow} ref=12 {st}')

# GATE-3: flag ON logs projected shells; kf4 must project (NN injects there)
r3b = json.loads((gate / 'rollout_apost_g3_kf4_on.json').read_text())
m = r3b.get('closure_proj_shells_mean')
st = 'PASS' if (m is not None and m > 0) else 'FAIL'
ok &= m is not None and m > 0
print(f'[GATE-3] kf4 ON proj_shells_mean={m} max='
      f'{r3b.get("closure_proj_shells_max")} removed_frac_mean='
      f'{r3b.get("closure_proj_removed_frac_mean")} {st}')
print(f'[GATE-3] kf4 ON closure verdict={r3b.get("closure_verdict")} '
      f'blowup={r3b.get("closure_blowup_step")} (OFF reference: step 12) '
      f'final={r3b["final_relL2"].get("closure")}')
r3a = json.loads((gate / 'rollout_apost_g3_b2_on.json').read_text())
print(f'[GATE-3] b2 ON proj_shells_mean='
      f'{r3a.get("closure_proj_shells_mean")} removed_frac_mean='
      f'{r3a.get("closure_proj_removed_frac_mean")} '
      f'final={r3a["final_relL2"].get("closure")} (OFF {ref1["closure"]:.4e})')

if ok:
    (new / 'GATE_PASS').write_text('G3 PASS\n')
    print('[GATE] ALL PASS -> GATE_PASS written')
else:
    print('[GATE] FAILURE -- ladder jobs will refuse to start')
    sys.exit(1)
PYEOF
echo "[dissproj_gate] done $(date -u +%FT%TZ)"
