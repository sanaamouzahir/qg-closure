#!/bin/bash
# rollout_gates_job.sh - acceptance gates for train_deriv_rollout.py
# (exp/wiener-conditioning worktree). Runs the three gates in one GPU job:
#
#   G-R1  trainer unroll vs run_arm: bare bit-exact + zero-NN closure vs
#         r3only bit-exact (prints max|d omega| over 4 steps)
#   G-R2  M=1 step-1 residual fraction on kf4 val windows at dT
#         5e-3/1e-2/1.5e-2 with the cond_v2 ep63 warm start -- must
#         reproduce ~0.0575/0.0586/0.0646 (4-arm table, session 7f)
#   G-R3  tiny-overfit: 2 windows, M=2, 50 Adam steps at --gate-lr 1e-3
#         (probe lr, NOT training; production cap stays 5e-5) -- loss
#         must drop >10x
#
# Submit (GPU rule: exactly -q ibgpu.q -l gpu=1; never -l h_vmem, never ibamd.q):
#   cd $QG_ROOT/qg-wiener-conditioning
#   qsub -q ibgpu.q -l gpu=1 -N rollout_gates scripts/sge/rollout_gates_job.sh
#
# Exit status: 0 only if ALL three gates PASS.
#$ -o /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/$JOB_NAME.$JOB_ID.log
#$ -e /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/$JOB_NAME.$JOB_ID.err
#$ -m ea
#$ -M sanaamz@mit.edu
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
source "$QG_ROOT/qg-env/bin/activate"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

SCRIPT_DIR="$QG_ROOT/qg-wiener-conditioning/training"
cd "$SCRIPT_DIR"

KF4=data/ensemble_N5/FRC-kf4/forced_turbulence_dT_5em3
COND=data/ensemble_N5_7lag/training_runs/deriv7_cond_local_v2/best.pt

echo "[rollout_gates_job] hostname: $HOSTNAME"
echo "[rollout_gates_job] date:     $(date -u +%FT%TZ)"
echo "[rollout_gates_job] cwd:      $PWD  (worktree)"
echo "[rollout_gates_job] git HEAD: $(/opt/rocks/bin/git rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "------------------------------------------------------------"

STATUS=0

# G-R1: model-independent scheme equivalence (physics init suffices)
python -u train_deriv_rollout.py --deep-roots "$KF4" \
    --model cheap_deriv --gate r1 || STATUS=1

# G-R2: cond ep63 warm start on kf4 -- the offline-consistency reproduction
python -u train_deriv_rollout.py --deep-roots "$KF4" \
    --init-ckpt "$COND" --model auto --gate r2 --gate-windows 16 || STATUS=1

# G-R3: gradient-flow sanity from physics init (probe lr 1e-3)
python -u train_deriv_rollout.py --deep-roots "$KF4" \
    --model cheap_deriv --gate r3 --gate-lr 1e-3 --grad-clip 1.0 || STATUS=1

echo "------------------------------------------------------------"
if [ "$STATUS" -eq 0 ]; then
    echo "[rollout_gates_job] ALL GATES PASS  ($(date -u +%FT%TZ))"
else
    echo "[rollout_gates_job] GATE FAILURE (see above)  ($(date -u +%FT%TZ))"
fi
exit $STATUS
