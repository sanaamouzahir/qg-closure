#!/bin/bash
# train_deriv_rollout_job.sh - SGE worker for the rollout-in-the-loss trainer
# (train_deriv_rollout.py) ON THE WORKTREE (exp/wiener-conditioning).
#
# Loss = mean_m relL2(omega_rollout_m, omega_truth_m) over an M-step unrolled
# AB2CN2+closure rollout; truth = the deep 28-mark builds; the stepper is the
# EXACT validated rollout_aposteriori arm (return_stepper export). Gates
# (scripts/sge/rollout_gates_job.sh) must PASS before any smoke is submitted.
#
# PREPARED 2026-07-09 -- DO NOT SUBMIT the smokes before Sanaa's GO.
#
# Every training qsub is a THREE-JOB UNIT (CHARTER v1.3 I18a: trainer + LIVE
# monitor + FINALIZE monitor; names distinct in the first 10 chars; the
# [QG][SUBMIT][log] email carries ALL THREE ids). Baseline card:
# diagnostics/baseline_cards/T2_rollout.json (first-run card -- recalibrate
# from the phys arm). GPU rule: exactly -q ibgpu.q -l gpu=1; never -l h_vmem,
# never ibamd.q. Monitors ride all.q inside monitor_training_job.sh (repo
# convention; they are not GPU jobs).
#
#   cd $QG_ROOT/qg-wiener-conditioning
#   CARD=$PWD/diagnostics/baseline_cards/T2_rollout.json
#
#   # ---- smoke arm 1: cond_v2 ep63 warm start (rollc_*) ----
#   TRAIN=$(qsub -terse -q ibgpu.q -l gpu=1 -N rollc_TRN \
#        scripts/sge/train_deriv_rollout_job.sh \
#        --deep-roots data/ensemble_N5/FRC-kf4/forced_turbulence_dT_5em3 \
#                     data/ensemble_N5_7lag/FRC-256/forced_turbulence_dT_5em3 \
#        --init-ckpt data/ensemble_N5_7lag/training_runs/deriv7_cond_local_v2/best.pt \
#        --model auto \
#        --unroll-schedule 1:10,2:10,4:10 --epochs 30 --grad-mode full \
#        --lr 5e-5 --batch-size 1 --grad-clip 1.0 \
#        --windows-per-epoch 64 --val-windows 16 \
#        --out-root data/ensemble_N5_7lag --run-name rollout_ft_cond)
#   RUN=$PWD/training/data/ensemble_N5_7lag/training_runs/rollout_ft_cond
#   LIVE=$(qsub -terse -N rollc_MONL \
#        -o "$PWD/logs/\$JOB_NAME.\$JOB_ID.log" -e "$PWD/logs/\$JOB_NAME.\$JOB_ID.err" \
#        scripts/sge/monitor_training_job.sh "$RUN" wiener-conditioning $TRAIN \
#        "$CARD" "$PWD/logs/rollc_TRN.$TRAIN.log")
#   FINAL=$(qsub -terse -N rollc_MONF -hold_jid $TRAIN -v QG_MONITOR_FINALIZE=1 \
#        -o "$PWD/logs/\$JOB_NAME.\$JOB_ID.log" -e "$PWD/logs/\$JOB_NAME.\$JOB_ID.err" \
#        scripts/sge/monitor_training_job.sh "$RUN" wiener-conditioning $TRAIN \
#        "$CARD" "$PWD/logs/rollc_TRN.$TRAIN.log")
#   echo "trainer=$TRAIN live=$LIVE final=$FINAL"   # -> [QG][SUBMIT][log] email
#
#   # ---- smoke arm 2: physics init, control arm (rollp_*) ----
#   TRAIN=$(qsub -terse -q ibgpu.q -l gpu=1 -N rollp_TRN \
#        scripts/sge/train_deriv_rollout_job.sh \
#        --deep-roots data/ensemble_N5/FRC-kf4/forced_turbulence_dT_5em3 \
#                     data/ensemble_N5_7lag/FRC-256/forced_turbulence_dT_5em3 \
#        --model cheap_deriv \
#        --unroll-schedule 1:10,2:10,4:10 --epochs 30 --grad-mode full \
#        --lr 5e-5 --batch-size 1 --grad-clip 1.0 \
#        --windows-per-epoch 64 --val-windows 16 \
#        --out-root data/ensemble_N5_7lag --run-name rollout_ft_phys)
#   RUN=$PWD/training/data/ensemble_N5_7lag/training_runs/rollout_ft_phys
#   LIVE=$(qsub -terse -N rollp_MONL \
#        -o "$PWD/logs/\$JOB_NAME.\$JOB_ID.log" -e "$PWD/logs/\$JOB_NAME.\$JOB_ID.err" \
#        scripts/sge/monitor_training_job.sh "$RUN" wiener-conditioning $TRAIN \
#        "$CARD" "$PWD/logs/rollp_TRN.$TRAIN.log")
#   FINAL=$(qsub -terse -N rollp_MONF -hold_jid $TRAIN -v QG_MONITOR_FINALIZE=1 \
#        -o "$PWD/logs/\$JOB_NAME.\$JOB_ID.log" -e "$PWD/logs/\$JOB_NAME.\$JOB_ID.err" \
#        scripts/sge/monitor_training_job.sh "$RUN" wiener-conditioning $TRAIN \
#        "$CARD" "$PWD/logs/rollp_TRN.$TRAIN.log")
#   echo "trainer=$TRAIN live=$LIVE final=$FINAL"   # -> [QG][SUBMIT][log] email
#
# lr cap 5e-5 (quiescent postmortem rule -- never higher for training runs).
# kf4/Re25k/combo deep dirs live under ensemble_N5/ (not ensemble_N5_7lag/).
# --deep-roots are resolved relative to training/ (the job cd's there).
#$ -o /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/$JOB_NAME.$JOB_ID.log
#$ -e /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/$JOB_NAME.$JOB_ID.err
#$ -m ea
#$ -M sanaamz@mit.edu
set -e
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
source "$QG_ROOT/qg-env/bin/activate"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# Run from the WORKTREE training/ so the rollout-loss code is the code that runs.
SCRIPT_DIR="$QG_ROOT/qg-wiener-conditioning/training"
cd "$SCRIPT_DIR"

echo "[train_deriv_rollout_job] hostname: $HOSTNAME"
echo "[train_deriv_rollout_job] date:     $(date -u +%FT%TZ)"
echo "[train_deriv_rollout_job] cuda dev: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "[train_deriv_rollout_job] cwd:      $PWD  (worktree)"
echo "[train_deriv_rollout_job] git HEAD: $(/opt/rocks/bin/git rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "------------------------------------------------------------"

python -u train_deriv_rollout.py "$@"

echo "------------------------------------------------------------"
echo "[train_deriv_rollout_job] done at $(date -u +%FT%TZ)"
