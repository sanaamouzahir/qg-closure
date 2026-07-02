================================================================================
 Shallow_NN_enssemble_fixedGrid_v2  --  derivative-loss closure, single-grid
================================================================================
The pre-6.1.2 model, on the ensemble. The cheap_deriv network predicts the LOCAL
N-time-derivatives [Ndot, Nddot, N3dot] DIRECTLY; the L^k truncation weightings
are applied analytically at assembly/inference, never learned. NO corrector, NO
delta-assembly in the loss -- the error-propagation analysis showed the closure
error == the per-operator error (no amplification), so the derivative loss IS the
closure objective, and Nddot's accuracy sets the rollout ceiling.

This is the v2 lock. (v1 = the delta-R pivot: train_delta.py + corrector. Kept
separately in Shallow_NN_enssemble_fixedGrid/.)

--------------------------------------------------------------------------------
 FILES IN THIS SNAPSHOT
--------------------------------------------------------------------------------
  model_deriv_closure.py   cheap_deriv net. Input (B, 2*n_time, H, W) =
                           [omega_0..m3, psi_0..m3]; output (B, 3, H, W) =
                           [Ndot, Nddot, N3dot]. 4 stages: time-FD -> spatial
                           grads -> 16 Jacobian features -> 1x1 mix (physics-init
                           to chain-rule binomials). Corrector OFF (hidden=0).
  add_deriv_targets.py     Builds packed/deriv_anal_f64.npy (N,3,Ny,Nx) per
                           sweep dir = analytic [Ndot,Nddot,N3dot] from omega_0
                           via the chain rule. Forcing rebuilt from the manifest
                           (exact for FRC, F=0 for DEC). Run ONCE before training.
  deriv_dataset.py         Pools sweep_dT_* dirs, serves (x, [Ndot,Nddot,N3dot],
                           regime=[dT,beta,nu,mu]). Single-grid (one grid/run).
  train_deriv.py           Trainer. Loss = per-sample per-channel rel-L2 (rel_l2),
                           cosine LR, dealias-pred ON, physics-init ON, learnable
                           SPATIAL stencils ON. Restricts to one grid; drops
                           off-grid roots.
  train_deriv_job.sh       SGE worker. -j y (one joined log). cd's to training/.
  README_SNAPSHOT.txt      this file.

--------------------------------------------------------------------------------
 EXTERNAL DEPENDENCIES  (NOT snapshotted; resolve from .../training/)
--------------------------------------------------------------------------------
  concat_dataset.py            GridHomogeneousBatchSampler, snapshot_input_fields
                               (imported by deriv_dataset.py)
  build_training_data_fixD_v2  build_L_hat, J_phys, L_op,
                               compute_n_dot_analytical, compute_n_ddot_analytical
                               (imported by add_deriv_targets.py)
  qg  (installed package)      CartesianGrid, Derivative, to_spectral/to_physical
  DATA: data/ensemble_N5/<MEMBER>/sweep_dT_<tag>/{manifest.json, split.npz,
        packed/{inputs.npy, deriv_anal_f64.npy}}   (sliced by slice_delta_sweep.py,
        targets added by add_deriv_targets.py)

  ALL scripts must be run from .../training/ (so the above import). Running from
  inside this snapshot dir gives ModuleNotFoundError: concat_dataset /
  build_training_data_fixD_v2 / qg. Flatten the snapshot into training/ to run,
  keep this dir as the archival lock.

--------------------------------------------------------------------------------
 EXACT TRAINING CONFIG (the run that produced the reference rollout)
--------------------------------------------------------------------------------
  input_fields  : omega_0 omega_m1 omega_m2 omega_m3 psi_0 psi_m1 psi_m2 psi_m3
  target_fields : N_dot_0_anal N_ddot_0_anal N_3dot_0_anal   (== deriv_anal_f64)
  model         : cheap_deriv     out_orders 3   n_time 4 (8 channels)
  grad_kernel   : 15              physics_init true   corrector_channels 0
  loss          : rel_l2 (per-sample, per-channel)   dealias_pred true
  normalize     : false
  batch_size 4   epochs 200   lr 5e-5   weight_decay 1e-4   lr_schedule cosine
  compute_dtype : float64   seed 0

--------------------------------------------------------------------------------
 RUN
--------------------------------------------------------------------------------
  cd .../training
  # 1) build targets once (idempotent; --overwrite to rebuild)
  python add_deriv_targets.py data/ensemble_N5 --device cuda
  # 2) train (FRC-512^2/4pi pool; omit FRC-256 so it is one grid/one domain)
  qsub -N deriv_frc -q ibgpu.q -l gpu=1 train_deriv_job.sh \
    --sweep-roots data/ensemble_N5/FRC-{b0,b05,b075,b1,b2,b25,kf4,Re25k,combo}/sweep_dT_* \
    --n-snapshots 4 --out-orders 3 --grad-kernel 15 \
    --epochs 200 --lr 5e-5 --weight-decay 1e-4 --batch-size 4 \
    --num-workers 8 --compute-dtype float64 --seed 0

--------------------------------------------------------------------------------
 LOCKED-IN FACTS (so the snapshot is self-documenting)
--------------------------------------------------------------------------------
  * Targets are recomputed analytically from omega_0 (= input channel 0), the
    SAME quantity build_training_data_fixD_v2 stored as N_*_0_anal. The slicer
    never saved them (it built delta = snapshot - AB2 step), so add_deriv_targets
    reconstructs them. Supervision is exactly consistent with the model input.
  * Forcing is time-independent and lives in each manifest (FRC: A cos(Bx)+D cos(Ey);
    DEC: absent -> F=0). It enters Nddot/N3dot via omega_dot, so it is load-bearing
    for the targets -- and it is exact, no external file.
  * TIME-FD stencils: across a dT sweep the model takes the per-sample dt^-k W_unit
    path (exact FD at each dt), NOT the learnable self.weight. So time stencils are
    frozen-exact here regardless of --learnable-stencils. self.weight is an inert
    trainable parameter (no gradient) and just pads the param count.
  * SPATIAL stencils ARE learnable (default), matching the single-trajectory run:
    the width-15 stencil refines toward the spectral derivative, tightening the
    high-k Nddot gap that sets the rollout ceiling.
  * Precision: training is full float64 (model.to(float64)); the float32-spatial
    mixed-precision path in the model is an INFERENCE/rollout optimization only
    (fires when params are float32 and inputs float64). Original training was f64
    too, so this matches.
  * The Nddot val rel-L2 ~ the rollout error floor (~1/improvement); it is the
    number to watch. The other orders barely move the rollout.
================================================================================
