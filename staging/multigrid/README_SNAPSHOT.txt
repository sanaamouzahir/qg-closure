================================================================================
 Shallow_NN_enssemble_MultiGrid  --  derivative-loss closure, MULTIGRID
================================================================================
Exactly the v2 derivative-loss model, generalized to pool ANY mix of grids and
domains in one training run. The objective, loss, precision, physics-init, and
targets are unchanged. ONE cheap_deriv network is trained across all grids.

On a SINGLE grid this is bit-identical to v2 (the per-sample spatial rescale is
x1). So this supersedes v2; v2 is kept as the locked single-grid reference.

--------------------------------------------------------------------------------
 WHAT IS DIFFERENT FROM v2  (only these four things changed)
--------------------------------------------------------------------------------
 DIFF 1  model_deriv_closure.py :: SpatialGrad
   The spatial stencil bakes 1/dx0 at a REFERENCE spacing. On a grid of spacing
   dx_i the true gradient is (conv)*(dx0/dx_i). forward() now takes per-sample
   (dx, dy) (B,) tensors and rescales the gradient outputs by dx0/dx_i, dy0/dy_i.
   - One learnable DIMENSIONLESS operator, applied at every grid's spacing.
   - dx0/dx_i is a constant -> Adam still steps the well-conditioned dx0 operator.
   - Rescale is PER-SAMPLE, not per-batch: a shape-homogeneous batch can still
     mix domains of unequal dx (e.g. 512^2/2pi has dx=2pi/512, 512^2/4pi has
     dx=4pi/512 -- same shape, different dx). Per-batch would be WRONG.
   - dx=dy=None -> x1 -> identical to v2. (TimeFD already did per-sample dt; the
     time path is unchanged.)
   CheapDerivClosureNet.forward gained (dx, dy) and passes them to self.grad.

 DIFF 2  deriv_dataset.py :: regime vector
   regime grows from [dT, beta, nu, mu] to [dT, beta, nu, mu, dx, dy], so each
   sample carries its grid spacing (dx=Lx/Nx, dy=Ly/Ny). Reading regime[:,0] for
   dT is unchanged; the trainer reads regime[:,4:6] for the spatial rescale.

 DIFF 3  train_deriv.py :: no grid restriction
   - KEEPS all roots (v2 dropped everything off the most-common shape).
   - Picks a REFERENCE full grid (Ny,Nx,Lx,Ly) = the most common; builds the model
     at its dx0,dy0 (dt0 only sets the inert TimeFD self.weight). All other grids
     rescale to it.
   - Builds a dealias projection PER SHAPE {(Ny,Nx): fn} (the 2/3 mask is mode-
     index based -> depends on (Ny,Nx) only, NOT on L -> shared across domains of
     equal shape). run_epoch selects it by the batch's shape x.shape[-2:].
   - run_epoch passes per-sample dx,dy (regime[:,4], regime[:,5]) into the model.
   - --grid 'NyxNx' still works as a SHAPE filter (e.g. pin to 512^2 only).

 DIFF 4  (no code; the trap to remember)
   GridHomogeneousBatchSampler groups by SHAPE, which does NOT imply equal dx.
   That is the whole reason DIFF 1 is per-sample. If you ever switch to a
   per-batch rescale it will silently corrupt mixed-domain batches.

 EVERYTHING ELSE is byte-for-byte v2: add_deriv_targets.py (targets are per-member,
 already grid-correct), train_deriv_job.sh, the loss (rel_l2 per-sample/channel),
 cosine, dealias-pred, physics-init, learnable spatial stencils, float64 training.

--------------------------------------------------------------------------------
 FILES / EXTERNAL DEPS / CONFIG
--------------------------------------------------------------------------------
 Same as v2 (see Shallow_NN_enssemble_fixedGrid_v2/README_SNAPSHOT.txt). External
 deps still resolve from .../training/: concat_dataset.py,
 build_training_data_fixD_v2.py, qg. Run from training/. Targets built once by
 add_deriv_targets.py over the WHOLE ensemble (it already handles every member).

--------------------------------------------------------------------------------
 RUN  (all grids/domains pooled)
--------------------------------------------------------------------------------
  cd .../training
  python add_deriv_targets.py data/ensemble_N5 --device cuda     # once, all members

  qsub -N deriv_mg -q ibgpu.q -l gpu=1 train_deriv_job.sh \
    --sweep-roots data/ensemble_N5/*/sweep_dT_* \
    --n-snapshots 4 --out-orders 3 --grad-kernel 15 \
    --epochs 200 --lr 5e-5 --weight-decay 1e-4 --batch-size 4 \
    --num-workers 8 --compute-dtype float64 --seed 0
  # --grid 512x512 to pin to one shape; omit to pool 256^2 and 512^2 together.

  SMOKE TEST first (cheapest real multigrid check -- two members, two shapes):
  qsub -N deriv_mg_smoke -q ibgpu.q -l gpu=1 train_deriv_job.sh \
    --sweep-roots data/ensemble_N5/FRC-b05/sweep_dT_* data/ensemble_N5/FRC-256/sweep_dT_* \
    --n-snapshots 4 --out-orders 3 --grad-kernel 15 --epochs 5 \
    --lr 5e-5 --batch-size 4 --num-workers 8 --compute-dtype float64
  Confirm the startup line shows shapes=[(256,256),(512,512)] and >1 full-grids,
  and that epoch-0 Nddot is already < 1 (physics-init survives the rescale).

--------------------------------------------------------------------------------
 SANITY CHECKS SPECIFIC TO MULTIGRID
--------------------------------------------------------------------------------
  * Single-grid equivalence: run this trainer on a single member and confirm the
    log matches v2 to ~machine precision (rescale must be x1 on the reference grid).
  * The startup print reports reference(Ny,Nx,Lx,Ly), the shape list, and dx0.
    If 'full-grids' == #shapes you have no same-shape/different-domain collisions;
    if 'full-grids' > #shapes you DO (e.g. 512^2/2pi + 512^2/4pi) -- that is the
    case DIFF 1 exists for, so it is fine, just be aware.
  * Param count is identical to v2 (the rescale adds no parameters; dx0,dy0 are
    plain floats on SpatialGrad).
================================================================================
