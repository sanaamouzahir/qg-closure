================================================================================
 Shallow_NN_enssemble_fixedGrid  --  single-grid delta-closure trainer
================================================================================

WHAT THIS IS
------------
Trains a temporal CLOSURE for one coarse AB2CN2 step. Instead of bumping the
scheme order, we add a per-step correction delta so that

      omega^{n+1} = Phi_AB2CN2(omega^n, omega^{n-1}; DeltaT) + delta

makes the cheap coarse step behave like a chosen REFERENCE propagator.

   delta = Phi_ref(omega; DeltaT) - Phi_AB2CN2(omega, omega_{-1}; DeltaT)

Two references (trained together via --reference both; a 1-bit flag selects):
   - exact : ref = fine RK4 (effectively the exact flow)  -> match exact
   - rk4   : ref = one coarse RK4 step                     -> match RK4
   delta_exact - delta_rk4 = tau_RK4 (RK4's own truncation), learned empirically.

This is the SINGLE-GRID version: it restricts to one resolution (--grid, default
the most common in the pool) and drops off-grid members. The mixed-grid trainer
(256^2 + 512^2 in one model) lives in Shallow_NN_enssemble_MultiGrid. Use THIS
one first as the clean fallback.

FILES
-----
   train_delta.py          driver: data -> model -> loss -> train loop
   delta_dataset.py        per-member .npz reader; regime = [dT, beta, nu, mu]
   model_deriv_closure.py  cheap_deriv: FD N-derivatives + analytic R3,R4 assembly
   closure_operators.py    verified AB2CN2 / AB4CN2 closure coefficients
   train_delta_job.sh      SGE worker (qsub -q ibgpu.q -l gpu=1)


ARCHITECTURE (one coarse step, as the code runs it)
---------------------------------------------------
  INPUT   x = lag stack: n_snapshots=4 frames of (omega, psi) -> 8 channels
          regime = [DeltaT, beta, nu, mu]

  STEP 1  cheap_deriv (PHYSICS front end)
          15x15 FROZEN FD stencils (grad_kernel=15) + time-FD over the 4 lags
          -> N-derivatives  nd = (Ndot, Nddot, N3dot)        (frozen = no params)

  STEP 2  analytic closure assembly -> delta_anal           (skipped if pure)
          N0   = -J(psi0, omega0)               (Jacobian, shared with base step)
          Lhat = Lhat(k) from (nu, mu, beta)    (diagonal spectral op, free)
          delta_anal = -(DeltaT^3/12)(Lhat^3 w + Lhat^2 N + Lhat Ndot - 5 Nddot) + R4

  STEP 3  feat = [ lags , N-derivs , delta_anal ]            (channel concat)

  STEP 4  corrector  DeltaTailNet  (the LEARNED tail; ~1.1K params)
          small CNN, kernel 3, tail_hidden=8, tail_depth=2,
          FiLM-conditioned on normalized regime (+ 1 ref-bit in 'both' mode)
          -> predicts the DeltaT^3-normalized residual

  STEP 5  delta = delta_anal + DeltaT^3 * corrector(feat, regime)

  STEP 6  loss = relative_L2( delta_pred / DeltaT^3 , delta_target / DeltaT^3 )
          batch-aggregate; summed over (exact, rk4) legs for backprop;
          REPORTED as mean-per-leg (0..1 scale; ~1.0 = predicting zero).

  One line:  lags -> FD N-derivs -> analytic R3R4 -> concat -> FiLM corrector
             -> delta_anal + DeltaT^3 * tail.


THE THREE RUNS (what differs -- ONLY steps 1-2)
-----------------------------------------------
All three share identical flags except the one config flag + run-name:
   --reference both --grid 512x512 --n-snapshots 4 --batch-size 24
   --num-workers 8 --epochs 40 --grad-kernel 15 --tail-hidden 8 --tail-depth 2

  (1) pure              --pure-empirical
        delta_anal := 0. No L^k assembly, no amplification. Corrector learns
        delta directly from [lags, N-derivs]. Epoch-0 mean relL2/leg == 1.0 by
        construction. The clean "does the physics prior help at all?" baseline.

  (2) hybrid_unfrozen   (no extra flag)
        Keeps delta_anal AND trains cheap_deriv under the delta-loss (the
        N-derivatives get reshaped by the actual closure objective). The
        principled hybrid: "old model + corrector, trained on delta."

  (3) hybrid_frozen     --freeze-physics
        Keeps delta_anal but FREEZES cheap_deriv -> delta_anal is a fixed prior
        and the corrector carries all learning. Isolates whether refitting the
        front end under the delta-loss matters. Also makes
        corrector(exact) - corrector(rk4) exactly the empirical tau_RK4.

  How to read the ranking (mean relL2/leg, lower = better):
    - pure ~= unfrozen     -> physics prior adds little; take pure (simplest).
    - unfrozen << pure     -> the analytic prior earns its keep.
    - frozen highest       -> refitting the front end matters (amplification of a
                              fixed, derivative-fit prior is a liability).


NOTE ON THE CORRECTOR KERNEL
----------------------------
The corrector here is kernel-3 ON PURPOSE for this comparison: pure NEEDS a
receptive field (it differentiates the lags itself), so all three use kernel-3 to
stay apples-to-apples. The kernel-1 (pointwise) corrector is a follow-on
optimization (Option A below), valid only once the basis channels are precomputed.


================================================================================
 NEXT, ONCE THESE FINISH:  if a HYBRID wins, build BOTH and compare
================================================================================
The three-way above answers "pure vs hybrid, and does refitting the front end
matter". If a hybrid wins, we then ship one of two endpoints. We will build BOTH
and pick by the delta-error gap between them.

OPTION A -- pointwise corrector + Lhat-basis  (interpretable, ~0.1x step)
  Old front end -> analytic R3R4 -> PRECOMPUTE the Lhat-power basis channels
  { Lhat w, Lhat^2 w, Lhat^3 w, Lhat^2 N, Lhat Ndot, ... }  (all free, diagonal
  spectral mults) -> feed them to a KERNEL-1 (pointwise) corrector that just
  MIXES them:   delta_tail(x) = sum_i c_i(regime) * basis_i(x).
  - corrector adds NO spatial structure (pointwise) -> ~0.1x a base step
  - optionally INIT the mix to the known R5 coeffs (13,13,13,-17,8,-7)/240
  - the only genuinely-learned piece is N''''(4th deriv, not constructible from
    4 lags) + R6+ -- small and smooth, no receptive field needed.
  KEY: grad_kernel (cheap_deriv FD) STAYS at 15 -- the physics differentiation is
  unchanged; only the CORRECTOR kernel goes 3 -> 1. The basis channels supply the
  spatial structure pointwise, which is exactly what makes kernel-1 sufficient.
  In this form the model is literally "what we had before + a pointwise corrector
  + the delta-loss" -- nothing spatial is added.
  Implementation: train_delta.py only -- add --tail-kernel (default 3, set 1),
  precompute the basis in physics_part, widen the corrector's in-channels.

OPTION B -- "what we had before" + modified loss  (NO corrector at all)
  Existing cheap_deriv -> analytic R3R4 assembly -> delta_anal, with the
  corrector contribution ZEROED, but trained on the DELTA-LOSS (not the old
  per-derivative loss). The ONLY change from the original model is what the loss
  is computed against: training on delta penalizes the Lhat-amplification the old
  derivative-loss left unmeasured, so the SAME assembly is refit to minimize the
  actual closure error rather than the per-derivative error.
  - answers: does analytic R3R4 ALONE, refit under delta-loss, already suffice?
  - no NN tail -> strongest "pure physics, right objective" story if it holds.
  Implementation: a --no-corrector mode (keep delta_anal, drop the tail).

DECISION between A and B (after running both):
    B-error  ~=  unfrozen-error  -> ship B (no tail needed; cleanest).
    B-error  >   unfrozen-error  -> ship A (the gap IS what the corrector buys;
                                    the pointwise A recovers it at ~0.1x step).
  hybrid_frozen from the three-way already previews this: it is delta_anal as a
  fixed prior + corrector, so (frozen vs B) shows how much the corrector alone
  adds on top of a fixed analytic closure.

WHAT I (CLAUDE) NEED FROM YOU TO BUILD A AND B:
  paste the three best= lines (pure / hybrid_unfrozen / hybrid_frozen). That
  ranking decides whether we proceed to A+B at all, and B's target to beat.
================================================================================