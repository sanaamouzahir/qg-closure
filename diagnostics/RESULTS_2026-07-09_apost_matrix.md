# RESULTS 2026-07-09 — apost ladder matrix (2 ckpt x 2 variant x 3 dT) + dealias/FFT audit

Job 1828239 `apmx0709` (ibgpu.q, GPU, ~5 min). Member FRC-kf4 (beta=1, nu=1.025e-4,
mu=0.02), 512^2, L=4pi, IC packed row 837, M=16 coarse steps, gamma=1, NO remediation.
Truth: RK4 h_fine=1e-5 (K=1500/1000/500); 1.5e-2 refs REUSED from
apost_smoke3/apost_refs_ladderrefs.npz. Ckpts: UNCOND=deriv7_filtered_lr5e-5/best.pt
(cheap_deriv, ep8); COND=deriv7_cond_local_v2/frozen_eval_20260709/best.pt (cond_local,
EPOCH 63, val 0.2139 — frozen copy; job 1827306 still training). Variant B = new
`--drop-nddot` flag (R3 without the -5*Nddot term; L^3 w implicit, L^2 N, L*Ndot kept).
One npz per case (Sanaa's output discipline) in diagnostics/Results/apost_ladder_20260709/;
index = ladder_matrix_summary.csv.

## TASK 0 — dealias + FFT audit of the rollout harness: NO BUG, one convention correction

- f_NN IS end-projected before the IMEX step (rollout_aposteriori.py:390-391, then
  rhs -= coef*f_nn, then /denom). Same for f_anal and R4. `--dealias-nn` default True.
- SAME mask object as training: `Derivative.alias_mask` (train_deriv.py:229 vs
  `_dealias_mul`, rollout_timed_pareto.py:92-102). Train/inference CONSISTENT; nothing changed.
- CONVENTION CORRECTION (Jul-8 log said "square per-axis 2/3, cutoff mode 170"): the
  solver mask is RADIAL — k_cut = sqrt(2)*(2/3)*min(kx_max,ky_max)
  (qg/solver/opt/derivative.py:30-32) = mode radius 241.36 at 512^2. Observationally
  identical for "shells >241 empty", different inside: on-axis modes up to 241 are RETAINED.
- CONSEQUENCE: the annulus 170.7 < |k| <= 241.4 is above the alias-safe 2/3 radius
  (N/3=170.7); quadratic products alias in that band. The blow-up seeds (|k|=184-240,
  Jul-8 sigma analysis) sit exactly in the alias-contaminated annulus. Shared by
  truth/training/rollout (self-consistent); changing it = solver-convention change (RED).
- FFT checks: rfftn/irfftn norm='forward' round trip exact; Hermitian multipliers
  symmetric; Parseval E/Z path shared by all arms; float64 end-to-end; rollout NN call =
  training signature model(x, dt, dx, dy); psi history = inv_laplacian both sides;
  IC row 837 bit-identical across the three sweeps.

## Matrix results (final rel-L2 at t=16*dT vs RK4 truth; improvement = bare/closure)

| ckpt   | variant   | dT     | blowup step | final relL2 (bare)      | improvement | corner sigma drift (t_last) |
|--------|-----------|--------|-------------|-------------------------|-------------|------------------------------|
| uncond | full      | 1.5e-2 | 12          | — (3.80e-2)             | —           | 4.53 |
| cond   | full      | 1.5e-2 | **7**       | — (3.80e-2)             | —           | 3.50 |
| uncond | dropnddot | 1.5e-2 | none        | 3.83e-2 (3.80e-2)       | 0.99x       | 2.90 |
| cond   | dropnddot | 1.5e-2 | none        | 3.82e-2 (3.80e-2)       | 1.00x       | 2.90 |
| uncond | full      | 1e-2   | none        | 1.09e-2 (1.75e-3)       | 0.16x       | 2.15 |
| cond   | full      | 1e-2   | **13**      | — (1.75e-3)             | —           | 6.91 |
| uncond | dropnddot | 1e-2   | none        | 1.7466e-3 (1.7481e-3)   | 1.00x       | 0.038 |
| cond   | dropnddot | 1e-2   | none        | 1.7464e-3 (1.7481e-3)   | 1.00x       | 0.038 |
| uncond | full      | 5e-3   | none        | 3.60e-3 (2.10e-4)       | 0.06x       | 0.108 |
| cond   | full      | 5e-3   | none        | 2.91e-4 (2.10e-4)       | 0.72x       | 0.028 |
| uncond | dropnddot | 5e-3   | none        | 2.0945e-4 (2.0950e-4)   | 1.00x       | 0.024 |
| cond   | dropnddot | 5e-3   | none        | 2.0943e-4 (2.0950e-4)   | 1.00x       | 0.024 |

corner drift = median over shells |k| in [184,240] of |sigma(t,k)/sigma(0,k)-1| at the
last logged checkpoint (case npz carries the full curves; low band |k|<60 stays 1-4%
everywhere except inside a blow-up cascade).

## Reading

1. The N-ddot term IS the (only) destabilizer: every dropnddot arm is stable at every dT,
   and every blow-up has the -5*Nddot term in the loop. But dropnddot also removes ALL
   gain (1.0x everywhere) — consistent with the error budget (N-ddot carries ~100% of the
   closure signal; L*Ndot alone is worthless AND harmless).
2. Conditioning does NOT fix the tail natively — it makes the feedback FASTER at large dT
   (blow-up step 7 vs 12 at 1.5e-2; 13 vs stable-but-poor at 1e-2). At 5e-3 it is a big
   a-priori win (0.72x vs 0.06x final; transient: t=0.005 cond 7.6e-7 vs bare 1.3e-5 =
   17x BETTER, uncond 1.8e-6 = 7.5x) yet still loses the horizon to the same compounding
   tail: cond error grows x3.7/step at the end while bare grows linearly.
3. Transient-vs-horizon: the closure is genuinely more accurate for ~10 steps at 5e-3,
   then the corner-band feedback overtakes. The instability lives in the alias-contaminated
   annulus (Task 0) — a training-blind band the stencil cannot regulate: training targets
   are m>=1 derivatives of DEALIASED products, while the rollout feeds the NN its own
   corner-band-polluted history.
4. t=0 LTE regression row (charter eval protocol, job 1828240): see appended line below.

## Files
- diagnostics/Results/apost_ladder_20260709/case_<ckpt>_<variant>_<dT>.npz (12) +
  ladder_matrix_summary.csv + apost_refs_full_{1em2,5em3}.npz (truth reuse).
- Commit 6077fa3: --drop-nddot flag, consolidate_apost_cases.py, apost_matrix_job.sh.

## t=0 LTE regression row (job 1828240, kf4@1.5e-2 IC837, --track-lte, n_steps=1)
- UNCOND: rel_Ndot 0.117, rel_Nddot(t=0) 0.172 (== the known 0.19 plateau), rms_inj 4.65e-5.
- COND ep63: rel_Ndot 0.101, rel_Nddot(t=0) **0.136**, rms_inj 2.24e-5 — better than control
  but OUTSIDE the 0.023-0.05 acceptance band of the 07-08 addendum => per protocol the COND
  rollout rows carry the caveat MID-TRAINING CKPT (ep63/300, conditioning not yet at floor);
  matches the training log (val_Nddot ~0.136 at ep63), so NOT a wiring regression — the run
  simply is not converged. Protocol-order violation acknowledged: this row should have run
  BEFORE the matrix; it ran after (same conclusion either way, rows labeled).
- After ONE closure step on its own state: COND rel_Nddot 0.447 vs UNCOND 0.208; rms_inj
  flips (COND 1.37e-4 > UNCOND 6.17e-5). The cond model's advantage inverts as soon as its
  own corner-band-polluted history feeds back — mechanism for the earlier blow-up (step 7 vs 12).

---

# PART 2 (same day) — alias-safe projection matrix + ANALYTIC closure arms (job 1828315)

Sanaa rulings executed: solver mask UNTOUCHED (RED held); new `--nn-project-radius`
(commit df2297a) projects the R3 CORRECTION alone onto |k| <= (2/3)*min(kx_max,ky_max)
(mode radius 170.67 at 512^2). Same member/IC/horizon; ALL truth refs reused. COND =
frozen ep63 (1827306 still running, best.pt unchanged since freeze — best_val plateau).
18 cases in diagnostics/Results/apost_ladder_20260709_p170/ (one npz each + summary CSV).

## Analytic closure (r3anal: EXACT chain-rule Ndot/Nddot, no NN — Sanaa's question)

| variant | dT | blowup | final relL2 (bare) | impr | corner drift |
|---|---|---|---|---|---|
| analytic, std dealias | 1.5e-2 | none | 2.868e-4 (3.803e-2) | **132.6x** | 0.040 |
| analytic, std dealias | 1e-2   | none | 4.974e-5 (1.748e-3) | **35.1x** | 0.030 |
| analytic, std dealias | 5e-3   | none | 2.935e-6 (2.095e-4) | **71.4x** | 0.023 |
| analytic + proj170    | 1.5e-2 | none | 3.757e-2            | 1.01x | 2.884 |
| analytic + proj170    | 1e-2   | none | 1.244e-3            | 1.40x | 0.041 |
| analytic + proj170    | 5e-3   | none | 1.457e-4            | 1.44x | 0.024 |

**Verdicts.** (1) The analytic closure is STABLE and hugely accurate at ALL dT incl.
1.5e-2 — the instability is 100% NN-SPECIFIC, not intrinsic to the scheme or to the
aliasing annulus. The exact Nddot actively REGULATES the corner band (drift 0.04 vs
2.9 for the un-corrected drop arms at 1.5e-2). (2) Projecting the correction out of
the annulus destroys nearly ALL of its value (132.6x -> 1.01x at 1.5e-2): the
170.7-241.4 band is where the R3 correction's mass and benefit live (L^k weighting).
The annulus is simultaneously the value band and the NN-feedback band.

## NN matrix WITH the 170.67 projection (vs Part-1 unprojected in parens)

| ckpt | variant | dT | blowup | final relL2 | impr |
|---|---|---|---|---|---|
| uncond | full+proj | 1.5e-2 | none* | 1.543e+0 | 0.02x (was: blowup s12) |
| cond   | full+proj | 1.5e-2 | s8   | —        | (was: blowup s7) |
| uncond | full+proj | 1e-2   | none | 2.178e-3 | 0.80x (was 0.16x) |
| cond   | full+proj | 1e-2   | s16  | —        | (was: blowup s13) |
| uncond | full+proj | 5e-3   | none | 3.472e-3 | 0.06x (was 0.06x) |
| cond   | full+proj | 5e-3   | none | 9.562e-4 | 0.22x (was 0.72x) |
| both   | dropnddot+proj | all | none | = bare | 1.00x (unchanged) |

*no formal Z-blowup but rel-L2 1.54 and low-band drift 5.6 = accuracy collapse in
slow motion.

**Verdict.** The projection is NOT the fix: it delays/softens the feedback (cond
blowup 13->16 at 1e-2; uncond 0.16x->0.80x) but never beats bare, and at 5e-3 it
HURTS cond (0.72x->0.22x) because it amputates the band where the correction works.
Combined with the analytic rows: the NN must get the annulus RIGHT (its ~14-17%
Nddot error there feeds back), not be excluded from it. Points to training-side
fixes (rollout-aware/noise-injected fine-tune, annulus-weighted loss, or spectral
conditioning of the head) over inference-time masking.

## Upstream-fork note (independently verified on main the same day)
derivative.py:30-32: k_cut = sqrt(2)*(2/3)*min(kmax) — RADIAL. The solver RETAINS
the aliasing annulus (2/3)kmax < |k| <= 0.943 kmax. This is upstream fork code and
is self-consistent with ALL training data; changing it means rebuilding data =
flagged decision (RED), not an action.

---

# PART 3 (same day, Sanaa full mandate) — dealias-procedure audit (the "how many times" question)

## (a) The analytic chain — every dealias application, file:line

SOLVER (the fork; generates ALL training data):
- qg/solver/opt/operator/jacobian.py:12-18 — jacobian_pq computes the products
  u*q, v*q and assembles the Jacobian with NO internal mask.
- qg/solver/opt/operator/__init__.py:48 — Operator.source applies ONE dealias to
  the SUM of all RHS patches (jacobian + forcing + bc). So the solver dealiases
  ONCE PER RHS EVALUATION, post-sum.

HARNESS (rollouts + truth):
- rollout_timed_pareto.py:121-122 (_N_core) — uq_h, vq_h masked PER PRODUCT
  (2 per Jacobian); the AB2CN2 N eval of every arm. 2 mask-multiplies/step.
- rollout_timed_pareto.py:82 (J_phys) — same, inside rk4_step: RK4 truth = 4 RHS
  evals/step x 2 = 8 masked products/fine step.
- rollout_perfect_closure.py:86 (J_phys) + 113-114 (_N_core);
  analytic_n_derivs_hat (:171-199) — order k needs (k+1) Jacobians, each per-
  product masked. Intermediates w^(k), psi^(k) are NOT re-masked (see below).
- rollout_aposteriori.py — f_anal masked once after assembly; f_NN masked once
  (end-projection); e_r4 masked once.

TARGETS (training data):
- build_training_data_mmap.py:102-103 — per-product dealias in the deep builds.
- slice_deriv_from_deep.py:45-47 (J_phys) + compute_derivatives (:66-81) — per-
  product dealias per Jacobian term per order. Building [N, Ndot, Nddot, N3dot]
  = 2+4+6+8 = 20 masked products per anchor.

CORRECTION to Sanaa's stated understanding: products ARE masked after each
product (or once on the RHS sum — exactly equivalent by linearity: the mask
commutes with ik and distributes over sums). But the INTERMEDIATES
(w-dot, psi-dot, ...) are NOT explicitly masked before entering the next
product — and need not be: w^(k) = L w^(k-1) + N^(k-1) and psi^(k) =
inv_lap w^(k) are diagonal (band-preserving) images of already-masked
quantities plus the band-limited state, so they never exceed the retained
ball. Masking happens ONLY on products / RHS sums. What NO placement of the
mask does: remove fold-back pollution landing INSIDE the ball — see (c).

## (b) The model chain — end-projection only, justification confirmed

- model_deriv_closure.py / model_cond_local.py: ZERO dealias inside the model
  (cheap_deriv has no FFTs at all; cond_local's 2 rFFTs at
  model_cond_local.py:340-341 are the sigma-hat context only, no field masking).
- train_deriv.py:229-232: ONE end-projection of the PREDICTION before the loss.
- rollout_aposteriori.py: ONE end-projection of f_NN at inference.
- Structure: TimeFD (linear) -> SpatialGrad convs (linear) -> 16 Jacobian
  products (the SINGLE quadratic level) -> 1x1 mix (linear). End-projection
  == 16 per-product projections EXACTLY (mask commutes with the linear mix and
  distributes over the sum). The locked-fact justification is CONFIRMED — with
  one addendum: that equivalence holds for ANY mask radius; it guarantees
  ALIAS-FREEDOM only when inputs are band-limited to (2/3)kmax, which the
  sqrt2-world inputs are NOT.

## (c) The fold math — where aliased pollution enters (N=512, kmax=256)

- sqrt2 convention: state band-limited to R = sqrt2*(2/3)*kmax = 0.943 kmax
  (mode 241.4). A pointwise product of two such fields has true support to
  2R = 1.886 kmax (mode 482.7). On the N-grid, sums past Nyquist alias:
  fold floor = 2*kmax - 2R = 0.114 kmax (mode 29.3). Folds land EVERYWHERE in
  29.3 <= |k| <= 241.4 — INSIDE the retained ball — and NO post-hoc mask (per-
  product or end) can remove them: they are numerically indistinguishable from
  resolved content.
- strict 2/3: R = (2/3)kmax -> 2R = (4/3)kmax -> fold floor = 2kmax - (4/3)kmax
  = (2/3)kmax = R. ALL folds land at/above the cut and the product mask removes
  them EXACTLY. This is the classical 2/3 rule; the fork's sqrt2 radius buys
  44% more retained modes at the price of alias pollution down to 0.114 kmax.
- WHERE it enters, sqrt2 world: (i) every solver/truth/analytic Jacobian
  product — and it COMPOUNDS with derivative order (order k's Jacobians consume
  order k-1's polluted outputs); (ii) the NN's 16 products fold identically —
  but the TARGETS fold the same way (same field pairings), so a-priori the
  pollution is a deterministic, largely learnable component; at ROLLOUT the
  NN's folds act on its own self-generated history (the feedback loop).
  Truth and arms share the same polluted dynamics, so sqrt2-world comparisons
  are self-consistent — aliasing is a property of the dynamical system being
  simulated, not a scoring error.

---

# PART 4 (same day) — spectral error profiles (mandate Part 1) + the 2/3-world rollout (Part 3), job 1828403

## Part 1 — eps(k) profiles (val samples, 6/root, median; 512^2 FRC members; predictions
## end-projected with the solver mask = training convention)

Ckpts: cond = deriv7_cond_local_v2 frozen ep63; control = deriv7_filtered_floor0.1 (the
STANDING UNCOND REFERENCE per the brief, Nddot 0.186); rollout_unc = deriv7_filtered_lr5e-5
(ep8; the ckpt used in all rollout ladders — uniformly the weakest of the three).

Pooled (median over 14 root rows), Nddot / bands low(k<60) | mid(60-170) | annulus(171-241):
  cond         0.108 | 0.056 | 0.094      (Ndot 0.041/0.024/0.070; N3dot 0.367/0.106/0.189)
  control      0.290 | 0.095 | 0.129      (N3dot low 2.53 — inflated by tiny low-k target energy)
  rollout_unc  0.188 | 0.151 | 0.161

FINDINGS:
1. The profile is U-SHAPED, not a high-k cliff: rel error largest at LOW k (eps(30):
   cond 0.18, control 0.45), minimum at k~100-200 (0.043 / 0.072 / 0.114), rising toward
   the mask edge (eps(240): 0.16 / 0.21 / 0.27). KNEE (2x-min crossing): cond k=209,
   control k=232, rollout_unc k=235 — inside the annulus but late.
2. The annulus is only MODESTLY worse than mid a-priori (x1.4-1.7) — the rollout
   catastrophe is feedback amplification of a modest a-priori error, consistent with the
   stable analytic arms.
3. Conditioning (ep63) bent the profile mostly at LOW k: low 0.290->0.108 (2.7x), mid
   1.7x, annulus only 1.4x. Exceptions: b1/b2 @1.5e-2 where cond is WORSE than control
   in the annulus (0.31/0.52 vs 0.12/0.41) — the near-wall large-dt rows.
4. rollout_unc (ep8) is the weakest ckpt everywhere — today's sqrt2-world UNCOND rollout
   rows understate what the CONTROL ckpt would do.
5. ANNULUS-WEIGHTING PROPOSAL (design only, NO training launched): per-shell floored
   rel loss with w(kappa) = 1 + lambda*1[171<=kappa<=241], lambda ~ 3 (calibrated so the
   annulus term rises from its ~x1.4 error share to parity with low+mid at init), PLUS
   rollout-aware noise injection (I16 R4) — the profile says a-priori weighting alone
   attacks the wrong band ordering (low-k is worse a-priori), so weighting should be
   paired with feedback-aware training, not replace it.
   Data: diagnostics/Results/spectral_error_profile_20260709/spectral_error_profile.npz.

## Part 3 — the 2/3 WORLD (whole harness incl. RK4 truth under radial (2/3)min(kmax);
## IC projected; NEW refs apost_refs_w23_*; kf4 IC837 M=16; ckpts trained on sqrt2 data
## => train/eval MASK MISMATCH, stated caveat)

| arm | dT | blowup | final relL2 (bare) | impr | low-k drift |
|---|---|---|---|---|---|
| analytic | 1.5e-2 | none | 2.710e-4 (6.400e-3) | 23.6x | 0.036 |
| analytic | 1e-2   | none | 8.050e-5 (1.310e-3) | 16.3x | 0.025 |
| analytic | 5e-3   | none | 1.673e-5 (1.587e-4) | 9.5x  | 0.013 |
| uncond   | 1.5e-2 | s12  | — | — | 4.50 |
| uncond   | 1e-2   | none | 4.042e-1 | 0.00x (collapse) | 2.81 |
| uncond   | 5e-3   | s14  | — | — | 23.3 |
| cond     | 1.5e-2 | s6   | — | — | 2.52 |
| cond     | 1e-2   | s7   | — | — | 0.30 |
| cond     | 5e-3   | s6   | — | — | 0.015 |

(corner-band sigma drift is MEANINGLESS in this world — shells >170.7 are empty by
construction; low-k (k<60) drift is the health metric here.)

ANSWER TO THE KEY QUESTION: **NO — in the alias-clean world the ML closure does NOT get
close to the analytic closure; it gets strictly WORSE than in the sqrt2 world** (uncond
now blows even at 5e-3 where it was stable; cond blows at all three dT). The analytic
closure stays stable and strong in BOTH worlds (23.6x/16.3x/9.5x here; gains smaller
than sqrt2's 133x/35x/71x because BARE improves ~6x in the clean world — the annulus
was a large share of bare's error). Interpretation under the stated caveat: the
train/eval mask mismatch dominates — the sqrt2-trained stencils absorb that world's
band structure (the Wiener mechanism) and break on 2/3-band states. CONSEQUENCES:
(1) aliasing per se is NOT the NN's problem (removing it entirely makes the NN worse,
analytic fine both ways); (2) since the answer is NO, the mandate's conditional
("if YES -> FFT-free per-product dealiasing design note") is NOT triggered; (3) the
clean discriminating test requires a mask-MATCHED model: retrain (or fine-tune) on
2/3-world data, or rollout-aware fine-tune in the sqrt2 world — both training-side.
Low-k answer: in the 2/3 world low-k stays clean under the analytic closure (1.3-3.6%
drift); NN arms corrupt low-k through error feedback, not through aliasing; a-priori,
low-k is every model's WORST relative band (Part 1) independent of aliasing.

Files: diagnostics/Results/apost_ladder_20260709_third23/ (9 case npz + summary CSV +
3 new refs; originals untouched).
