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
