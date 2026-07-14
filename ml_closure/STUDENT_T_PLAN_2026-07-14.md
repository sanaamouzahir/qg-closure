# Student-t likelihood plan (sigma fix) — HELD until Sanaa reviews the per-member recal plots

Sanaa direction (2026-07-14 chat): "I like the t-student idea. We could start
with the pretrained gaussian model and move to the heavy tailed one."

## Why (the evidence chain)
1. Structural prior sigma^2 = sp(a) + sp(b)*g^2 has the right SHAPE
   (Spearman sigma-vs-|grad| 0.75-0.92) but inflated LEVEL.
2. The 2-param recal fixed the level (held-out NLL -2.22->-2.73 cyl /
   -1.73->-2.29 cape) but coverage@1sigma stalls at 0.96-0.98: the fitted
   floor lands BELOW the validated clamp (1e-3 std-var) AND the residuals
   have kurtosis ~1e4-2e4 — a Gaussian cannot be simultaneously
   NLL-optimal and quantile-calibrated on such tails. Both trainers'
   B-item flags said the same (e.g. LOMO fold-1: "kurtosis 1129 -> RAISE
   B-ITEM (heteroscedastic / Student-t)").

## Design
- Custom gpytorch likelihood: one-dimensional StudentT with
  scale^2(x) = het_noise(g) (the EXISTING structural head, reused) and a
  single learned dof parameter nu (softplus(raw_nu) + 2 so variance
  exists; init from residual-kurtosis moment match, nu ~ 4-6 as fallback).
  expected_log_prob via Gauss-Hermite quadrature (gpytorch
  _OneDimensionalLikelihood default machinery); predict_physical already
  reduces sample-dim marginals (the arm-C comment in model_piff.py).
- WARM START from piff_{fpc,cape}_gjs best.pt (Gaussian structural):
  everything loads except the likelihood object; noise_a/noise_b/g2_scale
  carry over 1:1 (same head). Short retrain (~50-100 ep, grid-winner
  lr/wd, cosine) — likelihood swap + joint refinement, per Sanaa
  "start with the pretrained gaussian model and move to the heavy tailed".
- Config knob: model.likelihood: gaussian | student_t (default gaussian —
  no behavior change for existing configs). Trainer/eval need: NLL under
  t; coverage vs t-quantiles (1sigma-equivalent = 68.27% central interval
  of the t at learned nu, NOT the Gaussian z=1) — else calibration is
  judged against the wrong yardstick.
- The arm-F clamp stays on the SCALE head (band re-check is part of the
  acceptance: with honest tails the scale should not need to sit on the
  floor).

## Acceptance (pre-registered)
1. Held-out central-68% coverage in [0.63, 0.73] on BOTH geometries
   (t-quantile yardstick).
2. Held-out NLL beats the recalibrated Gaussian (-2.73 cyl / -2.29 cape).
3. Wake R2 of the mean unchanged within noise (+-0.005) — the likelihood
   swap must not degrade the mean.

## Cost
~2 GPU-h/geometry (100 ep at gjs epoch times) + the likelihood code
(G4-reviewed). NOT FIRED — awaiting Sanaa's plot review.
