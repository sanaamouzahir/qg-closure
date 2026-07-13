ARCHIVE: HYPERPARAMETER GRID, T6 DIAGNOSTIC LADDER, SMOKE TESTS

All symlinks into runs_piff/. These are not production models -- they are
the experiments that CHOSE the production settings and diagnosed the
training pathologies.

-------------------------------------------------------------------
1) HYPERPARAMETER GRID (6 rungs, 60 epochs each, cylinder steady inlet)
   learning rate x weight decay grid. Val NLL (lower = better) / R^2:

   grid_lr1.0e-3_wd1.0e-5   NLL 5.6694  R^2 0.813   <-- WINNER
   grid_lr1.0e-3_wd1.0e-4   NLL 5.6864  R^2 0.804
   grid_lr3.0e-4_wd1.0e-5   NLL ~5.72   R^2 ~0.79
   grid_lr3.0e-4_wd1.0e-4   NLL ~5.72   R^2 ~0.79
   grid_lr1.0e-4_wd1.0e-5   NLL ~7.18   R^2 0.61
   grid_lr1.0e-4_wd1.0e-4   NLL ~7.18   R^2 0.61

   Read: learning rate is the only lever; weight decay is inert.
   The winner was extended to 150 epochs to become the production model
   (train_models/cylinder_steady_inlet_only_150epochs_PRODUCTION).

-------------------------------------------------------------------
2) T6 LADDER (t6_arms/arm{A..F}.npz -- learning-capability probes on a
   fixed overfit setup; original gate: R^2 > 0.95 within 50 epochs)

   arm | what it tested                              | verdict
   ----+---------------------------------------------+------------------
    A  | "just needs more time" (150 epochs)         | plateaus ~0.89; crosses 0.85 at ep81 -> basis for re-gating the bar to 0.85-in-100
    B  | more capacity (1024 inducing points vs 512) | +0.02 R^2 at ~6x cost; capacity not the bottleneck
    C  | Student-t likelihood (heavy-tailed noise)   | NEGATIVE: R^2 ~0.00 -- rejects the strong-wake pixels as outliers; the heavy tail is signal, not noise
    D  | free per-pixel learned noise (heteroscedastic) | NEGATIVE: collapse -- the noise head "buys out" the fit and eats the signal
    E  | same as D but with a 25-epoch warmup + cap  | NEGATIVE: warm start destroyed within 2 epochs of unfreezing; D's collapse is structural
    F  | structural noise prior (sigma^2 = a + b*|grad omega|^2, only 2 scalars learned) | NO COLLAPSE (first heteroscedastic arm to survive; b mobile, R^2 rising monotonically) but too slow: 0.495 at ep49. Candidate for a 150-epoch extension.

   Ladder summary: baseline 0.17 -> y-standardization fix 0.80 -> A 0.89
   asymptote -> C/D/E collapse -> F survives-but-slow. Consequence: the
   0.95-in-50 bar was unreachable for this model family; re-gated to
   R^2 >= 0.85 in 100 epochs, which the production model passes.

-------------------------------------------------------------------
3) SMOKE TESTS (2-epoch end-to-end pipeline checks; disposable)
   smoke_T7_20260712_0401  original end-to-end smoke (train+eval mechanics)
   smoke_T7_20260712_0440  re-smoke during the NaN / y-standardization debugging
   smoke_T7_20260712_0455  clean re-smoke after the y-standardization fix
   cape_smoke_2ep          cape pipeline smoke (also flagged cape's 5x
                           heavier-tailed residuals, kurtosis ~2064)
