# RESULTS 2026-07-08 (smoke3) — r3only verified, K fixed, blowup mechanism ISOLATED

Session scope: Sanaa's four checks on the smoke-2 report (r3only story; full analytic LTE
diagnostic; developed-flow / prefactor / sign triple-check; K too coarse). All four done.
Artifacts: `diagnostics/Results/apost_smoke3/` (job 1827061). Old smoke2 artifacts kept.

## 1. r3only ≈ bare is PHYSICAL (verified two independent ways)

- Independent numpy recompute at IC kf4/820 (no driver code): `c12·rms(L³ω) = 1.420e-8`
  matches the driver diag to all printed digits. Predicted 4-step analytic-term effect
  3.16e-7 == observed bare→r3only rel-L2 gap 3.0e-7.
- Why: kf4/b2 have ν=1.025e-4, μ=0.02 → rms(N̈)≈259 (kf4@1.5e-2) vs rms(L²N)≈0.4.
  The R3 bracket is ≥99.9% N̈ for low-ν members even at ΔT=1.5e-2 — same budget as the
  error-prop result (L-terms ≈ 0% of ‖δ‖). Removing only the L-pieces cannot move the error.
- Functional proof: the new `r3anal` arm (below) swaps EXACT chain-rule Ṅ/N̈ into the
  identical assembly and the error collapses 20–120×. Signs/prefactors are right.

## 2. Prefactors / signs / developed flow — all check out

- Assembly verified term-for-term against validated `rollout_perfect_closure`:
  coef=ΔT³ (no (1−1/K²); K = truth refinement only), f_NN=(1/12)(LṄ−5N̈) subtracted,
  implicit L³ fold algebraically equal to the explicit form at the claimed order
  (O(h⁴) difference BY DESIGN — noted in-code). L̂ = −ν|k|²−μ+iβk_x/|k|² exact.
- ΔT³/12 cross-check: diag 5·N̈ term 3.56e-4 == (ΔT³/12)·5·rms(N̈_target=259) ✓.
- Developed flow: IC rows 820/837 (kf4), 964/934 (b2) are all KEPT by the quiescent
  filter, target norms 0.83–1.12× member median. (Smoke-2's original pathological IC was
  row 0, filter-dropped — already reported.)
- closure-reviewer + sge-checker audits: clean (two low-severity items fixed:
  closure2-without-ckpt2 fast-fail; job-script mail directives).

## 3. K fixed — and the old K=20 truth was NOT materially polluted

- New rule (driver docstring + runtime warning at h_fine>2.5e-5): accuracy runs need
  h_fine=ΔT/K ≤ ~1e-5 — the truth stands in for the ANALYTIC flow because we deliberately
  do NOT model τ_RK4 (T5 formula → too-deep network).
- Measured old-truth error, rel-L2(K=20 truth, h_fine=1e-5 truth) at matching checkpoints:
  ≤1.7e-9 across all five smokes — 5 orders below the closure errors. Old smoke-2 gains
  stand; the K fix is hygiene + principled requirement, not a correction of the conclusions.
- Truth refs saved (`apost_refs_smoke3*.npz`) — reuse with `--load-refs`, the truth is
  the expensive leg (24k RK4 steps @512² ≈ 1.7 min GPU per run).

## 4. NEW: r3anal arm + --track-lte (the full-analytic-LTE diagnostic)

- `r3anal`: AB2CN2 + full analytic R3 (exact chain-rule Ṅ/N̈ per step, no NN) through the
  IDENTICAL driver assembly — only the [Ṅ,N̈] source differs from `closure`.
- `--track-lte` → `lte_<tag>_<arm>.csv`: per checkpoint, per-term rms of the analytic LTE
  at the arm's own state; closure arms add rel-L2(NN vs analytic) per head and the injected
  error `inj = ΔT³·rms(f_NN − f_anal)`.

## 5. smoke3 results (h_fine=1e-5 truth; bare / r3only / r3anal / closure)

| run | member@ΔT (IC) | CFL | bare final | r3anal final | closure |
|---|---|---|---|---|---|
| 3a | kf4@1.5e-2 (820) | 0.85 | 3.78e-2 (t=.24) | 3.18e-4 (**119×**) | 6.4× at step 1 → BLOWUP step 12 |
| 3a_val | kf4@1.5e-2 (837) | 0.85 | 3.80e-2 | 2.87e-4 (**133×**) | 7.3× step 1 → BLOWUP step 12 |
| 3b | b2@5e-3 (964) | 0.73 | 2.48e-3 (t=.06) | 9.70e-5 (**26×**) | 5.1× step 1 → BLOWUP step 11 |
| 3b_val | b2@5e-3 (934) | 0.72 | 2.55e-3 | 9.50e-5 (**27×**) | 5.5× step 1 → BLOWUP step 11 |
| 3c | kf4@1e-2 (820) | 0.56 | 1.90e-3 (t=.16) | 5.59e-5 (**34×**) | 6.4× step 1 → 0.2× by t=.16, no blowup ≤16 |

r3only ≈ bare everywhere (expected, §1). Blowup steps identical to smoke2 (truth-independent).

## 6. VERDICT: the blowup is NN-INJECTED noise feedback, NOT marginal-AB2

The smoke-2 ambiguity ("pure NN feedback" vs "NN kick tips marginal AB2 at CFL 0.85") is
resolved without the dt-sweep discriminating run:

1. **r3anal is stable at the same CFL 0.85** carrying the same anti-dissipative −5N̈
   correction at full magnitude, for the full horizon, 119–133× better than bare.
   The scheme + correction structure is fine; only the NN version detonates.
2. **closure diverges exponentially at CFL 0.56 too** (3c): rel-N̈ error 0.19 → 29 by
   step 16, crossover t≈0.13, blowup extrapolates to step ~18–19. No CFL cliff involved.
3. **LTE track quantifies the loop**: NN error at the closure's own state is flat at the
   val level (rel N̈ ≈ 0.19–0.25) for the first 2–4 steps, then grows exponentially —
   doubling per ~1.5 steps (3a), ~1 step (3b), ~2 steps (3c). The injected correction
   error `inj` crosses the true LTE τ ≈ 3.6e-4/2.9e-4/1.1e-4 at steps ~7/5/12; Z-blowup
   follows 3–5 steps later. Blowup horizon ≈ (steps until inj ≈ τ) + ~4.
4. Structural reading: r3anal needs only the CURRENT state (chain rule); the NN reads the
   7-lag HISTORY through frozen TimeFD rows with 1/dt², 1/dt³ weights. NN-written high-k
   noise enters the history, is lag-decorrelated, gets 1/dt^k-amplified into the deriv
   features, and returns as a noisier correction — a history-contamination loop the
   analytic closure cannot have. Low-ν members (no viscous sink at high k) are the worst
   case, consistent with §1.

## 7. Consequences

- **Per-step gain is real and generalizes** (5–7× step-1 on train AND val rows; r3anal
  ceiling 26–133×). The per-step NN error (val N̈ rel-L2) remains THE target — cond_local
  + enlarged ensemble attack exactly this; run 1827034 in flight.
- **Blowup horizon is now measurable ex-ante**: track `inj/τ` from the LTE CSV; the arm
  dies ~4 steps after inj/τ→1. Add both (horizon + inj/τ growth rate) to the cond_local
  eval alongside rel-L2.
- Open (not implemented, for discussion): inference-side noise control in the closure arm
  (e.g. spectral floor / stronger dealias on NN outputs, or history written from the
  UNcorrected step) — would test the loop-gain story directly but changes the method.
