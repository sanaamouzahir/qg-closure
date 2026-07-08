# BRANCH_LOG — Physics-conditioned spatial stencil  (branch: exp/wiener-conditioning)

Running record. Supervisor updates this at the end of every session. Newest entry on top.

## 2026-07-08 — session 3 (global work order: driver rework, first cond_local submission, smokes)
- TASK A (54b734e, ACCEPTED by Sanaa): rollout_aposteriori truth = RK4 @ dT/K (imports
  rollout_fine); coef = dT³ / coef4 = dT⁴ ALL arms (no (1−1/K^n)); K = truth refinement only.
  Minimal-FFT step (bare 5 / closure 8 via N_spectral_fields), untimed warmup (old 8.4s closure
  smoke walltime was lazy-init), Parseval E/Z (zero FFTs, IC guard). σ̂-from-stepper:
  cond_grad.sigma_hat_spec + cond_local forward(cond_feats=...) — zero extra FFTs, training
  path bit-identical. Reviewer caught 3b calling the deleted API → ported. Smoke re-PASS.
- TASK B: deriv7_cond_local submitted (41 roots = 17 control + new8×3, minus Re25k@1.5e-2).
  Job 1826982 CRASHED 4 min in — the MULTIGRID TRAP: 512²/2π (DEC-512) batched with 512²/4π
  (FRC); cond_local's grid-uniform-batch σ̂ guard threw. FIX (b1cee46): shells are mode-index
  shells for square domains (kmag~1/L cancels) → per-sample squareness guard + canonical-L
  context; bit-identical (init gate re-PASS exact 0.0; mixed-dx real-data batch == per-sample
  to 0.0). RESUBMITTED as 1827034 (13:52, -j y this time); monitors 1827035 (-hold_jid) +
  1827036 (live watchdog). cond_deriv has the same guard flaw — not fixed (dead instrument).
- SMOKES 2a/2b/2c + val-IC (b6989c8): closure (deriv7_filtered ckpt) beats bare EARLY on
  developed ICs — 7.7–8.2× kf4@1.5e-2, 4.0–4.3× b2@5e-3 — then BLOWS UP step ~11–12 (Z 10²–10³×);
  r3only stable ≈ bare. kf4@1e-2: no blowup ≤16 steps, crossover by t=0.16 → divergence rate
  grows with dT. Yesterday's b2 smoke IC (row 0) is DROPPED by the quiescent filter from every
  split — pathological zonal state, explains bare 2e-8. VAL-row reruns match train-row (no
  leakage). --diag: NN = ~100% of correction mass, 5·N̈ = 99.3–99.7% (low-ν members, no viscous
  sink). physics-sanity: MIXED leaning physical (NN-noise feedback vs NN-kick-on-marginal-AB2 at
  CFL 0.85 not separable yet; its discriminator = --restart-ic dT sweep at fixed physical horizon,
  needs GO). NEW METRIC for the cond eval: blowup horizon alongside rel-L2.
- D ITEMS 1–6 all approved & LANDED (b6989c8): --save/load-refs, --pareto (reviewer MAJOR: bare
  dtb leg needs RK4 back-step seed, else dt¹ startup floor flatters the closure; same flaw exists
  in rollout_timed_pareto's sweep — flagged, not fixed there), --profile-step (+3b flag),
  σ̂(κ,t) checkpoint CSVs, --freeze-sigma, --ckpt2/'closure2'. improvement_x per closure arm.
- Emails: LANDED (Task A), PROPOSE (D costs) → all approved, SUBMIT (1826982), RESUBMIT
  (1827034), LANDED (smokes), LANDED (D items). NEW EMAIL FORMAT per Sanaa (ADHD-friendly:
  CAPS+bold titles, indented spaced numbered points) — saved to agent memory.
- STATE: 1827034 running (~800s/epoch expected, ~2.8 days). Watch val_Nddot; success bar:
  kf4@1.5e-2 ≤0.023, FRC-256@1.5e-2 ≤0.037, FRC-256@1e-2 ≤0.0055, pooled ~0.04–0.05 vs 0.19.
  Tomorrow: eval via the ONE driver (cached truth + live/frozen-σ̂/control legs + drift CSVs).

## 2026-07-06 — session 1 (cond_deriv integration + acceptance, branch supervisor)
- Synced origin/main into the worktree (merge 0866cc0): brought in `Theoretical_guarantees/`
  {cond_grad.py, conditioned_parameterization_note.md, THEORETICAL_GUARANTEES.md, checks}.
  Note: system git 1.8.3.1 cannot drive this worktree — use `/opt/rocks/bin/git` (2.9.2).
  Symlinked `training/data` → package-stable `.../src/qg/training/data` (data is gitignored;
  worktree shares code only). Excluded locally; never commit the symlink.
- Job 1 (Fable, `[fable-authored]` 1033f14): `build_model('cond_deriv')` = cheap_deriv pipeline
  with SpatialGrad→SpectralCondGrad. New `CondDerivClosureNet`; `training/cond_grad.py` (prod
  copy of the design module); `--model {cheap_deriv,cond_deriv}` in train_deriv.py. ORDER CLIP +
  frozen binomial mix preserved; context computed once/forward; NO local stencils. 2932 params
  (SpectralCondGrad 2832 + mix 51 + inert TimeFD 49). cheap_deriv unaffected (still 3,700).
- STEP A — ACCEPTANCE: **PASS.** `diagnostics/diagnose_cond_init_sanity.py` (new): 4/4 probes
  (FRC-256@5e-3 256², kf4@1e-2, combo@5e-3, Re25k@1e-2 512²) → **rel(model, exact-spectral-
  advective) ≈ 5e-16** on all of N1/N2/N3. SpectralCondGrad zero-init exactness is bit-exact
  end-to-end; Fable's wiring is clean. layer.grad == solver spectral derivative to 2.3e-16.
- SCIENTIFIC CORRECTION to the kickoff's STEP A phrasing: cond_deriv does NOT (and cannot) match
  the FLUX-form `[spec]` floors to 1e-12 — cond_deriv is ADVECTIVE-form. The gap (rel(M,[spec]) =
  7e-3/1.4e-2/1.0e-1) is the **advective-vs-flux discrete form difference** (CLAUDE deferred item,
  shared with cheap_deriv), NOT a wiring bug. The true zero-init test is vs a spectral-ADVECTIVE
  reference (→5e-16, above). diagnose_one_sample.py gained `--model`; its `[model]` row = 0.0073/
  0.0135/0.0886 vs target (norm ratio 1.000) — the exact-spectral-advective floor.
- MINOR (noted, not fixed — preserve control comparability): TimeFD's `W_unit` buffer is float32
  (`.to(torch.float32)` at construction), injecting ~2.7e-4 into assembled N2dot vs a fully-f64
  pipeline (psi order-2 field differs 14% vs f64 vandermonde). Pre-existing cheap_deriv behaviour,
  far below the ~0.04–0.19 science floor. Candidate one-line f64 fix for a FUTURE run; would break
  bit-comparability with the control, so out of scope here.
- Control to beat (`deriv7_filtered_floor0.1`): Ndot 0.058 | **Nddot 0.186** | N3dot 0.563.
- Decided next: STEP B — sge-checker audit + draft/propose the deriv7_cond qsub (FRC-* minus
  Re25k@1.5e-2, 300ep lr5e-5 f64 bs4 rel-floor0.1), then chain the monitor.
- STEP B — SUBMISSION DRAFTED + PROPOSED (gated on Sanaa): `scripts/sge/train_deriv_cond_job.sh`
  (worktree-scoped worker — the shared wrapper cd's into package-stable which lacks cond_deriv) +
  `scripts/sge/submit_deriv7_cond.sh` (dry-run default, `--go` to fire; asserts 17 roots; -j y;
  -m ea). 17 FRC roots minus Re25k@1.5e-2. sge-checker PASS on all hard rules. CPU trainer smoke
  (FRC-256@5e-3, 1 ep) ep0 val 0.160 — trains clean, no explosion. `[QG][SUBMIT]` sent; awaiting go.
  NOTE: guard hook substring-matches — never put the literal forbidden queue/mem tokens in a bash
  command (even inside an email body) or it blocks the whole call.
- GREEN-LANE (diagnostics/diagnose_sigma_drift.py): sigma-hat conditioning input is STABLE
  anchor-to-anchor — median |dx|/x = 0.5-0.85% across FRC-256/kf4/Re25k and dt 5e-3..1.5e-2
  (band = x >= 1% of per-window max). Data-conditioning is well-posed in the learnable regime.
  Heavy tail (p99 30-55%, max >10x) sits at near-saturation shells (x -> pi, arcsin cap) = the
  near-wall Prop-2 region already flagged unlearnable; watch that the MLP does not overfit it.
- Emails: `[QG][LANDED][wiener-conditioning]` acceptance passed; `[QG][PROPOSE][wiener-conditioning]`
  adopt diagnose_cond_init_sanity as the pre-training gate; `[QG][SUBMIT][wiener-conditioning]`
  deriv7_cond primary run (awaiting approval).
- STATE: blocked on submit approval. On go: `submit_deriv7_cond.sh --go` then chain STEP C monitor.

---
## 2026-07-06 — session 0 (seed, by global supervisor)
- Sanaa asked for: create branch + seed brief, then STOP (theory-first).
- Ran / submitted (job ids): nothing — branch is theory-gated.
- Results: n/a.
- Flags from physics-sanity: none.
- Decided next: on receipt of the Wiener parameterization, scope the conditioned-stencil module
  `δ_θ(k)=dT^(S−m)·[analytic] × g_θ(k;Re,β,μ)`; until then, no work.
- What Sanaa wants to see next check-in: the parameterization delivered → first design proposal.
- First email to send: `[QG][BLOCKED][wiener-conditioning] awaiting parameterization from Sanaa`.

---
## Seed
- Hypothesis: δ★(k) ∝ −ik·C_m·(dT·σ(k;Re,β,μ))^(S−m); conditioning on (Re,β,μ) removes the
  pooled-variance floor the control plateaued at (the dT^(S−m) factor is analytic, only g_θ learned).
- Success criterion: pooled TEST Nddot < control 0.186 at equal data.
- Control to beat (`deriv7_filtered_floor0.1`): Ndot 0.058 | **Nddot 0.186** | N3dot 0.563 (Nddot = ceiling).
- Status: BLOCKED — theory first; no code/data until Sanaa delivers the parameterization.
