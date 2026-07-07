# BRANCH_LOG — Physics-conditioned spatial stencil  (branch: exp/wiener-conditioning)

Running record. Supervisor updates this at the end of every session. Newest entry on top.

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
