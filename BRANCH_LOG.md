# BRANCH_LOG — Physics-conditioned spatial stencil  (branch: exp/wiener-conditioning)

Running record. Supervisor updates this at the end of every session. Newest entry on top.

## 2026-07-08 — session 5 (ORDER 1 resumption after supervisor disconnect)
- Triage job 1827252 (diagnostics/diagnose_condlocal_triage.py, 42 roots, best.pt of incident
  1827034 vs fresh physics-init "zero", --max-per-root 48 --d2) COMPLETED 21:14Z; evidence
  committed as diagnostics/triage_condlocal_D1.csv.
- D1 VERDICT: REAL BREAKAGE (not an unfloored-eval artifact). med≈mean and raw≈floored on
  every root; best.pt median Ndot > 1 on 16/42 roots. Pooled med-of-root-medians, best vs
  zero: Ndot 0.284 vs 0.196, Nddot 0.132 vs 0.257, N3dot 0.562 vs 0.327. Structure: FRC-only
  Ndot blowup scaling ~dt^-3 (FRC Ndot med best/zero = 18.6x @5e-3, 2.4x @1e-2, 1.1x @1.5e-2;
  DEC improves, 0.5-0.7x) => wrong ΔT power in cond_local's m=1 (Ndot) head conditioning,
  anchored at the largest dt. N3dot best WORSE than zero on 27/42 roots (global, not DEC-512
  quirk). Nddot is the one order training genuinely improved (0.26 -> 0.13).
- D2 VERDICT: ALIGNED, all 10 probed TASK-0c members — recompute-from-ch0 relerr <= 9.5e-14
  (mostly bit-exact 0); ch1 off-by-one probe clearly distinct (2e-2..8.7e-1); psi0-consistency
  ~3e-8 == float32-disk quantization (sanctioned). No finalize_partial_build off-by-one.
- D3 VERDICT: CLEAN — zero-init raw medians uniform across all four grid identities
  (Ndot 0.16-0.23, Nddot 0.24-0.27); no per-sample dx-rescale bug. (256²/4π N3dot 0.73 is the
  DEC-loRe/base small-target tail, not a grid effect.)
- Secondary finding (floor-audit): regime[6:9] stored member-medians are pre-filter low-row
  estimates and are 21x-46,343x too SMALL on FRC-b0/b05/b075/b1 => the 0.1-floor is inert for
  those members and their val splits still contain ~100-2000x-under-median target samples.
  Candidate follow-up: recompute regime[6:9] post-filter (not done this session).
- F1 fix (training/train_deriv.py): confirmed val/best-selection ALREADY uses the floored
  denominator (same as loss) — the suspected unfloored-val corruption does not exist. Added
  per-order MEDIAN (same floored denominator) computed over all val samples, printed each
  epoch next to the floored-mean and APPENDED to log.csv as val_med_{Ndot,Nddot,N3dot} after
  elapsed_s (existing columns/positions preserved). New helper relative_l2_persample();
  run_epoch now returns (mean, per-order mean, per-order median). CPU smoke PASSED: 1 epoch,
  DEC-loRe sweep_dT_5em3 (FRC-256 killed — 1263 samples too slow on CPU), f64, params=3,700,
  ep0 print "[mean: Ndot=4.537e-01 ...] [med: Ndot=4.578e-01 ...]", TEST line + log.csv
  epoch/test rows carry val_med_{Ndot,Nddot,N3dot}. (DEC-loRe N3dot ~18-20 = that member's
  known small-target tail, matches triage zero-init 24.8 — not a regression.)
- F2 authority sits with the GLOBAL supervisor. Branch verdict reported: job 1827216 reruns
  the incident config unfixed and will reproduce the dt^-3 Ndot divergence (incident last.pt
  ep6 val 3.476 > ep0); recommend qdel 1827216/1827217 and fix the cond_local Ndot-head ΔT
  conditioning first. Hygiene control 1827225/1827226 unaffected — keep running.
- No pushes, no emails (global supervisor consolidates).

## 2026-07-08 — session 4c (I16 playbook run: STEP-1 no-bug, ladder, control reframe, resubmits)
- STEP 1 (bug hunt) — NO BUG: (a) lte_smoke3a_val_closure.csv rel_Ndot/Nddot(t=0) =
  0.117/0.172 (a sign-flip reads ~2.0); (b) NEW diagnostics/diagnose_head_sign.py (DECISIONS.md):
  signed corr per head on 3 val samples = +0.9935/+0.9851/+0.8543 → SIGN-OK; (c) inj/τ crosses 1
  at step 4→5 (3a_val), blowup at 12 ≈ crossover+7 (same +7 in 3b: cross 3→4, blow 11).
- Sanaa's addendum ADOPTED: ckpt is the CONTROL; rel_Nddot(0)=0.172 == the 0.19 pooled plateau
  (per-tier gap vs kf4's raw floor 0.031 = the (ii) compromise). NO ANOMALY. Ladder reframed as
  characterizing the control (paper "before" leg). Tomorrow's cond_local eval: run the t=0 LTE
  row FIRST as a regression detector — acceptance rel_Nddot(0) ∈ 0.023–0.05 on kf4@1.5e-2/IC837;
  ~0.17 ⇒ training regression, FLAG before any rollout conclusion.
- Driver: --nn-kcut (R1) / --nn-gamma (R2) / --nn-clip (R3) added; closure-reviewer G3 PASS
  (F1 ladder crash-masking fixed; F2/F3 informational). LTE rms_inj stays RAW by design.
- Ladder run 1 (1827104): all rungs crashed — apost_smoke3 npz/csv/json WORKING COPIES were
  externally deleted ~15:30–16:26 (committed csv/json restored via checkout-index from 1035414;
  refs npz regenerating). Ladder rerun = 1827220 (refs regen + crash-abort fix).
- Queue wipe explained: Sanaa qdel'ed the deriv7_con* trio (qstat truncates all three names to
  'deriv7_con' — looked like duplicates; 1827034 exit 137 at ep5/300). freeW 1825543 also gone
  (ep65, best val 0.291) — other branch's call, NOT resubmitted here. RESUBMITTED (I14, single
  instances, DISTINCT names): condlocal_train=1827216 (+condlocal_mon 1827217 hold; live
  watchdog dropped — it caused the duplicate confusion). Naming convention adopted: trainer and
  monitor names must differ in the first 10 chars (qstat truncation).
- Hygiene ablation (Sanaa PROPOSE, decided GO): first submit 1827218 died 13 s in — MY flag
  error (train_deriv.py has no --model; exit 2 argparse, pre-dates any qdel). Corrected
  resubmit: hygiene_train=1827225 (+hygiene_mon 1827226), 17 control roots minus Re25k@1.5e-2,
  unconditioned, --rel-floor 0.1, run deriv7_hygiene. Predicted plateau ~0.05 pooled.
- CHARTER v1.2 (I16 anomaly playbook, I17 one-document rule) appended to SUPERVISOR_BRIEF;
  DECISIONS.md created (I17 bootstrap).

## 2026-07-08 — session 4b (CHARTER v1.1 retrofit, per Sanaa's [QG][GLOBAL] directive)
- NOTE: the verbatim amendment text (I12–I15, 6.1–6.3, git-visible status) is EMAIL-ONLY — not
  in any committed tree or checkout. Operational form encoded in SUPERVISOR_BRIEF.md §CHARTER
  v1.1 adoption; needs the real file landed on main (global supervisor) to replace it.
- logs/: branch-root `logs/` created; ALL 45 scripts/sge/*.sh rewired — `#$ -o/-e` (or qsub-arg
  `-o/-e`) → `qg-wiener-conditioning/logs/$JOB_NAME.$JOB_ID.log|.err` (free-time-fd pattern).
  33 modified, 12 legitimately untouched. bash -n all-pass; sge-checker full-dir audit run.
- sge-checker also caught PRE-EXISTING hard-rule violations in legacy wrappers — fixed: forbidden
  CPU queue + per-job vmem requests removed (submit_qg, submit_step1/2/3, submit_pi_ff → ibfdr.q),
  gpu=0→gpu=1 (submit_pi_ff + compute_pi_ff comment), missing -m ea -M added (submit_qg, train,
  train_v2, train_decay_fixD_v2_annealing, train_deriv_condlocal_job), 5e-5→5.0e-5-style float
  tokens (submit_deriv7_cond, train_decay_fixD_v2_annealing), stale forbidden-token comment
  examples scrubbed (qg_job.sh).
- I14: monitor_training.py EXPLODE verdicts now print "ACTION per I14: qdel + diagnose +
  resubmit (branch authority; not BLOCKED)".
- I15: this branch already tighter than K≥100 (h_fine ≤1e-5 rule + driver warning; smoke3 used
  K=1500/500/1000). Nothing queued at K<100. All smoke tables carry the smoke label in the name.
- Item 3 (SUBMIT emails for running jobs): NO-OP — 1827034/35/36 emailed (session 3),
  1825543/44 emailed (free-time-fd BRANCH_LOG line 13).
- Deferred to global: 7 sweep wrappers (submit_beta_sweep, submit_cyl_sweep_v4, ...) delegate to
  package-stable submitters OUTSIDE this worktree — their jobs still log to the old dir until
  those external copies are retrofitted or SUBMIT_SCRIPT is repointed. Charter decision needed.
- Ledgers pushed same-day (BRANCH_LOG + diagnostics/RESULTS_*; logs/ stays gitignored).

## 2026-07-08 — session 4 (Sanaa's smoke-2 checks: r3only verified, K fixed, blowup ISOLATED)
- Sanaa's directives: (1) check r3only≈bare, (2) add full-analytic-LTE diagnostic, (3) triple-check
  developed flow / ΔT³·(1/12)·L prefactors / signs, (4) K=20 too coarse — truth must be ~analytic.
  Also: act autonomously, report after (saved to memory).
- r3only≈bare is PHYSICAL: independent numpy recompute matches diag (c12·rms(L³ω)=1.420e-8 exact);
  predicted 4-step analytic effect 3.16e-7 == observed gap 3.0e-7; low-ν members → bracket ≥99.9% N̈.
- Prefactors/signs verified vs rollout_perfect_closure term-for-term; ICs 820/837/964/934 all
  filter-KEPT at 0.83–1.12× member-median target norms (developed).
- K: driver now warns h_fine>2.5e-5; rule = h_fine ≤1e-5 for accuracy runs (we don't model τ_RK4).
  Measured old K=20 truth error ≤1.7e-9 → smoke-2 tables were NOT materially polluted.
- NEW (driver): `r3anal` arm (exact chain-rule Ṅ/N̈ through the identical assembly, no NN) +
  `--track-lte` (per-checkpoint analytic-LTE budget + NN-vs-analytic drift + injected error inj).
  closure-reviewer clean; sge-checker soft-fixes applied.
- SMOKE3 (job 1827061, h_fine=1e-5, arms bare/r3only/r3anal/closure, train+val ICs):
  **blowup mechanism ISOLATED = NN-injected history-contamination loop, NOT marginal-AB2.**
  r3anal stable at CFL 0.85 and 26–133× better than bare; closure diverges exponentially even at
  CFL 0.56 (3c); LTE track: rel-N̈ flat at val level ~0.2 for 2–4 steps then doubles per ~1–2 steps;
  blowup ≈ 4 steps after inj/τ→1. Mechanism: NN noise → 7-lag history → 1/dt^k TimeFD amplification
  → noisier correction. See diagnostics/RESULTS_2026-07-08_smoke3.md.
- STATE: 1827034 (cond_local) still the main event; its eval must now report blowup horizon +
  inj/τ growth alongside per-(member,dt) rel-L2. Truth refs cached for reuse (--load-refs).

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
