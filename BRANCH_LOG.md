# BRANCH_LOG — Physics-conditioned spatial stencil  (branch: exp/wiener-conditioning)

Running record. Supervisor updates this at the end of every session. Newest entry on top.

## 2026-07-11 — session 9 (Sanaa ruling: OPTION 2 → implemented, gated, SUBMITTED; global supervisor Fable)
- RULING (chat ~15:15 EDT, full authorization): rollout fork = OPTION 2 (truth-free annulus
  stability term; option 1 deeper builds NOT authorized). Sanaa's question "start from the
  model with the e-5 val error?" answered NO: warm start = cond_local_v2 ep63 FROZEN, not
  rollout_ft_cond (2.98e-05 is a short-horizon ROLLOUT-loss scale, not comparable to offline
  pooled 0.214; that ckpt DEGRADED a-posteriori accuracy at 5e-3, 0.05x vs 0.72x pre-FT, and
  its M<=4 stability gain is subsumed by the longer curriculum).
- IMPLEMENTATION (train_deriv_rollout.py, no new files): after M supervised steps keep rolling
  K = max(0, 16 - M) TRUTH-FREE steps; penalty = free_weight * mean_f relu(log Z_ann(f)/
  Z_ann(f-1)), Z_ann = annulus enstrophy (mode radius > (2/3)(N/2); 170.67@512²). Hinge
  penalizes growth only (draining the annulus is not rewarded — the p170 lesson). Supervised→
  free boundary always detaches (the approved truncated gradient); in trunc mode every closed
  segment backward()s immediately → activation memory ≤ trunc_k steps at any M+K. Val adds a
  free-roll probe (16 truth-free steps: blow-up fraction fb_s*, median max log-growth ag_s*)
  — THE stability number the supervised val cannot see (M_max=3 at stride 3). Gate r4 added.
- VERIFICATION: gates r1 bit-exact (0.000e+00 both arms), r2 healthy, r4 PASS (hinge ACTIVE on
  warm model, |grad| 3.4e-3); closure-reviewer areas 1-5 sound, 1 CRITICAL caught (np.mean on
  grad tensors would crash default full-mode — my trunc launch was immune; FIXED + both modes
  re-smoked); sge-checker PASS all checks. Micro-smoke: ag_s3 = 4.62 vs ag_s1 = 4.4e-4 — the
  truth-free hinge SEES the 1.5e-2 instability, exactly the observed failure mode.
- SUBMITTED (~16:0x EDT): 1830425/26/27 — **epoch-0 INCIDENT, caught in ~2 min by the log
  watch**: annulus enstrophy overflows to inf mid-blow-up while the field is still finite →
  log(inf) → backward(inf) → NaN weights (stab=inf, 383/384 blown, val all-NaN). The K=4 CPU
  smoke could not see it (e^18 fits f64, the production K=12's e^55 does not). qdel'd.
  FIX (3 guards, DECISIONS row): non-finite-Z → blown before the log; hinge clamp_max
  (--free-cap 10.0); optimizer step skipped on non-finite grad norm (n_skip logged). Poison
  path REPRODUCED on CPU with guards → finite, un-poisoned, honest counts.
- RESUBMITTED: rollout_ft_opt2_cond = ro2c_TRN **1830428** + ro2c_MONL **1830429** + ro2c_MONF
  **1830430**. kf4+FRC-256, strides 1,2,3, M schedule 4:6,8:8,12:8,16:8,21:10 (40 ep,
  per-stride clamp 21/7/3), trunc:4, free-horizon 16, free-cap 10.0, lr 5e-5, f64. ~2-3 GPU-h.
  Success criteria: fb_s3 → 0 with ag_s3 falling AND 5e-3 accuracy NOT degraded (the 7g
  failure); then a-posteriori M=16 ladder vs pre-FT.
- ALSO this session (cross-branch): inbound-mail root cause (mseas.mit.edu has NO MX; MIT
  border blocks direct SMTP from Exchange Online → Sanaa's replies can never arrive; needs
  IS&T ticket; chat = only ruling channel — emails one-way, "reply-approvable" retired);
  [QG][INFO] diagnosis email spooled. CP-ML-1 build running in parallel (sgs branch).
- Next: LIVE monitor watches 1830425; on landing → per-root eval + a-posteriori M=16 kf4
  ladder vs pre-FT ep63 → [QG][LANDED][WIENER] with the fork's success verdict.

## 2026-07-10 — session 8 (resume: report session 7g's unlogged jobs + reporting-chain fixes)
- CONTEXT: Sanaa reported (1) no more reports, (2) no daily report, (3) reply-channel unknown.
  Root causes found: session 7g (evening 07-09, jobs 1828852/1828855/1828863) finished 22:20 EDT
  but ended WITHOUT emailing or logging; the daily report had never been installed (no crontab);
  the reply channel had never been built.
- SESSION 7g RECONSTRUCTED FROM LOGS (it wrote no ledger entry): rollout-aware fine-tune,
  strides [1,2,3], M 1→4, grad_mode=full, 30 ep, f64, roots kf4@5e-3 + FRC-256@5e-3.
  COND-FT (warm ep63): best rollout val 2.98e-05, rf med s1/s2/s3 = .033/.030/.038.
  PHYS-FT (cheap_deriv): 3.94e-05, .060/.051/.044. Ckpts training_runs/rollout_ft_{cond,phys}.
- A-POSTERIORI SMOKE (kf4, M=16, full variant): **NEGATIVE RESULT — FT does not transfer.**
  1.5e-2 both unstable (phys s10 [pre-FT 12, worse], cond s11 [pre-FT 7, delayed]); 1e-2 both
  unstable s15/s16 (pre-FT uncond SURVIVED — phys-FT regression; cond 13→16); 5e-3 stable but
  ≤ bare (phys 0.00x, cond 0.05x vs pre-FT cond 0.72x — accuracy degraded; cond corner drift
  0.249 vs phys 8.84). Tiny a-priori rollout loss + a-post failure ⇒ M≤4 @5e-3 doesn't reach
  the M=16 feedback. Next lever remains the annulus-weighted loss (7d design, λ~3).
  Results/apost_rollout_smoke/ (one npz per case + ladder_matrix_summary.csv).
- REPORTING FIXES (cross-branch, live): QG-closure/reporting/daily_report.sh + crontab
  0 8 * * * on mseas (deterministic shell; queue + replies + branch states + 24h results/logs);
  reply channel = mseas spool → reporting/inbox/, on-cluster loop verified, MIT→mseas leg
  awaiting Sanaa's "TEST" reply. Emails: [QG][DAILY] 2026-07-10 (live test) +
  [QG][LANDED][WIENER] overnight results (outbox: reporting/outbox/landed_20260710_rolloutft.txt).
- Decided next: await Sanaa on (a) TEST reply, (b) annulus-weighted-loss GO vs other lever.

## 2026-07-09 — session 7e (Sanaa question: which dealias world, data vs network — CONFIRMED)
- Sanaa (chat): (1) was the mmap build sqrt(2)*(2/3) or strict 2/3? (2) does the network
  2/3-dealias internally ("a dealiasing layer" in her memory) or sqrt(2)-something?
- AUDIT (code-level, extends session 7d Part 2): DATA = sqrt2-world RADIAL everywhere —
  solver k_cut = sqrt(2)*(2/3)*kmax_axis (derivative.py:29-32, radial ball, keep radius
  241.36/256 at 512², 120.68/128 at 256²; once per RHS sum, operator/__init__.py:48);
  build+slice harness imports the SOLVER'S OWN Derivative (build_training_data_mmap.py:67,
  514,90-105; slice_deriv_from_deep.py:34-49,66-82) → targets use the IDENTICAL sqrt2 mask,
  per-product (2/Jacobian, 20/anchor). NETWORK = zero internal masks/FFTs on the field path
  (cheap_deriv AND cond_local; cond_local's 2 rFFTs are the σ̂ scalar context only); the
  remembered "dealias layer" = the END-PROJECTION on the output, train_deriv.py:229 +
  296-297 (--dealias-pred default ON), SAME sqrt2 solver mask per shape; == per-product
  exactly (quadratic-then-linear). NO dt in the mask path ("sqrt(2)/dt" does not exist —
  the sqrt2 is corner-vs-axis geometry). Comment strings at train_deriv.py:30,215,233 say
  "2/3 rule" loosely — the object is sqrt2; flagged for a comment fix (not applied, RED-free
  but zero-urgency). third23 experiment changed NO committed defaults (both override flags
  default None).
- Verdict: pipeline internally CONSISTENT, uniformly sqrt2-world; consistently aliased in
  the 170.7–241.4 annulus (512² mode units, folds ≥ 0.114 kmax irremovable). Session 7d's
  2/3-world NN failure = train/eval mask MISMATCH, not aliasing per se (unchanged reading).
- [QG][AUDIT][WIENER] sent from mseas rc=0 17:39 EDT; outbox
  diagnostics/Results/outbox_20260709_dealias_audit.txt.

## 2026-07-09 — session 7d (Sanaa full mandate: eps(k) profiles + dealias audit + the 2/3 world; job 1828403)
- PART 1 (spectral_error_profile.py, 3 ckpts x 14 (member,dt) roots x 6 val samples): profile
  is U-SHAPED — worst at LOW k (Nddot eps(30): cond 0.18 / control 0.45), min at k~100-200,
  knee k=209 (cond) / 232 (control); annulus only x1.4-1.7 worse than mid a-priori. ep63
  conditioning bends LOW k 2.7x, annulus only 1.4x (worse-than-control at b1/b2@1.5e-2).
  rollout_unc (lr5e-5 ep8) uniformly weakest. Annulus-weighting proposal drafted (lambda~3
  per-shell weight + rollout-aware injection) — design only, NO training launched.
- PART 2 (dealias audit, file:line in RESULTS Part 3 section): solver = ONE mask per RHS SUM
  (operator/__init__.py:48; jacobian patch unmasked internally); harness/targets = per-product
  (2 per Jacobian; 20 masked products per [N..N3dot] anchor); model = ZERO internal, ONE
  end-projection (train_deriv.py:229; == 16 per-product exactly, single quadratic level).
  Correction to Sanaa's picture: intermediates w^(k),psi^(k) are NOT re-masked (diagonal
  images, band-preserving — no mask needed). FOLD MATH: sqrt2 world folds land >= 0.114 kmax
  (mode 29.3) INSIDE the ball, irremovable at any placement; strict 2/3 folds land >= (2/3)kmax
  = removed exactly.
- PART 3 (the 2/3 WORLD, whole harness + truth + IC under radial (2/3)min(kmax), new refs):
  ANSWER = NO. Analytic stays stable/strong (23.6x/16.3x/9.5x; bare itself ~6x better —
  the annulus was much of bare's error). NN gets WORSE than sqrt2 world: uncond blows even
  at 5e-3 (s14), cond blows at all dT (s6/s7/s6). Train/eval mask mismatch dominates (stated
  caveat) => aliasing per se is NOT the NN's problem; per-product-dealias design note NOT
  triggered; clean test needs a mask-matched model (retrain on 2/3 data) or rollout-aware
  fine-tune. Low-k: clean under analytic in 2/3 world; NN corrupts it via feedback, not alias.
- 9 case npz + summary in Results/apost_ladder_20260709_third23/; profile npz in
  Results/spectral_error_profile_20260709/. Emails: [QG][SUBMIT][WIENER] + [QG][LANDED][WIENER]
  (full tables + Part-2 writeup inline, mailx from mseas).


## 2026-07-09 — session 7c (Sanaa order: kill plateaued trainings + final report)
- qdel EXACTLY 1827225 (hygiene_train) + 1827306 (condlocal_train) at 15:05:58 EDT after
  qstat -j identity verification. Reason: plateaued, ordered by Sanaa. GPU slots freed for
  SGS Phase-B (expected).
- Final states (ckpts/configs/logs verified intact before AND after):
  deriv7_hygiene: killed ep112/300 (~22 h); best ep107 pooled 0.24953, Ndot 0.0594 /
  Nddot 0.1780 / N3dot 0.5112; no new best ep108-112, Nddot flat ~0.178 (= control 0.186
  ballpark — hygiene ablation did not move the ceiling).
  deriv7_cond_local_v2: killed ep78/300 (~21 h); best ep63 pooled 0.21389, Ndot 0.0744 /
  Nddot 0.1378 / N3dot 0.4296 (medians 0.0536/0.0929/0.1784); 15 epochs (64-78) no new
  best. Nddot 0.138 vs control 0.186 is NOT equal-data (41-root pool incl. DEC vs 17) —
  per-root eval before any scoreboard claim. best.pt == frozen_eval_20260709 (today's
  matrix ckpt) byte-identical.
- Monitor topology: FINALIZE monF_hyg (1827288) fired+mailed; monF_cond2 (1827308) fired,
  delivery failed on-node -> verdict RELAYED manually from outbox; stale hygiene_mon
  (1827226) died on old CLI (exit 2, superseded); LIVE monL_hyg/monL_cond2 self-exited
  15:07:24. No monitor qdel needed.
- Emails: [QG][LANDED][WIENER] kill report (full metrics tables inline) + relayed
  [QG][MONITOR] cond_local_v2 postmortem. Outbox copies: diagnostics/Results/outbox_20260709_kill/.
- Next: per-root eval of both final ckpts; annulus-fix PROPOSE pending Sanaa.


## 2026-07-09 — session 7b (Sanaa rulings: f_NN-only projection + analytic arms; job 1828315)
- Rulings: solver mask RED/untouched; remediate on the correction alone. Implemented
  --nn-project-radius (df2297a): R3 correction projected to |k| <= (2/3)min(kmax) = 170.67
  at 512^2 (f_NN of closure arms; f_anal when r3anal runs with it). r3anal sigma logging added.
- ANALYTIC ANSWER (r3anal, exact derivs): STABLE at all 3 dT with 132.6x / 35.1x / 71.4x
  improvement over bare (1.5e-2/1e-2/5e-3) — the instability is 100% NN-specific, NOT the
  scheme's and NOT intrinsic to the aliasing annulus; exact Nddot REGULATES the corner band.
  Analytic+proj170 collapses to 1.01x/1.40x/1.44x => the annulus CARRIES the closure's value.
- NN+proj170: not the fix. Softens feedback (cond blowup s13->s16 at 1e-2; uncond 0.16x->0.80x)
  but never beats bare; at 5e-3 HURTS cond (0.72x->0.22x); uncond@1.5e-2 avoids formal blowup
  yet rel-L2 1.54 (accuracy collapse). dropnddot+proj = 1.0x everywhere.
- 18 cases: diagnostics/Results/apost_ladder_20260709_p170/ (one npz each + summary CSV).
  RESULTS_2026-07-09_apost_matrix.md Part 2. COND still frozen ep63 (best.pt unchanged).
- Email-format corrections adopted: [QG][SUBMIT|LANDED][WIENER] order; full tables INLINE.
- Next: training-side fix for the annulus (rollout-aware fine-tune / annulus-weighted loss /
  spectral head conditioning) — the inference-time mask family is exhausted as a fix.


## 2026-07-09 — session 7 (branch supervisor: apost matrix + dealias/FFT audit; Sanaa away, green light)
- TASK 0 (dealias/FFT triple-check of rollout_aposteriori.py): NO train/inference bug — f_NN
  end-projected before the IMEX step with the IDENTICAL Derivative.alias_mask training uses.
  CONVENTION CORRECTION: the solver mask is RADIAL, k_cut=sqrt(2)*(2/3)*k_max = mode radius
  241.36 at 512^2 (derivative.py:30-32), NOT square per-axis 170. The 170.7<|k|<=241.4 annulus
  is above the alias-safe N/3 radius -> quadratic products ALIAS exactly where the blow-up
  seeds (184-240). Solver-level, self-consistent everywhere; fixing it = RED (solver convention).
- Ran job 1828239 `apmx0709`: 2 ckpt (UNCOND deriv7_filtered_lr5e-5 ep8 / COND
  deriv7_cond_local_v2 FROZEN ep63) x 2 variant (full / new --drop-nddot) x 3 dT, kf4 IC837,
  M=16, gamma=1, no remediation. 1.5e-2 truth REUSED (ladderrefs); 1e-2/1p5 K per h_fine=1e-5.
- RESULT (full table: diagnostics/RESULTS_2026-07-09_apost_matrix.md +
  Results/apost_ladder_20260709/ladder_matrix_summary.csv; ONE npz per case, intermediates
  deleted per Sanaa's output discipline): N-ddot term is the sole destabilizer (all dropnddot
  arms stable, all blow-ups have it) but also the sole value (dropnddot = 1.0x everywhere).
  COND does NOT fix the tail natively: blows EARLIER at 1.5e-2 (step 7 vs 12), blows at 1e-2
  (step 13) where UNCOND survives; at 5e-3 COND 0.72x vs UNCOND 0.06x final and up to ~17x
  transient gain before the corner-band feedback overtakes.
- t=0 LTE row (job 1828240, post-hoc — protocol wanted it first): COND rel_Nddot(t=0)=0.136,
  outside 0.023-0.05 acceptance => COND rows labeled MID-TRAINING (ep63/300, val agrees; not
  a wiring regression). After 1 closure step COND rel_Nddot 0.447 vs UNCOND 0.208 (advantage
  inverts on self-generated history).
- Commits: 6077fa3 (--drop-nddot, consolidate_apost_cases.py, apost_matrix_job.sh) + ledgers.
- Emails: [QG][WIENER][SUBMIT] (params + TASK0 verdict), [QG][WIENER][LANDED] (12-case table).
- Decided next: (a) let 1827306 finish, rerun COND legs from the converged ckpt; (b) the tail
  fix is not conditioning — candidates: alias-safe corner treatment of f_NN (kcut at the TRUE
  2/3 radius 170.7 = principled, not remediation), or R4 rollout-aware fine-tune; (c) formalize
  the Wiener doc before the next model change.


## 2026-07-08 — session 6 (global supervisor: incident ROOT CAUSE + fixes + resubmission)
- ROOT CAUSE (revises session-5's "wrong dT power in the m=1 head" attribution): the
  conditioning path was numerically DEAD in the incident ckpt — head weights O(1e-1) x
  amp dT^(S-k) <= 5e-8 => ~1e-9 relative contribution; the Ndot damage lives in the BASE
  path (mix max|w| 3 -> 3.59 (best) / 4.06 (last), stencils drifted). Poison mechanism:
  deriv_dataset.py sampled floor medians from RAW packed rows, but the quiescent filter
  edits only split.npz -> on late-developing members (FRC-b0/b05/b075/b1) the median
  landed on quiescent rows, floors 21x-46,343x too small, floor inert, rule-16
  prediction-distortion of the shared base path. The dt^-3-looking FRC-Ndot signature is
  the base-path distortion surfacing where targets are relatively smallest, not an amp law.
- FIXES ([fable-authored]): (1) deriv_dataset floor median from the UNION of split.npz
  kept indices (+ empty-split fallback); (2) cond_local amp = (dT/dT_ref)^(S-k),
  DT_REF_COND=1.5e-2 — Adam-visible modulation at the anchor dt, identical scaling LAW,
  fixes the dead-path defect; (3) amp = 0 on k=0 channels (eps_0 = 0, pure-noise DOF
  removed); (4) triage loader strict=False whitelisting dt_ref_cond (old ckpts);
  (5) init-gate whitelist dt_ref_cond; (6) monitor emails now follow Sanaa's format
  convention (PARAMETERS first, bold-caps titles, indented numbered spaced points).
- GATES: model self-smoke ALL PASS (zero-init exactly 0.0; spectral-context 7.7e-17;
  L-invariance 0.0; mixed-dx 7.9e-17); G1 OVERALL PASS (FRC-256@5e-3, kf4@1e-2,
  DEC-512@1e-2, b05@1e-2: A worst rel 0.0, B cond==ctrl medians); post-fix floors sane
  (b05 Ndot 8.77e3, b0 2.59e4 vs FRC-256 8.03e3, kf4 6.49e3); G4 closure-reviewer PASS
  (both recommendations applied); G5 monitor-script PASS.
- RESUBMISSION (I18 three-job unit): run deriv7_cond_local_v2 (T1, 41 roots, same config;
  new run-name preserves the incident run dir). Job ids in the [QG][SUBMIT][log] email.
- NOT touched: session-4c uncommitted work (rollout_aposteriori R1-R3 flags, sge log
  rewiring leftovers, SUPERVISOR_BRIEF) — needs its own G3/G5 before its own submissions.

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
