# BRANCH_LOG — SGS spatial closure, Phase 1  (branch: exp/sgs-closure)

Running record. Supervisor updates this at the end of every session. Newest entry on top.

## 2026-07-13 — FIELD-PLOT TRIAGE P1-P4 LANDED, NO FLAG (branch supervisor Fable; late-afternoon session)
- Sanaa's four GREEN triage items on the eval field panels, all landed (jobs 1832249-58,
  ~0.15 GPU-h total; trainers 1832221/1832231 untouched).
- P2 (decisive, ran first): the vertical streaks in the Pi panels are sinc/Gibbs RINGING
  of the LES filter's sharp axis-wise cutoff on the compact high-amplitude Brinkman
  source — f64 A/B recompute (t=81, both geometries): streak cross collapses ~100-240x
  under an all-Gaussian filter; sharp recompute matches stored piff_s4 to rel-L2 ~3e-8.
  BENIGN — no mask/dealias bug, NO FLAG, running trainings consume exactly the defined
  sharp-filter Pi_FF. omega_bar carries the same ringing at only ~0.2% of field scale.
  Filter-definition discussion item recorded.
- P1: replot_eval_fields.py (reusable) — 5-panel symlog field5_*.png next to the old
  4-panel linear figures for prod_ext150 + cape_base_100ep; wake now visible.
- P3: sponge audit — ALREADY excluded everywhere (loss/metrics/kurtosis/eval/diagnostics)
  via RunData.valid; chain with file:line in yamls/valid_pixel_mask_sponge_audit/;
  mask figures per geometry (cylinder 92.1% valid, cape 70.9%).
- P4: sigma decomposition — cape model noise-dominated 493-1133x on ALL 5 members;
  cylinder model 577x on its training member, posterior GP variance inflates only on
  the 4 unseen modulation classes (epistemic OOD response). Homoscedastic-collapse
  "before" figures at paper quality.
- NEW STANDING ORDER APPLIED: canonical diagnostics tree ml_closure/pngs/<name>/ (+.txt),
  yamls/<name>/, csvs_and_npz/ — populated for all four diagnostics; legacy copies kept.
- [QG][LANDED][sgs] email spooled (landed_20260713_plot_triage.mail), relay-verified.

## 2026-07-13 — ORDERS 1-3 EXECUTED; CONDITIONED ENSEMBLE FLEET IN FLIGHT (global supervisor Fable; afternoon session)
- SANAA RULINGS (chat): three [QG][GLOBAL] ORDERS (architecture email / results map +
  diagnostics / ensemble training under new template T4) + mid-session STANDING FULL
  INDEPENDENCE ("no need to ask me for any approval anymore... report everything") —
  recorded in DECISIONS + memory. Also: video-rerender directive — fleet 1832129-34
  fired via pipeline-runner, standing landing-chain rule saved to memory.
- O1 LANDED 12:50:01 (architecture email, relay-verified). O2 LANDED 13:00:01 (results
  map + all 8 diagnostics; see the 13:05 entry below for detail).
- O3 DATA GAP CLOSED: FPC-telS-A Pi_FF chain had NEVER fired (chain-gap, video-family) —
  1832176-80 + step0 sine/ramp/ou 1832181-83, all clean; 5/5 FPC members Step-0 complete.
- O3 MODEL (commits 017bd52, ddd811b, [fable-authored]): zeta_dot (table-Re, T_shed=2.992
  cumsum boxcar, recorded zdot_sd buffer) into FiLM (cond_dim 2) + ARD; |grad omega_bar|*
  (recorded g_scale) as ARD dim; GP inputs 16+3=19; flags-off = exact legacy (old ckpts
  load strict). Gates: T1-T5 + T8/T8b PASS; legacy in-test T6 0.814 = known family plateau
  (arm-2 hit 0.80 on the same path) — operative bar is the re-gated run-level one. G4:
  1 CONFIRMED MEDIUM (kmeans before conditioning stats) FIXED pre-submit; G5 PASS;
  template T4 appended to charter §4 (Sanaa's order = the RED approval).
- CROSS-EVAL BEFORE-ROW (1832208-12; 1832203-07 qdel'd in 1 min, I14 --config fix):
  prod_ext150 per-member R2 const .858 / ou .842 / ramp .587 / sine .491 / telS-A .365 —
  SPREAD 0.493; degradation tracks the Re-excursion of each modulation class.
- IN FLIGHT (~16 GPU-h): piff_fpc_ens unit 1832221/22/23 (+evals 1832224-30, startup
  verified: zdot_sd 0.1818, g_scale 1.1048, 883 train frames); piff_cape_cond unit
  1832231/32/33 (+1832234/35, zdot_sd 0.1934, g_scale 2.0588); cape LOMO ladder 1832241;
  arm-F 150 ep 1832242 (held). Predictions P1-P3 pre-recorded
  (ml_closure/PREDICTIONS_ensemble_2026-07-13.md); [QG][SUBMIT][log] spooled.
- SUPERVISOR DECISIONS (independence ruling): prod_ext150 PROMOTED; cape convergence
  tier YES (fires after the fleet); Wiener levers queue behind the SGS fleet.

## 2026-07-13 — OVERNIGHT FLEET LANDED 7/7 CLEAN (global supervisor Fable; Monday resume session)
- All 7 jobs exit 0 (22:32 Sat – 04:50 Sun EDT). The live-session watcher died with the
  07-12 session, so the landing chain ran at Monday resume; consolidated [QG][LANDED]
  email spooled 11:58 (landed_20260713_overnight_fleet.mail).
- EXT150 (pF_ext150 1831561 -> pEv 1831562 -> pCal 1831575): winner extended to 150 ep.
  Full-frame val R2 0.8584 / RMSE 33.1 / NLL 5.16 — **PASSES the re-gated 0.85-in-100-ep
  bar** (was 0.8323 @ ep59). Recal: s=0.521, test NLL 5.17->4.91, 1sig cov 0.974->0.954
  vs 0.683 nominal — shape-not-scale CONFIRMED on the production candidate. zeta_ls
  0.6931 = init (expected, const-only). Package: ml_closure/runs_piff/prod_ext150/eval.
- ARM F (pT6_F 1831574): gate FAIL (R2 0.495 @ ep49 vs 0.95-in-50) BUT **no collapse** —
  first hetero arm that survives: b mobile (->6.35), sigma median stable 35-38, R2
  monotone rising at cutoff. READ: structural prior closes the ELBO buy-out channel
  (D/E's death), cost = ~4x slower mean learning. Candidate follow-on: 150-ep extension
  (arm-A precedent). Curve: ml_closure/runs_piff/t6_arms/armF.npz.
- CAPE BASELINE (pCape_base 1831571 -> pEv 1831572 -> pCal 1831573): 5-member, 100 ep.
  Pooled val R2 0.800 / RMSE 94.8 / NLL 6.25 (best val NLL 6.493). **zeta ARD ls = 2.016
  — first time OFF the 0.6931 init**: multi-member training makes zeta identifiable,
  LOMO arm is now meaningful. Recal: s=0.508, NLL 6.26->5.98, cov 0.969->0.908.
  Package: ml_closure/runs_piff/cape_base_100ep/eval.
- HELD FOR SANAA (Monday list, in the email): promote prod_ext150; cape LOMO arm;
  arm-F 150-ep extension; cape convergence-tier scope; Wiener option-1 + formalization.
- Nothing in flight on our side. Status cron confirmed reverted to 12/16/20.

## 2026-07-12 — SANAA GO (night session): winner extension + recalibration + arm F + CAPE ML all FIRED (global supervisor Fable)
- RULING (chat ~21:45, recorded in DECISIONS): GO on re-gate (reversal window closed),
  structural-prior B-item, winner extension, sigma recalibration, AND the Pi_FF closure in
  parallel on the cape cases. Autonomous overnight; Sanaa checks tomorrow.
- RECALIBRATION (calibrate_piff.py NEW; pCal_win 1831569 done): NLL-optimal scalar s has the
  closed form s^2 = mean(z^2); fit on val[100,110), tested on val[110,120]. WINNER RESULT:
  s = 0.503 (sigma exactly halved, mean sigma 34.7 ~ RMSE 35.1), test NLL 5.29 -> 5.01, but
  1-sigma coverage only 0.974 -> 0.955 vs 0.683 nominal. READ: the miscalibration is
  PER-PIXEL SHAPE, not scale — no global scalar can fix it (rare huge wake errors dominate
  RMSE; typical pixel error is far smaller). This is the quantitative motivation for arm F.
- ARM F (structural noise prior, t6_arm.py extended; closure-reviewer: 7/7 PASS + 1 MEDIUM
  fixed): sigma^2(x) = softplus(a) + softplus(b)*s_feat(x), s_feat = FIXED train-mean-
  normalized |grad omega*|^2 of the input field; only scalars a,b learn; cap [1e-3,10] std
  space. The reviewer's MEDIUM (b init at softplus^-1(1e-4) is saturation-frozen ~40/50 ep —
  would confound the verdict) fixed: init softplus(b)=0.01 (+10% mean s2, negligible) + b at
  10x lr (b must be MOBILE — if the collapse channel exists we want to SEE it). pT6_F 1831574.
- CAPE Pi_FF ML: Step-0 canonical artifacts built for all 5 members (make_dataset_manifest.py
  gained a cape-geometry branch — D := L_cape = 1, x_c = 0.2*Lx, y_c = tip height 4.0;
  reviewer 5/5 PASS; step0_CA* 1831563-67). conf_piff_cape.yaml: 5 runs, winner lr/wd,
  100 ep, zeta keeps the FPC normalization (table-Re = shared modulation coordinate; manifest
  caveat records Re_cape = table_Re/1.2566). SMOKE 1831570 CLEAN (2 ep, R2 0.21 -> 0.43,
  145M train px, 61 s/ep) — and residual kurtosis 2064 (vs FPC 395): cape is MORE
  heteroscedastic; the arm-F question is even more load-bearing here.
- IN FLIGHT overnight: pF_ext150 1831561 -> pEv_ext150 1831562 -> pCal_ext 1831575;
  pCape_base 1831571 -> pEv_cape 1831572 -> pCal_cape 1831573; pT6_F 1831574. All -m ea.
  New scripts: piff_step0_job.sh, piff_tool_job.sh (generic ml_closure tool wrapper).
- Note for the record: one malformed submission (1831568, --help as ckpt arg) was qdel'd
  within a minute; no side effects.

## 2026-07-12 — CAPE Pi_FF CHAIN LANDED CLEAN; audit verdicts RELAX x5; wake diags submitted (global supervisor Fable; evening session, cont.)
- CHAIN (1831531-51) landed ~21:24 EDT, ~35 min wall / ~0.75 GPU-h: DNS_LES x15 on disk
  (2.0G/504M/127M per s2/s4/s8 per member), audit_A x5, restart_ic_t30.npy (t=30.15).
  Only log noise = the known benign imageio_ffmpeg pre-check (lines 1-3 every job log).
- AUDIT_A VERDICTS: **RELAX all five** (S8/A rule: 2tau_int(s4) 0.56-2.80 >> 0.5 threshold ->
  modulated dt_save=0.5 sufficient; n_pp 8.5-11.8 >= 8 OK). All tau_int fits resolved; phase
  coverage no empty bins (tel 24-bin uniformity 5.5 = worst, still no gaps). Pi_FF is
  OVERSAMPLED 2.5-9.4x everywhere -> N_eff 21-81 per member/scale vs theory 360 (sine worst:
  21-38, slow sinusoidal modulation inflates tau_int). Context notes, no gates tripped:
  l_corr s4/s8 slightly below the theory bracket lower edge in all members; sine U_c ratio
  0.48 vs 0.85*U_mid (modulated-inlet lag bias suspect).
- WAKE DIAGNOSTICS submitted per FPC precedent (wake_FPC* 1829699-702): wake_CA{co,si,ra,ou,te}
  1831552-56, all.q, --scales 2 4 8 --t-min 30.
- STILL FOR SANAA MONDAY: cape convergence-tier scope (does the 07-08 five-grid directive
  extend to cape? IC is ready), per-member RUN report emails vs one consolidated (leaning
  consolidated per her one-npz-per-case convention).

## 2026-07-12 — CAPE-A WAVE LANDED 5/5 CLEAN; shedding verdicts; Pi_FF+audit chain FIRED (global supervisor Fable; evening session)
- LANDED ~19:16 EDT (jobs 1830412-20, ~27 h/member as budgeted): all five FPCape members
  ran 960k steps to T=120, NaN-guard FINAL CHECK CLEAN — the dt=1.25e-4 fix holds at full
  duration. Shedding trackers (1830413-21) all clean: single Welch peak per member
  (f_sh 0.149-0.176), drag locked at 2f_sh (0.80-0.86), no third harmonic, alias-free,
  zero WARN/NaN. f_sh sits just below the cylinder band [0.189,0.48] in all five — expected
  (bottom-attached asymmetric cape wake; cylinder St context-only per charter). Telegraph
  member: mild estimator spread (Cl_mid 0.1485 vs Cl_inst 0.1587), within one Welch bin.
- PI_FF CHAIN FIRED (sge-runner, submit_piff_capeA.sh authored: dry-run default, preflight
  no-DNS_LES-overwrite guard): per member piff_s2->s4->s8 chained via hold_jid (max 5
  concurrent GPU — courtesy cap) -> audA_<tag> on all.q holding on s8. Jobs 1831531-1831550;
  + icx_CAco_t30 1831551 (FPCape-const index 67 -> t=30.15, precedent-matched IC for the
  convergence tier). Cost ~0.75 GPU-h. DEVIATION from FPC precedent (documented): scales
  chained sequentially per member instead of concurrent, to respect the 6-GPU-job ceiling;
  audit holds on s8 (== all three scales via the chain). One bash-4.2 empty-array/set -u
  bug in the new submitter caught on dry-run path, fixed before any submission.
- [QG][LANDED][SGS] email spooled (shedding table + chain manifest). NEXT at chain landing
  (~22:30 EDT): verify DNS_LES x15 + audit_A x5 + restart IC, then per-member [QG][RUN][SGS]
  reports + wake diagnostics + convergence-tier decision for Sanaa Monday.

## 2026-07-12 — GRID LANDED, WINNER SELECTED, S4 EVAL PACKAGE DONE (global supervisor Fable; autonomy window, evening session)
- GRID (1830888-93): all 6 exit 0, landed 07:13-09:24 EDT, ~6.5 GPU-h as budgeted. Val NLL /
  R2 at ep59 (best epoch = FINAL epoch on all six rungs — none saturated at 60 ep):
  lr1e-3/wd1e-5 5.6694/0.8133 (WINNER, lowest NLL per spec S3.3); lr1e-3/wd1e-4 5.6864/0.8043;
  lr3e-4 pair ~5.72/~0.79; lr1e-4 pair ~7.18/0.61. READ: lr is the only lever; wd inert.
- S4 EVAL on the winner (piff_eval_winner 1831530, ~5 min GPU; new reusable
  scripts/sge/piff_eval_job.sh, sge-checker PASS): full-frame val R2 0.8323 / RMSE 36.0 /
  NLL 5.286 (full frames beat crop-val as expected). FLAG — CALIBRATION OVERDISPERSED:
  coverage 0.975/0.990/0.995 vs nominal 0.683/0.954/0.997; mean sigma 69 ~ 1.9x RMSE.
  zeta ARD ls 0.6931 = exact init (zeta constant on FPC-const — unidentifiable by
  construction; meaningful only multi-run). Package in runs_piff/grid_lr1.0e-3_wd1.0e-5/eval/.
- PROPOSALS to Sanaa (email, not actions): (i) extend winner to 100-150 ep as production ckpt
  (re-gated 0.85-in-100-ep bar on trajectory: 0.813@59 climbing, arm-A precedent crossed 0.85
  at ep81); (ii) post-hoc scalar sigma recalibration on val (no retrain; pairs with the
  structural-noise-prior B-item).
- [QG][MILESTONE][SGS-CLOSURE] S4 email spooled via pending_mail relay.

## 2026-07-12 — ARM E conclusive; RE-GATE executed; GRID FIRED staggered (branch supervisor Fable; orchestrator ruling 3 under Sanaa's autonomy window — reversible Monday)
- RULING 3 (reasoning recorded in DECISIONS: Sanaa away + inbound mail broken; reply-approval
  would idle the track ~2 days): arm E (warmup heteroscedastic retry), then fire the grid
  either way — E >= 0.95 via B-item config, else re-gate at R2 >= 0.85 in 100 ep on the
  baseline (A-curve evidence) and fire with the arm-2 baseline config.
- ARM E (commit 5b0cd94; pT6_E 1830865): 25-ep mean-warmup with the noise head frozen
  (sigma.detach, outside optimizer — no stale grads), then unfreeze with sigma^2 hard-capped to
  [0.1 e^-2, 0.1 e^2]. RESULT: warmup healthy (R2 ~0.71 by ep 21, arm-2 trajectory); AT
  UNFREEZE the warm mean was destroyed in 2 epochs (0.38 -> 0.02 -> 0.0000), sigma_max slammed
  to the cap while median stayed at init. D's collapse is STRUCTURAL, not optimization order:
  any input-dependent sigma freedom lets the ELBO buy out the signal pixels and shrink the
  mean to the prior — the same prediction-shrinking family as the temporal track's
  quiescent-window collapse (rule 16). Heteroscedastic-as-free-head B-item: CLOSED negative
  (C plain StudentT / D joint / E warmup+cap). A structural noise prior is the surviving idea.
- RE-GATE (executed, autonomy window, Sanaa can reverse Monday): T6 acceptance := R2 >= 0.85
  in 100 ep on the y-standardized baseline. Evidence: arm A = the baseline at 150 ep crosses
  0.85 at ep 81; the 5-arm ladder shows ~0.9 family saturation (0.95-in-50 uncalibrated).
  Baseline satisfies -> grid released on the arm-2 config (T7 smoke 1830736 already green on
  exactly this config, so no re-smoke needed).
- GRID FIRED 05:05 (submit_piff_grid.sh --go --stagger 2; commit b88f6af added --stagger):
  pF_1e4_1e5 1830888 + pF_1e4_1e4 1830889 running; pF_3e4_1e5 1830890 <- 88, pF_3e4_1e4
  1830891 <- 89, pF_1e3_1e5 1830892 <- 90, pF_1e3_1e4 1830893 <- 91 (hold chains verified via
  qstat -j). GPU COURTESY: all 6 GPUs carry production (CAPE-A x5 + telS-A) -> max 2 grid
  co-tenants at a time, ~3.5 h total, ~6.5 GPU-h. I18 monitor unit not wired for ~1.1 h rungs
  (deviation noted; -m ea per job + in-session supervision instead).
- NEXT: lowest-val-NLL selection + eval_piff on the winner (S4 package) when the grid lands.

## 2026-07-12 — ARM D (heteroscedastic B-item): noise head ABSORBS the signal — ladder complete, grid HELD (branch supervisor Fable; orchestrator ruling 2)
- RULING 2 (recorded in DECISIONS): try the heteroscedastic-Gaussian B-item (arm C's own
  conclusion) against the UNMODIFIED 0.95 gate before putting the re-gate question to Sanaa.
- IMPLEMENTATION (commit d5de2ef): sigma^2(x) = softplus(linear(F+1 GP inputs)) + 1e-4 floor,
  never a function of y; FixedNoiseGaussianLikelihood(learn_additional_noise=False) with the
  per-batch noise= kwarg — plumbing verified in installed gpytorch 1.13 source (live tensor ->
  DiagLinearOperator, kwargs forwarded; gradients reach the head — confirmed on CPU). Head init
  = exactly the arm-2 data-informed noise. All else arm-2 baseline.
- RESULT (pT6_D 1830755, exit 0): R2 flat 0.0000 all 50 ep, RMSE frozen 167.5. NOT a crash —
  the logged sigma stats expose the mechanism: max sigma(x) 2125 -> 3050 physical (~35x target
  std), min 28 -> 4. Jointly trained from scratch, the noise head EATS THE SIGNAL: the ELBO
  prefers inflating sigma on large-|Pi| pixels over fitting them with the GP mean. The exact
  collapse family the spec mandates logging, never auto-switching. No resubmit (that allowance
  is for wiring crashes; this ran as designed).
- LADDER COMPLETE: baseline 0.17 -> standardized 0.80 -> A(time,150ep) 0.89 asymptote ->
  B(M=1024) 0.81 -> C(StudentT) 0.00 (rejects signal as outliers) -> D(hetero joint) 0.00
  (noise absorbs signal). VERDICT: rule 3 — the 0.95-in-50-ep bar is unreachable for the
  specced family under every tested variation; the standing evidence is the A curve.
- PROPOSALS for Sanaa (not actions): (i) re-gate T6 at R2 >= 0.85 in 100 ep (A crosses 0.85 at
  ep 81) — unblocks the lr x wd grid immediately; (ii) heteroscedastic RETRY with mean-warmup
  (freeze noise at 0.1 for ~30 ep, then unfreeze head) and/or a sigma cap — the D collapse is
  an optimization-order pathology, not necessarily a dead end for the B-item.
- GRID: still HELD. [QG][GATE-ML][SGS-CLOSURE] ladder email sent; relay-verified.

## 2026-07-12 — 3-ARM T6 DISCRIMINATION: rule 3 fires — gate bar suspect, grid HELD (branch supervisor Fable; orchestrator ruling under Sanaa's autonomy window)
- ORCHESTRATOR RULING (recorded in DECISIONS): keep the gate moving — 3 concurrent T6 arms from
  the arm-2 y-standardized baseline, pre-committed decision rule (any >= 0.95 -> minimal arm +
  T7 + grid; C >= 0.90 -> propose C+epochs; all ~0.8 -> gate bar suspect).
- Runner ml_closure/t6_arm.py + scripts/sge/piff_t6arm_job.sh (commit b3ff7bb). Jobs pT6_A
  1830743 / pT6_B 1830744 / pT6_C 1830745 -> rerun 1830750.
- ARM A (150 ep): best R2 0.8910 at ep 145, final 0.8652 — slow asymptote to ~0.89; "needs
  time" gains 0.80 -> 0.89, cannot reach 0.95.
- ARM B (M=1024, 50 ep): 0.8146 at ep 49 (+0.02 over M=512 at ~6x per-epoch cost) — capacity
  is not the binding constraint.
- ARM C (StudentT B-ITEM EXPERIMENT): first run 1830745 crashed on wiring — gpytorch MC-samples
  non-Gaussian marginals, pred moments 2-D; fixed in predict_physical (law-of-total-variance
  reduction, commit b48622a; also fixes a silent single-chunk flatten), resubmitted once per the
  ruling. Rerun 1830750 exit 0: R2 FLAT ~0.0000 all 50 ep (RMSE frozen at 167.5 =
  constant-mean prediction) — SCIENTIFIC NEGATIVE, mechanism understood: StudentT's redescending
  influence bounds the gradient from large residuals, and large-|Pi| pixels ARE the signal; the
  wake structures get rejected as outliers. Kurtosis 395 = SIGNAL HETEROSCEDASTICITY, not noise
  heavy-tails. Plain StudentT is the wrong B-item variant; input-dependent (heteroscedastic)
  Gaussian noise is the right one.
- VERDICT (rule 3): no arm >= 0.95, C < 0.90 -> the 0.95-in-50-ep T6 bar is itself suspect —
  the specced model family saturates ~0.9 under BOTH time and capacity extension. GRID HELD.
- PROPOSAL for Sanaa (not an action): (i) re-gate T6 at R2 >= 0.85 in 100 ep on the arm-2
  config (evidence-based bar: A first crosses 0.85 at ep 81), or (ii) rule the
  heteroscedastic-Gaussian B-item first and re-gate after. Grid fires on whichever passes.
- Commits: b3ff7bb (arms), b48622a (predict_physical fix), b9749d3 + this (ledger).
  [QG][GATE-ML][SGS-CLOSURE] RESULT email sent with the three curves; relay-verified.

## 2026-07-12 — AUTONOMY WINDOW: T6 fix, two arms, BOTH miss 0.95 — grid HELD, escalated (branch supervisor Fable; Sanaa ~23:55 ruling)
- RULING (Sanaa chat ~23:55 07-11, via coordinator, recorded verbatim in DECISIONS): full-autonomy
  window tonight + all of 07-12, gates waived, act-and-report; implement data-informed GP init;
  y-standardization as fallback; fire the grid on T6 PASS; if BOTH arms fail, stop and report.
- ARM 1 (data-informed init in RAW target space, commit 2a31e9b): target_stats exact f64 over all
  62,548,500 valid train pixels (mean -1.35e-3, var 7.617e3); outputscale 6.856e3 / noise 762.
  T6 job 1830733: T5 PASS, T6 NaN FROM EP 0 — float32 K_zz Cholesky goes singular (gpytorch
  jitter is ABSOLUTE 1e-6 ~ 1e-10 relative at that scale). Smoke 1830734 (raced onto arm-1 code
  via its hold) NaN'd identically and failed loudly at eval (no best.pt — desired behavior).
- ARM 2 (recorded invertible y-STANDARDIZATION, commit 5a75893): y_mu/y_sd buffers in every ckpt;
  GP trained on (y-y_mu)/y_sd; init in standardized space (outputscale 0.9, noise 0.1);
  predict_physical() inverts exactly — all reported metrics stay in PHYSICAL units; S1.2 target
  definition untouched. T6 job 1830735: NO NaN, healthy climb R2 0.029 -> 0.798 (RMSE 165 -> 75)
  but plateaus ~0.79-0.80 from ep ~39: GATE 0.95 NOT MET. T7 smoke 1830736: mechanically CLEAN,
  2-ep val R2 0.086 -> 0.158 (was 0.003), constants logged, eval package good; kurtosis 394.8
  B-item raised again.
- VERDICT: both pre-authorized arms exhausted -> STOP per the ruling. submit_piff_grid.sh NOT
  fired. [QG][GATE-ML][SGS-CLOSURE] escalation email sent (full T1-T7 table + analysis).
- ANALYSIS for the ruling: kurtosis ~395 = homoscedastic Gaussian badly misspecified (heavy
  tails); the ELBO inflates noise instead of fitting the mean — plausibly the SAME mechanism as
  the 0.8 plateau, i.e. the open heteroscedastic/Student-t B-item and the T6 miss are one issue.
  Alternatives: more epochs (curve still +0.001-0.002/ep at 49), larger M, T6 gate revisit.
- Commits this window: 2a31e9b (arm 1), 5a75893 (arm 2), a43c424 + this (ledger).

## 2026-07-12 — FPC-telS postmortem: impulse hypothesis DISPROVEN, dt-edge confirmed; FPC-telS-A submitted at dt 1.25e-4 (incident agent Fable; Sanaa 07-12 autonomy window, chat)
- POSTMORTEM (job 1830422, NaN-guard kill at step 270251, detection 03:59:36Z): actual NaN
  onset in scalars.npz at t=66.8150 (step 267260), not the guard's t≈67.56. Onset is 12.34
  time units INTO the long Re=5600 dwell [54.472, 68.862] and 2.05 BEFORE the next switch;
  the original hard-switch FPC-tel blew at t=68.615 inside the SAME dwell. Both variants die
  mid-dwell, far from any ramp — the switch-impulse hypothesis is disproven.
- SIGNATURE = cape dt-edge, not physics: enstrophy Z grew exponentially with shrinking
  doubling time (3.44 at t=66.740 → 32.0 at 66.8075 → 164 at 66.810 → 7.6e191 at 66.8125)
  while E stayed FLAT at 4.314 (grid-scale blowup, energy at high-k only). Cd normal (~2.2)
  until the final record. No secular E/Cd growth over the dwell (E 4.31-4.36 for 12 units).
- Re CONTEXT: dwell Re=5600 → U_inlet 2.872 = 1.44x the max U 2.0 (Re 3900) that the clean
  FPC-{const,sine,ramp,ou} members ever saw at 2.5e-4/2048^2; probe |u| ~2.2-2.45 at onset.
  Nuance stated plainly: the earlier Re=5600 dwell [39.51, 49.82] (10.3 units) survived —
  the edge crossing is a local/instantaneous-velocity rare event, consistent with the cape
  precedent (capeSmk: 2048^2 dt-unstable at 2.5e-4, clean at 1.25e-4; penalty exonerated).
- DECISION (autonomy window; CAPE-A playbook): rerun as FPC-telS-A at dt 1.25e-4.
  Table telegraphS20_dt1p25e-4_T120.npz generated (modulation.py, seed 20260707,
  --switch-smooth-steps 20 = SAME physical 0.0025 ramp as 10 steps at 2.5e-4; switch times
  unchanged). VALIDATED: bitwise equal to telegraphS10_dt2p5e-4 at ALL 480001 shared times
  (ramp windows included); 7 ramps, physical duration 0.0025 each (20/21-step grid spans).
- SCRIPTS: phaseB_A_job.sh (phaseB_job.sh with dt 1.25e-4 baked, NaN-guard verbatim — the
  CAPE-A pattern) + submit_FPCtelSA.sh (save_rate 3600 = dt_save 0.45, scalar_rate 10,
  flush_every 500, new dir FPC-telS-A, no-overwrite guard; partial FPC-telS t<66.81 KEPT).
  sge-checker PASS all rules. SUBMITTED: sgsB_teSA 1830737 (ibgpu.q gpu=1, running
  immediately) + shed_CteSA 1830738 (all.q, hold_jid). Cost ~32 GPU-h (960k steps at the
  measured 8.3 it/s of 1830422).
- Email [QG][FLAG][SGS] spooled via reporting/pending_mail.

## 2026-07-12 — CP-ML-1 build COMMITTED; qg-env-piff repaired+completed; T1–T4 green; test jobs submitted (branch supervisor Fable; Sanaa blanket GO)
- BLANKET GO (Sanaa chat, 07-11 evening): "I have approved everything you sent me" — covers the
  CP-ML-1 build (already approved ~15:15 with its three defaults) and this session's submissions.
- VENV qg-env-piff: the clone was MISWIRED — bin/activate + 32 console-script shebangs still
  pointed at the shared qg-env (a pip through them would have poisoned qg-env; the grid
  submitter's dry-run guard was right to refuse). Fixed: all bin/ paths repointed to
  qg-env-piff (bin/python symlink + full site-packages copy were already correct).
  gpytorch install: attempt 1 failed on a pip-23.2.1 sdist build-dep subprocess; attempt 2
  with --only-binary :all: SUCCEEDED. Resolved pins: gpytorch 1.13, linear-operator 0.6.1,
  jaxtyping 0.2.19, scikit-learn 1.7.2, joblib 1.5.3, threadpoolctl 3.6.0, typeguard 4.5.2
  against the copied torch 2.4.1+cu118 / numpy 2.2.6 / scipy 1.13.1. Shared qg-env verified
  UNTOUCHED post-install (gpytorch not importable there; zero new dists).
- T1–T4 pre-commit rerun from ml_closure/ in the completed venv: 4 passed, 0 skipped (70 s CPU).
- TEST JOBS (self-audit vs hard SGE rules passed; cpu=all.q, gpu=exactly ibgpu.q gpu=1):
  piffT_cpu 1830729 (T1–T4) + piffT_gpu 1830730 (T5–T6) + piff_smoke 1830731 (T7 2-epoch
  end-to-end, hold_jid 1830730). GPU arm co-scheduled IMMEDIATELY on ibgpu-compute-0-0 next to
  the running CAPE-A wave (jobscript picks idlest GPU by memory) — no queue wait after all;
  smoke follows on its hold.
- RESULTS (all landed same session): T1–T4 PASS (1830729); T5 PASS (gpytorch works on the
  node); T6 FAILED twice — 1830730 on a harness bug (small_conf duplicate-kwarg, fixed
  0381586, T1–T4 re-verified), resubmit 1830732 on SUBSTANCE: train R2 0.1700 < 0.95 after
  50 epochs, glacial monotone climb. T7 smoke 1830731 mechanically CLEAN (full train+eval
  package in ml_closure/runs_piff/smoke_T7_20260712_0401/; kurtosis 352.9 B-item flagged).
- T6 DIAGNOSIS: spec-S1.2 physics nondimensionalization leaves y at RMS ~165 vs O(1)
  outputscale/noise init — ~2.7e4 variance gap the ELBO climbs too slowly. Fix options
  (a) data-informed hyperparameter init from train-y variance [recommended] or (b) recorded
  y-standardization constant — HELD for Sanaa in the gate email; no unilateral model change.
- GATE HELD: submit_piff_grid.sh NOT fired; [QG][GATE-ML][SGS-CLOSURE] follows with the full
  T1–T7 table + the T6 fix decision; recommendation: grid does not fire until T6 passes.
- Email: [QG][SUBMIT][SGS-CLOSURE] spooled via reporting/pending_mail (10-min cron relay).

## 2026-07-11 — CAPE-A + telS submitted, NaN-guard live, NaN outputs purged (global supervisor Fable; Sanaa chat rulings)
- RULINGS (chat ~14:40-14:50 EDT, full authorization per convention): CAPE-A; NaN-guard 3(i)
  "good"; QUESTIONS 1+2 "go for it" (= tel option (b) smoothed-switch rerun + dt=5e-4 rung
  drop); "rm -f the folders of the cape that have nans"; ALSO: always double-check emails
  actually sent (new standing rule, saved to agent memory).
- DELETED (~173 GB, NaN verified in every scalars.npz first): FPCape-{const,sine,ramp,ou,tel}
  (34 GB each, 99.9% NaN) + FPCape-smokeP2048 (94.2% NaN). Kept: smokeY1024, smokeD2048, all FPC-*.
- NaN-GUARD (phaseB_job.sh + phaseCape_job.sh): solver backgrounded; watcher polls the
  atomically-rewritten scalars.npz every 300 s, SIGTERM+KILL on NaN; post-run check exits 99
  on a NaN record. Submitters pass +qg.diag.flush_every=500 (rewrite every 5000 steps ~8 min
  wall) so detection latency ~15 min. sge-checker PASS (2 non-blocking advisories: orphan
  sleep <=300 s post-exit; guard-subshell rc unused).
- CAPE-A WAVE: phaseCape_job.sh dt -> 1.25e-4 (baked, commented to the smoke verdict); 5 new
  tables {const,sine,ramp,ou,telegraph}_dt1p25e-4_T120.npz generated + validated (bitwise
  dt-consistent with the 2.5e-4 set at shared times; OU exact by micro-grid construction).
  submit_phaseCapeA_wave.sh (hard-requires tables, save_rate 3600, no-overwrite guard):
  sgsCapA_co/si/ra/ou/te = 1830412/14/16/18/20, shed_CA* = 1830413/15/17/19/21 (hold_jid).
  Cost ~27 h/member (~135 GPU-h), 960k steps each.
- FPC-telS: modulation.py gained --switch-smooth-steps (telegraph): every jump incl. the
  T_wait entry jump (t=30, the first Cd~1378 impulse) -> 10-dt linear ramp; max per-step
  dRe 3400->340; switch times/seed unchanged; equals hard table outside the 7 ramp windows.
  closure-reviewer confirmed 1 edge (ramp truncated at t[-1] would end mid-jump silently)
  -> loud-reject guard added, table regenerated byte-identical. NOTE for the record: the old
  tel blowup t=68.615 sits ~0.25 BEFORE the 68.86 switch — impulse hypothesis not a lock;
  telS is the discriminating experiment. submit_FPCtelS.sh: sgsB_teS 1830422 + shed_CteS
  1830423 (~11 h). Old partial FPC-tel kept.
- dt=5e-4 convergence rung: DROPPED per ruling; no rerun. dt study = 2 points + 1.9e-6
  early-window agreement.
- ML SPEC 01 committed verbatim (was on disk since 07-10, uncommitted). Next: CP-ML-1
  implementation-plan email, then WAIT for Sanaa's approval before building.
- Emails this session: [QG][SUBMIT][SGS] (this wave, cost-first) + [QG][PLAN][SGS-CLOSURE]
  CP-ML-1; both via pending_mail spool -> outgoing.mit.edu relay, delivery VERIFIED in
  relay.log per the new standing rule.

## 2026-07-11 — FPCape WAVE FAILED (5/5 NaN at t=0.1475) + mail-chain root cause fixed (global supervisor Fable)
- CONTEXT: Sanaa received ZERO emails since 07-09 and could not reply. ROOT CAUSE (definitive):
  mseas postfix has no relayhost -> direct-to-Outlook delivery silently junked; the 07-10
  "selftest PASS" mailed sanaamz@mseas.mit.edu = LOCAL loop, never testing off-node. FIX (live,
  tested): all mail via outgoing.mit.edu (reporting/daily_report.sh + send_pending.sh, explicit
  From/Reply-To=sanaamz@mseas.mit.edu); daily-report tqdm bloat fixed (6.5MB -> 2.9KB, \r-line
  sanitization + 500KB cap); cron relay verified end-to-end 12:50 EDT. Inbound mseas:25 verified
  listening on 18.18.38.35; final leg awaits Sanaa's TEST reply. All owed reports RESENT
  ([QG][POSTMORTEM] / [QG][REPORT][WIENER] with the 3-option fork / [QG][REPORT][SGS] / [QG][DAILY]).
- FPCape WAVE (1829708/10/12/14/16): all five ran 480k steps, "Simulation complete" — but ALL
  scalars/fields NaN from t=0.1475 (sample 59/48000; 99.88% of record). BIT-IDENTICAL pre-NaN
  trajectories across members (E 2.000->6.433, Cd_inst startup spike 21320) => deterministic
  startup-transient failure, inlet-independent. Shedding trackers (shed_Cp*) correctly refused
  ("0 samples at t>=30"). Third silent-NaN incident family (after FPC-tel, cnv dt=5e-4).
- DISCRIMINATORS vs healthy FPC wave: cape penalty 1.025 -> dt/(2 eta)=0.49 (cylinder 1.25 ->
  0.40); post-spike Cd ~50 with E climbing (cylinder ~0.5, E flat). Cape yaml values are the
  paper's 1024^2 / dt 1e-4 / Re 400 setup, run verbatim at 2048^2 / dt 2.5e-4 / Re<=4456; the
  pre-launch CFL check (0.041) did not re-derive the penalty margin.
- SMOKES SUBMITTED (sge-checker PASS; T=3, save_rate 500 to snapshot t=0.125 pre-onset):
  capeSmkP 1830334 (2048^2, penalty+sponge 1.25 — penalty-stiffness arm) and capeSmkY 1830335
  (1024^2, yaml 1.025 — resolution arm). Verdict email follows their landing.
- SMOKE VERDICT (same day, 3 arms landed by 13:11): smkY 1024^2@2.5e-4 CLEAN (E<=2.517, Cd~23);
  smkP 2048^2@2.5e-4 pen1.25 BLOWS t=0.1725 (delay only; E 7.6e28 pre-NaN); smkD 1830340
  2048^2@dt 1.25e-4 pen1.025 CLEAN (E<=2.695, Cd~27). => PENALTY EXONERATED; the cape at 2048^2
  is dt-unstable at 2.5e-4 (resolved tip velocities cross the local advective edge); dt<=1.25e-4
  required at 2048^2. T=3 clean != T=120 clean — NaN-guard goes in before any rerun. Rerun fork
  emailed ([QG][LANDED] 13:1x): CAPE-A 2048^2@1.25e-4 (~135 GPU-h, tables regen at new dt) vs
  CAPE-B 1024^2@2.5e-4 (charter row, ~15 GPU-h, tables reusable). HELD for Sanaa.
- HELD FOR SANAA (reply-approvable, in the SGS email): NaN-guard (i) in job wrapper (now 7
  silent-NaN completions, ~55 GPU-h burned on the cape wave), cape-wave rerun with the fixed
  parameter, FPC-tel rerun choice, dt=5e-4 rung drop, ~23 GB NaN-field cleanup.

## 2026-07-10 — FPCape PRODUCTION WAVE SUBMITTED (global supervisor Fable; Sanaa order)
- Sanaa ORDER (chat): same five cases as the FPC ensemble, flow past cape, 2048^2 — supersedes
  the charter S4.1 cape row (1024^2, "CAPE-" naming) and its T=15 dt-smoke note. Justification
  recorded in DECISIONS: CFL(2048^2, dt 2.5e-4, U=2) = 0.041, identical to the 5 landed FPC
  runs; eta = factor*dt convention makes the penalty/sponge terms dt-invariant.
- Scripts ([fable-authored], sge-checker PASS 10/10): scripts/sge/phaseCape_job.sh (copy of
  phaseB_job.sh, single change scenario=flow_past_cape; commons dt 2.5e-4 / T 120 /
  nu 6.4443e-4 / f64 unchanged; cape geometry from conf/scenario/flow_past_cape.yaml verbatim:
  mask x_c=0.2, y_base 0, x_scale 1, y_scale 4, x_support 2, penalty 1.025, bc width 0.1,
  sponge 1.025) and scripts/sge/submit_phaseCape_wave.sh (wave-2 pattern; DRY-RUN default,
  --go to fire; DNS_FR never-overwrite guard; job names sgsCape_*/shed_Cp* unique in qstat's
  10 chars).
- SUBMITTED 19:08 EDT: sgsCape_co/si/ra/ou/te 1829708/10/12/14/16 (ibgpu.q gpu=1, save_rate
  1800 = dt_save 0.45, recorder rate 10 with per-run +qg.diag.out) -> shed_Cpco/si/ra/ou/te
  1829709/11/13/15/17 (all.q, hold_jid, --t-min 30, --tag FPCape-<case>). Hold chains verified
  via qstat -j. Queued behind current GPU load; overnight landing, Sanaa reviews tomorrow.
- Cape recorder adaptation (no science-code edits needed): scalars.py requires qg.diag.length
  for non-circular masks -> +qg.diag.length=1.0 (L_cape=1; Re_cape(t)=U(t)/nu in [1751,4456],
  mid 3103 — report with the cape length scale, never call it "Re 3900", charter line 29);
  probes = the APPROVED cape lee set (2026-07-07 entry below): wake (x_c+{1,2,3}L, 4.0) =
  (6.0265/7.0265/8.0265, 4.0), cross-stream (6.0265, 4.5/3.5), 6th recirculation (6.0265, 2.0).
- Inlet tables REUSED from the FPC waves (verified geometry-independent: modulation.py
  generates pure U(t) with no geometry inputs; bc.Flow.const_x_flow reads U[n] only and the
  cape bc function const-outlet-vorticity-rtd forwards inlet_table through **kwargs exactly
  like the cylinder path; dt match enforced at table load). No tables job needed — submitter
  keeps the conditional-regeneration branch as a guard.
- Shedding-tracker caveat (submitted anyway, per order): Cd/Cl/PSD/f_sh outputs are generic
  (Brinkman-reaction force on chi + Welch), but the T_sh(Re) theory table, St_ref 0.21 and the
  default band [0.15,0.55] are CYLINDER-referenced — cape St/T_sh comparisons are context-only
  until a cape-specific reference is ruled. Cape wake is bottom-attached/asymmetric; nonzero
  mean Cl expected.
- Cost: upper bound ~55 GPU-h (5 x ~11 h co-scheduled wall, wave-2 precedent); exclusive-run
  precedent (FPC-const) is 2.75 h/run. Storage ~4.5 GB/run f32 x 5 = ~23 GB.

## 2026-07-09 — GATE D-1 RULED: PASS WITH FINDINGS; St estimator fix SHIPPED (resumed session)
- Sanaa (chat, ~17:35 EDT): Gate D-1 = **PASS WITH FINDINGS**; Cd/Cl-low deferred with explicit
  trigger (low Cd persisting at higher-Re cases ⇒ investigate); **Welch/zero-x St computation
  APPROVED**; FPC-const re-confirmed. Recorded in DECISIONS (698c59f).
- St fix implemented in shedding_tracker.py: gate + headline St from Welch peak (zero-x
  cross-check); Hilbert retained ONLY for phase/T_sh(t) exports, its St kept as
  st_inlet_hilbert_deprecated; new keys St_inlet_welch / St_inlet_zero_crossing / St_cyl_welch
  in yaml+npz; NO existing key renamed/removed (st_verdict_gate1.py + gate1_overlays.py read
  raw scalars.npz only — verified untouched). G4 review PASS (1 LOW cosmetic: unguarded U_med
  divisor, pre-existing style; NaN-safe downstream).
- Smokes (all.q, §3-compliant): selftest 1828698 **12/12 PASS** (2 new St checks, err <3e-5);
  gd1 re-analysis 1828699 → shedding_v2_welchSt: **st_measured_inlet 0.19157, zero-x 0.19078,
  st_pass TRUE** (Hilbert-deprecated 0.15941 recorded). Gate D-1 CLOSED as pass-with-findings.
- Findings carried forward: Cd 1.136 / rmsCl 0.257 low (suspects: penalty-force
  under-prediction, wake re-entry); watch Cd at the higher-Re Phase-B cases per the trigger.

## 2026-07-09 — GATE D-1 LANDED + VETTED; [QG][GATE-D1][SGS] SENT; awaiting ruling (resumed session)
- Chain 1828396/1828397/1828398 landed 17:00 EDT (~45 min GPU, 221.5 it/s). Wake DEVELOPED:
  16 shedding cycles in t 250–1500, drag line at 2.0018×f_sh, third harmonic absent — the
  T=120 lesson closed. Tracker gate verdict as coded: FAIL ×3 (St 0.159 / Cd 1.136 / rmsCl
  0.257 vs canonical 0.195–0.20 / 1.3–1.4 / 0.4–0.7).
- physics-sanity vet (code-level, scalars.py + shedding_tracker.py) — findings:
  (1) St FAIL is an ESTIMATOR ARTIFACT: St built from Hilbert-median f (0.01301) which has
  ~3.5 cycles retrograde phase slip; Welch 0.01564 / zero-x 0.01557 / wake-probe 0.01548
  cluster → corrected St_inlet 0.1916/0.1908 = PASS inside the tracker's own ±5% band
  (shedding_tracker.py:343-344). (2) St_cyl 0.197 = two low biases cancelling (Welch f →
  0.237); U_cyl measured 1.5D UPSTREAM (scalars.py:242-248), Re_cyl=162 → route void.
  (3) Cd/rmsCl TRUSTWORTHY (U_inlet² norm verified, scalars.py:170-175) and genuinely LOW —
  self-consistent under-forcing; suspects: Brinkman-penalty force under-prediction, periodic
  wake re-entry (~6 domain flushes). (4) Δf resolution ±20% of f_sh — any future St gate
  needs a longer window or an error bar.
- [QG][GATE-D1][SGS] sent DIRECTLY from mseas (rc=0, 17:21 EDT; outbox
  2026-07-09_QG_GATED1_SGS_verdict.txt): ruling requested (pass-with-findings vs
  hold-for-root-cause), tracker St-fix proposed (Welch/zero-x for St; Hilbert retained for
  phase/T_sh exports only), 3 optional probes costed (penalty-drag audit / wake-re-entry A/B
  windows / long-window St error bar). DECISIONS updated. GATE D-1 REMAINS OPEN — no tracker
  edit, no probe submitted without her GO.
- FPC-const 1828324 ~83% at 17:14, landing ~17:40; watcher armed for the landing chain
  (Π_FF s{2,4,8} → audit_A → [QG][RUN][SGS] → t=30 IC extract).

## 2026-07-09 — Gate D-1 OPTION B SUBMITTED; EMAIL CHANNEL SWITCHED TO MSEAS (branch supervisor)
- Sanaa (chat, via coordinator): OPTION B APPROVED for Gate D-1 incl. EXPLICIT dt sign-off
  (2.5e-3, deviation from chartered 2.5e-4) — [red-approved] in DECISIONS (e1b8f99).
- Submitted (scripts e1b8f99, sge-checker PASS x3, GATED1_RELEASE guard): gd1_tab 1828396
  (all.q, const Re=200 table at run dt, T=1500, exit 0) -> sgs_gd1_re200 1828397
  (ibgpu.q gpu=1: 1024^2, dt 2.5e-3, T=1500, T_wait 250, save_rate 2800 [dt_save 7.0 =
  phase-binning T_sh/8=7.8 minus the S5 commensurability trap: T_sh/7.0=8.93], recorder
  rate 10 per-run diag.out; running 16:11 EDT, 221.5 it/s, ETA ~45 min) -> shed_gd1
  1828398 (all.q hold, --gate-d1 --t-min 250). Targets: Cd 1.3-1.4 / St 0.195-0.20 /
  rmsCl 0.4-0.7 (the ONE permitted literature comparison). [QG][GATE-D1][SGS] at landing.
- **EMAIL DELIVERY INCIDENT**: Sanaa did NOT receive [QG][FLAG][SGS]. Diagnosis: all FIVE
  SGS emails today went via mailx jobs on ibfdr-compute-0-0; rc=0 there = local-MTA
  handoff only, evidently no off-node relay — all five presumed lost. The one confirmed-
  delivered email today went via mailx on mseas.mit.edu (wiener agent). **CANONICAL
  CHANNEL NOW: mailx directly from mseas.mit.edu**, outbox file copy always; recorded in
  DECISIONS. Consolidated resend sent from mseas (rc=0): option-B SUBMIT + full FLAG
  content + Gate-1 results recap + Phase-B wave-1 status + incident report, inline
  tables, subject [QG][SUBMIT][SGS]. An ibfdr duplicate (job 1828402) doubles as a
  channel cross-check. Subject convention re-adopted: [QG][<established family>][<qualifier>].
- FPC-const 1828324 unaffected, running (54.8 it/s, ~2.5-3 h).

## 2026-07-09 — GATE-1 APPROVED; PHASE-B WAVE 1 SUBMITTED; Gate D-1 FLAGGED (branch supervisor)
- Sanaa approved the Gate-1 report (chat, via coordinator) — S3.4 hard stop LIFTED
  (7a6eb3b). She ordered wiener trainings 1827225/1827306 killed (their supervisor
  executed; this branch did not touch them; ibgpu host now free).
- Wave 1 submitted per charter S4 + Amendment 02 (scripts 501cb9f/3f4a81c, sge-checker
  PASS x3, PHASEB_RELEASE guard): phaseB_tab 1828323 (all.q, const table T=120 dt 2.5e-4,
  done) -> sgs_FPC_const 1828324 (ibgpu.q gpu=1: 2048^2, T=120, save_rate 1080 per the
  commensurability fix, recorder rate 10 with per-run diag.out, f64 solve/f32 write;
  running since 15:13 EDT, measured 54.8 it/s -> ~2.5-3 h) -> shed_FPCc 1828325 (all.q,
  hold, --t-min 30). Quota check: 57 TB free vs ~7.5 GB this wave.
- Theory doc read IN FULL (Amendment 02 gate). Consequences applied: audit_A NOT blind-
  chained (its S8/A decision rule needs Pi_FF tau_int(s=4) -> Pi_FF + audit_A submitted AT
  LANDING after the conditional mmap-prep decision); 4 modulated runs stay held behind
  Audit A; convergence tier held behind the t=30 IC extract.
- **Gate D-1 HELD + [QG][FLAG][SGS] sent**: chartered T=120 at Re=200 gives U=0.1026 ->
  9.8 convective units, 0.49 domain flushes, ~1.4 shedding periods (T_sh~62) — an
  undeveloped wake by construction (the Gate-1 St lesson). Options emailed: A) T~1500 at
  chartered dt (~15-17 h GPU); B) dt 2.5e-3 (CFL 0.010) T~1500 (~2 h, needs her dt
  sign-off); C) as-chartered plumbing-only run. Table generation ready on her ruling.
- Emails (corrected convention [QG][<family>][SGS], inline tables): [QG][SUBMIT][SGS]
  cost-first with job/parameter tables; [QG][FLAG][SGS] with the timescale table.
- At landing (next session / SGE end-mail): Pi_FF s{2,4,8} -> audit_A -> per-run
  [QG][RUN][SGS] report (S4.3: walltime, CFL, E/Z traces, U-overlay, snapshots at t=30/120,
  NaN check) -> t=30 IC extract for the convergence tier.

## 2026-07-09 — St verdict + Gate-1 overlays COMPLETE, [QG][GATE1][SGS] sent, HARD STOP (branch supervisor)
- Sanaa GO (chat, via coordinator; RED, recorded in DECISIONS): overlays + GATE1 report
  CONDITIONAL on first justifying St 0.11-0.12 vs 0.21. Email conventions corrected:
  subject order [QG][<family>][<qualifier>] (family second, SGS qualifier last); result
  tables INLINE in the body.
- ST VERDICT: **UNDEVELOPED WAKE** (her hypothesis), not normalization, not a tracker bug.
  Suspects cleared in shedding_tracker.py: St=f*D/U_inlet(t), U recorded (2.0 exact, not
  1); D=2r=0.4pi consistent with Re := U*D/nu = 3900; f from the LIFT PSD (drag only
  cross-checked near 2f_sh — not circular). Development numbers (diagnostics/
  st_verdict_gate1.py, all.q job 1828287 exit 0): Cl envelope thirds 0.29->0.67->1.48
  (const_rec, 5.0x and still growing at window end; ou 1.87x; sine rms 2.5x), observed
  cycles 2.00 vs expected 3.3-4.1, window 15.9-19.4 convective D/U units after only
  8-10 units of wait (limit cycle at Re~3900 needs O(50+)), f_sh(t) monotone ramp
  0.05->0.28 toward f_qs=0.334 with St_cyl(t) CROSSING 0.21 late-window (peak 0.235),
  PSD shows NO discrete line and Welch df=0.1 puts the 0.11-vs-0.21 gap at 1.6-1.9 bins.
  Caveats: 512^2 = 25.6 pts/D under-resolved at Re 3900 (rule 12 anchor; shifts absolute
  St but is not needed to explain the median); blockage D/(8pi) = 5%, minor. Smoke St is
  a NON-MEASUREMENT; production T=120 windows give ~30 cycles at df~0.011 (tracker
  in-design). No tracker fix required.
- GATE-1 OVERLAYS (diagnostics/gate1_overlays.py, all.q job 1828314 exit 0):
  * U_inlet-vs-table EXACT overlay: max|U_rec - U_table[step]| = 0.0 for const_rec,
    sine, ou (6000 rows each); sine/ou uniquely pin alignment offset 0 (the bc-doc
    semantic). Figures fig_gate1_u_vs_table.png per case dir.
  * dt-consistency STRICT IDENTITY: max|U_2p5[k] - U_1p25[2k]| = 0.0 over all 60001
    compared samples for BOTH ou and telegraph pairs (micro-grid subsample identity as
    designed). Figures + gate1_overlays_summary.yaml under outputs/SGS_closure_gate1/.
- GATE-1 CHECKLIST COMPLETE: bit-identity byte-exact PASS; recorder-on byte-exact
  (non-invasive); 5/5 smokes exit 0 no-NaN; per-case scalars sane (U=2.0/Re=3899.9955
  exact on const); shedding smoke coherent; U-vs-table exact; dt-consistency identity;
  St justified. [QG][GATE1][SGS] report emailed with inline tables; **HARD STOP** —
  no Phase-B submission until Sanaa approves the Gate-1 report (charter S3.4).
- Session note: a network drop killed the previous session mid-record; nothing was lost
  (DECISIONS entries + scripts + job outputs all on disk); resumed and completed.

## 2026-07-09 — Gate-1 GPU smokes SUBMITTED (branch supervisor; Sanaa green light, chat)
- Sanaa directive (chat, ~13:00 EDT): full green light to submit the held Gate-1 smokes on
  any FREE ibgpu slot; wiener trainings 1827225/1827306 untouchable. qstat -F gpu showed
  hc:gpu=6 free on ibgpu-compute-0-0 — submitted immediately, no queueing behind wiener.
- Pre-submission fixes (sge-checker PASS on both): job names sgs_gate1_* -> g1_* (all five
  collided in qstat's 10-char truncation; wrong-qdel hazard) = be2a0b0.
- ATTEMPT 1 FAILED: jobs 1828225-29 died ~5 s in at hydra composition — "+scenario=" append
  clashes with package-stable's scenario DEFAULT (decaying_turbulence in conf/config.yaml);
  qacct exit_status=1 x5, no partial output. The main-repo "+scenario=" convention applies
  to the fork's config (no scenario default), NOT to package-stable. Fix: plain
  "scenario=flow_past_cylinder_sponge" (matches outputs/flow_past_cylinder_re1000's
  overrides.yaml) = b0e455c.
- ATTEMPT 2 RUNNING: 1828230 g1_legacy / 1828231 g1_table_const / 1828232 g1_const_rec /
  1828233 g1_sine / 1828234 g1_ou, all r on ibgpu-compute-0-0 since 13:24 EDT, all picked
  GPU 0 (idle-pick race; benign at 512^2, wiener GPUs carry memory so never selected).
  ~205 it/s on 60000 steps => minutes-scale runs. Smokes are simulations, NOT training-class
  — I18 three-job monitor unit does not apply (payload run_qg.py; sge-checker concurred).
- CPU workers shedding_job.sh / audit_A_job.sh: NOT submitted — their inputs (scalars.npz
  from recorder runs / FPC-const production) do not exist until the smokes land; selftests
  already green 2026-07-09. Gate = data, not queue.
- [QG][SUBMIT][SGS-CLOSURE] sent via mailx job 1828235 pinned to ibfdr-compute-0-0 (the
  known-working mail node); body archived logs/outbox/2026-07-09_QG_SUBMIT_gate1_smokes.txt
  + copy in outputs/SGS_closure_gate1/outbox/.
- Next in-session: babysit to completion, qacct x5, [QG][LANDED] email, log results here.
- RESULTS (updated in-session):
  * legacy 1828230 + table_const 1828231: exit 0, 328 s wallclock each, full DNS output.
    **BIT-IDENTITY PASS AT BYTE LEVEL**: cmp legacy/table_const DNS.npy AND DNS_FR.npz
    both bit-identical (stronger than the max|dw|=0.0 criterion; cmp is not a frontend
    .py execution). The C.3 bit-identity arm of Gate-1 is green.
  * Recorder arms (const_rec/sine/ou) took TWO more bugs, both now fixed git-visibly:
    (BUG 1, attempt 2 = 1828232-34, exit 1 x3 at step 20000 = first flush,
    flush_every 2000 x rate 10): np.savez appends '.npz' to any filename not ending in
    it, so savez('scalars.npz.tmp') wrote 'scalars.npz.tmp.npz' and os.replace died
    FileNotFoundError. Fix 585c0bd: write through an open file handle (live package +
    solver_patches mirror, byte-exact cmp).
    (BUG 2, attempt 3 of sine/ou = 1828237/38 exit 1, const_rec 1828236 exit 0 but
    poisoned): hydra does NOT chdir — the recorder's default relative out
    'scalars.npz' resolved against launch cwd $QG_DIR: wrong location AND one shared
    scalars.npz(.tmp) across all concurrent recorder cases -> os.replace race (a
    sibling consumes your tmp). const_rec "succeeded" as last-writer only; its run dir
    had no scalars.npz. Fix d48fcae: submitter passes per-case
    +qg.diag.out=$QG_DIR/outputs/SGS_closure_gate1/<id>/scalars.npz (solver untouched;
    recorder already honored diag.out). PHASE-B RELEVANT: every future recorder run
    MUST carry a per-run diag.out.
  * Mixed-writer artifacts + const_rec run-1 quarantined (moved, never deleted;
    DNS_FR.npz intact) to outputs/SGS_closure_gate1/quarantine_2026-07-09_shared_cwd_scalars/
    with README.txt.
  * Attempt 4 (recorder arms only): jobs 1828241 const_rec / 1828242 sine / 1828243 ou —
    ALL exit 0 (272 s each), per-case scalars.npz (1.16 MB, 4000 rows) in each run dir,
    no strays in $QG_DIR. **const_rec (recorder ON) is ALSO byte-identical to legacy**
    (cmp DNS.npy + DNS_FR.npz) — recorder provably non-invasive. GATE-1 SMOKE MATRIX
    FULLY GREEN: 5/5 cases landed, bit-identity byte-exact, recorder arms recording.
  * Shedding tracker on the 3 recorder scalars (first real-data run, gate=data now
    clear): jobs 1828244 shed_g1cr / 1828245 shed_g1sn / 1828246 shed_g1ou, all.q,
    exit 0 x3, --t-min 5.0 (T_WAIT; the 30.0 default exceeds the T=15 smoke horizon).
    Outputs <case>/shedding/{summary.yaml,npz + PSD/instfreq PNGs}. Physically coherent:
    clean lift peaks, f_drag~2f_sh (ratio 0.81-0.83), no 3rd harmonic, alias-free,
    Hilbert phase advance ~2.0 cycles; const arm U_inlet_median=2.0 EXACT,
    Re_inlet_median=3899.9955. St_inlet 0.11-0.12 vs 0.21 ref = context only (2 cycles
    in the 10 t.u. window; 2D-truth framing, not a validation target).
  * audit_A_job NOT submitted: input = Phase-B FPC-const production run (post-Gate-1
    approval). Gate = data, not queue.
  * [QG][LANDED][SGS-CLOSURE] sent via mail job (ibfdr-compute-0-0); body archived in
    logs/outbox/2026-07-09_QG_LANDED_gate1_smokes.txt + outputs outbox copy.
  * Wiener trainings 1827225/1827306 verified untouched throughout.
  * REMAINING for Gate-1 report (awaiting Sanaa GO): U_inlet-vs-table exact overlay,
    dt-consistency overlay (ou 2.5e-4 vs 1.25e-4 table), [QG][GATE1] report, HARD STOP.

## 2026-07-09 — session close: CP-1 module set COMPLETE, selftests green (branch supervisor)
- Landed: 82832aa (CP-1 approval + ledger reconciliation), a58e1aa (modules 1-3),
  02a5c47 (audit_resolution.py + CPU workers shedding_job.sh/audit_A_job.sh;
  bash -n clean, sge-checker PASS). CP-1 module list is now fully authored AND committed.
- audit_resolution.py notes: 5-grid-aware (--grids, partial-tier tolerant; 256/512
  labeled under-resolved lower-bound anchors; 2-deg rule on the 4096-vs-2048 pair only).
  eta terminology reconciled per S7.2: theory doc's eta_phys = RATE 1/tau_eta; the
  YAML/scalars eta = TIMESCALE tau_eta = penalty*dt as applied (scalars.py meta
  verified); delta_eta = sqrt(nu*tau_eta) = sqrt(nu/eta_phys_rate). Fixed-physical-eta
  tier check compares tau_eta across grids.
- Selftests on all.q (Amendment 02 §3 — the previously phantom-recorded step, now real):
  shed_st 1828217 PASS 10/10; audA_st 1828218 PASS 12/12; wake_st 1828219 PASS 10/10;
  audB_st 1828220 PASS 28/28. qacct: failed=0, exit_status=0 on all four. Known-benign
  env-activation .err noise (imageio_ffmpeg + dirname, session-2b note). Minor: numpy
  DeprecationWarning on np.trapz in audit_decorrelation — harmless, swap to
  np.trapezoid at next touch.
- Root-level SGS_closure_supervisor_brief.md duplicate removed (byte-identical to the
  canonical docs/briefs/ copy; was untracked).
- Gate-1 GPU smokes remain APPROVED-PENDING-GPU (entry below); ibgpu still held by
  wiener jobs 1827225/1827306 at session close.
- Next: CP-2 submission plan (must include the 256^2/512^2 convergence runs per
  e9a2b2d) → relay to Sanaa → Gate-1 smokes when GPUs free.

## 2026-07-09 — CP-1 APPROVED (Sanaa, chat) + ledger reconciliation (branch supervisor)
- CP-1 (audit/diagnostics code plan, emailed [QG][PROPOSE][SGS-CLOSURE] 2026-07-08) is
  APPROVED by Sanaa 2026-07-09 via chat. Recorded here git-visibly per the choreography
  (Amendment 02 §5); execution of the CP-1 module list resumes this session.
- INTEGRITY RECONCILIATION: the 2026-07-08 evening session died mid-work. DECISIONS.md
  had recorded (a) scripts/sge/shedding_job.sh + audit_A_job.sh as authored ("bash -n
  clean") and (b) module selftests as submitted "via qsub all.q: see logs/". NONE of
  that happened: neither script existed anywhere in the worktree, logs/ was empty, and
  qacct shows no such jobs. The three diagnostics modules (audit_decorrelation.py,
  shedding_tracker.py, diagnostics_wake.py) DID exist, complete but uncommitted. The
  phantom DECISIONS.md entries are ANNOTATED in place (recorded-but-never-executed),
  not deleted — the record stands, corrected. Remaining CP-1 gap: audit_resolution.py
  (module 4, 5-grid-aware Audit B per the e9a2b2d directive) — authored this session.
- Gate-1 GPU smokes: APPROVED-PENDING-GPU (Sanaa, chat 2026-07-09). Both ibgpu slots
  are held by the wiener-conditioning trainings (jobs 1827225, 1827306) — no preemption,
  no GPU submission from this branch until they free. Then GATE1_RELEASE=1 per session-2b.

## 2026-07-08 — DIRECTIVE (Sanaa, chat): convergence tier extended to FIVE grids
- The convergence grid study now covers the FULL sweep {256^2, 512^2, 1024^2, 2048^2,
  4096^2} (was {1024^2, 2048^2, 4096^2}). All MOD-const, shared-IC, fixed-physical-eta.
- BRANCH SUPERVISOR: this changes YOUR CP-2 submission plan — the convergence tier gains
  the 256^2 and 512^2 runs (t=30 IC extract spectrally regridded DOWN to both; storage
  +~0.7 GB total, negligible; dt/CFL per grid to be stated per-run in the plan). Audit B
  (audit_resolution.py) is being authored 5-grid-aware (--grids CLI, partial-tier
  tolerant, under-resolved coarse anchors labeled as lower bounds, 2-degree rule stays
  on the 4096-vs-2048 fine pair only).
- Theory context: pts/delta at Re_mid ~ 0.20 (256^2) and 0.41 (512^2) — deeply
  under-resolved BY DESIGN; they anchor the coarse end of the convergence curve.
  (Repo rule 12 — 512^2 under-resolved for cylinder at Re >= 600 — is exactly why they
  belong in a convergence study and exactly why they are NOT production grids.)

## 2026-07-08 — Supervisor_simulation.md delivered + committed; CP-1 plan emailed (global supervisor)
- Blocker resolved: Sanaa delivered the theory doc; committed verbatim as
  docs/briefs/Supervisor_simulation.md next to AMENDMENT_02_workflow.md. Read in full.
- CP-1 plan emailed ([QG][PROPOSE][SGS-CLOSURE]): module list (audit_decorrelation.py,
  audit_resolution.py, shedding_tracker.py, diagnostics_wake.py + batch wrappers),
  per-module outlines with the exact theory-doc quantity each measurement compares
  against, authoring order, §3-compliant execution plan, storage estimate. WAITING for
  approval — no diagnostics/audit code authored until CP-1 clears (choreography).
- Gate-1 smokes remain HELD (unchanged); Phase A artifacts unchanged.

## 2026-07-08 — AMENDMENT 02 received + committed (global supervisor); CP-1 BLOCKED on missing theory doc
- AMENDMENT 02 committed verbatim: docs/briefs/AMENDMENT_02_workflow.md. Precedence now
  Amendment 02 > Amendment 01 > charter. Adopted immediately: CYLINDER-ONLY scope (all
  FPC-cape items SUSPENDED, not cancelled); f64 solve / f32-at-write storage; dt_save
  0.25 t.u. (save_rate 1000), MOD-const runs 0.27 (save_rate 1080); scalars every 10
  steps; §3 ABSOLUTE rule: no .py execution on the login/frontend node (qsub/qlogin
  only; submit-only shell scripts exempt; violations are FLAG-level).
- BLOCKER: the accompanying theory document `Supervisor_simulation.md` was NOT attached
  and exists nowhere on the filesystem (searched QG_ROOT, all worktrees, docs/, home).
  Amendment 02 gates ALL planning (CP-1 included) on reading it in full — audits A/B,
  the commensurability rates, the §7.1 tables, and the §9 storage rule all reference
  its sections. [QG][FLAG][SGS-CLOSURE] sent; per the choreography the team WAITS.
- NOT started (correctly, per the gate): CP-1 plan, audit_decorrelation.py,
  audit_resolution.py, storage re-estimate. Phase A artifacts + Gate-1 hold state
  unchanged (see session entries below).

## 2026-07-08 — charter adoption record + I18 tooling (written by global supervisor)
- CHARTER v1.1 is CANONICAL ON MAIN (2056b46; merged into this branch as 5cc2e82
  2026-07-08). The email-appended v1.1 text is superseded by the file; canonical wins on
  any conflict. Known defect: the file carries the v1.1 block twice (merge artifact of
  284c702+2056b46) — dedup is in the v1.3 draft; charter edits are RED, Sanaa pushes.
- v1.2 (I16 anomaly playbook, I17 one-document rule) adopted operationally 2026-07-08;
  v1.3 (I18 monitoring-is-part-of-the-submission, I19 branch->global escalation) DRAFTED,
  RED-pending. Proof case: deriv7_cond_local job 1827034 ran 6 epochs order-inverted
  with no agent detection (P1 postmortem: main DECISIONS.md 2026-07-08).
- I18 tooling landed in THIS commit: diagnostics/monitor_training.py v2 (LIVE/FINALIZE,
  [QG][MONITOR] cadence first-val/every-5/on-trigger, ORDER-INVERSION vs physics-init
  medians 0.19/0.26/0.33, baseline card), scripts/sge/monitor_training_job.sh,
  diagnostics/baseline_cards/T1_deriv7.json, sge-checker G5 refusal (training qsub
  without the LIVE+FINALIZE monitor pair = REFUSED). Every future training submission
  from this branch is a three-job unit; [QG][SUBMIT][log] carries all ids.
- This branch's supervisor: CONFIRM adoption in your next digest (ORDER 3).

## 2026-07-07 — session 2b (global supervisor Fable: gate1 tables LANDED)
- Sanaa GO for the tables CPU job ONLY (explicitly no other qsub; smokes wait until the
  mmap builds free the GPUs). Submitted job 1826260 sgs_gate1_tables — 27 s, clean exit.
- 6/6 tables + PNGs under qg-simple-package-stable/src/qg/outputs/SGS_closure_gate1/tables/,
  all stamped sha 4cfb6dca4, all U(t=0)=2.000000:
  const 2.5e-4 (U == 2.0 everywhere — bit-identity arm exact); sine 2.5e-4 Re[2418,5600]
  (min unreached by design: 2/3 period in the 10 t.u. window); ou 2.5e-4/1.25e-4 seed
  20260707 Re[~3854,4225] N 60001/120001 (dt-consistency pair); telegraph 2.5e-4/1.25e-4
  rails [2200,5600] hit.
- Benign pre-script log noise (imageio_ffmpeg import error + dirname complaint from env
  activation under -V) — no effect; note for future log readers.
- Emails: [QG][LANDED][SGS-CLOSURE] sent 20:23Z (after session-2 PROPOSE at 20:13Z).
- HELD: the 5 GPU smokes (GATE1_RELEASE=1 submitter ready). Trigger = mmap builds done +
  Sanaa's next GO. Then: bit-identity diff → Cd/Cl + overlays → GATE1 report → HARD STOP.

## 2026-07-07 — session 2 (global supervisor Fable: post-disconnect resume, Gate-1 scripts released, PROPOSE sent)
- Session context lost to a connection drop; state fully recovered from git + this log
  (nothing on the branch was lost — all Phase A commits intact).
- Gate-1 scripts: TBDs filled and committed (3efca02) — modulation.py path
  $QG_ROOT/qg-sgs-closure/training/modulation.py (+ existence guard), T_WAIT=5.0,
  hydra APPEND keys as implemented (+qg.bc.inlet_table=<npz>, +qg.diag.scalar_rate=10;
  legacy arm qg.bc.inlet_velocity=2.0 — the config composes plain YAML, new keys need
  the + prefix; verified vs conf/config.yaml + register_configs). Table-path grep in
  the submitter re-verified against the + form. sge-checker re-audit post-edit: PASS x3.
- Bit-identity precondition confirmed in code: U_PER_RE = 2/3900 exactly, modulation.py
  asserts U(3900) == 2.0 bitwise, so table_const vs legacy can be exactly 0.0.
- SUBMISSION RULE (Sanaa, this session): NO qsub without her explicit per-step GO —
  email the report/PROPOSE first, wait for approval. A prepared sgs_gate1_tables qsub
  was countermanded before submission; NOTHING has been submitted on this branch.
- Email #1 for the branch sent: [QG][PROPOSE][SGS-CLOSURE] "Gate-1 release ready" to
  sanaamz@mit.edu, 2026-07-07T20:13Z (plan: tables job -> GATE1_RELEASE=1 5 smokes ->
  analysis -> GATE1 report -> HARD STOP).
- Queue note: running build_mmap/deriv7/monitor jobs belong to the wiener/main track, not SGS.
- Next (on Sanaa's GO): tables job -> smokes -> bit-identity diff + Cd/Cl + overlays ->
  [QG][GATE1][SGS-CLOSURE] report -> hard stop before Phase B.

---
## 2026-07-07 — session 1b (global supervisor Fable: Phase A code handoff)
- AMENDMENT 01 committed verbatim (docs/briefs/AMENDMENT_01_diagnostics.md).
- [fable-authored] training/modulation.py + training/spectral_regrid.py committed on branch.
  OU is realized on a DT_MICRO=1.25e-4 micro-grid and exactly subsampled → the Gate-1
  dt-consistency overlay is a strict identity, not statistical. `--re-const 200` covers
  Gate D-1. Regrid zeroes Nyquist both ways; `--self-test` prints the charter round-trip number.
- Solver hooks LIVE in shared qg-simple-package-stable (Sanaa ruling A.1), keys:
  `qg.bc.inlet_table` (bc.py, Flow.const_x_flow; table dt must equal solver dt, exhaustion
  guarded) and `qg.diag.scalar_rate` (+ optional out/length/u_mid/flush_every/probes) driving
  NEW src/qg/_output/scalars.py; 3-line stash in obstacle.py hands the recorder the exact
  discrete momentum sink (source-time, mean flow included — bc patch precedes penalty in the
  patch list). Zero extra FFTs: reductions only on sampled steps; Z via Parseval on qh(t_n).
  Mirror: solver_patches/sgs_hooks_2026-07-07.patch (byte-exact round-trip verified) +
  solver_patches/src/qg/_output/scalars.py; PORTING.md updated.
- Solver findings for the Gate-1 report (config audit §1.4): (a) `circular` mask IGNORES
  YAML x_center/y_center — obstacle sits at DOMAIN CENTER (Lx/2, Ly/2); recorder locates it
  from the applied chi centroid. (b) `mask.tol` key is inert (code reads `tolerance`).
  (c) nu/penalty/sponge/B/mu have NO float() casts in the solver — explicit-mantissa YAML
  mandatory. (d) post-step snapshots have ZERO-MEAN u (mean inlet lives only inside the step);
  Π_FF/wake stats from snapshots must re-add U_inlet — documented in scalars.py; scalars' E
  includes the mean-flow energy, DNS_FR_diagnostics' E does not.
- Answers to supervisor TBDs: t-wait 5.0 approved for Gate-1 smokes; modulation.py lives in
  training/ (flat); key names as above. Cape probe proposal approved incl. 6th recirculation
  probe — pass as qg.diag.probes + qg.diag.length=1.0 for cape.
- Next: supervisor fills TBDs → tables job → Gate-1 smokes → GATE1 report → HARD STOP.

---
## 2026-07-07 — session 1 (branch supervisor: charter ACK + AMENDMENT 01 receipt + Gate-1 prep)
- AMENDMENT 01 received and read in full (docs/briefs/AMENDMENT_01_diagnostics.md). Supersedes
  charter §7: ALL diagnostics code (scalar recorder, shedding_tracker.py, diagnostics_wake.py)
  is now Fable-authored; Sanaa authors only the later ML closure. Phase A RELEASED; extended
  Gate 1 per §C.3 (bit-identity with recorder absent + recorder-on Cd/Cl shedding smoke +
  U_inlet-vs-table exact overlay). NEW Gate D-1 (§F): FPC Re=200, 1024², T=120 — the one
  permitted literature comparison. Email enum ruling: charter codes for named reports
  (GATE1/RUN/CONV/PIFF/MILESTONE/GATE-D1), global codes (SUBMIT/LANDED/FLAG/PROPOSE/BLOCKED)
  for state changes, ISSUE→FLAG. Nothing submitted this session (per Fable: hold for handoff).
- Gate-1 job scripts DRAFTED (sge-runner authored under direction, per §8 matrix), HOLD state:
  scripts/sge/gate1_make_tables.sh, gate1_job.sh, submit_gate1_smokes.sh. Five smokes
  (legacy / table_const / const_rec / sine / ou), 512², dt 2.5e-4, T=15, nu=6.4443e-4,
  qg.grid.precision=float64 explicit (solver DEFAULT is float32 — key confirmed in qg/config.py).
  Submit guard: GATE1_RELEASE=1 required before any qsub. sge-checker audit: PASS ×3, no fixes;
  two operational notes (modulation.py path placeholder; run tables job to completion before the
  interactive smokes submitter — soft-skip, not hold_jid).
- TBD at Fable handoff: Gate-1 --t-wait (drafts use 5.0, flagged); modulation.py final path;
  bc.inlet_table + diag.scalar_rate key names as implemented.
- Cape lee-probe proposal (§C.1 equivalent set; from mask geometry, x_c=0.2·Lx=5.0265,
  tip height y_scale=4.0, L=L_cape=1): wake probes (x_c+{1,2,3}L, 4.0) = (6.0265, 4.0),
  (7.0265, 4.0), (8.0265, 4.0); cross-stream pair (x_c+1L, 4.0±0.5L) = (6.0265, 4.5/3.5);
  optional 6th recirculation probe (6.0265, 2.0) — cape wake is bottom-attached/asymmetric.
  Land clearance checked: cape height at x_c+1 is 0.736 (probe at 3.5 clears by 2.76); support
  ends at x_c+2. Incident-velocity window (U_cape analog of U_cyl): mean streamwise u over
  window centered (x_c−1.5L, 4.0), |y−4.0|≤0.5L, excluding masked points — clear of the left
  sponge (extends to 0.1·Lx=2.513) and of the cape support (starts x_c−2=3.027).
- DATASET_MANIFEST.md template planned (§G), to be instantiated per case at Phase D: case/geom/
  modid; git SHAs ([fable-authored] code + run commit); per-file path/shape/dtype for DNS_FR,
  DNS_LES_s{2,4,8}, U_of_t, scalars; grid (N, L, dx), dt, save_rate, scalar_rate; nu and the
  frozen PHYSICAL eta values; filter definition (scale, alpha, width, operator + code ref);
  Π_FF sign/normalization convention AS IMPLEMENTED (code line ref); usable window [30,120];
  seeds; probe coordinate table; NaN-check result; caveats. Target: zero-archaeology handoff.
- Next: await Fable's [fable-authored] handoff SHAs → fill TBDs → tables job → Gate-1 smokes
  (GATE1_RELEASE=1, SUBMIT email) → Gate-1 report → HARD STOP for Sanaa approval.

---
## 2026-07-07 — session 0 (branch instantiation, global supervisor Fable)
- Sanaa asked for: worktree setup, charter transmission to Opus branch supervisor, team
  acknowledgment of the §8 matrix / qlogin+milestone-email rules / Gate 1 stop, restatement
  + conflict report back to her. Phase A code authorship HELD until Sanaa acks the restatement.
- Done: worktree `qg-sgs-closure` added tracking origin/exp/sgs-closure (base = current main
  78c0ca4); charter read in full; SUPERVISOR_BRIEF.md instantiated with the §8 override matrix;
  Opus 4.8 branch supervisor briefed and acknowledgment collected.
- Environment notes for all future sessions: system git 1.8.3.1 cannot drive worktrees — use
  `/opt/rocks/bin/git` (2.9.2). Live solver = shared editable install qg-simple-package-stable
  (v0.2.1, OUTSIDE this worktree) — the bc.py hook location needs Sanaa's ruling (see conflict
  report). /gdata shows 58 TB free at fs level (96% full) — per-user quota to be verified in
  Phase 0 against the ~100 GB Phase B estimate.
- Config audit (preliminary, §1.4): `flow_past_cylinder_sponge.yaml` confirms the traps —
  `nu: 5e-3`, `tol: 1e-3` (PyYAML strings); `mask.r = 0.628318530717959` confirmed ✓;
  default grid 512², dt 1e-3, `penalty: 1.25` and `sponge: 1.25` (× dt convention) — physical
  eta values must be recorded per §4.1 and FROZEN for Phase C per §5.2.
- Decided next: await Sanaa's ack of the restatement → Phase 0 execution (full config audit,
  quota number, submodule/venv ruling) → Phase A (Fable authors modulation.py + bc hook).
- What Sanaa wants to see next check-in: her ack; rulings on the conflict list (email
  categories, bc.py hook location, 512² Gate-1 smoke grid).

---
## Seed
- Hypothesis (Phase 1, no ML yet): non-stationary inlet-modulated flow-past-obstacle cases
  give a controlled testbed for a priori SGS closure; deliverable = 10-run FR ensemble +
  convergence tier + Π_FF at scales {2,4,8}, ready for Sanaa's diagnostics_wake.py.
- Success criterion: Gate 1 pass (bit-identity + smoke stability + dt-consistency), clean
  production ensemble (no NaN, CFL reported), monotone-or-flagged short-horizon convergence,
  complete DNS_LES_s{2,4,8}.npz set per case.
- Baseline/control: MOD-const (Re = 3900) per geometry.
- Truth framing: 2D fine-grid penalized-obstacle solution ONLY; Cd≈0.99 / St≈0.21 are
  context, never validation targets.

## 2026-07-13 ORDER 2 (SGS supervisor): results map + 8 diagnostics — LANDED
- New standalone tool `ml_closure/diagnose_piff.py` (imports dataset_piff/model_piff as-is; train/eval helpers copied in). Jobs: 1832173 (all.q, item 7 log parsing), 1832174/1832175 (ibgpu.q gpu=1; FPC ckpt prod_ext150 on FPC-const, cape ckpt on 5 FPCape members). All clean.
- Outputs: runs_piff/prod_ext150/diagnostics/ + runs_piff/cape_base_100ep/diagnostics/ (+ zeta_ls curves in every grid/smoke run dir).
- Headlines: (1) sigma FLAT across |grad omega| deciles (1.0002x) vs |err| 39.6x (FPC) — Arm F prior quantified; (2) overdispersion lives in the FREESTREAM (cov1 free 0.96-0.99 vs wake 0.85-0.96); (3) zeta_ls frozen at ln2 for ALL FPC-const runs, moves 0.877->2.308 on cape = data property, not optimizer; (4) kurtosis tail lives in FPCape-sine (1428); (5) cape error fields temporally COHERENT across the shedding cycle (corr 0.85-0.92 at lag 0.45) vs FPC fast decorrelation.
- FPC-sine/ramp/ou/tel model diagnostics impossible until their Step-0 packages exist (no DNS_LES_s4/U_of_t/manifest).
- Email: reporting/pending_mail/landed_20260713_results_map.mail ([QG][LANDED][sgs]).
