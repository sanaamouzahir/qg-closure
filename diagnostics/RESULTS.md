# RESULTS — wiener-conditioning (LIVING DOC)

**Convention (Sanaa, 2026-07-09): this is THE results file. Overwrite it in place
every session — never create a new dated RESULTS_*.md. Per diagnostic run: ONE
consolidated .npz + ONE readable text block here. Dated RESULTS_2026-07-*.md files
are frozen archives; git history preserves every prior state of this file.**

_Last overwrite: 2026-07-09 evening (session 7f: diagnostics run-status audit + eval wave)._

## Diagnostics run-status (Sanaa's question: "did you run all of these?" — answer: NO)

Ran on current data AND cond_local: spectral_error_profile (ep63, 14 FRC roots),
rollout_aposteriori (3 matrices, but ONE member/ONE IC, kf4/IC837),
consolidate_apost_cases, diagnose_condlocal_init (G1 gate ×2), filter_quiescent_windows
(41-root pool). Triage ran on the INCIDENT ckpt only.

NOT run on cond_local (holes, ranked):
1. **eval_deriv_by_root on the final ckpts** — the "before any scoreboard claim" gate.
   → CLOSED THIS SESSION: jobs 1828724 (cond_v2 ep63, 41 roots) / 1828725 (hygiene
   ep107, 17) / 1828726 (floor0.1 regression vs Jul-6 CSV, validates the --model auto
   patch). eval_deriv_by_root.py patched: --model auto reads config.json next to ckpt.
2. A-priori debug ladder never on cond_local: diagnose_error_distribution (control-era
   Jul 3 only), diagnose_one_sample, diagnose_head_sign (control ckpt only).
3. All rollout evidence = kf4/IC837 at MID-TRAINING ep63; aposteriori_accuracy/
   stability ran on smokes only (control ckpt, one later-dropped quiescent IC);
   benchmark_walltime_closure never timed cond_local.
4. closure_error_propagation second pass (real ε after training) never ran — only the
   eps=1 geometry pass (Jun 29, 3 members).
5. Everything in analysis/ (numerics suite) + perfect-closure/pareto/multistep predates
   the S=7 pool and cond_local entirely (May–mid-June); pareto has a known unfixed flaw
   (bare dtb leg). σ̂-drift/frozen-σ̂ A/B are among the un-GO'd D-item ports.

Data-quality footing is solid: filter (41 roots), triage D2 byte-alignment (10 members
≤9.5e-14), G1 init gates. Gaps there: mark-noise/sliced-inputs scripts never pointed at
the TASK-0c deep builds; FD floor (temporal_fd_floor_deep) only on Re25k/combo/kf4.

npz count in Results/: 59 (three ladder dirs 12+2/18/9+3 case npz + summary CSVs;
smoke dirs hold the pre-discipline litter, apost_smoke3 alone ~67 csv/json).

## Scoreboard (final ckpts, trainings killed 2026-07-09 on Sanaa's order)

| run | killed at | best | pooled val | Ndot / Nddot / N3dot | medians |
|---|---|---|---|---|---|
| deriv7_cond_local_v2 (41 roots) | ep78/300 | ep63 | 0.21389 | 0.0744 / 0.1378 / 0.4296 | 0.0536 / 0.0929 / 0.1784 |
| deriv7_hygiene (17 control roots) | ep112/300 | ep107 | 0.24953 | 0.0594 / 0.1780 / 0.5112 | — |
| control deriv7_filtered_floor0.1 | done | — | ~0.19 pooled | Nddot ~0.186 | — |

**PER-ROOT EVAL DONE (jobs 1828724-26, 2026-07-09 18:30): the conditioning advantage
is REAL on equal data.** On the 17 shared roots, cond_v2 ep63 beats hygiene ep107 on
Nddot nearly uniformly (~20-40%): kf4 0.058/0.065/0.057 vs 0.085/0.088/0.086
(1e-2/1.5e-2/5e-3); FRC-256 0.064/0.087/0.062 vs 0.084/0.094/0.084; b2/b25/combo all
better; sole exception Re25k@1e-2 (0.197 vs 0.180). N3dot better still (kf4 0.10-0.19
vs 0.33). BUT the theoretical success bars are NOT met: kf4@1.5e-2 Nddot 0.065 vs bar
0.023-0.05; FRC-256@1e-2 0.064 vs 0.0055. Pathologies: DEC-loRe N3dot exploded per-root
means (8.50@5e-3, 2.01@1e-2; Ndot/Nddot healthy) — suspect within-root small-denominator
poisoning (rule 16: needs median check); FRC-b0@1.5e-2 broadly bad (Nddot 0.947).
Regression: control CSV reproduced to ~1e-10 rel (GPU reduction noise) — --model auto
patch clean. CSVs: eval_by_root_val.csv next to each ckpt.
Hygiene ablation did not move the Nddot ceiling (0.178≈0.186).

## ROLLOUT-IN-THE-LOSS: built, gated, smoked (session 7g, overnight 07-09/10)

Trainer `train_deriv_rollout.py` drives the EXACT validated a-posteriori stepper
(G-R1: max|Δω| = 0.0 vs bare AND r3only arms). G-R2 reproduced the offline-consistency
numbers (0.0575/0.0587 vs 0.0575/0.0586). Smokes (kf4+FRC-256, curriculum M 1→2→4,
30 ep, full BPTT): cond arm step-1 residual 0.058→0.031-0.038 (beat its offline
parent), phys arm 0.194→0.045, n_blown_val = 0 throughout.

**16-step rerun verdict (apost_rollout_smoke/): horizon-limited.** Blow-up pushed out
by ≈ the trained horizon (1.5e-2: step 7→11; 1e-2: 13→16-with-large-error) but NOT
cured; 5e-3 16-step error DEGRADED (2.9e-4→4.2e-3) — no extrapolation past trained M.
**Data limit: 28 marks ⇒ trainable M ≤ 21/7/3 per stride — 16-step@1.5e-2 is
untrainable from existing deep builds.** Options emailed (deeper builds / truth-free
annulus-energy regularizer past the marks / both); AWAITING SANAA'S RULING — no
training runs until then.

## THE 4-ARM TABLE + the no-bug verdict (session 7f, kf4/IC837, 16 steps, RK4 truth)

| arm \ dT | 5e-3 (1step → end) | 1e-2 | 1.5e-2 |
|---|---|---|---|
| AB2CN2 bare | 1.33e-5 → 2.10e-4 | 1.06e-4 → 1.75e-3 | 3.53e-4 → 3.80e-2 |
| + full analytic LTE | 1.87e-7 → 2.93e-6 (71×) | 2.99e-6 → 4.97e-5 (35×) | 1.54e-5 → 2.87e-4 (133×) |
| + conditioned NN (ep63) | 7.63e-7 → 2.91e-4 | 6.18e-6 → **BLOWS step 13** | 2.28e-5 → **BLOWS step 7** |
| + unconditioned NN (ep8) | 1.77e-6 → 3.60e-3 | 1.41e-5 → 1.09e-2 | 4.87e-5 → **BLOWS step 12** |

**Sanaa's bug-vs-rollout check: NO BUG.** Step-1 residual fraction == offline Nddot
rel-L2 essentially exactly (cond: .0575/.0586/.0646 rollout vs .0567/.0578/.0646
offline; uncond: .134/.134/.138 vs .125/.125/.130). The blow-ups are pure rollout
feedback (annulus pumping). Offline accuracy does NOT order rollout stability: cond is
2.2× better offline yet blows earlier than uncond at dT ≥ 1e-2 — stability is governed
by WHERE in k the residual sits, not its size. Analytic arm = the ceiling is reachable.
(4-arm email 2026-07-09; sources: apost_ladder_20260709{,_p170} + eval CSVs.)

## Latest findings (sessions 7b–7e, 2026-07-09)

1. **ε(k) profile is U-SHAPED** (spectral_error_profile.py, 3 ckpts × 14 roots ×
   6 val samples): worst at LOW k (Nddot ε(30): cond 0.18 / control 0.45), min at
   k~100–200, knee k=209 (cond) / 232 (control); aliasing annulus only ×1.4–1.7 worse
   than mid. Conditioning helps low-k ×2.7, annulus only ×1.4 (worse-than-control at
   b1/b2@1.5e-2). npz: Results/spectral_error_profile_20260709/.
2. **A-posteriori instability is 100% NN-specific**: r3anal (exact derivs) STABLE at
   all 3 dT with 132.6×/35.1×/71.4× improvement over bare; exact Nddot regulates the
   corner band. NN arms blow up via the annulus; --nn-project-radius (2/3 on the
   correction only) implemented per Sanaa's ruling (solver mask RED/untouched).
3. **The 2/3 WORLD answers NO**: re-masking the whole harness+truth+IC at radial
   (2/3)min(kmax) makes the NN WORSE (train/eval mask mismatch dominates); analytic
   stays strong (23.6×/16.3×/9.5×). Aliasing per se is NOT the NN's problem. Clean
   test would need a mask-matched retrain. npz: Results/apost_ladder_20260709_third23/.
4. **Dealias worlds CONFIRMED for Sanaa (session 7e)**: data + targets + model
   end-projection all sqrt(2)·(2/3) RADIAL (solver derivative.py:29-32; harness
   per-product; model single end-projection train_deriv.py:229 — no internal layer,
   no dt in the mask path). Pipeline internally consistent; annulus 170.7–241.4
   (512² mode units) consistently aliased. [QG][AUDIT][WIENER] sent.
5. **cond_local incident history** (resolved 2026-07-08): floors sampled pre-filter
   (21×–46,343× too small on FRC-b0..b1) → rule-16 poisoning; fixed d4c0179 (post-filter
   floors; amp (dT/dT_ref)^(S−k), k=0 zeroed). v2 was the first run where conditioning
   actually acted.

## Open items

- **NEW DIRECTION (Sanaa ruling 2026-07-09): rollout in the loss.** Either full backprop
  through the autodifferentiable solver, or an x-dt unrolled rollout-loss term. Design
  proposal emailed ([QG][PROPOSE][WIENER] 2026-07-09); trainer build + smoke gates next;
  production training awaits her GO.
- **Mask ruling (same date): the sqrt2 aliasing is across the board — comparisons are
  fair; do NOT special-case it.** Annulus-weighted loss WITHDRAWN; per-product-dealias
  and 2/3-retrain items CLOSED.
- DEC-loRe N3dot verdict: uniformly bad root (median 9.33 ≈ mean), stack roughness
  ~1e-5 ≈ f32 floor territory — input-information-limited (slow flow + f32 disk),
  NOT a model bug; harmless for rollout. Re-slice f64 only if a clean number is needed.
- ~~Per-root eval of both final ckpts~~ DONE (see scoreboard above).
- σ̂-drift CSV, frozen-σ̂ A/B, control-as-5th-arm, save-refs, pareto, profile-step —
  six D-item ports PROPOSED with costs, awaiting per-item GO.
- Wiener filter theory formalization (iPad) before the next conditioned model.

## Archive pointers (frozen)

RESULTS_2026-07-03.md (quiescent-window investigation) · RESULTS_2026-07-08_smoke3.md
(apost smoke) · RESULTS_2026-07-09_apost_matrix.md (ladder matrix + dealias/FFT audit).
Consolidated npz: Results/apost_ladder_20260709*/ (one per case + summary CSV),
Results/spectral_error_profile_20260709/.
