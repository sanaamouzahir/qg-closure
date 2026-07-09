# RESULTS — wiener-conditioning (LIVING DOC)

**Convention (Sanaa, 2026-07-09): this is THE results file. Overwrite it in place
every session — never create a new dated RESULTS_*.md. Per diagnostic run: ONE
consolidated .npz + ONE readable text block here. Dated RESULTS_2026-07-*.md files
are frozen archives; git history preserves every prior state of this file.**

_Last overwrite: 2026-07-09 evening (session 7e)._

## Scoreboard (final ckpts, trainings killed 2026-07-09 on Sanaa's order)

| run | killed at | best | pooled val | Ndot / Nddot / N3dot | medians |
|---|---|---|---|---|---|
| deriv7_cond_local_v2 (41 roots) | ep78/300 | ep63 | 0.21389 | 0.0744 / 0.1378 / 0.4296 | 0.0536 / 0.0929 / 0.1784 |
| deriv7_hygiene (17 control roots) | ep112/300 | ep107 | 0.24953 | 0.0594 / 0.1780 / 0.5112 | — |
| control deriv7_filtered_floor0.1 | done | — | ~0.19 pooled | Nddot ~0.186 | — |

Nddot 0.138 (cond) vs 0.186 (control) is NOT equal-data (41 vs 17 roots incl. DEC);
per-root eval (eval_deriv_by_root.py) REQUIRED before any scoreboard claim — NOT RUN
yet on the final ckpts. Hygiene ablation did not move the Nddot ceiling (0.178≈0.186).

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

- Per-root eval of both final ckpts (the standing "before any scoreboard claim" gate).
- Annulus-weighted loss proposal (λ~3 per-shell + rollout-aware injection) — design
  only, PROPOSED, no training without Sanaa's GO.
- σ̂-drift CSV, frozen-σ̂ A/B, control-as-5th-arm, save-refs, pareto, profile-step —
  six D-item ports PROPOSED with costs, awaiting per-item GO.
- Wiener filter theory formalization (iPad) before the next conditioned model.

## Archive pointers (frozen)

RESULTS_2026-07-03.md (quiescent-window investigation) · RESULTS_2026-07-08_smoke3.md
(apost smoke) · RESULTS_2026-07-09_apost_matrix.md (ladder matrix + dealias/FFT audit).
Consolidated npz: Results/apost_ladder_20260709*/ (one per case + summary CSV),
Results/spectral_error_profile_20260709/.
