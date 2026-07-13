WHAT THIS FOLDER SHOWS
======================
The TRUE (measured, not assumed) sizes of the AB2CN2 truncation operators
R_3, R_4, R_5, R_6 on developed-flow snapshots of the three replication
members (FRC-kf4, FRC-256, FRC-combo), and whether the truncation series
predicts the improvement factors we MEASURED with the exact analytic R3
closure in 16-step rollouts.

INPUTS
  - omega snapshots: raw ensemble DNS, 4 per member at t~30/50/70/90
    (/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-simple-package-stable/src/qg/outputs/Step_size_resolution_closure_ensemble/<member>/DNS_FR.npz, float32 disk, upcast; all compute float64)
  - forcing rebuilt exactly from each member's hydra config
    (F = A cos(Bx) + D cos(Ey))
  - measured improvements: Results/apost_ladder_20260709_p170 (full world)
    and Results/apost_ladder_20260709_third23 (2/3 world) summary CSVs
    (16-step r3anal rollouts, FRC-kf4 IC837, truth RK4 h_fine=1e-5, I4).

FORMULAS
  tau = -(h^3/12)R3 - (h^4/24)R4 - (h^5/240)R5 - (h^6/1440)R6 + O(h^7)
  term_p = (dT^p / D_p) * ||R_p||,  D = {3:12, 4:24, 5:240, 6:1440}
  predicted ceiling of a PERFECT R3-only closure (one-step LTE-dominated,
  no accumulation/feedback):
    ceiling(dT) = (term3 + term4 + term5) / (term4 + term5)

OUTPUTS
  pngs/truncation_term_ratios_vs_measured_improvement/per_order_weighted_lte_contributions.png
      grouped bars, term_p per dT per member, log scale.
  pngs/truncation_term_ratios_vs_measured_improvement/predicted_ceiling_vs_measured_improvement.png
      predicted ceiling (3 members) vs measured r3anal improvement (kf4,
      full world + 2/3 world), per dT, log scale.
  yamls/truncation_term_ratios_vs_measured_improvement/summary.yaml  -- all numbers incl. FD-validation medians.
  yamls/truncation_term_ratios_vs_measured_improvement/truncation_term_ratios_consolidated.npz -- per-snapshot raw.

PURPOSE
Sanaa 2026-07-13 item 2: sanity-check the measured 132.6x/35.1x/71.4x
(1.5e-2/1e-2/5e-3) analytic-closure improvements against what the
truncation series itself allows, and locate where the comparison gaps
(rollout accumulation vs one-step LTE; per-dT bare baselines; FD trust
in N^(4)/N^(5)).
