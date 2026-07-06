# BRANCH_LOG — Time derivative as spatiotemporal convolution  (branch: exp/time-conv)

Running record. Supervisor updates this at the end of every session. Newest entry on top.

## 2026-07-06 — session 0 (seed, by global supervisor)
- Sanaa asked for: create branch + seed brief; BUILD AFTER free-time-fd's first result (not in parallel).
- Ran / submitted (job ids): nothing — gated on free-time-fd landing a first Nddot.
- Results: n/a.
- Flags from physics-sanity: none.
- Decided next: while gated, read code + scope the per-lag spatiotemporal-conv module
  (ω^(k)=Σ_i K_i^(k) * ω_{−i}, init Vandermonde × spatial-delta so init == control). Do NOT submit.
- What Sanaa wants to see next check-in: free-time-fd's first number, then a cost estimate for this branch.

---
## Seed
- Hypothesis: per-order spatiotemporal convolution (a spatial kernel PER LAG across the 7 levels) is
  the most expressive form of the time-FD-as-Wiener idea (free-time-fd is its pointwise special case);
  it beats free-time-fd on Nddot enough to justify the parameter/compute increase.
- Success criterion: beats free-time-fd on Nddot, cost-adjusted (measure params / GPU-h / memory / walltime).
- Control ref (`deriv7_filtered_floor0.1`): Ndot 0.058 | **Nddot 0.186** | N3dot 0.563.
- Invariants: init == control (Vandermonde × delta); analytic 1/dt^k kept; ORDER CLIP unchanged;
  no physics conditioning; GATE — don't submit until free-time-fd's first result exists.
