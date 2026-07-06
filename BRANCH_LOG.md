# BRANCH_LOG — Exact recursion, no ML  (branch: exp/recursion-noml)

Running record. Supervisor updates this at the end of every session. Newest entry on top.

## 2026-07-06 — session 0 (seed, by global supervisor)
- Sanaa asked for: create branch + seed brief; PARKED — no work until she says go.
- Ran / submitted (job ids): nothing — parked.
- Results: n/a.
- Flags from physics-sanity: none.
- Decided next: nothing until Sanaa un-parks. First decisions when live: D⁻¹ mechanism (exact-FFT
  ~3 transforms/order vs learned-local) and flux-vs-advective Jacobian form.
- What Sanaa wants to see next check-in: her go-signal + her speed ideas for the transforms count.

---
## Seed
- Hypothesis (design, not learned): exact recursion ω^(k)=Lω^(k−1)+N^(k−1) as the closure — no
  time-FD, no inner wall, dT only in analytic prefactors. Cost floor ≈ 5(m+1) transforms/order with
  direction-grouped accumulators; Sanaa has speed ideas to bring.
- Success criterion: TBD by Sanaa — this is the no-ML CEILING reference; compare COST vs learned branches.
- Control ref (`deriv7_filtered_floor0.1`): Ndot 0.058 | **Nddot 0.186** | N3dot 0.563.
- Status: PARKED — no code, no data, no jobs until Sanaa says go.
