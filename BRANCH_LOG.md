# BRANCH_LOG — Physics-conditioned spatial stencil  (branch: exp/wiener-conditioning)

Running record. Supervisor updates this at the end of every session. Newest entry on top.

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
