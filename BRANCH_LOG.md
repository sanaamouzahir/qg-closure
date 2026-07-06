# BRANCH_LOG — Learnable time-FD coefficients  (branch: exp/free-time-fd)

Running record. Supervisor updates this at the end of every session. Newest entry on top.

## 2026-07-06 — session 0 (seed, by global supervisor)
- Sanaa asked for: create branch + seed brief; this branch STARTS NOW (same trainer, one module change).
- Ran / submitted (job ids): nothing yet — branch just created.
- Results: n/a.
- Flags from physics-sanity: none yet (watch for odd-even/checkerboard in learned rows once trained).
- Decided next: (1) implement W_unit rows 1..3 → nn.Parameter, Vandermonde init, row 0 frozen, keep
  analytic 1/dt^k scaling; (2) closure-reviewer on the diff; (3) sge-checker → sge-runner the control
  config (S=7, grad_kernel 15, lr 5e-5, 300 ep, --rel-floor 0.1, filtered splits, all members).
- What Sanaa wants to see next check-in: first `deriv7_freeW` run landed, Nddot vs control 0.186.

---
## Seed
- Hypothesis: the Vandermonde time-FD rows are optimal for noiseless truncation but NOT the optimal
  linear estimator on real pooled data (Sanaa's small-dt regression: optimal coeffs ≠ Vandermonde,
  the Wiener-in-time mechanism). Learning rows 1..3 beats fixed Vandermonde on Nddot.
- Success criterion: pooled TEST Nddot < control 0.186 at EQUAL data.
- Control to beat (`deriv7_filtered_floor0.1`): Ndot 0.058 | **Nddot 0.186** | N3dot 0.563 (Nddot = ceiling).
- Invariants: row 0 frozen; analytic 1/dt^k kept (learn dimensionless only, dt-portability survives);
  ORDER CLIP unchanged; no physics conditioning; change ONLY W_unit rows so the delta is attributable.
