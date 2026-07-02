# Theoretical Guarantees for the AB2CN2 Temporal Closure

*QG ML temporal closure (δR). Four diagnostics on three deep survivors
FRC-{Re25k, combo, kf4}. float64; dealiased (2/3) Jacobians; 512²; forced
turbulence; marks at dt_fine = 5e-3.*

---

## 0. Summary

The closure has a ΔT-dependent domain of validity, set by two nested walls.

- **Outer wall** — series radius ΔT★. The truncation series converges for ΔT < ΔT★.
- **Inner wall** — finite-lag stencil ΔT_inner(k). A depth-k time-stencil resolves
  the N-derivatives only for ΔT below ΔT_inner, hit first.

Both reduce to one timescale: the eddy turnover time τ_eddy = 1/σ.

> ΔT★ = C · τ_eddy,   C = 2.08 ± 0.15   (universal across 2.5× Re, 2× β)

The inner wall comes from mark decorrelation: the oldest of k marks sits at lag
(k−1)ΔT and must satisfy (k−1)ΔT ≲ ΔT★, giving ΔT_inner ~ ΔT★/(k−1).

In the trained range (ΔT ≤ 1.5e-2) every member is inside ΔT★, so the binding limit
is the inner wall — fixable by more lags. Error propagation is benign: at ΔT ≤ 5e-3
the increment is one un-amplified term (N̈ in R₃), so derivative error = closure
error, 1:1.

---

## 1. Setup

Coarse QG vorticity:

> ∂_t ω̄ = L ω̄ + N,   N = −J(ψ̄, ω̄) + F,   ψ̄ = ∇⁻²ω̄,   Ḟ = 0.

N-time-derivatives by the dealiased chain rule:

> ω̄⁽ᵏ⁾ = L ω̄⁽ᵏ⁻¹⁾ + N⁽ᵏ⁻¹⁾,   N⁽ᵐ⁾ = −Σⱼ C(m,j) J(ψ̄⁽ᵐ⁻ʲ⁾, ω̄⁽ʲ⁾).

AB2CN2 truncation defect as a ΔT-series:

> δ(ΔT) = Σ_{p≥3} c_p ΔTᵖ,   c_p = R_p / D_p,
> R₃ = L³ω̄ + L²N + L Ṅ − 5 N̈,   D₃ = 12,
> R₄ = 2L⁴ω̄ + 2L³N + 2L²Ṅ − 4L N̈ + N⃛,   D₄ = 24.   (D₅=240, D₆=1440)

Each c_p is dominated by its top N-derivative N⁽ᵖ⁻¹⁾. The −5 on N̈ is the AB2
stencil's blind spot to N̈; it makes the learned N̈ term necessary and dominant (§5).

---

## 2. Outer wall — radius ΔT★

**Lemma 1 (Cauchy–Hadamard).** The series δ = Σ c_p ΔTᵖ has radius

> 1/ΔT★ = limsup_p ‖c_p‖^{1/p}.

The ratio ‖c_p‖/‖c_{p+1}‖ only **brackets** ΔT★ (Rudin 3.37); the c_p are
non-geometric, so it is not an estimate. Headline ΔT★ = root-test median, p = 3..6.

**Measurement** (p = 3..6, 32 anchors):

| member | ν | β | Re | N-growth ‖Nᵐ⁺¹‖/‖Nᵐ‖ (m=0→1…4→5) | ΔT★ (root) | ratio bracket |
|---|---|---|---|---|---|---|
| Re25k | 4.0e-5 | 1.0 | 2.5e4 | 34, 58, 72, 84, 93 | **0.066** | [0.017, 0.136] |
| combo | 4.0e-5 | 0.5 | 2.5e4 | 15, 27, 37, 45, 51 | **0.139** | [0.033, 0.270] |
| kf4   | 1.0e-4 | 1.0 | 9.8e3 | 10, 19, 28, 33, 37 | **0.199** | [0.043, 0.366] |

> **Proposition 1.** ΔT★(Re,β) = 1/limsup‖c_p‖^{1/p} is finite, set by N-derivative
> growth. For ΔT > ΔT★ no finite-order analytic closure converges.

**β at fixed Re.** Re25k and combo share ν. They differ only in β (1.0 vs 0.5).
The data:

| | β | σ = ‖Ṅ‖/‖N‖ | τ_eddy = 1/σ | ΔT★ |
|---|---|---|---|---|
| Re25k | 1.0 | 34.2 | 0.029 | 0.066 |
| combo | 0.5 | 14.7 | 0.068 | 0.139 |

Higher β → 2.3× the strain rate σ → 2.1× narrower ΔT★ (via Prop. 3, ΔT★ = C/σ).
So **ΔT★ depends on (Re, β) through σ(Re,β)**, not on Re alone. The mechanism
linking β to σ is not resolved here; only the σ-to-ΔT★ link is (§4).

---

## 3. Inner wall — finite-lag stencil

**Facts.** A depth-k backward FD for ω̈ is the curvature of the degree-(k−1)
polynomial through marks at lags {0, ΔT, …, (k−1)ΔT}. Oldest mark: lag (k−1)ΔT.

**Heuristic (decorrelation).** A mark is informative only if it has not
decorrelated: lag < τ_decorr ~ ΔT★. Applied to the oldest mark:

> (k−1) ΔT ≲ ΔT★   ⟹   **ΔT_inner(k) ~ ΔT★ / (k−1).**

This is a scaling law with an O(1) prefactor, not a theorem. Larger k reaches
further back, so larger k needs smaller ΔT. This is the "more lags, more
restrictive" effect.

**Two opposing effects of adding a lag.**

| effect | direction | regime |
|---|---|---|
| higher order, O(ΔT^{k−2}) | helps | ΔT < ΔT_inner |
| longer reach, (k−1)ΔT | hurts (decorrelation) | ΔT > ΔT_inner |

The inner wall is the crossover.

**Evidence — direction (fd_floor, Re25k N̈ floor, perfect spatial ops).**
Read across k at each ΔT; k\* = deepest lag that still lowers the floor:

| ΔT | N̈ floor, n = 3→7 | k\* | (k\*−1)ΔT |
|---|---|---|---|
| 0.005 | 35.9 → 14.9 → 6.6 → 3.4 → 2.0 | >7 (all help) | >0.030 |
| 0.010 | 68.9 → 50.0 → 41.1 → 40.5 → 41.1 | 6 | 0.050 |
| 0.015 | 96.6 → 98.7 → 107.7 → 124.4 → 151.0 | 3 | 0.030 |

k\* falls (7+ → 6 → 3) as ΔT rises. Predicted direction. (k\*−1)ΔT ≈ 0.03–0.05 ≈
0.5–0.8 · ΔT★, order-consistent with the heuristic. Coarse: integer k, 3 ΔT values.

**Evidence — magnitude (k=7 wall vs prediction).** Inner wall = ΔT where
relL2(ω̈_FD, ω̈_true) crosses 0.5 (a convention; the ramp is smooth, ranking is
threshold-robust):

| member | ΔT★/(k−1), k=7 | measured ΔT_inner | ratio |
|---|---|---|---|
| Re25k | 0.011 | **0.013** | 1.17 |
| combo | 0.023 | >0.020 (censored) | — |
| kf4   | 0.033 | >0.020 (censored) | — |

Censoring: a k=7 stencil reaches only ΔT ≤ 0.02 (6j ≤ 27 marks); combo/kf4 never
cross 0.5 in range, so their walls are lower bounds.

> **Proposition 2.** Deeper stencils lower the floor for ΔT < ΔT_inner(k) and raise
> it for ΔT > ΔT_inner(k). The wall obeys (k−1)ΔT_inner ~ ΔT★ (order of magnitude).
> Status: direction confirmed (k\* falls with ΔT); magnitude pinned only at
> Re25k, k=7 (ratio 1.17); the exact 1/(k−1) k-scaling is unconfirmed (censoring).

**Cross-check.** n=7 floor = fd_depth_check k=7, member-by-member (Re25k 1e-2 41.1%;
combo 1e-2 1.41%; kf4 1.5e-2 3.08%).

---

## 4. Decorrelation time

**Definition.** σ ≡ ‖Ṅ‖/‖N‖, the m=0→1 N-growth ratio. Units 1/time. Effective
strain/turnover rate (Ṅ/N = fractional rate of change of the advective term).
Turnover time: τ_eddy ≡ 1/σ. One derivative — cheap.

**Lemma 2 (radius ∝ turnover time).** The advective core has one timescale, τ_eddy.
Each ∂_t pulls one factor σ: ‖ω̄⁽ᵐ⁾‖ ~ aₘ σᵐ ‖ω̄‖. Then

> 1/ΔT★ = limsup ‖ω̄⁽ᵐ⁾/m!‖^{1/m} = σ · limsup(aₘ/m!)^{1/m} ≡ σ / C.

Hence ΔT★ = C · τ_eddy, with C dimensionless and a-priori O(1). Dimensional analysis
forces the proportionality; the empirical content is that C is (Re,β)-universal.

**Measurement.**

| member | σ (1/t) | τ_eddy (theory) | ΔT★ (measured) | C = ΔT★/τ_eddy |
|---|---|---|---|---|
| Re25k | 34.2 | 0.029 | 0.066 | 2.27 |
| combo | 14.7 | 0.068 | 0.139 | 2.05 |
| kf4   | 9.65 | 0.104 | 0.199 | 1.92 |

> **C = 2.08 ± 0.15.** Constant to 7% across 2.5× Re and 2× β.

The one-derivative turnover time predicts the six-order radius up to one universal
constant. Analytic horizon ≈ 2 turnovers.

> **Proposition 3.** ΔT★ = C · τ_eddy, C ≈ 2.1 independent of (Re,β),
> τ_eddy = ‖N‖/‖Ṅ‖. ΔT★(Re,β) inherits its parameter dependence from σ(Re,β).

**Consistency — ΔT★ three ways.**

| route | estimator | Re25k | combo | kf4 |
|---|---|---|---|---|
| series radius | ΔT★ root test | 0.066 | 0.139 | 0.199 |
| theory | C · τ_eddy, C=2.08 | 0.061 | 0.142 | 0.216 |
| stencil span | (k−1) · ΔT_inner | 0.077 | ≥0.120 | ≥0.120 |

All agree within the finite-p band (~30%).

---

## 5. Amplification / error propagation

**Setup.** δ = Σ_p (ΔTᵖ/D_p) R_p, R_p = Σ_k a_{p,k} L^{p−k} field_k. Learned terms:
field_{k≥2} = N⁽ᵏ⁻¹⁾, relative error ε. Error of term (p,k):
(ΔTᵖ/D_p)|a_{p,k}| ε ‖L^{p−k} field‖. High L-power ⟹ amplification.

**Result (ΔT = 5e-3, ε = 0.026/0.030/0.040 on Ṅ/N̈/N⃛).**

| member | ‖δ‖ | dominant term | err/‖δ‖ | RMS / L1 / corr |
|---|---|---|---|---|
| Re25k | 4.07e-5 | L⁰ N̈ (R₃) = 4.07e-5 | 3.00% | 3.00 / 3.05 / 3.00% |
| combo | 7.78e-5 | L⁰ N̈ = 7.77e-5 | 3.00% | 3.00 / 3.05 / 3.00% |
| kf4   | 9.98e-6 | L⁰ N̈ = 9.92e-6 | 2.98% | 2.98 / 3.10 / 2.98% |

The L⁰ N̈ term = ‖δ‖ to 3 figures, all members. It carries no L, so 3% on N̈ → 3% of
‖δ‖, un-amplified. Amplified R₄ terms (L²Ṅ, L¹N̈) are 5–7 orders down → 0.00–0.03%.
One term dominates, so RMS = L1 = corr.

> **Proposition 4.** At ΔT ≤ 5e-3, ‖δ‖ ≈ (ΔT³/12)·5·‖N̈‖ and closure error = relative
> N̈ error, un-amplified. The N̈ accuracy from deeper stencils (§3) converts to
> closure-error reduction 1:1.

**Caveat.** R₄/R₃ ~ ΔT, so amplified terms grow ∝ ΔT. At 1.5e-2 they are 2–3× larger
(still 5 orders down). Confirm with a ΔT sweep (pending). Result is ΔT=5e-3.

---

## 6. Operating envelope

> **Corollary.** For depth k, operate at ΔT ≲ ΔT★(Re,β)/(k−1). Then: series converges
> (Prop. 1), stencil resolves the N-derivatives (Prop. 2), accuracy → closure error
> 1:1 (Prop. 4). High-Re / low-β members cap first (Re25k inner ≈ 0.013). A pooled
> (Re,β)-blind closure caps at the worst member's inner wall.

**Recommendations.**

- Regenerate at k=7 for ΔT ≤ 1e-2 ensemble-wide; through 1.5e-2 for combo/kf4; cap
  Re25k at ΔT ≈ 1e-2.
- Keep N⃛ out of the loss at coarse ΔT (worst-conditioned: >100% at Re25k/1e-2,
  climbs with n). `--loss-weights 1 1 0` justified.
- Only N̈ is learned; it is the un-amplified dominant term. Closure is well-posed.

---

## 7. Caveats

- Finite-p (p=3..6): ±20–30%; trust ranking. Root test principled; ratio brackets.
- Inner-wall 1/(k−1) k-scaling: direction confirmed, magnitude only at Re25k k=7.
- combo/kf4 inner walls censored at ΔT ≤ 0.02 (k=7 reach); lower bounds.
- Amplification at single ΔT = 5e-3; envelope sweep pending.
- F = 0 in error_propagation (cascade-dominated; representative for ranking).
- Scope: 3 deep survivors. epoch0_faithfulness deferred (old packed schema; belongs
  on 7-lag slices).

---

## Appendix — provenance

| diagnostic | script | provides |
|---|---|---|
| convergence_radius | convergence_radius.py | ΔT★ (root), bracket, N-growth, inner wall |
| fd_depth_check | fd_depth_check.py | ω̈/N̈ floor, k=4 vs 7 |
| fd_floor | temporal_fd_floor_diagnostic.py | per-order floor, n=3→7 |
| error_propagation | closure_error_propagation.py | per-op ε → δ |

Archived under `Results/<diagnostic>/forced_turbulence/<run>/` (report.md +
console.txt + meta.json). float64; dealiased Jacobians matching the solver RHS;
truth = analytic chain rule; metrics relative-L2 in the resolved band.
