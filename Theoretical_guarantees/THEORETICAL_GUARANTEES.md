# Theoretical Guarantees for the AB2CN2 Temporal Closure

*Coarse-grid QG turbulence, ML temporal closure (δR). All results float64,
dealiased (2/3-rule) Jacobians matching the solver RHS, 512² (FRC-256: 256²)
forced turbulence, trajectory marks at dt_fine = 5·10⁻³. Truth = analytic
spectral chain rule; metrics = relative L2 in the resolved band. Members:
FRC-Re25k (ν=4.0·10⁻⁵, β=1), FRC-combo (ν=4.0·10⁻⁵, β=0.5),
FRC-kf4 (ν=1.02·10⁻⁴, β=1), FRC-256 (ν=1.02·10⁻⁴, β=1, 256²).*

---

## 0. Summary

The closure has a ΔT-dependent domain of validity set by two nested limits.

- **Outer wall** — the truncation series has a finite convergence radius ΔT★.
- **Inner wall** — a finite-lag time stencil stops resolving the N-derivatives
  at a smaller step; it is hit first.

Both are governed by one measurable timescale: the **Taylor microscale of the
nonlinear term**, τ_λ = 1/σ with σ = ‖Ṅ‖_rms/‖N‖_rms (§5, derived and
verified). Empirically ΔT★ ≈ 2·τ_λ across all measured regimes (§6). Error
propagation is benign across the whole trained envelope ΔT ≤ 1.5·10⁻²: the
closure error equals the N̈ approximation error 1:1, un-amplified (§7).

---

## 1. Governing equation and N-derivatives

Coarse QG vorticity:

> ∂_t ω̄ = L ω̄ + N,   N = −J(ψ̄, ω̄) + F,   ψ̄ = ∇⁻²ω̄,   Ḟ = 0,
> L̂(k) = −ν|k|² − μ + iβ k_x/|k|².

Time derivatives of the state and of the nonlinear term follow from the chain
rule and the bilinearity of J:

> ω̄⁽ᵏ⁾ = L ω̄⁽ᵏ⁻¹⁾ + N⁽ᵏ⁻¹⁾,   ψ̄⁽ᵏ⁾ = ∇⁻²ω̄⁽ᵏ⁾,
> N⁽ᵐ⁾ = −Σ_{j=0}^{m} C(m,j) · J(ψ̄⁽ᵐ⁻ʲ⁾, ω̄⁽ʲ⁾)   (m ≥ 1; F drops out).

Every quantity below is built from these exact relations.

## 2. The truncation series

The defect of one AB2CN2 step of size ΔT against the true flow is a power
series in ΔT:

> δ(ΔT) = Σ_{p≥3} c_p ΔTᵖ,   c_p = R_p / D_p,
> R₃ = L³ω̄ + L²N + L Ṅ − 5 N̈,   D₃ = 12,
> R₄ = 2L⁴ω̄ + 2L³N + 2L²Ṅ − 4L N̈ + N⃛,   D₄ = 24   (D₅ = 240, D₆ = 1440).

Each coefficient is dominated by its highest N-derivative N⁽ᵖ⁻¹⁾. The −5
weight on N̈ in R₃ is the AB2 stencil's structural blind spot to N̈: it makes
the N̈ term the dominant, irreducible part of the closure (quantified in §7).

## 3. Outer wall — the convergence radius ΔT★

**Lemma 1 (Cauchy–Hadamard).** The series δ = Σ c_p ΔTᵖ converges for
ΔT < ΔT★ with

> 1/ΔT★ = limsup_p ‖c_p‖^{1/p}.

The successive-ratio ‖c_p‖/‖c_{p+1}‖ only brackets ΔT★ (the c_p are not
geometric); the headline number is the root-test median over p = 3…6.

**Measurement** (p = 3…6, 32 anchors per member):

| member | ν | β | N-growth ‖N⁽ᵐ⁺¹⁾‖/‖N⁽ᵐ⁾‖ (m = 0→1 … 4→5) | ΔT★ (root test) | ratio bracket |
|---|---|---|---|---|---|
| Re25k | 4.0·10⁻⁵ | 1.0 | 34, 58, 72, 84, 93 | **0.066** | [0.017, 0.136] |
| combo | 4.0·10⁻⁵ | 0.5 | 15, 27, 37, 45, 51 | **0.139** | [0.033, 0.270] |
| kf4   | 1.0·10⁻⁴ | 1.0 | 10, 19, 28, 33, 37 | **0.199** | [0.043, 0.366] |

> **Proposition 1.** ΔT★ is finite and set by the growth of the N-derivatives.
> For ΔT > ΔT★ no finite-order analytic closure converges.

**β-dependence at fixed Re.** Re25k and combo share ν and differ only in β
(1.0 vs 0.5): halving β roughly doubles ΔT★ (0.066 → 0.139). The radius
depends on (Re, β) jointly through the flow's rate content, made precise in
§5–6. Note the per-order growth ratios in the table **increase with m**
(34 → 93 for Re25k): higher derivatives grow faster than any single rate.
§5 derives why.

## 4. Inner wall — the finite-lag time stencil

The N-derivatives entering the closure are estimated from k stored marks at
lags {0, ΔT, …, (k−1)ΔT} (a backward finite difference). Two opposing effects
of stencil depth:

| effect | direction |
|---|---|
| higher approximation order, O(ΔT^{k−m}) for the m-th derivative | deeper helps |
| longer reach back in time, (k−1)ΔT | deeper hurts once old marks are stale |

**Measurement** — N̈ relative error of the depth-4 vs depth-7 stencil, exact
spectral spatial operators, three regimes × three steps:

| member | ΔT | N̈ error, k=4 | N̈ error, k=7 | deeper stencil… |
|---|---|---|---|---|
| Re25k | 0.005 | 0.149 | **0.020** | helps 7.6× |
| Re25k | 0.010 | 0.500 | **0.411** | helps, marginal |
| Re25k | 0.015 | **0.987** | 1.510 | **hurts** |
| combo | 0.005 | 0.033 | **0.001** | helps 25× |
| combo | 0.010 | 0.126 | **0.014** | helps 9× |
| combo | 0.015 | 0.273 | **0.092** | helps 3× |
| kf4   | 0.005 | 0.021 | **0.003** | helps 7.6× |
| kf4   | 0.010 | 0.084 | **0.004** | helps 19× |
| kf4   | 0.015 | 0.183 | **0.031** | helps 6× |

The crossover is directly observed: for the narrow-radius member (Re25k,
ΔT★ = 0.066) the depth-7 stencil helps at ΔT = 5·10⁻³, barely helps at 10⁻²,
and **hurts** at 1.5·10⁻² — where its span 6·ΔT = 0.09 exceeds ΔT★. For the
wide-radius members (combo, kf4) depth-7 wins at every measured step. The
inner wall is where a stencil's span becomes comparable to the member's ΔT★;
its precise location is regime-dependent and, for combo/kf4, lies beyond the
measured range.

> **Proposition 2.** Deeper stencils lower the N-derivative error while the
> stencil span stays well inside ΔT★ and raise it once the span exceeds ΔT★.
> Observed at Re25k: help at 5·10⁻³ (span 0.03 ≪ 0.066), harm at 1.5·10⁻²
> (span 0.09 > 0.066).

## 5. The correlation function and the Taylor microscale

The two walls above invoke "staleness" of old marks. This section makes that
precise, from first principles, and verifies it.

**Setup.** Along a developed-flow trajectory, view the nonlinear term as a
process N(t) with the domain inner product ⟨a,b⟩ = ∫ a b dx. Assume
statistical stationarity (S): E‖N(t)‖² constant over the analysis window
(spin-up windows are excluded). Define the autocorrelation

> ρ(τ) = E⟨N(t), N(t+τ)⟩ / E‖N(t)‖².

**Derivation.** Taylor-expand N(t+τ) inside the correlation:

> E⟨N, N(t+τ)⟩ = E‖N‖² + τ·E⟨N, Ṅ⟩ + (τ²/2)·E⟨N, N̈⟩ + O(τ³).

The first-order term vanishes: ⟨N, Ṅ⟩ = ½ d/dt ‖N‖², whose stationary average
is zero. Differentiating E⟨N, Ṅ⟩ = 0 once more in t gives
E⟨N, N̈⟩ = −E‖Ṅ‖². Hence

> **ρ(τ) = 1 − (τ²/2)·σ² + O(τ⁴),   σ² = E‖Ṅ‖²/E‖N‖²,**

and the curvature timescale — the **Taylor microscale** of the N-process —

> **τ_λ = 1/σ,   σ = ‖Ṅ‖_rms/‖N‖_rms   (exact identity under (S);**
> **no Gaussianity, no spectral assumption).**

The same derivation holds verbatim for the state ω̄ with its own
σ_ω = ‖ω̄̇‖_rms/‖ω̄‖_rms, and per wavenumber shell κ with
σ(κ) = ‖Ṅ_κ‖/‖N_κ‖.

**Per-order laddering.** Define σ_m = ‖N⁽ᵐ⁺¹⁾‖_rms/‖N⁽ᵐ⁾‖_rms. Because each
time derivative weights the spectrum by another factor of the local rate,

> σ_m² = ∫ σ(κ)^{2(m+1)} E_N(κ) dκ / ∫ σ(κ)^{2m} E_N(κ) dκ,

the (m+1)-th moment ratio of σ(κ) under the N-spectrum. Since σ(κ) increases
with κ, **σ_m increases with m**: higher derivatives progressively sample the
fast small-scale tail. This is exactly the per-order growth escalation seen
in the §3 table (34 → 93).

**Verification** (analytic derivatives, no finite differences; curvature fit
of the measured ρ(τ) on the first lags vs the identity 1/σ):

| member | τ_λ(N) fit | 1/σ_N | τ_λ·σ_N | τ_λ(ω) fit | 1/σ_ω | τ_λ·σ_ω | σ₀, σ₁, σ₂ |
|---|---|---|---|---|---|---|---|
| Re25k   | 0.0329 | 0.0287 | **1.15** | 0.310 | 0.317 | **0.98** | 34.8, 60.4, 76.4 |
| kf4     | 0.1113 | 0.1020 | **1.09** | 0.743 | 0.768 | **0.97** | 9.8, 20.6, 29.4 |
| FRC-256 | — | 0.0670 | — | — | 0.560 | — | (Results CSV) |

The identity holds: τ_λ·σ = 1 within 15% on the N-process and within 3% on
ω. The residual +10–15% on N is the expected O(τ⁴) mixture effect — ρ is a
superposition of per-shell parabolas and the fast shells die first, so the
pooled curve decays slightly slower than the single-σ parabola. The
laddering is monotone (σ₀ < σ₁ < σ₂) in every member, as derived.

![Re25k: measured ρ(τ) vs the identity parabolas](img/rho_vs_identity_Re25k.png)

![kf4: measured ρ(τ) vs the identity parabolas](img/rho_vs_identity_kf4.png)

![FRC-256: measured ρ(τ) vs the identity parabolas](img/rho_vs_identity_FRC256.png)

**Two further measured facts.**

1. **The state decorrelates an order of magnitude slower than its nonlinear
   term:** σ_N/σ_ω ≈ 7.5–11 across members. Over the full 27-lag span, ρ_ω
   stays ≥ 0.96 while ρ_N falls to 0.2–0.6. A time stencil applied to the
   ω-history therefore sees smooth data; the difficulty of estimating
   N-derivatives lives in the derivative ladder (σ_m growth), not in bulk
   state decorrelation.
2. **ρ_N plateaus at a positive value** (≈ 0.2 for Re25k) instead of decaying
   to zero: the static component of N (the time-independent forcing plus the
   time-mean of −J) never decorrelates. This does not bias the identity —
   the static part is absent from ‖Ṅ‖ and inflates ‖N‖ exactly as it inflates
   the measured curvature denominator — but integral or 1/e decorrelation
   times would be distorted by it. τ_λ is the clean object.

> **Proposition 3.** Under stationarity, ρ(τ) = 1 − τ²σ²/2 + O(τ⁴) with
> σ = ‖Ṅ‖_rms/‖N‖_rms exactly; verified to 15% (N) and 3% (ω) across the
> measured regimes. σ_m grows monotonically with m as the (m+1)-th moment
> ratio of σ(κ); verified in every member.

## 6. The radius is two Taylor microscales

Sections 3 and 5 measure two independent timescales: the six-order series
radius ΔT★ and the one-derivative microscale τ_λ. They are proportional:

| member | ΔT★ (root test) | τ_λ = 1/σ_N | C = ΔT★·σ_N |
|---|---|---|---|
| Re25k | 0.066 | 0.029 | 2.27 |
| combo | 0.139 | 0.068 | 2.05 |
| kf4   | 0.199 | 0.104 | 1.92 |

> **C = 2.08 ± 0.15,** constant to 7% across 2.5× in Re and 2× in β.
> Using the directly fitted τ_λ instead of 1/σ moves C to 2.00 (Re25k) and
> 1.79 (kf4) — same band.

> **Proposition 4.** ΔT★ = C·τ_λ with C ≈ 2 independent of (Re, β): the
> analytic closure's convergence horizon is approximately **two Taylor
> microscales of the nonlinear term**. All (Re, β)-dependence of ΔT★ enters
> through the single measurable rate σ(Re, β).

This also grounds the inner wall of §4: the depth-7 stencil at Re25k starts
hurting precisely when its span (6ΔT = 0.09) exceeds ΔT★ = 2τ_λ — i.e. when
its oldest marks are several microscales stale.

## 7. Error propagation is benign across the envelope

**Setup.** δ = Σ_p (ΔTᵖ/D_p) R_p with R_p = Σ_k a_{p,k} L^{p−k}·field_k. The
learned quantities are the N-derivatives; an error ε on field_k contributes
(ΔTᵖ/D_p)|a_{p,k}|·ε·‖L^{p−k} field_k‖ — terms carrying powers of L are
potentially amplified.

**Measurement 1 — operating step ΔT = 5·10⁻³** (measured per-order errors
ε = 2.6/3.0/4.0% on Ṅ/N̈/N⃛):

| member | ‖δ‖ | dominant term | error/‖δ‖ |
|---|---|---|---|
| Re25k | 4.07·10⁻⁵ | L⁰N̈ (R₃) = 4.07·10⁻⁵ | 3.00% |
| combo | 7.78·10⁻⁵ | L⁰N̈ = 7.77·10⁻⁵ | 3.00% |
| kf4   | 9.98·10⁻⁶ | L⁰N̈ = 9.92·10⁻⁶ | 2.98% |

The un-amplified L⁰N̈ term equals ‖δ‖ to three figures in every member; the
amplified terms (L²Ṅ, L¹N̈) are 5–7 orders of magnitude smaller.

**Measurement 2 — envelope edge ΔT = 1.5·10⁻²** (unit per-term errors, pure
amplification geometry, Re25k — the fastest-cascade member):

| term | contribution to ‖δ‖ |
|---|---|
| L⁰N̈ (R₃) | **100.2%** |
| L⁰N⃛ (R₄) | 2.6% |
| L¹Ṅ (R₃) | 0.5% |
| L¹N̈ (R₄) | 0.2% |
| L²Ṅ (R₄) | 0.00% |

The feared L-amplification of the R₄ terms does not materialize even at the
largest trained step: the closure error tracks the N̈ error one-to-one
(correlated-field total: 99.98%), with a 2.6% additive N⃛ contribution.

> **Proposition 5.** Across the entire trained envelope ΔT ≤ 1.5·10⁻², the
> closure error equals the relative N̈ error, un-amplified:
> ‖Δδ‖/‖δ‖ ≈ ε_N̈. Any improvement in N̈ accuracy converts to closure-error
> reduction 1:1, and N̈ accuracy is the single quantity that sets the
> rollout ceiling.

## 8. Operating envelope

> **Corollary.** For a depth-k stencil, operate where the stencil span stays
> well inside the member's radius: (k−1)ΔT comfortably below
> ΔT★(Re, β) = 2τ_λ. Then the series converges (Prop. 1), the stencil
> resolves the N-derivatives (Prop. 2), and derivative accuracy converts to
> closure accuracy 1:1 (Prop. 5). Narrow-radius members (high Re, high β)
> cap the pooled envelope first: at k = 7, Re25k caps near ΔT ≈ 10⁻² while
> combo and kf4 remain inside through 1.5·10⁻².

## 9. Caveats

- Finite truncation depth (p = 3…6) puts a ±20–30% band on ΔT★; rankings are
  robust, absolute values carry the band.
- The inner-wall crossover is directly observed only at Re25k (its wall lies
  inside the measured ΔT range); combo/kf4 walls lie beyond ΔT = 1.5·10⁻²
  and are bounded, not located.
- The τ_λ identity is verified on the three members shown; decaying
  (non-stationary) members require detrending before the identity applies
  and are not yet measured.
- Scope: forced-turbulence members listed in the header; all statements are
  for the resolved (dealiased) band.
