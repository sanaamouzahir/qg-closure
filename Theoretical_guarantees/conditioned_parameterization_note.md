# Conditioned Spatial Layer — Parameterization Note (step 2 design)

*For sign-off before implementation. Everything here follows from the 1.a/1.b
derivations and the v3 measured checks (CHECK 1–3).*

---

## 1. What the measurements fixed

- The per-(member, dT) optimal correction is a per-shell transfer
  r⁽ᵏ⁾(κ) = C_k · (ΔT·σ_ω(κ))^{S−k} · e^{iφ(κ)} + h.o.t. — a function of the
  **single dimensionless group x(κ) = ΔT·σ_ω(κ)**, times an exactly known
  ΔT-power per channel.
- CHECK 1: the ΔT^{S−k} factorization collapses the measured transfers where
  truncation dominates (Re25k spread 1.64×); h.o.t. matter only near the wall.
- CHECK 2: σ̂_ω(κ) from 2 marks matches the analytic σ_ω(κ) to 0.0–0.1% →
  **data-conditioning is lossless**: the model can read its regime off its own
  input stack. No (Re, β, μ) scalars anywhere.
- CHECK 3: the absorbable slice under the full Jacobian-slot basis is 36–44%
  (healthy tiers), capped partly by the width-15 reachability (iii = 0.31).
  A **spectral** correction has no reachability deficit — (iii) → 0 by
  construction.

## 2. The design

**Base operator: exact spectral gradients.** ik_x, ik_y (solver `Derivative`
multipliers). No learned local stencils in this branch. Init behavior is then
the measured [spec] floors: 0.0003 / 0.008 / 0.10 at 5e-3 — already at the
control's trained level before any learning.

**Correction branch: per-channel conditioned spectral transfer.** For channel
k (fields ω⁽ᵏ⁾ and ψ⁽ᵏ⁾, k = 0..3), the gradient operator applied inside the
Jacobian assembly is

    D̂_d^{(k)}(k⃗) = i k_d · [ 1 + ΔT^{S−k} · g^A_θ(x(κ), κ̃) ]
                  +   k_d ·   ΔT^{S−k} · g^B_θ(x(κ), κ̃)

with
- d ∈ {x, y}; the k_d prefactor keeps the correct parity in both terms;
- **ΔT^{S−k} analytic, per channel** (the TimeFD trick a third time: never
  learn a known power);
- g^A = in-phase correction, g^B = quadrature (the e^{iφ} of the measured
  transfer; the odd-S−k advective phase lives here);
- inputs: x(κ) = ΔT·σ̂_ω(κ) (the theory's group) and κ̃ = κ/κ_cut (grid-portable
  shell coordinate, handles dealias-band shaping);
- g^A_θ, g^B_θ: one tiny MLP per channel, 2→16→16→2, tanh; ~600 params × 8
  channels ≈ 4.8k trainable (control: 3.7k).

**Init: g_θ ≡ 0** (zero-init last layer) → exact spectral operators →
init eval must reproduce the [spec] floors to float64 round-off. That is the
integration acceptance test.

## 3. The conditioning input σ̂_ω(κ), de-biased

Per sample, from its own two newest marks:

    σ̂_raw(κ) = ‖(ω̂₀ − ω̂₋₁)_κ‖ / ( ΔT · ‖(ω̂₀)_κ‖ )

The 2-mark FD of a mode rotating at rate σ has response
(2/ΔT)·sin(σΔT/2), so invert analytically:

    σ̂(κ) = (2/ΔT) · arcsin( min( σ̂_raw(κ)·ΔT/2 , 1 ) )

- at small σΔT this is σ̂_raw (validated 0.0–0.1% at 5e-3);
- at coarse ΔT it removes the known FD saturation bias — no learning, no bias.
- cost: 2 rFFTs per sample (reusable from the dealias path), one shell
  reduction.

## 4. Training configuration

- Branch: exp/wiener-conditioning. Trainer unchanged except `--model
  cond_deriv`.
- Data: filtered splits, norm-floored loss (--rel-floor 0.1), all members,
  **minus the Re25k 1.5e-2 tier** (past-wall, unlearnable — Prop 2; approved).
- Optimizer: lr 5e-5, 300 ep, f64, batch 4 — identical to control for a clean
  comparison.
- ORDER CLIP and the physics-init binomial mix untouched.

## 5. Success criteria (from CHECK 3, per tier)

| tier | raw floor | conditioned ceiling (width-15 basis) | target |
|---|---|---|---|
| kf4 @1.5e-2 | 0.031 | 0.023 | ≤ 0.023 (spectral may beat it) |
| FRC-256 @1.5e-2 | 0.047 | 0.037 | ≤ 0.037 |
| FRC-256 @1e-2 | 0.0068 | 0.0055 | ≤ 0.0055 |
| Re25k @1e-2 | 0.252 | 0.243 | ≈ 0.24 (near-wall, expect little) |

Pooled (minus the dropped tier): predicted plateau ≈ 0.04 vs control 0.19.
The per-tier table is the honest scoreboard — the pooled number is dominated
by tier composition.

## 6. Two flagged limitations (carry into the report)

1. Near-wall tiers (x(κ) ≳ 0.5 in the tail) violate the leading-order
   factorization (CHECK 1 h.o.t.); the conditioned model helps little there.
   That is a property of the problem (Prop 2), not the parameterization.
2. The noise tiers (5e-3 for kf4/256) are float32-storage limited: nothing to
   absorb, nothing needed (raw 0.002–0.004 already below target).
