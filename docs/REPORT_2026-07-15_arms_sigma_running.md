# Plain-English Report — 2026-07-15

Three parts: (1) Wiener — what the "arms" are, the multi-objective loss, and exactly how the von Neumann penalty enters; (2) SGS — σ stages 2 and 3; (3) what is running right now.

Sources: `training/train_deriv_rollout.py`, `training/wiener_certificate.py`, `scripts/sge/submit_w31_p1.sh`, `training/accept_ft_gate.py` (exp/wiener-conditioning); `ml_closure/train_crps_head.py`, `recalibrate_structural_sigma.py`, `sigma_conformal_prototype.py`, BRANCH_LOG (exp/sgs-closure); HANDOFF.md + qstat at 12:03 EDT.

---

## Part 1 — Wiener: the arms, the multi-objective optimization, the von Neumann penalty

### 1.1 What "the arms" are

The w31 stencil was trained a-priori (predict the derivatives). The **fine-tune (FT)** then trains it a-posteriori: during training we actually *run the closed solver* — unroll M coarse AB2CN2+closure steps from a developed-flow window and grade the trajectory against the fine reference. The "arms" are simply **two copies of this fine-tune, launched side by side, identical in every setting except one knob**: how hard the training defends the model's original derivative accuracy (the "anchor weight").

- **Arm A** (`w31p1a`, job 1834629): anchor λ = 3·10⁻² — anchor term sized to the rollout-loss scale ("balanced").
- **Arm B** (`w31p1b`, job 1834632): anchor λ = 3·10⁻¹ — anchor dominates ("hard anchor").

Why two: the previous FT generation (`p1lam01`) taught us that a stabilizing fine-tune can quietly *degrade* per-member N̈ accuracy — and N̈ accuracy IS the rollout floor (benign amplification, 1:1). So the two arms bracket the trade: A should move the rollout metrics more; B should defend N̈ better. Both warm-start from w31 `best.pt` (ep32), run 20 epochs on 7 rollout roots with **combo and b25 held out** as out-of-distribution checks.

**Acceptance is mechanical** (`accept_ft_gate.py`, your option 4): per-member, per-dT a-priori N̈ before vs after; PASS only if `after ≤ before × 1.10` for **every** (member, dT) cell and the overall median. Exit 3 on FAIL. However stable an arm looks, if it pays for stability with >10% N̈ anywhere, it is rejected.

### 1.2 The multi-objective optimization

Each optimizer step minimizes a sum of four terms:

```
L  =  L_roll  +  w_free · L_free  +  λ_vn · L_stab  +  λ_anc · L_anchor
        (1)          (2)                 (3)                (4)
w_free = 1e-2,  λ_vn = 0.1,  λ_anc = 3e-2 (arm A) or 3e-1 (arm B)
```

**(1) Supervised rollout loss** — "track the true trajectory."

```
L_roll = (1/M) Σ_{m=1..M}  ‖ω_m^closure − ω_m^truth‖₂ / ‖ω_m^truth‖₂
```

Truth = the deep 28-mark builds (RK4-ultrafine reference marks spaced 5·10⁻³). Stride s ∈ {1,2,3} gives dT ∈ {5e-3, 1e-2, 1.5e-2}; the truth supply per window is M_max = 21/7/3 steps respectively. The stepper is the *exact validated inference stepper* (8-FFT closure path), reused, autograd-safe — not a reimplementation. Unroll length ramps 16 steps for 6 epochs, then 21 for 14 (`--unroll-schedule 16:6,21:14`); gradients are truncated every 4 steps (`trunc:4`) so backprop depth (and memory) stays bounded.

**(2) Truth-free "analytic" term** — "stay physical after the reference runs out."
Every window keeps rolling to a total horizon of 16 steps even after its truth marks are exhausted (K = 16 − M extra steps; this is how dT=1.5e-2, with only 3 truth steps, gets long-horizon behaviour into training at all). In `analytic` mode (what the arms run), during those free steps the model's three heads are graded against the *exact analytic* N-derivatives of the state it is rolling through:

```
L_free = (1/K) Σ_{f=1..K}  min( (1/3) Σ_{c=1..3} relL2(pred_c, N^(c)_analytic) , 10 )
```

The supervised→free boundary always detaches (the approved truncated gradient). A blown-up free segment still teaches: the finite growth terms collected before the blow-up keep their gradient.

**(3) von Neumann certificate penalty** — section 1.3.

**(4) A-priori accuracy anchor** — "keep doing the original homework."
One batch per optimizer step drawn from the **full 41-root a-priori pool** (the warm checkpoint's own training data), evaluated with the *exact* original train_deriv objective — floored per-order relative L2 on the [Ṅ, N̈, N⃛] targets, rel_floor 0.1, dealias-projected:

```
L_anchor = relL2_floored( model(stencil), [Ṅ, N̈, N⃛]_targets )
```

This is the term whose weight differs between the arms. It is the in-training version of the acceptance gate: the gate rejects after the fact; the anchor prevents during.

In plain English, the four pulls are: *match the fine solution while truth exists* (1), *keep producing correct physics after it ends* (2), *be provably linearly stable at every step size and scale* (3), *never forget the a-priori derivative accuracy that sets your rollout floor* (4).

### 1.3 Exactly how the von Neumann penalty is incorporated

**Plain English first.** Classic von Neumann analysis asks: if I freeze the coefficients of my scheme and look at one Fourier mode, does one time step amplify it (unstable) or not (|G| ≤ 1)? We do this *to the closed scheme including the learned closure*, per training window, at the window's initial state, per radial wavenumber shell — and we make it **differentiable in the model's parameters**, so "don't be linearly unstable" becomes a loss term. It fires only where amplification exceeds 0.98, and only on shells where the frozen-coefficient linearization is actually valid.

**The math** (`wiener_certificate.py`, evaluated per sample and per shell κ):

Frozen linearized dynamics: `D(k) = L̂(k) + iσ(k)`, where σ(k) is the per-shell advection frequency read from the model's own σ̂ context (per-sample, from the window's initial spectral state).

The learned stencils enter through their trig-polynomial symbols. With θ = kΔx/√2 (isotropic evaluation):

```
ρ_c(θ) = Ŵ_{base+δ}(θ) / (iθ)          — symbol of (base taps + conditioner tap-deltas)
                                          over the exact ik; ρ ≡ 1 for exact stencils
```

(The 2026-07-14 "tap-read certificate" fix is here: the base-tap symbol is now read from the central row of the depthwise kernel — the old top-row read was zero at physics init and left the certificate nearly blind to the base taps. The arms are the first FT with the fixed read.)

Learned closure transfer per output order c (Wiener freeze: time-orders → Dʲ, Jacobians → iσ):

```
T_c(k) = Σ_{ij} mix[c,ij] · iσ(k) · ρ_ij(k) · D(k)^{i+j},   ρ_ij = ½(ρ^ψ_i + ρ^ω_j)
```

Effective explicit symbol of the closed step (rollout convention: coef = ΔT³, no (1−1/K²); fold constant c_f = ΔT²/12):

```
E(k) = iσ·(1 + c_f·L̂²)  +  c_f·( L̂·T₁ − 5·T₂ )
       [advection + S⁻¹L²N fold]   [learned (1/12)(L·Ṅ − 5N̈) transfer]
```

Implicit side exactly as the scheme folds it (L³ fully implicit):

```
r = 1 / (1 − ΔT/2·L̂ + ΔT³/12·L̂³),   a = r(1 + ΔT/2·L̂)
b = 1.5·ΔT·r·E,   c₂ = −0.5·ΔT·r·E
G_eff(k) = max | ((a+b) ± √((a+b)² + 4c₂)) / 2 |     — AB2 companion spectral radius
```

The penalty:

```
L_stab = λ_vn · mean_{k valid}  relu( |G_eff(k)| − (1−ε) )²,   ε = 0.02, λ_vn = 0.1
valid: |ΔT·σ(k)| ≤ 0.5   — outside this the frozen-linear model provably diverges
                            from the true dynamics, so penalizing there is noise
```

**How it is wired in**: computed once per training window (before the unroll), from the window's initial two marks; its small graph (conditioner MLP + tap deltas + base stencils + mix — the base stencils are deliberately NOT detached, they must feel the penalty too) is backwarded independently into the same optimizer step. The per-window max |G| is logged as a histogram column. Calibration anchor: with exact stencils and zero modulation the certificate reproduces the analytic arm's stability, so any excess |G| > 1 is attributable to the learned taps. Honest caveats (as stated in the code): linearized (no r₂/r₃), isotropic-shell evaluation — which is why the dissipative projection (P0) stays the unconditional inference backstop.

### 1.4 What to expect from the arms

- Land ~2026-07-16 07:00–08:00 (~3630 s/epoch × 20). FINALIZE monitors email verdicts with no agent present.
- The ep0 `train = inf` you saw is a **startup artifact, not a blow-up**: the warm model is unstable in free rollout *before* fine-tuning (n_blown 1163 at ep0 — exactly the disease the FT treats). Val and anchor columns were finite and sane (arm A ep0 val 1.89e-3, anc_N̈ 0.116).
- Success reads as: rollout val falling, n_blown → 0, the |G| histogram settling under ~1, **and** `accept_ft_gate.py` PASS (every (member,dT) N̈ within 10% of w31-ep32's per-root eval). Then pick between arms on rollout val + the OOD (combo, b25) read; expect A better on rollout, B better on the gate.

---

## Part 2 — SGS: σ master plan, stages 2 and 3

One-line recap of the diagnosis behind all three stages: the residuals are **heavy-tailed**, so no Gaussian-NLL fit can give nominal coverage — the NLL optimum over-widens the quiet bulk to pay for the tails; meanwhile the structural head *saturates* on the failure axis (it responds 2.7–4.2× in the top gradient decile while |error| rises 8.8×).

Stage 1 (landed, you know it): the 3-param recalibration

```
var′(x) = τ_pv · var_GP(x) + clamp( s_a·softplus(a) + s_b·softplus(b)·g²/g2_scale, 1e-3, 10 )
```

NLL-fit of (s_a, s_b, τ_pv) on the fit half of the val window, then **stratified split-conformal** on top: per gradient-decile stratum s and level p,

```
q_s(p) = Quantile_p( |r_i| / σ_recal,i ),   intervals = q_s(p) · σ_recal
```

→ held-out coverage within 0.6% of nominal at 68/95/99.7 on both geometries. Deployable q-tables in `runs_piff/*/conformal_calibration.yaml`.

### Stage 2 — the CRPS head (built and trained; refit-on-ylp75 is your pending ruling #2)

**Plain English.** The structural σ-head is a 2-degree-of-freedom affine in g² — too rigid: one floor and one slope cannot serve both the quiet freestream and the near-body tail. Stage 2 replaces it (as a *sidecar*, nothing existing modified) with a small neural head that predicts σ per pixel from three features, trained on the **frozen** mean model's residuals — and trained with **CRPS instead of NLL**, because CRPS doesn't panic about tails.

**Steps** (`train_crps_head.py`):
1. Load the finished checkpoint (mean + GP + FiLM-CNN + structural head), freeze it entirely (`eval()`, `requires_grad_(False)`); `best.pt` is never touched.
2. One pass under `no_grad`: collect per-pixel residuals r = y_std − μ_GP and features, cache in memory.
3. Train the head: MLP 3 → 32 → 32 → 1, softplus output, floor 10⁻³, standardized σ units. Features: `g²/g2_scale` (the same standardization the structural head used; `log1p` inside the head to condition the heavy tail), `sdf*` (clipped signed-distance plane), `ζ` (frame conditioning scalar).
4. Honesty protocol: val window split at t_mid; the fit half is the per-epoch monitor, the honesty half is touched only by the final eval report. Monotonicity of σ in g is *not* hard-constrained (your spec) — it is checked post-hoc.
5. Stage-1 conformal is then **re-fit on top of the new σ** (it composes; it needs only a σ ranking).

**The loss** — closed-form CRPS of a Gaussian at the observed residual:

```
CRPS( N(0,σ²); r ) = σ · [ z·(2Φ(z) − 1) + 2φ(z) − 1/√π ],   z = r/σ
```

mean over train-split valid pixels (same masking as training). Φ, φ = standard normal CDF/PDF.

**Why CRPS beats NLL here, in one breath**: NLL grows *quadratically* in the tail (z²/2 term), so a few huge residuals bully σ upward everywhere; CRPS grows *linearly* in |r|, so the bulk stays sharp while the tail still pushes. It is a proper scoring rule, so minimizing it still targets the right predictive distribution.

**What to expect**: sharper (smaller) σ in the quiet flow, a real σ response in the top gradient decile where the structural head saturated, 1-σ coverage closer to 0.68 *before* conformal, and tighter conformal q's after (the q-table's job shrinks when the base σ ranks better). The numbers to beat: the existing heads — fit on the **pre-ylp75** gjs checkpoints — give cov1 = 0.713 (FPC) / 0.749 (CAPE) vs nominal 0.68. That mismatch of checkpoint generations is exactly why ruling #2 (refit the heads on the ylp75 models — a cheap GPU job) is on your queue.

### Stage 3 — retire the GP posterior-variance pathway (proposal only; nothing built or run)

**Plain English.** The total predictive variance is currently `τ_pv·var_GP + aleatoric`. Stage 1's own NLL fit chose **τ_pv ≈ 0.017 (FPC) / 0.006–0.009 (CAPE)** — the data assigns the GP posterior-variance pathway essentially zero weight. Stage 3 draws the conclusion: drop the pathway; intervals come from the (stage-2) aleatoric head plus conformal, full stop.

```
before:  σ_tot²(x) = τ_pv · var_GP(x) + σ_ale²(x)      (τ_pv ≈ 0.01 → first term ~noise)
after:   σ_tot²(x) = σ_ale²(x)                          (+ conformal q_s on top)
```

**What to expect**: NLL and coverage essentially unchanged (you are removing a term the fit already zeroed); inference gets simpler and cheaper (no GP posterior-variance evaluation); one fewer moving part to calibrate. The one honest risk: the GP variance is the only term that grows *from the model's own uncertainty* rather than from input features, so it could in principle matter out-of-distribution where the features under-signal — worth one held-out-geometry check before deletion. It awaits your GO; that is ruling #3 on your queue.

---

## Part 3 — What is running right now (qstat 12:03 EDT, reconciled against HANDOFF — exact match)

**1833569 `w31_TRN`** — the width-31 conditioned stencil retrain (`train_deriv` family), epoch ~46 of 150.
*Input*: the 41-root filtered S=7 a-priori sweep pool. *Output*: `training_runs/deriv7_cond_local_w31/{best.pt, last.pt, log.csv}`. *Purpose*: none anymore, strictly speaking — its best is **frozen at ep32** (pooled val 0.0605, beats cond_v2 in 40/41 cells) and the arms already warm-started from it. It runs only in case a later epoch beats ep32; nothing depends on its exit; you may qdel it at will.

**1833570 `w31_L` / 1833571 `w31_F`** — its LIVE and FINALIZE monitors (v2). *Input*: the trainer's `log.csv`. *Output*: ~2-hourly [QG][MONITOR] emails; FINALIZE (held on the trainer) emails the postmortem verdict. Agent-free via the relay cron.

**1834629 `w31p1a_TRN` (arm A)** and **1834632 `w31p1b_TRN` (arm B)** — the two rollout fine-tunes described in Part 1 (anchor λ 3e-2 vs 3e-1; vn 0.1; free-analytic 1e-2; trunc:4; 20 epochs; ~3630 s/ep → land 07-16 07:00–08:00).
*Input*: w31 `best.pt` (ep32) + deep 28-mark windows of 7 rollout roots (combo, b25 held out) + the 41-root anchor pool. *Output*: `training_runs/rollout_ft_w31_p1a/` and `..._p1b/` (`best.pt`, `log.csv` with anchor columns, |G| histogram data). *Purpose*: make the w31 closure stable in free rollout without paying more than 10% N̈ anywhere.

**1834631 `w31p1a_F` / 1834634 `w31p1b_F`** — the arms' FINALIZE monitors (held on their trainers). They email the landing verdicts with no agent present. After landing, the next session runs `eval_deriv_by_root.py` then `accept_ft_gate.py --tol-rel 0.10` from the wiener worktree.

**1834691 `w31p1a_L2` / 1834692 `w31p1b_L2`** — replacement LIVE monitors, queued with `-a 12:50` (start 12:50). The originals emailed `ep0 EXPLODE` and exited *by design* — the trigger was the ep0 `train = inf` startup artifact explained above. If ep1 is also non-finite these will EXPLODE-and-exit again; the arms would then run live-blind, but the FINALIZE verdicts still fire. Treat a second EXPLODE pair as "check the arms by hand."

**Agent-free crontab on mseas** (not in qstat, always on): `send_pending.sh` every 10 min (the mail relay — `relay.log` is the only proof of delivery); `daily_report.sh` 08:00; `status_report.sh` 12/16/20h; `autofire_check.sh` every 10 min, currently **inert** (both w31 and ylp75 markers set).

My reflex-ladder test jobs (all.q, this morning) have finished and left the queue; today's charter landing touches **none** of the jobs above — the new ladder guards only future submissions made through the retrofitted templates.
