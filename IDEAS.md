# IDEAS — Innovation Ledger

Agent-proposed and human-proposed research ideas, each assessed against explicit
criteria before any compute is spent, and closed with a verdict. Maintained by the
global supervisor; PURSUE verdicts spawn an `exp/*` branch (git worktree) with a
branch supervisor. Assessments and verdicts survive here even when the idea dies —
a killed idea with a documented reason is a result.

## Assessment protocol (charter I29-adjacent)

Every entry must answer, before launch:
- **Claim**: one falsifiable sentence.
- **Prior art**: has someone pursued this? One-sentence differentiation; closest
  papers filed in `docs/papers/`. No PURSUE without this field (LAB_DOCTRINE §7.5).
- **Tri-objective**: expected effect on cost, accuracy, AND stability — all
  three, before launch (LAB_DOCTRINE §2).
- **Expected evidence**: the named plot/metric/table that would confirm or kill it.
- **Cost**: GPU-hours + wall-clock + my-attention estimate.
- **Kill condition**: the observable that ends it (pre-registered, not post-hoc).
- **Verdict**: PURSUE (→ branch) / HOLD (blocked on dependency) / KILL (+ reason).

Overnight (NIGHT mode, charter v1.5): the supervisor may run assessments and
green-tier experiments on PURSUE ideas autonomously; anything YELLOW/RED waits for
the 07:00 handoff. Every overnight pursuit appears in the morning digest under
"attempted / worked / killed and why."

---

## Ledger

### IDEA-001 — Wiener conditioning of stencil taps on σ̂(κ)
- **Origin**: agent (from the exact error analysis: pooled floor dominated by
  tier-to-tier variance of the optimal filter).
- **Claim**: conditioning the learned taps on the measurable per-shell
  decorrelation rate σ̂(κ) removes the pooled-variance term of the training floor.
- **Expected evidence**: ΔT-collapse of the scaled residuals across the sweep;
  conditioned floor below the unconditioned plateau on shared roots.
- **Cost**: ~20 GPU-h offline + analysis.
- **Kill condition**: no collapse of r_j/ΔT^{S−k} across the sweep.
- **Verdict**: **PURSUE** → `exp/wiener-conditioning`. Collapse confirmed
  (1.6× vs 243× raw); now the paper's conditioning section.

### IDEA-002 — Dissipative spectral projection as a rollout stabilizer
- **Origin**: agent (after diagnosing rollout blow-ups).
- **Claim**: projecting the correction onto a dissipative subspace per shell
  stops the autoregressive error loop.
- **Expected evidence**: extended stable horizon at the envelope edge.
- **Kill condition**: no horizon gain on the standard 10-seed protocol.
- **Verdict**: **KILL**. Mechanism was within-shell phase error; an
  energy-based projection cannot see it. Lesson encoded (see LESSONS below).

### IDEA-003 — R4 assembly via the implicit fold
- **Origin**: joint (agent proposed extending the assembled order; the exact
  fold identity was derived together on iPad).
- **Claim**: evaluating the folded L³ correction at the implicit level supplies
  the analytic R4 content exactly, so R4 costs only the derivative bracket.
- **Expected evidence**: symbolic identity of the fold expansion; a posteriori
  parity or better vs R3-only closure.
- **Kill condition**: fold residual ≠ (0,0) at O(h⁴).
- **Verdict**: **PURSUE**. Identity verified symbolically (residuals (0,0));
  now §5 of the paper. A posteriori evaluation queued for the new campaign.

### IDEA-004 — Weight-tied recursion cell (any-order N^(p), no inner wall)
- **Origin**: agent.
- **Claim**: one cell unrolled p times with the exact ∂t recursion replaces the
  depth-S stencil and removes the inner wall.
- **Cost**: one Δ⁻¹ per order — the exact-FFT vs learned-local decision is open.
- **Verdict**: **HOLD** (blocked on the Δ⁻¹ decision; listed as future work in
  the paper).

### IDEA-005 — Von Neumann certificate as a differentiable training penalty
- **Origin**: agent (from the frozen-coefficient stability analysis).
- **Claim**: penalizing |G_eff|>1−ε inside the linearization's validity shells
  converts the horizon-limited gain into a per-step contraction property.
- **Verdict**: **HOLD** — λ-sweep ran; the certificate section is under
  revision and the idea re-enters assessment with the new campaign.

### IDEA-006 — Energy-budget loss term (−⟨ψ̄Π⟩) for the SGS closure
- **Origin**: agent (from the Jakhar PRL/JAMES reading, 2026-07-23); demoted
  by Sanaa's setting objection the same day.
- **Claim**: matching the interscale ENERGY transfer per crop (alongside the
  enstrophy EnsCon term) removes a stability-critical budget error the
  enstrophy term cannot see.
- **Prior art**: Jakhar et al. 2024/2026 — energy transfer is the
  stability-critical budget in 2D FHIT; ours differs in setting (obstacle
  wakes, per-crop), which is exactly the problem: in wakes the large-scale
  energy budget is dominated by shedding/body/sponge terms, and per-crop
  ⟨ψ̄Π⟩ is gauge-sensitive and crop-capped.
- **Tri-objective**: cost negligible; accuracy neutral; stability plausible
  but setting-unproven.
- **Expected evidence**: budget-error diagnostic on existing v1/v2 residuals —
  signed ⟨ω̄ε⟩ vs ⟨ψ̄′ε⟩ (crop-demeaned) per (member × near-wall/wake/far).
- **Cost**: analysis-only, no GPU training.
- **Kill condition**: energy-budget error small next to the enstrophy one in
  the wake band.
- **Verdict**: **HOLD** — blocked on the diagnostic; no design change on
  imported FHIT priors (BUDGET_LOSS_DOCTRINE §5, INCIDENT_LOG 2026-07-23).

### IDEA-007 — Sign-split EnsCon (diffusion/backscatter matched separately)
- **Origin**: agent (from the PRL discovery-criterion structure).
- **Claim**: replacing the net-transfer scalar with the pair (Σ(ω̄Π)₊, Σ(ω̄Π)₋)
  matched independently kills the diffusion/backscatter cancellation channel
  at negligible cost.
- **Prior art**: Jakhar/Guan/Hassanzadeh PRL 2026 evaluate P>0 and P<0
  separately in their equation-discovery criterion; ours embeds the split in
  a CNN training loss per crop.
- **Tri-objective**: cost negligible; accuracy neutral; stability ≥ net-only
  by construction (strictly more constraint).
- **Expected evidence**: signed transfer ratios → 1 per member with R²/per-pixel
  medians unchanged.
- **Cost**: one loss-term edit + one training run.
- **Kill condition**: accuracy regression on the per-member table at matched β.
- **Verdict**: **PURSUE** (staged) — after the V3 main run lands; the
  fire-ready chain is not edited (BUDGET_LOSS_DOCTRINE §7).

---

## LESSONS (recurring lessons → playbook updates)

Every closed idea and every operational failure is mined for a lesson; lessons
that recur are promoted to numbered charter invariants or amendments, so the
playbook is executable, not aspirational. Examples now in force:

- **L-01** Blow-up forensics before stabilizer design: diagnose the mechanism
  (phase vs amplitude) before proposing any fix. (From IDEA-002.)
- **L-02** Float64 everywhere in closure math — the target is below float32
  epsilon. → charter invariant.
- **L-03** Convergence sweeps restart from a shared developed-flow snapshot;
  spinup chaos voids cross-ΔT comparisons. → pipeline rule.
- **L-04** Self-reported "done" is never terminal for deliverables; evidence
  packs (diff stat, compile proof, claims-vs-source table) required. → I28.
- **L-05** Written plan with expected evidence + abort criteria before any
  qsub/training launch. → I29.
- **L-06** Every error closes the session as FIXED or FLAGGED-to-handoff;
  silent unresolved errors are a charter violation. → I30.
- **L-07** Session identity check at open: state model + role; refuse tasking
  on mismatch. → I31. (From the direct-Opus-as-supervisor incident.)
- **L-08** Citations ship with the exact passage to verify; claims about
  others' work ship with their justification; unverified items are flagged,
  never smoothed over. → claims-audit protocol (CLAIMS_AUDIT.md).
