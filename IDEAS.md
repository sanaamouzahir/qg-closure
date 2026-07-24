# IDEAS — Innovation Ledger

Agent-proposed and human-proposed research ideas, each assessed against explicit
criteria before any compute is spent, and closed with a verdict. Maintained by the
global supervisor; PURSUE verdicts spawn an `exp/*` branch (git worktree) with a
branch supervisor. Assessments and verdicts survive here even when the idea dies —
a killed idea with a documented reason is a result.

## Assessment protocol (charter I29-adjacent)

Every entry must answer, before launch:
- **Claim**: one falsifiable sentence.
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
