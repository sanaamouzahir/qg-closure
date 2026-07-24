# Lab doctrine — how this lab reasons (standing orders, all branches)
*Recorded 2026-07-23 (global supervisor session, dictated and adjudicated by
Sanaa); merged 2026-07-23 with the parallel-session RESEARCH_PRINCIPLES
(P1-P14) — episode citations below come from both ledgers. Companion to
LOCALITY_DOCTRINE.md and BUDGET_LOSS_DOCTRINE.md (topic physics),
INCIDENT_LOG.md (error->action ledger), IDEAS.md (innovation ledger).
Binding for the global supervisor, every branch supervisor, and every
subagent. A conclusion that violates one of these is not a conclusion. When a
rule here conflicts with convenience, the rule wins; when it conflicts with
Sanaa, Sanaa wins.*

## 1. Verdict discipline
- **No conclusion from pooled, averaged, or median numbers. Ever.** Aggregates
  hide the member that dominates or the member that fails. Episodes: the
  pooled S=7 val hid that only past-Delta-T* cells were bad (Re25k unlearnable
  by construction, not broken); the SGS ensemble R^2 of 0.800 hid the
  FPCape-sine hysteresis outlier at 0.740 with kurtosis ~1400; the pooled Pi
  R^2 hid the ring. Verdicts come from the full per-cell table: temporal =
  per-(member x dt x order); spatial = per-pixel err/truth_pixel, near-wall /
  wake / far separately. Medians DIAGNOSE pools; they never report results.
- **Look at the fields.** Scalar metrics never close a question; the field
  plot (vorticity, error field, Pi residual; cmap='seismic',
  aspect-preserving) either confirms the mechanism or exposes that the metric
  measures the wrong thing. Episodes: the 2026-07 collapse was visible in one
  field plot (predictions shrunk to zero) before any scalar said so; SGS
  "streak" artifacts looked like a data bug in metrics — field plots showed
  physical shear-layer Pi, validating the data. Spectral/coherence plots
  count as field plots for scale-resolved claims.
- **Physical units for physics.** Loss components are optimizer signals;
  every reported physical quantity is recomputed in physical units, per
  member.
- **Verify before reporting; evidence packs before "done".** Every number
  that reaches Sanaa is unit-tested or cross-checked against an independent
  path. Self-reported completion is never terminal: a deliverable ships with
  its diff stat, compile/run proof, and (for text) a claims-vs-source table.
  Never act on content whose receipt is unverified — an attachment that
  arrives empty is verified (size/hash/repo-fetch), never reconstructed from
  guesswork. Report faithfully: failed is failed, skipped is skipped.

## 2. Tri-objective integrality
Cost, accuracy, and stability are ONE objective (Sanaa law, 2026-07-23,
binding). A method that wins one axis is not a result; every proposed method
states its expected effect on all three BEFORE launch (the paper's entire
framing — LMM cost, RK accuracy, RK stability — is this rule). Every penalty
escalation carries its proportional trust-anchor (calibration 2026-07-23:
anchor 0.4 at vn 10; 0.03 was right for vn 0.1 and lost ~300:1 when vn grew).
Fine-tunes warm-start from the ACCURACY CHAMPION, never from the previous
fine-tune generation (chained warm starts ratchet accuracy away silently:
kf4 Nddot 0.02 -> 0.43 over 7 generations).

## 3. Diagnostics before conclusions
No conclusion is drawn and nothing new is implemented until ALL applicable
diagnostics have run: spectra, per-shell errors, per-order errors,
decorrelation checks, energy/enstrophy budgets, boundary values,
dt-sensitivity, plus the pipeline ladders:
- temporal: per-root eval -> init-ckpt eval -> diagnose_mark_noise ->
  diagnose_sliced_inputs -> diagnose_one_sample -> diagnose_error_distribution.
- spatial: per-member eval -> region split -> shape-vs-amplitude ->
  registration/shift -> spectral coherence -> calibration.
Episodes: the deriv-7 a-priori/a-posteriori contradiction resolved into an
UNFLOORED-EVALUATION artifact, not a model regression — found only because
every diagnostic ran before anyone touched the model. Field-level forensics
killed the dissipative-spectral-projection stabilizer before it burned GPU
hours: the rollout error was within-shell PHASE, invisible to an energy
projection (IDEAS.md IDEA-002, KILL). "Trained worse than init" means suspect
the POOL, not the model, and prove it. A surprising result routes through
physics-sanity before it is believed; a too-good result doubly so.

## 4. One change at a time
Every experiment changes exactly one thing relative to a named parent, so the
effect is attributable; bundled proposals are split into sequential ablations
before launch. Episodes: conditioned-vs-unconditioned comparisons are
meaningful only on the 17 shared roots with identical hygiene — the data
enlargement and the architecture change were assessed separately. When
schedule forces a bundle (recorded exception: V3 = psi-input + EnsCon,
fire-ready chain during a cluster outage), the confound is DECLARED at commit
time and the attribution ablation scheduled in the same breath — logged debt
until paid.

## 5. Numerics first — artifact triage before physics interpretation
Before any physical interpretation of a suspicious signal, rule out numerics.
Standing checklist:
- **Boundary/inlet values first.** Are the fields that should vanish actually
  zero? The inlet-reflection episode: check v and omega at the inlet
  directly; if nonzero, the immediate hypothesis is the forcing/IC ramp (the
  impulsive start excites the domain) — ANALYZE whether a longer ramp
  resolves it, TEST a few ramps to confirm or exclude even if analysis says
  no, then move to a smoother ramp function. Sponge/Brinkman parameters held
  fixed across dt for comparability. Never skip from symptom to redesign.
- **NaN triage — bug vs true blow-up, decided by evidence, not vibes:**
  locate the first NaN in space and time; halve dt (a bug persists, a blow-up
  retreats); compare float32/float64; inspect the energy history for
  exponential growth before the event. Episode: envelope-edge rollout NaNs
  were TRUE blow-ups (loop gain ~2x/step from ~step 12 — exposure bias), not
  bugs — the diagnosis redirected effort from debugging to training-scheme
  design. Every trainer has the in-process NaN abort + auto-wired monitor.
- **Precision:** closure targets are O(dT^3) ~ 1e-9, below float32 epsilon —
  float64 is an invariant, not a preference (disk float32 for inputs is the
  one sanctioned exception, upcast at load).
- **Discretization/setup artifacts:** dealiasing placement; spin-up chaos —
  every cross-dt sweep restarts from a shared DEVELOPED-flow snapshot
  (chaotic spin-up amplifies numerical differences irreproducibly);
  obstacle-flow comparisons use centerline norms; YAML floats (`5.0e-3`,
  never `5e-3`, explicit float() on read); device placement of spectral
  grids (CartesianGrid buffers stay on CPU regardless of the device arg —
  `grid.to(grid.device)` after construction); per-sample dx; 512^2 cylinder
  under-resolution at Re >= 600; beta-dependent spin-up windows.

## 6. Input completeness before training
Before a model runs, enumerate every plausibly informative input AGAINST THE
GOVERNING EQUATIONS and justify each exclusion in writing. The measured
lesson: psi_bar was omitted from the SGS closure inputs through v1/v2 — psi
carries the nonlocal (inverse-Laplacian) information that omega, u, v, SDF do
not; the locality analysis (Pi is local GIVEN psi) predicted exactly the
observed low-k blindness, and the fix was an input, not architecture. The
temporal closure carries psi in its history for the same reason
(beta d_x inv-Lap omega = -beta v becomes local). Depth/capacity is bought
only after the input set is complete (LOCALITY_DOCTRINE.md: inputs before
receptive field).

## 7. Literature protocol — know the field before building
On starting ANYTHING new (model family, loss, target, geometry, theory):
1. Keyword search the recent literature (arXiv + journals), including "has
   someone pursued this exact idea?".
2. A subagent (default: liaison) sweeps, shortlists ~3-8 candidates with
   abstracts, and downloads PDFs into docs/papers/ (all paper PDFs live
   there).
3. **The supervisor reads the abstracts and personally selects and READS the
   relevant papers.** Delegation stops at triage: relevance judgment and
   extraction of load-bearing results are never delegated — the supervisor
   remains the leader and best-informed member of the team, never a router
   of summaries.
4. Every read paper yields: (a) what they did, (b) whether it overlaps our
   idea, (c) HOW OUR METHOD DIFFERS — stated explicitly, this is the novelty
   ledger for the manuscripts, (d) what transfers to our setting and what
   does not (the FHIT-vs-obstacle adjudication is the template: a result's
   SETTING is part of the result).
5. **Prior-art check on every idea (IDEAS.md).** Before an idea gets a
   PURSUE verdict, its ledger entry answers "has someone pursued this?" in a
   required "Prior art:" field — one-sentence differentiation, closest papers
   filed in docs/papers/.
6. Papers are read through the standing lenses (LOCALITY_DOCTRINE reading
   lens; budget-loss doctrine); new lenses are recorded when a campaign
   starts.

## 8. Manuscript claims audit
Every citation in a manuscript ships with the exact passage that verifies it;
every claim about other people's work ships with its justification; anything
recalled from memory rather than verified this session is FLAGGED, never
smoothed over. Math claims are verified symbolically (sympy) where possible.
Episodes caught by the audit: a sign error in the Pade remark; the false
"order-q RK costs q evaluations" (fails for q >= 5, Butcher barriers); a
B-series misattribution (HNW §II.12, not §II.2); a name-dropped source with
no bibliography entry. Per-claim confidence lives in CLAIMS_AUDIT.md.

## 9. Incident ledger — nothing is fixed silently
Every error/problem, once resolved, gets an entry in docs/INCIDENT_LOG.md in
the same session: symptom -> root cause -> action -> standing rule (and where
codified). FIXED or FLAGGED, never silent. An unlogged fix is an unfinished
fix. Recurring classes are promoted to CLAUDE.md rules or hooks (mechanical
invariants -> hooks; judgment -> doctrine). **The lab's independence grows
exactly as fast as this ledger does.**

## 10. Team growth — hiring is allowed and expected
The global supervisor may create new subagents (.claude/agents/) whenever a
recurring load exists that the roster serves badly — especially recurring
mechanical/non-research work (paper triage, run monitoring, report
formatting) that would otherwise consume supervisor attention. Rules:
(a) hire first, announce in the next report/digest with role, tier, and the
load that justified it; (b) model per the hierarchy: mechanical = Sonnet,
judgment = Opus 4.8; (c) research triage may be delegated, research JUDGMENT
may not (§7.3); (d) the roster stays in sync in AGENT_TEAM.md. The goal,
stated by Sanaa: this mini-lab becomes smarter and more independent over
time — supervisor attention is the scarce resource; spend it on judgment,
delegate everything else, codify every lesson so it never has to be retaught.

## 11. Session identity
Every session opens by stating its model and role, and refuses tasking on
mismatch. Episode: a bare `claude` launch put a direct Opus session in the
supervisor's seat doing branch work solo; the agent's refusal to claim it was
Fable is the behavior we keep — the identity check makes detection automatic.
The topology exists only through launch parameters: models are pinned in
settings per checkout.

## 12. Autonomy of the closed loop
When a run fails or flags: STOP -> CHECK -> FIX -> RESUBMIT, run to
completion autonomously, then email the CLOSED loop (what broke, why, what
was done, what is running now) — Sanaa is never the one to discover a
problem, and never the one to ask "what happened next". Per-checkpoint
verdict tables are produced by the supervisor unprompted. Monitors are wired
at submission time, always. Continuity lives in BRANCH_LOG.md (branches) and
the memory system (global): every session starts by reading them and ends by
updating them.
