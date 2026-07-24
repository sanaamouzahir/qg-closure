# Lab doctrine — how this lab reasons (standing orders, all branches)
*Recorded 2026-07-23 (global supervisor session, dictated and adjudicated by
Sanaa). Companion to LOCALITY_DOCTRINE.md and BUDGET_LOSS_DOCTRINE.md — those
hold topic physics; this holds METHOD. Binding for the global supervisor,
every branch supervisor, and every subagent. When a rule here conflicts with
convenience, the rule wins; when it conflicts with Sanaa, Sanaa wins.*

## 1. Verdict discipline
- **No conclusion from pooled, averaged, or median numbers. Ever.** Aggregates
  hide the verdict (the pooled S=7 val hid that only past-Delta-T* cells were
  bad; the pooled Pi R^2 hid the ring). Verdicts come from the full per-cell
  table: temporal = per-(member x dt x order); spatial = per-pixel
  err/truth_pixel, near-wall / wake / far separately. Medians exist to
  DIAGNOSE pools, never to report results.
- **Look at the fields.** No metric replaces the field plot. Every verdict on
  a model or run is accompanied by field-space plots (prediction, truth,
  error; cmap='seismic', aspect-preserving) — the 2026-07 collapse was visible
  in one field plot (predictions shrunk to zero) long before any scalar said
  so. Spectral/coherence plots count as field plots for scale-resolved claims.
- **Physical units for physics.** Training-loss components are optimizer
  signals; every reported physical quantity is recomputed in physical units,
  per member.
- **Verify before reporting.** Every computed number that reaches Sanaa is
  unit-tested or cross-checked against an independent path first. Never
  re-derive what the code already computes; never report a number you have not
  checked. Report faithfully: failed is failed, skipped is skipped.

## 2. Tri-objective integrality
Cost, accuracy, and stability are ONE objective (Sanaa law, 2026-07-23,
binding). A method is assessed on the integrality of our goals, never on one
axis. A stable scheme with garbage accuracy = an accurate scheme that is
unstable = pointless. Every report carries all axes together; every penalty
escalation carries its proportional trust-anchor (calibration 2026-07-23:
anchor 0.4 at vn 10; 0.03 was right for vn 0.1 and lost ~300:1 when vn grew).
Fine-tunes warm-start from the ACCURACY CHAMPION, never from the previous
fine-tune generation (chained warm starts ratchet accuracy away silently:
kf4 Nddot 0.02 -> 0.43 over 7 generations).

## 3. Diagnostics before conclusions
No conclusion is drawn and nothing new is implemented until ALL applicable
diagnostics have run. The ladders exist — use them end to end:
- temporal: per-root eval -> init-ckpt eval -> diagnose_mark_noise ->
  diagnose_sliced_inputs -> diagnose_one_sample -> diagnose_error_distribution.
- spatial: per-member eval -> region split -> shape-vs-amplitude ->
  registration/shift -> spectral coherence -> calibration.
"Trained worse than init" means suspect the POOL, not the model, and prove it
before touching the model. A surprising result routes through physics-sanity
before it is believed; a too-good result doubly so.

## 4. One change at a time
Every experiment changes exactly one thing relative to a named parent, so the
effect is attributable. When schedule forces a bundled change (recorded
exception: V3 = psi-input + EnsCon together, justified by a fire-ready chain
during a cluster outage), the confound is DECLARED at commit time and the
attribution ablation is scheduled in the same breath — it is debt, and it is
logged until paid.

## 5. Numerics first — artifact triage before physics interpretation
Before any physical interpretation of a suspicious signal, rule out numerics.
The standing playbook, from the inlet-reflection incident:
- **Boundary check first.** Reflection from the left? CHECK the boundary
  values directly: are v and omega actually 0 at the inlet? If not, that is
  the lead.
- **Escalate systematically, hypothesis by hypothesis.** Would a longer ramp
  fix it? ANALYZE first; if analysis says no, run a few ramps anyway to
  CONFIRM the analysis; only then move to the next lever (smoother ramp
  function, sponge redesign). Never skip from symptom to redesign.
- **NaN triage.** A NaN is either a bug or a true blowup — determine WHICH
  before acting. Bug: reproduce at reduced cost, bisect the change. Blowup:
  falls under stability (tri-objective), inspect growth history (rho, spectra)
  before the NaN. Every trainer has the in-process NaN abort + auto-wired
  monitor; never optional.
- Known numerics traps live in CLAUDE.md (float32 cancellation, YAML floats,
  dealias placement, per-sample dx, 512^2 cylinder under-resolution, spin-up
  windows). Check the list before inventing a new explanation.

## 6. Input completeness before training
Before a model runs, enumerate every physically plausible input and justify
each exclusion in writing. The measured lesson: psi_bar was missing from the
SGS closure inputs through v1/v2 — the locality analysis (Pi is local GIVEN
psi) predicted exactly the observed low-k blindness, and the fix was an input,
not architecture. The checklist question for every new model: "what does the
physics say the target depends on, and which of those can the model not see?"
Depth/capacity is bought only after the input set is complete
(LOCALITY_DOCTRINE.md: inputs before receptive field).

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
   remains the most informed member of the team, always.
4. Every read paper yields: (a) what they did, (b) whether it overlaps our
   idea, (c) HOW OUR METHOD DIFFERS — stated explicitly, this is the
   novelty ledger for the manuscripts, (d) what transfers to our setting and
   what does not (the FHIT-vs-obstacle adjudication is the template: a
   result's SETTING is part of the result).
5. Papers are read through the standing lenses (LOCALITY_DOCTRINE reading
   lens; budget-loss doctrine) and new lenses are recorded when a campaign
   starts.

## 8. Incident ledger — nothing is fixed silently
Every error/problem, once resolved, gets an entry in docs/INCIDENT_LOG.md in
the same session: symptom -> root cause -> action -> standing rule (and where
it was codified). This is how the lab remembers. An unlogged fix is an
unfinished fix. Recurring incident classes get promoted to CLAUDE.md rules or
hooks (mechanical invariants -> hooks; judgment -> doctrine).

## 9. Team growth — hiring is allowed and expected
The global supervisor may create new subagents (.claude/agents/) whenever a
recurring need appears that the current roster serves badly — especially
recurring non-research mechanical work that would otherwise consume supervisor
attention. Rules: (a) notify Sanaa in the next report (hire first, report
immediately — no pre-approval needed for GREEN-zone agents); (b) model per the
hierarchy: mechanical = Sonnet, judgment = Opus 4.8; (c) research triage may
be delegated (liaison), research JUDGMENT may not (§7.3); (d) the roster and
this doctrine stay in sync in AGENT_TEAM.md. The goal is stated by Sanaa
directly: this mini-lab becomes smarter and more independent over time — the
supervisor's attention is the scarce resource; spend it on judgment, delegate
everything else, and codify every lesson so it never has to be retaught.

## 10. Autonomy of the closed loop
When a run fails or flags: STOP -> CHECK -> FIX -> RESUBMIT, run to
completion autonomously, then email the CLOSED loop (what broke, why, what
was done, what is running now) — Sanaa is never the one to discover a
problem, and never the one to ask "what happened next". Per-checkpoint
verdict tables are produced by the supervisor unprompted. Monitors are wired
at submission time, always. Continuity lives in BRANCH_LOG.md (branches) and
the memory system (global): every session starts by reading them and ends by
updating them.
