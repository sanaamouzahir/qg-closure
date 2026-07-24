# IDEAS — innovation ledger (all branches)
*Mandated by LAB_DOCTRINE.md §7.5. Every idea gets an entry BEFORE work
starts; no PURSUE verdict without the "Prior art:" field answered (has
someone pursued this? how does our method differ — one sentence; closest
papers filed in docs/papers/). Verdicts: PURSUE / PARK / KILL, each with the
mechanism-level reason. Kills are as valuable as pursuits — they save
GPU-hours and prevent re-proposal. Append-only; update verdicts in place
with a dated note.*

Format:
```
## IDEA-nnn (date) — one-line title
What:      the idea
Prior art: who did something like it; how ours differs (one sentence)
Tri-obj:   expected effect on cost / accuracy / stability (all three)
Verdict:   PURSUE | PARK | KILL — mechanism-level reason
Status:    where it stands (runs, commits, or the killing evidence)
```

---

## IDEA-002 (2026-07) — Dissipative spectral projection as rollout stabilizer
What:      project rollout states onto a dissipative spectral envelope to
           stop long-rollout blow-ups.
Prior art: standard spectral-viscosity / clipping family (eddy-viscosity
           regularization of unstable closures); ours would have been a
           post-hoc energy-shell projection.
Tri-obj:   cost small; accuracy neutral-to-negative; stability claimed.
Verdict:   KILL — field-level forensics showed the rollout error is
           within-shell PHASE, invisible to an energy projection; the
           mechanism cannot address the failure mode (exposure bias, loop
           gain ~2x/step from ~step 12 — a training-scheme problem, not a
           spectral-amplitude problem).
Status:    closed; redirected effort to training-scheme design
           (INCIDENT_LOG E-04/E-07).

## IDEA-003 (2026-07-23) — Energy-budget loss term (-<psi Pi>) for the SGS closure
What:      add an interscale ENERGY-transfer consistency term next to the
           enstrophy EnsCon term.
Prior art: Jakhar et al. 2024/2026 (energy transfer is the stability-
           critical budget in 2D FHIT); ours differs in setting (obstacle
           wakes, per-crop) — which is exactly the problem.
Tri-obj:   cost negligible; accuracy neutral; stability plausible but
           setting-unproven.
Verdict:   PARK — FHIT priority ranking does not transfer to wakes
           (shedding/body/sponge dominate the large-scale energy budget);
           per-crop <psi Pi> is gauge-sensitive and crop-capped. Decision
           delegated to data: budget-error diagnostic <omega eps> vs
           <psi' eps> per (member x region) on existing v1/v2 residuals.
Status:    diagnostic pending; rule recorded in BUDGET_LOSS_DOCTRINE.md §5.

## IDEA-004 (2026-07-23) — Sign-split EnsCon (diffusion/backscatter separately)
What:      replace the net-transfer scalar with the pair (Sum(omega Pi)_+,
           Sum(omega Pi)_-) matched independently — two rank-one
           constraints, kills the cancellation channel.
Prior art: PRL 2026 (Jakhar/Guan/Hassanzadeh) evaluate P>0 and P<0
           separately in their discovery criterion; ours embeds the split
           in a CNN training loss per crop rather than an equation-discovery
           elbow.
Tri-obj:   cost negligible; accuracy neutral; stability strictly >= net-only.
Verdict:   PURSUE (staged) — after the V3 main run, not editing the
           fire-ready chain.
Status:    staged upgrade, BUDGET_LOSS_DOCTRINE.md §7.
