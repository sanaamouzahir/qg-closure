# Incident log — error/problem -> action (all branches)
*Mandated by LAB_DOCTRINE.md §8 (Sanaa order 2026-07-23). Every resolved
error/problem gets an entry IN THE SAME SESSION: symptom -> root cause ->
action -> standing rule (where codified). Append-only; newest at top.
An unlogged fix is an unfinished fix.*

Format:
```
## YYYY-MM-DD [branch] one-line title
Symptom:  what was observed
Cause:    root cause, once proven
Action:   what was done (with the verification)
Rule:     the standing rule that came out of it + where it is codified
```

---

## 2026-07-23 [sgs-closure] Energy-budget term nearly added on wrong-setting evidence
Symptom:  supervisor recommended adding an energy-transfer loss term on the
          strength of FHIT literature (Jakhar Table 6).
Cause:    priority ranking "energy transfer is THE stability-critical budget
          in 2D" is doubly-periodic forced-turbulence evidence; in obstacle
          wakes the large-scale energy budget is dominated by shedding/body/
          sponge terms. Also found: per-crop <psi Pi> is gauge-sensitive and
          crop-capped. Caught by Sanaa.
Action:   demoted to decide-on-own-data: budget-error diagnostic
          (<omega eps>, <psi' eps> per member x region) rules on it.
Rule:     a result's SETTING is part of the result; no design change on
          imported priors when our own residuals can answer.
          (BUDGET_LOSS_DOCTRINE.md §4-5, LAB_DOCTRINE.md §7.4)

## 2026-07-23 [sgs-closure] V3 bundles two changes (declared confound)
Symptom:  V3CNN turns on psi_bar input AND EnsCon loss in one run.
Cause:    fire-ready chain authored during cluster outage; schedule over
          purity, accepted knowingly.
Action:   confound declared at commit time; attribution ablation (psi on,
          --enscon-beta 0) scheduled after the main run. Debt logged here
          until paid.
Rule:     one change at a time; bundled changes must be declared + ablation
          scheduled in the same breath. (LAB_DOCTRINE.md §4)

## 2026-07-22 [sgs-closure] GP eval crash-in-waiting: hardcoded feature dim
Symptom:  v1-era GP trainer hardcoded 20 input dims; any v2+ eval would
          crash on state_dict shapes.
Cause:    dimension baked in instead of read from the artifact.
Action:   dims made dynamic — eval reads dim from saved inducing points
          (commit 51e4151).
Rule:     shapes/dims are read from artifacts, never hardcoded twice.

## 2026-07 [sgs-closure] cp1252 byte killed two eval jobs
Symptom:  SyntaxError on the cluster; two eval jobs dead on arrival.
Cause:    a Windows-side patch introduced a cp1252-encoded byte into a
          Python file; mseas Python choked.
Action:   file fixed; remote AST parse gate made mandatory before every
          qsub of freshly-patched code.
Rule:     no submission of patched code without a remote `python -m py_compile`
          / AST gate. (branch brief)

## 2026-07 [sgs-closure] Mis-pathed submission
Symptom:  job submitted with a wrong path.
Cause:    path not verified against the remote tree before qsub.
Action:   qdel'd pre-start; paths now verified in the submit script.
Rule:     submit scripts resolve and test paths before qsub (sge-checker).

## 2026-07 [sgs-closure] Same-minute commit sweep resolved by force-push
Symptom:  two commits collided in the same minute; history diverged.
Cause:    concurrent commit from two stations.
Action:   resolved with a one-off force-push on the experiment branch.
Rule:     force-push remains RED (never on main, hook-blocked); on experiment
          branches only with explicit acknowledgment, and logged.

## 2026-07 [sgs-closure] Hollow R^2: full-RHS target inflated by the ring
Symptom:  full-RHS closure target gave high R^2 that did not reflect closure
          skill; ring/sponge commutator dominated the variance.
Cause:    target included body-force/sponge commutator terms the closure
          should not own; R^2 against a variance-dominated component is
          hollow.
Action:   target redefined to the Jacobian-only filtered commutator
          (Sanaa 07-13); all metrics re-based.
Rule:     check WHAT dominates the target variance before trusting any
          skill metric; targets contain only what the closure must model.

## 2026-07 [sgs-closure] One SVGP cannot serve near-wall and far-field
Symptom:  arm C/D/E ladder: single SVGP mediocre everywhere; heteroscedastic
          heads absorbed the signal.
Cause:    ~40x rms gap between near and far bands; one stationary GP cannot
          straddle it.
Action:   CNN-first architecture (Sanaa order 07-22): CNN mean + frozen
          sigma_loc(sdf) weighting, GP re-added as residual head only.
Rule:     model the mean with the mean model; GPs get residuals; nothing
          learnable in the loss weighting.

## 2026-07 [sgs-closure/sims] Inlet reflection from the left boundary
Symptom:  reflected signal entering from the left (inlet) side of the domain
          in obstacle simulations.
Cause:    numerics, not physics — inlet/sponge handling; the FIRST check is
          whether v and omega are actually 0 at the inlet.
Action:   boundary values checked directly; ramp analysis (does a longer
          ramp fix it? analyze, then CONFIRM with a few ramp runs even if
          analysis says no), then smoother ramp function; sponge analysis
          written up (docs/sponge_analysis.tex/pdf).
Rule:     artifact triage before physics interpretation: boundary values
          first, then systematic lever-by-lever escalation with analysis
          preceding runs. (LAB_DOCTRINE.md §5)

## 2026-07 [sgs-closure] Cape near-wall stripes
Symptom:  striped near-wall error pattern at the cape, suspected reflection
          artifact.
Cause:    cape-tip separated shear layer — real flow feature, not the
          reflection; not post-hoc filterable either way.
Action:   diagnosed and recorded; near-wall band reported separately.
Rule:     near-wall and far-field are never mixed in one metric.
          (memory: cape-nearwall-stripes)

## 2026-07-23 [wiener/temporal] vn-penalty outweighed its trust anchor ~300:1
Symptom:  stability fine-tune drifting away from accuracy; anchor
          ineffective.
Cause:    lambda_anchor held at 0.03 (right for vn 0.1) while vn grew to 10.
Action:   anchor recalibrated to 0.4 at vn 10; proportional-scaling law
          declared binding.
Rule:     any penalty escalation carries a proportional trust-anchor
          escalation in the same change. (CLAUDE.md tri-objective law)

## 2026-07 [wiener/temporal] Chained warm-starts ratcheted accuracy away
Symptom:  kf4 Nddot rel error 0.02 -> 0.43 over 7 fine-tune generations,
          silently.
Cause:    each generation warm-started from the previous fine-tune; soft
          anchors hold a model near its warm PARENT, so drift compounds.
Action:   warm-start policy fixed: always from the accuracy-champion
          checkpoint.
Rule:     never chain warm starts. (CLAUDE.md tri-objective law)

## 2026-07 [temporal] Von Neumann certificate blind to the actual network
Symptom:  frozen-coefficient von Neumann analysis certified stability the
          rollout contradicted.
Cause:    the certificate linearized the analytic closure, not the trained
          network; the network Jacobian J_NN is what amplifies.
Action:   stability analysis + penalty moved to J_NN by autodiff
          (--vn-mode jnn); rho unit-tested against measured growth (Test A);
          paper corrected (commit d3c79c7).
Rule:     stability claims about an NN-augmented scheme use J_NN, never the
          analytic stand-in; every rho is validated against measured growth.

## 2026-07-03 [temporal] Quiescent-window poisoning destroyed a training run
Symptom:  deriv7_equalw_R3R4: destroyed feature mix, N3dot ~ 0.99, trained
          medians WORSE than init; Re25k looked catastrophic.
Cause:    9-33% of windows were quasi-zonal spin-up (J(psi,omega) ~ 0,
          targets 1e4-1e5x smaller); per-sample relative loss exploded on
          them; optimizer's cheapest move was shrinking ALL predictions
          toward zero. Re25k was the same artifact, not a bad member.
Action:   filter_quiescent_windows.py (target-norm + stack-roughness
          thresholds, splits backed up); retrain from physics init at
          lr 5e-5 with ALL members.
Rule:     never train/report on unfiltered per-sample relative metrics;
          spin-up is beta-dependent, t-start per member; filter is a
          mandatory pipeline stage. (CLAUDE.md rules 15-16)

## 2026-07 [temporal] Emitted unused time-orders exploded the model ~1e14
Symptom:  epoch-0 val ~ 1e14 after one Adam step.
Cause:    model emitted orders 4-6 (scaled 1/dt^4..6): ~1e18 Jacobian
          features; physics-init zeroed their mix weights but one Adam step
          put lr-sized weights on them.
Action:   ORDER CLIP: emit only orders 0..out_orders; all S snapshots kept
          per emitted row. Startup check: params=3,700, epoch-0 val O(1).
Rule:     never emit unused time-orders; never let gradients flow through
          1/dt^k-scaled features no output needs. (CLAUDE.md rule 14)

## 2026-07 [temporal] Pooled S=7 val misread as model regression
Symptom:  pooled sweep val O(0.1-1) vs the old single-(member,dt) 2.6-4%
          reference read as a collapse.
Cause:    not comparable: the reference was dt=1e-3 (time-FD error
          invisible); the pool contains near/past-Delta-T* cells that are
          unlearnable by ANY network (stencil span 6dt exceeds Re25k's
          Delta-T* at dt=1.5e-2).
Action:   eval_deriv_by_root.py per-(member,dt,order) made the standard
          read; convergence-radius theory documented.
Rule:     pooled numbers never carry a verdict; comparability requires same
          (member, dt, filter) cell. (CLAUDE.md rule 18)

## pre-2026-07 [infra] HDF5 dataset corruption/friction
Symptom:  HDF5-based data pipeline unreliable for the closure datasets.
Cause:    (historical) concurrent-access/complexity mismatch with cluster
          usage.
Action:   abandoned for per-sample .npz (fixD) and packed contiguous memmap
          .npy (ensemble).
Rule:     do not reintroduce HDF5. (CLAUDE.md Environment)

## pre-2026-07 [sims] 512^2 under-resolved for cylinder at Re >= 600
Symptom:  cylinder-flow results unreliable at 512^2.
Cause:    under-resolution.
Action:   1024^2 mandated for cylinder Re >= 600.
Rule:     CLAUDE.md rule 12.

## pre-2026-07 [infra] PyYAML parsed 5e-3 as a string
Symptom:  config float silently a string; downstream type errors/wrong runs.
Cause:    YAML 1.1 scientific-notation quirk.
Action:   always 5.0e-3 form + explicit float() casts on read.
Rule:     CLAUDE.md rule 4.
