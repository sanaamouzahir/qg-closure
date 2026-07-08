# BRANCH_LOG — Time derivative as spatiotemporal convolution  (branch: exp/time-conv)

Running record. Supervisor updates this at the end of every session. Newest entry on top.

## 2026-07-08 — charter adoption record + I18 tooling (written by global supervisor)
- CHARTER v1.1 is CANONICAL ON MAIN (2056b46; merged into this branch as ce6cf76
  2026-07-08). The email-appended v1.1 text is superseded by the file; canonical wins on
  any conflict. Known defect: the file carries the v1.1 block twice (merge artifact of
  284c702+2056b46) — dedup is in the v1.3 draft; charter edits are RED, Sanaa pushes.
- v1.2 (I16 anomaly playbook, I17 one-document rule) adopted operationally 2026-07-08;
  v1.3 (I18 monitoring-is-part-of-the-submission, I19 branch->global escalation) DRAFTED,
  RED-pending. Proof case: deriv7_cond_local job 1827034 ran 6 epochs order-inverted
  with no agent detection (P1 postmortem: main DECISIONS.md 2026-07-08).
- I18 tooling landed in THIS commit: diagnostics/monitor_training.py v2 (LIVE/FINALIZE,
  [QG][MONITOR] cadence first-val/every-5/on-trigger, ORDER-INVERSION vs physics-init
  medians 0.19/0.26/0.33, baseline card), scripts/sge/monitor_training_job.sh,
  diagnostics/baseline_cards/T1_deriv7.json, sge-checker G5 refusal (training qsub
  without the LIVE+FINALIZE monitor pair = REFUSED). Every future training submission
  from this branch is a three-job unit; [QG][SUBMIT][log] carries all ids.
- This branch's supervisor: CONFIRM adoption in your next digest (ORDER 3).

## 2026-07-06 — session 0 (seed, by global supervisor)
- Sanaa asked for: create branch + seed brief; BUILD AFTER free-time-fd's first result (not in parallel).
- Ran / submitted (job ids): nothing — gated on free-time-fd landing a first Nddot.
- Results: n/a.
- Flags from physics-sanity: none.
- Decided next: while gated, read code + scope the per-lag spatiotemporal-conv module
  (ω^(k)=Σ_i K_i^(k) * ω_{−i}, init Vandermonde × spatial-delta so init == control). Do NOT submit.
- What Sanaa wants to see next check-in: free-time-fd's first number, then a cost estimate for this branch.

---
## Seed
- Hypothesis: per-order spatiotemporal convolution (a spatial kernel PER LAG across the 7 levels) is
  the most expressive form of the time-FD-as-Wiener idea (free-time-fd is its pointwise special case);
  it beats free-time-fd on Nddot enough to justify the parameter/compute increase.
- Success criterion: beats free-time-fd on Nddot, cost-adjusted (measure params / GPU-h / memory / walltime).
- Control ref (`deriv7_filtered_floor0.1`): Ndot 0.058 | **Nddot 0.186** | N3dot 0.563.
- Invariants: init == control (Vandermonde × delta); analytic 1/dt^k kept; ORDER CLIP unchanged;
  no physics conditioning; GATE — don't submit until free-time-fd's first result exists.
