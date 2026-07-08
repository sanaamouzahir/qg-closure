# BRANCH_LOG — Exact recursion, no ML  (branch: exp/recursion-noml)

Running record. Supervisor updates this at the end of every session. Newest entry on top.

## 2026-07-08 — charter adoption record + I18 tooling (written by global supervisor)
- CHARTER v1.1 is CANONICAL ON MAIN (2056b46; merged into this branch as fb36072
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
- Sanaa asked for: create branch + seed brief; PARKED — no work until she says go.
- Ran / submitted (job ids): nothing — parked.
- Results: n/a.
- Flags from physics-sanity: none.
- Decided next: nothing until Sanaa un-parks. First decisions when live: D⁻¹ mechanism (exact-FFT
  ~3 transforms/order vs learned-local) and flux-vs-advective Jacobian form.
- What Sanaa wants to see next check-in: her go-signal + her speed ideas for the transforms count.

---
## Seed
- Hypothesis (design, not learned): exact recursion ω^(k)=Lω^(k−1)+N^(k−1) as the closure — no
  time-FD, no inner wall, dT only in analytic prefactors. Cost floor ≈ 5(m+1) transforms/order with
  direction-grouped accumulators; Sanaa has speed ideas to bring.
- Success criterion: TBD by Sanaa — this is the no-ML CEILING reference; compare COST vs learned branches.
- Control ref (`deriv7_filtered_floor0.1`): Ndot 0.058 | **Nddot 0.186** | N3dot 0.563.
- Status: PARKED — no code, no data, no jobs until Sanaa says go.
