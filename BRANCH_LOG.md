# BRANCH_LOG — Fewer lags at equal accuracy  (branch: exp/lean-stencil)

Running record. Supervisor updates this at the end of every session. Newest entry on top.

## 2026-07-08 — charter adoption record + I18 tooling (written by global supervisor)
- CHARTER v1.1 is CANONICAL ON MAIN (2056b46; merged into this branch as 1e5522b
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
- Sanaa asked for: create branch + seed brief; STARTS when TASK 1's builds land (reuses them, no new sim).
- Ran / submitted (job ids): nothing yet — gated on the data pipeline (build→slice→resplit→filter) for
  the new members completing.
- Results: n/a.
- Flags from physics-sanity: none.
- Decided next: when builds land — (1) re-slice S∈{4,5,6} from the SAME deep 28-mark builds
  (`slice_deriv_from_deep.py --n-snapshots S`), (2) resplit → filter each, (3) train control config per S,
  (4) results-summarizer accuracy-vs-S curve (Nddot median per S per dT tier) + memory/walltime per S.
  Check whether the slicer supports a non-uniform lag pattern; if not, PROPOSE the minimal change first.
- What Sanaa wants to see next check-in: the accuracy-vs-S curve + smallest S within 10% of control Nddot.

---
## Seed
- Hypothesis: most of S=7's value is in the Nddot jump; a smaller S (or non-contiguous lag pattern,
  e.g. {0,1,2,4,6}) keeps Nddot within ~10% of the S=7 control at much lower memory/I/O.
- Success criterion: an S<7 (or 5-lag pattern) with pooled TEST Nddot ≤ ~0.205 (within 10% of 0.186) at lower cost.
- Control to beat (`deriv7_filtered_floor0.1`): Ndot 0.058 | **Nddot 0.186** | N3dot 0.563 (Nddot = ceiling).
- Invariants: reuse existing deep builds (no new sim); resplit+filter each S; control config, change only S/lag-pattern.
