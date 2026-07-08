# BRANCH_LOG — Learnable time-FD coefficients  (branch: exp/free-time-fd)

Running record. Supervisor updates this at the end of every session. Newest entry on top.

## 2026-07-08 — charter adoption record + I18 tooling (written by global supervisor)
- CHARTER v1.1 is CANONICAL ON MAIN (2056b46; merged into this branch as cbac3b3
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
- Sanaa asked for: create branch + seed brief; this branch STARTS NOW (same trainer, one module change).
- Ran / submitted (job ids): [fable-authored] learnable time-FD rows (c2fdf93) + worktree wrapper
  repoint (3016cc4). Init-repro PASS (max|control-freeW|=0.000e+00, +21 params, W_learn grad live).
  closure-reviewer SAFE; sge-checker SAFE-TO-SUBMIT (18 equal-data roots = control's roots).
  Training job **deriv7_freeW = 1825543** (ibgpu.q, S=7, grad_kernel 15, lr 5e-5, 300 ep,
  --rel-floor 0.1, --learn-time-fd, f64). Notify job 1825544 (hold_jid 1825543) → emails LANDED
  with Nddot vs 0.186.
- Results: pending (training running). [QG][SUBMIT][free-time-fd] sent.
- Flags from physics-sanity: none yet — run physics-sanity on the learned W_learn rows once the ckpt
  lands (odd-even/checkerboard, moment-condition breakage — expected but can alias).
- Decided next: on LANDED → results-summarizer verdict + physics-sanity on the learned rows; if it
  beats 0.186, consider the wd=0 stencil ablation (## Proposed below).
- What Sanaa wants to see next check-in: deriv7_freeW pooled TEST Nddot vs control 0.186.

---
## Proposed
- **wd=0 stencil param-group** (from closure-reviewer, run-#1 review). AdamW wd=1e-4 decays W_learn
  (and the spatial wx/wy) toward ZERO, not toward Vandermonde — a mild prior AGAINST the very
  "is Vandermonde optimal?" question this branch tests. Put the stencils in a `weight_decay=0`
  group and rerun. Why it might beat control: removes a decay-toward-zero bias fighting the learned
  rows. Cost: ~1 GPU-run (≈ control walltime), no new storage. Kill: if Nddot moves <2% vs the
  wd=1e-4 freeW run, drop it. STATUS: gated on the deriv7_freeW result — only worth it if the
  learnable rows show signal. (Requires code from Fable per the authorship rule.)

---
## Seed
- Hypothesis: the Vandermonde time-FD rows are optimal for noiseless truncation but NOT the optimal
  linear estimator on real pooled data (Sanaa's small-dt regression: optimal coeffs ≠ Vandermonde,
  the Wiener-in-time mechanism). Learning rows 1..3 beats fixed Vandermonde on Nddot.
- Success criterion: pooled TEST Nddot < control 0.186 at EQUAL data.
- Control to beat (`deriv7_filtered_floor0.1`): Ndot 0.058 | **Nddot 0.186** | N3dot 0.563 (Nddot = ceiling).
- Invariants: row 0 frozen; analytic 1/dt^k kept (learn dimensionless only, dt-portability survives);
  ORDER CLIP unchanged; no physics conditioning; change ONLY W_unit rows so the delta is attributable.
