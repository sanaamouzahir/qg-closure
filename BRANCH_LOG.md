# BRANCH_LOG — Fewer lags at equal accuracy  (branch: exp/lean-stencil)

Running record. Supervisor updates this at the end of every session. Newest entry on top.

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
