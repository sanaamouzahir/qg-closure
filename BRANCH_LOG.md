# BRANCH_LOG — SGS spatial closure, Phase 1  (branch: exp/sgs-closure)

Running record. Supervisor updates this at the end of every session. Newest entry on top.

## 2026-07-07 — session 0 (branch instantiation, global supervisor Fable)
- Sanaa asked for: worktree setup, charter transmission to Opus branch supervisor, team
  acknowledgment of the §8 matrix / qlogin+milestone-email rules / Gate 1 stop, restatement
  + conflict report back to her. Phase A code authorship HELD until Sanaa acks the restatement.
- Done: worktree `qg-sgs-closure` added tracking origin/exp/sgs-closure (base = current main
  78c0ca4); charter read in full; SUPERVISOR_BRIEF.md instantiated with the §8 override matrix;
  Opus 4.8 branch supervisor briefed and acknowledgment collected.
- Environment notes for all future sessions: system git 1.8.3.1 cannot drive worktrees — use
  `/opt/rocks/bin/git` (2.9.2). Live solver = shared editable install qg-simple-package-stable
  (v0.2.1, OUTSIDE this worktree) — the bc.py hook location needs Sanaa's ruling (see conflict
  report). /gdata shows 58 TB free at fs level (96% full) — per-user quota to be verified in
  Phase 0 against the ~100 GB Phase B estimate.
- Config audit (preliminary, §1.4): `flow_past_cylinder_sponge.yaml` confirms the traps —
  `nu: 5e-3`, `tol: 1e-3` (PyYAML strings); `mask.r = 0.628318530717959` confirmed ✓;
  default grid 512², dt 1e-3, `penalty: 1.25` and `sponge: 1.25` (× dt convention) — physical
  eta values must be recorded per §4.1 and FROZEN for Phase C per §5.2.
- Decided next: await Sanaa's ack of the restatement → Phase 0 execution (full config audit,
  quota number, submodule/venv ruling) → Phase A (Fable authors modulation.py + bc hook).
- What Sanaa wants to see next check-in: her ack; rulings on the conflict list (email
  categories, bc.py hook location, 512² Gate-1 smoke grid).

---
## Seed
- Hypothesis (Phase 1, no ML yet): non-stationary inlet-modulated flow-past-obstacle cases
  give a controlled testbed for a priori SGS closure; deliverable = 10-run FR ensemble +
  convergence tier + Π_FF at scales {2,4,8}, ready for Sanaa's diagnostics_wake.py.
- Success criterion: Gate 1 pass (bit-identity + smoke stability + dt-consistency), clean
  production ensemble (no NaN, CFL reported), monotone-or-flagged short-horizon convergence,
  complete DNS_LES_s{2,4,8}.npz set per case.
- Baseline/control: MOD-const (Re = 3900) per geometry.
- Truth framing: 2D fine-grid penalized-obstacle solution ONLY; Cd≈0.99 / St≈0.21 are
  context, never validation targets.
