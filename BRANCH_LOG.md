# BRANCH_LOG — SGS spatial closure, Phase 1  (branch: exp/sgs-closure)

Running record. Supervisor updates this at the end of every session. Newest entry on top.

## 2026-07-07 — session 1b (global supervisor Fable: Phase A code handoff)
- AMENDMENT 01 committed verbatim (docs/briefs/AMENDMENT_01_diagnostics.md).
- [fable-authored] training/modulation.py + training/spectral_regrid.py committed on branch.
  OU is realized on a DT_MICRO=1.25e-4 micro-grid and exactly subsampled → the Gate-1
  dt-consistency overlay is a strict identity, not statistical. `--re-const 200` covers
  Gate D-1. Regrid zeroes Nyquist both ways; `--self-test` prints the charter round-trip number.
- Solver hooks LIVE in shared qg-simple-package-stable (Sanaa ruling A.1), keys:
  `qg.bc.inlet_table` (bc.py, Flow.const_x_flow; table dt must equal solver dt, exhaustion
  guarded) and `qg.diag.scalar_rate` (+ optional out/length/u_mid/flush_every/probes) driving
  NEW src/qg/_output/scalars.py; 3-line stash in obstacle.py hands the recorder the exact
  discrete momentum sink (source-time, mean flow included — bc patch precedes penalty in the
  patch list). Zero extra FFTs: reductions only on sampled steps; Z via Parseval on qh(t_n).
  Mirror: solver_patches/sgs_hooks_2026-07-07.patch (byte-exact round-trip verified) +
  solver_patches/src/qg/_output/scalars.py; PORTING.md updated.
- Solver findings for the Gate-1 report (config audit §1.4): (a) `circular` mask IGNORES
  YAML x_center/y_center — obstacle sits at DOMAIN CENTER (Lx/2, Ly/2); recorder locates it
  from the applied chi centroid. (b) `mask.tol` key is inert (code reads `tolerance`).
  (c) nu/penalty/sponge/B/mu have NO float() casts in the solver — explicit-mantissa YAML
  mandatory. (d) post-step snapshots have ZERO-MEAN u (mean inlet lives only inside the step);
  Π_FF/wake stats from snapshots must re-add U_inlet — documented in scalars.py; scalars' E
  includes the mean-flow energy, DNS_FR_diagnostics' E does not.
- Answers to supervisor TBDs: t-wait 5.0 approved for Gate-1 smokes; modulation.py lives in
  training/ (flat); key names as above. Cape probe proposal approved incl. 6th recirculation
  probe — pass as qg.diag.probes + qg.diag.length=1.0 for cape.
- Next: supervisor fills TBDs → tables job → Gate-1 smokes → GATE1 report → HARD STOP.

---
## 2026-07-07 — session 1 (branch supervisor: charter ACK + AMENDMENT 01 receipt + Gate-1 prep)
- AMENDMENT 01 received and read in full (docs/briefs/AMENDMENT_01_diagnostics.md). Supersedes
  charter §7: ALL diagnostics code (scalar recorder, shedding_tracker.py, diagnostics_wake.py)
  is now Fable-authored; Sanaa authors only the later ML closure. Phase A RELEASED; extended
  Gate 1 per §C.3 (bit-identity with recorder absent + recorder-on Cd/Cl shedding smoke +
  U_inlet-vs-table exact overlay). NEW Gate D-1 (§F): FPC Re=200, 1024², T=120 — the one
  permitted literature comparison. Email enum ruling: charter codes for named reports
  (GATE1/RUN/CONV/PIFF/MILESTONE/GATE-D1), global codes (SUBMIT/LANDED/FLAG/PROPOSE/BLOCKED)
  for state changes, ISSUE→FLAG. Nothing submitted this session (per Fable: hold for handoff).
- Gate-1 job scripts DRAFTED (sge-runner authored under direction, per §8 matrix), HOLD state:
  scripts/sge/gate1_make_tables.sh, gate1_job.sh, submit_gate1_smokes.sh. Five smokes
  (legacy / table_const / const_rec / sine / ou), 512², dt 2.5e-4, T=15, nu=6.4443e-4,
  qg.grid.precision=float64 explicit (solver DEFAULT is float32 — key confirmed in qg/config.py).
  Submit guard: GATE1_RELEASE=1 required before any qsub. sge-checker audit: PASS ×3, no fixes;
  two operational notes (modulation.py path placeholder; run tables job to completion before the
  interactive smokes submitter — soft-skip, not hold_jid).
- TBD at Fable handoff: Gate-1 --t-wait (drafts use 5.0, flagged); modulation.py final path;
  bc.inlet_table + diag.scalar_rate key names as implemented.
- Cape lee-probe proposal (§C.1 equivalent set; from mask geometry, x_c=0.2·Lx=5.0265,
  tip height y_scale=4.0, L=L_cape=1): wake probes (x_c+{1,2,3}L, 4.0) = (6.0265, 4.0),
  (7.0265, 4.0), (8.0265, 4.0); cross-stream pair (x_c+1L, 4.0±0.5L) = (6.0265, 4.5/3.5);
  optional 6th recirculation probe (6.0265, 2.0) — cape wake is bottom-attached/asymmetric.
  Land clearance checked: cape height at x_c+1 is 0.736 (probe at 3.5 clears by 2.76); support
  ends at x_c+2. Incident-velocity window (U_cape analog of U_cyl): mean streamwise u over
  window centered (x_c−1.5L, 4.0), |y−4.0|≤0.5L, excluding masked points — clear of the left
  sponge (extends to 0.1·Lx=2.513) and of the cape support (starts x_c−2=3.027).
- DATASET_MANIFEST.md template planned (§G), to be instantiated per case at Phase D: case/geom/
  modid; git SHAs ([fable-authored] code + run commit); per-file path/shape/dtype for DNS_FR,
  DNS_LES_s{2,4,8}, U_of_t, scalars; grid (N, L, dx), dt, save_rate, scalar_rate; nu and the
  frozen PHYSICAL eta values; filter definition (scale, alpha, width, operator + code ref);
  Π_FF sign/normalization convention AS IMPLEMENTED (code line ref); usable window [30,120];
  seeds; probe coordinate table; NaN-check result; caveats. Target: zero-archaeology handoff.
- Next: await Fable's [fable-authored] handoff SHAs → fill TBDs → tables job → Gate-1 smokes
  (GATE1_RELEASE=1, SUBMIT email) → Gate-1 report → HARD STOP for Sanaa approval.

---
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
