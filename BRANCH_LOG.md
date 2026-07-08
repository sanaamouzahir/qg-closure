# BRANCH_LOG — SGS spatial closure, Phase 1  (branch: exp/sgs-closure)

Running record. Supervisor updates this at the end of every session. Newest entry on top.

## 2026-07-08 — Supervisor_simulation.md delivered + committed; CP-1 plan emailed (global supervisor)
- Blocker resolved: Sanaa delivered the theory doc; committed verbatim as
  docs/briefs/Supervisor_simulation.md next to AMENDMENT_02_workflow.md. Read in full.
- CP-1 plan emailed ([QG][PROPOSE][SGS-CLOSURE]): module list (audit_decorrelation.py,
  audit_resolution.py, shedding_tracker.py, diagnostics_wake.py + batch wrappers),
  per-module outlines with the exact theory-doc quantity each measurement compares
  against, authoring order, §3-compliant execution plan, storage estimate. WAITING for
  approval — no diagnostics/audit code authored until CP-1 clears (choreography).
- Gate-1 smokes remain HELD (unchanged); Phase A artifacts unchanged.

## 2026-07-08 — AMENDMENT 02 received + committed (global supervisor); CP-1 BLOCKED on missing theory doc
- AMENDMENT 02 committed verbatim: docs/briefs/AMENDMENT_02_workflow.md. Precedence now
  Amendment 02 > Amendment 01 > charter. Adopted immediately: CYLINDER-ONLY scope (all
  FPC-cape items SUSPENDED, not cancelled); f64 solve / f32-at-write storage; dt_save
  0.25 t.u. (save_rate 1000), MOD-const runs 0.27 (save_rate 1080); scalars every 10
  steps; §3 ABSOLUTE rule: no .py execution on the login/frontend node (qsub/qlogin
  only; submit-only shell scripts exempt; violations are FLAG-level).
- BLOCKER: the accompanying theory document `Supervisor_simulation.md` was NOT attached
  and exists nowhere on the filesystem (searched QG_ROOT, all worktrees, docs/, home).
  Amendment 02 gates ALL planning (CP-1 included) on reading it in full — audits A/B,
  the commensurability rates, the §7.1 tables, and the §9 storage rule all reference
  its sections. [QG][FLAG][SGS-CLOSURE] sent; per the choreography the team WAITS.
- NOT started (correctly, per the gate): CP-1 plan, audit_decorrelation.py,
  audit_resolution.py, storage re-estimate. Phase A artifacts + Gate-1 hold state
  unchanged (see session entries below).

## 2026-07-08 — charter adoption record + I18 tooling (written by global supervisor)
- CHARTER v1.1 is CANONICAL ON MAIN (2056b46; merged into this branch as 5cc2e82
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

## 2026-07-07 — session 2b (global supervisor Fable: gate1 tables LANDED)
- Sanaa GO for the tables CPU job ONLY (explicitly no other qsub; smokes wait until the
  mmap builds free the GPUs). Submitted job 1826260 sgs_gate1_tables — 27 s, clean exit.
- 6/6 tables + PNGs under qg-simple-package-stable/src/qg/outputs/SGS_closure_gate1/tables/,
  all stamped sha 4cfb6dca4, all U(t=0)=2.000000:
  const 2.5e-4 (U == 2.0 everywhere — bit-identity arm exact); sine 2.5e-4 Re[2418,5600]
  (min unreached by design: 2/3 period in the 10 t.u. window); ou 2.5e-4/1.25e-4 seed
  20260707 Re[~3854,4225] N 60001/120001 (dt-consistency pair); telegraph 2.5e-4/1.25e-4
  rails [2200,5600] hit.
- Benign pre-script log noise (imageio_ffmpeg import error + dirname complaint from env
  activation under -V) — no effect; note for future log readers.
- Emails: [QG][LANDED][SGS-CLOSURE] sent 20:23Z (after session-2 PROPOSE at 20:13Z).
- HELD: the 5 GPU smokes (GATE1_RELEASE=1 submitter ready). Trigger = mmap builds done +
  Sanaa's next GO. Then: bit-identity diff → Cd/Cl + overlays → GATE1 report → HARD STOP.

## 2026-07-07 — session 2 (global supervisor Fable: post-disconnect resume, Gate-1 scripts released, PROPOSE sent)
- Session context lost to a connection drop; state fully recovered from git + this log
  (nothing on the branch was lost — all Phase A commits intact).
- Gate-1 scripts: TBDs filled and committed (3efca02) — modulation.py path
  $QG_ROOT/qg-sgs-closure/training/modulation.py (+ existence guard), T_WAIT=5.0,
  hydra APPEND keys as implemented (+qg.bc.inlet_table=<npz>, +qg.diag.scalar_rate=10;
  legacy arm qg.bc.inlet_velocity=2.0 — the config composes plain YAML, new keys need
  the + prefix; verified vs conf/config.yaml + register_configs). Table-path grep in
  the submitter re-verified against the + form. sge-checker re-audit post-edit: PASS x3.
- Bit-identity precondition confirmed in code: U_PER_RE = 2/3900 exactly, modulation.py
  asserts U(3900) == 2.0 bitwise, so table_const vs legacy can be exactly 0.0.
- SUBMISSION RULE (Sanaa, this session): NO qsub without her explicit per-step GO —
  email the report/PROPOSE first, wait for approval. A prepared sgs_gate1_tables qsub
  was countermanded before submission; NOTHING has been submitted on this branch.
- Email #1 for the branch sent: [QG][PROPOSE][SGS-CLOSURE] "Gate-1 release ready" to
  sanaamz@mit.edu, 2026-07-07T20:13Z (plan: tables job -> GATE1_RELEASE=1 5 smokes ->
  analysis -> GATE1 report -> HARD STOP).
- Queue note: running build_mmap/deriv7/monitor jobs belong to the wiener/main track, not SGS.
- Next (on Sanaa's GO): tables job -> smokes -> bit-identity diff + Cd/Cl + overlays ->
  [QG][GATE1][SGS-CLOSURE] report -> hard stop before Phase B.

---
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
