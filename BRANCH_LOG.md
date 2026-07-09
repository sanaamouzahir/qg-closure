# BRANCH_LOG — SGS spatial closure, Phase 1  (branch: exp/sgs-closure)

Running record. Supervisor updates this at the end of every session. Newest entry on top.

## 2026-07-09 — Gate-1 GPU smokes SUBMITTED (branch supervisor; Sanaa green light, chat)
- Sanaa directive (chat, ~13:00 EDT): full green light to submit the held Gate-1 smokes on
  any FREE ibgpu slot; wiener trainings 1827225/1827306 untouchable. qstat -F gpu showed
  hc:gpu=6 free on ibgpu-compute-0-0 — submitted immediately, no queueing behind wiener.
- Pre-submission fixes (sge-checker PASS on both): job names sgs_gate1_* -> g1_* (all five
  collided in qstat's 10-char truncation; wrong-qdel hazard) = be2a0b0.
- ATTEMPT 1 FAILED: jobs 1828225-29 died ~5 s in at hydra composition — "+scenario=" append
  clashes with package-stable's scenario DEFAULT (decaying_turbulence in conf/config.yaml);
  qacct exit_status=1 x5, no partial output. The main-repo "+scenario=" convention applies
  to the fork's config (no scenario default), NOT to package-stable. Fix: plain
  "scenario=flow_past_cylinder_sponge" (matches outputs/flow_past_cylinder_re1000's
  overrides.yaml) = b0e455c.
- ATTEMPT 2 RUNNING: 1828230 g1_legacy / 1828231 g1_table_const / 1828232 g1_const_rec /
  1828233 g1_sine / 1828234 g1_ou, all r on ibgpu-compute-0-0 since 13:24 EDT, all picked
  GPU 0 (idle-pick race; benign at 512^2, wiener GPUs carry memory so never selected).
  ~205 it/s on 60000 steps => minutes-scale runs. Smokes are simulations, NOT training-class
  — I18 three-job monitor unit does not apply (payload run_qg.py; sge-checker concurred).
- CPU workers shedding_job.sh / audit_A_job.sh: NOT submitted — their inputs (scalars.npz
  from recorder runs / FPC-const production) do not exist until the smokes land; selftests
  already green 2026-07-09. Gate = data, not queue.
- [QG][SUBMIT][SGS-CLOSURE] sent via mailx job 1828235 pinned to ibfdr-compute-0-0 (the
  known-working mail node); body archived logs/outbox/2026-07-09_QG_SUBMIT_gate1_smokes.txt
  + copy in outputs/SGS_closure_gate1/outbox/.
- Next in-session: babysit to completion, qacct x5, [QG][LANDED] email, log results here.
- RESULTS (updated in-session):
  * legacy 1828230 + table_const 1828231: exit 0, 328 s wallclock each, full DNS output.
    **BIT-IDENTITY PASS AT BYTE LEVEL**: cmp legacy/table_const DNS.npy AND DNS_FR.npz
    both bit-identical (stronger than the max|dw|=0.0 criterion; cmp is not a frontend
    .py execution). The C.3 bit-identity arm of Gate-1 is green.
  * Recorder arms (const_rec/sine/ou) took TWO more bugs, both now fixed git-visibly:
    (BUG 1, attempt 2 = 1828232-34, exit 1 x3 at step 20000 = first flush,
    flush_every 2000 x rate 10): np.savez appends '.npz' to any filename not ending in
    it, so savez('scalars.npz.tmp') wrote 'scalars.npz.tmp.npz' and os.replace died
    FileNotFoundError. Fix 585c0bd: write through an open file handle (live package +
    solver_patches mirror, byte-exact cmp).
    (BUG 2, attempt 3 of sine/ou = 1828237/38 exit 1, const_rec 1828236 exit 0 but
    poisoned): hydra does NOT chdir — the recorder's default relative out
    'scalars.npz' resolved against launch cwd $QG_DIR: wrong location AND one shared
    scalars.npz(.tmp) across all concurrent recorder cases -> os.replace race (a
    sibling consumes your tmp). const_rec "succeeded" as last-writer only; its run dir
    had no scalars.npz. Fix d48fcae: submitter passes per-case
    +qg.diag.out=$QG_DIR/outputs/SGS_closure_gate1/<id>/scalars.npz (solver untouched;
    recorder already honored diag.out). PHASE-B RELEVANT: every future recorder run
    MUST carry a per-run diag.out.
  * Mixed-writer artifacts + const_rec run-1 quarantined (moved, never deleted;
    DNS_FR.npz intact) to outputs/SGS_closure_gate1/quarantine_2026-07-09_shared_cwd_scalars/
    with README.txt.
  * Attempt 4 (recorder arms only): jobs 1828241 const_rec / 1828242 sine / 1828243 ou,
    submitted 13:5x EDT. Outcome logged below/next entry.

## 2026-07-09 — session close: CP-1 module set COMPLETE, selftests green (branch supervisor)
- Landed: 82832aa (CP-1 approval + ledger reconciliation), a58e1aa (modules 1-3),
  02a5c47 (audit_resolution.py + CPU workers shedding_job.sh/audit_A_job.sh;
  bash -n clean, sge-checker PASS). CP-1 module list is now fully authored AND committed.
- audit_resolution.py notes: 5-grid-aware (--grids, partial-tier tolerant; 256/512
  labeled under-resolved lower-bound anchors; 2-deg rule on the 4096-vs-2048 pair only).
  eta terminology reconciled per S7.2: theory doc's eta_phys = RATE 1/tau_eta; the
  YAML/scalars eta = TIMESCALE tau_eta = penalty*dt as applied (scalars.py meta
  verified); delta_eta = sqrt(nu*tau_eta) = sqrt(nu/eta_phys_rate). Fixed-physical-eta
  tier check compares tau_eta across grids.
- Selftests on all.q (Amendment 02 §3 — the previously phantom-recorded step, now real):
  shed_st 1828217 PASS 10/10; audA_st 1828218 PASS 12/12; wake_st 1828219 PASS 10/10;
  audB_st 1828220 PASS 28/28. qacct: failed=0, exit_status=0 on all four. Known-benign
  env-activation .err noise (imageio_ffmpeg + dirname, session-2b note). Minor: numpy
  DeprecationWarning on np.trapz in audit_decorrelation — harmless, swap to
  np.trapezoid at next touch.
- Root-level SGS_closure_supervisor_brief.md duplicate removed (byte-identical to the
  canonical docs/briefs/ copy; was untracked).
- Gate-1 GPU smokes remain APPROVED-PENDING-GPU (entry below); ibgpu still held by
  wiener jobs 1827225/1827306 at session close.
- Next: CP-2 submission plan (must include the 256^2/512^2 convergence runs per
  e9a2b2d) → relay to Sanaa → Gate-1 smokes when GPUs free.

## 2026-07-09 — CP-1 APPROVED (Sanaa, chat) + ledger reconciliation (branch supervisor)
- CP-1 (audit/diagnostics code plan, emailed [QG][PROPOSE][SGS-CLOSURE] 2026-07-08) is
  APPROVED by Sanaa 2026-07-09 via chat. Recorded here git-visibly per the choreography
  (Amendment 02 §5); execution of the CP-1 module list resumes this session.
- INTEGRITY RECONCILIATION: the 2026-07-08 evening session died mid-work. DECISIONS.md
  had recorded (a) scripts/sge/shedding_job.sh + audit_A_job.sh as authored ("bash -n
  clean") and (b) module selftests as submitted "via qsub all.q: see logs/". NONE of
  that happened: neither script existed anywhere in the worktree, logs/ was empty, and
  qacct shows no such jobs. The three diagnostics modules (audit_decorrelation.py,
  shedding_tracker.py, diagnostics_wake.py) DID exist, complete but uncommitted. The
  phantom DECISIONS.md entries are ANNOTATED in place (recorded-but-never-executed),
  not deleted — the record stands, corrected. Remaining CP-1 gap: audit_resolution.py
  (module 4, 5-grid-aware Audit B per the e9a2b2d directive) — authored this session.
- Gate-1 GPU smokes: APPROVED-PENDING-GPU (Sanaa, chat 2026-07-09). Both ibgpu slots
  are held by the wiener-conditioning trainings (jobs 1827225, 1827306) — no preemption,
  no GPU submission from this branch until they free. Then GATE1_RELEASE=1 per session-2b.

## 2026-07-08 — DIRECTIVE (Sanaa, chat): convergence tier extended to FIVE grids
- The convergence grid study now covers the FULL sweep {256^2, 512^2, 1024^2, 2048^2,
  4096^2} (was {1024^2, 2048^2, 4096^2}). All MOD-const, shared-IC, fixed-physical-eta.
- BRANCH SUPERVISOR: this changes YOUR CP-2 submission plan — the convergence tier gains
  the 256^2 and 512^2 runs (t=30 IC extract spectrally regridded DOWN to both; storage
  +~0.7 GB total, negligible; dt/CFL per grid to be stated per-run in the plan). Audit B
  (audit_resolution.py) is being authored 5-grid-aware (--grids CLI, partial-tier
  tolerant, under-resolved coarse anchors labeled as lower bounds, 2-degree rule stays
  on the 4096-vs-2048 fine pair only).
- Theory context: pts/delta at Re_mid ~ 0.20 (256^2) and 0.41 (512^2) — deeply
  under-resolved BY DESIGN; they anchor the coarse end of the convergence curve.
  (Repo rule 12 — 512^2 under-resolved for cylinder at Re >= 600 — is exactly why they
  belong in a convergence study and exactly why they are NOT production grids.)

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
