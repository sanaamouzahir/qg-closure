# [QG][BRIEF][SGS-CLOSURE] AMENDMENT 02 — cylinder-only scope, audit codes, and execution choreography

Global supervisor: first action, commit this file verbatim as `docs/briefs/AMENDMENT_02_workflow.md` and the accompanying theory document as `docs/briefs/Supervisor_simulation.md`, both on the branch. These amend the charter and Amendment 01; precedence: Amendment 02 > Amendment 01 > charter. Read `Supervisor_simulation.md` IN FULL before planning anything — it is the binding theoretical basis for every audit and every rate below.

## 1. Scope change: CYLINDER ONLY

All cape (FPC-cape) items in the charter and Amendment 01 are SUSPENDED, not cancelled: no cape configs, runs, Pi_FF, or diagnostics until Sanaa explicitly reopens them after the cylinder pipeline has run end to end. Case matrix is now 5 FPC production runs (const, sine, ramp, ou, telegraph) + the Gate D-1 Re=200 validation run + the 7-run convergence tier. Milestone emails adjust accordingly (no "first pair"; see §5).

## 2. Precision and rates (binding)

- Solver: float64. Storage of ALL outputs (snapshots, scalars, Pi_FF): float32, cast at write only (Supervisor_simulation.md §9). Update the storage estimate and re-verify quota.
- Snapshots: dt_save = 0.25 t.u. (save_rate 1000); MOD-const runs (production const AND Gate D-1) use dt_save = 0.27 (save_rate 1080) per the commensurability analysis (§5 of the theory doc).
- Scalars: every 10 steps, unchanged.

## 3. Execution environment rule (absolute)

NO .py code ever executes on the login/frontend node — not audits, not plotting, not Pi_FF, not rerendering, not "quick checks". Every Python execution goes through qsub batch submission or an interactive qlogin session. Shell scripts that merely SUBMIT .py jobs may run on the frontend; shell scripts that RUN .py may not. Violations are FLAG-level events.

## 4. New code to author (global supervisor, per the §8 authorship matrix)

In addition to Phase A artifacts already chartered (modulation.py, bc hook, scalar recorder, spectral_regrid.py) and the Amendment 01 diagnostics (shedding_tracker.py, diagnostics_wake.py):

1. `audit_decorrelation.py` — implements Audit A (theory doc §8/A) exactly: ACFs and tau_int of C_L, probe v, and Pi_FF at 5 wake points per scale; U_c from probe cross-correlation; spatial ACF of Pi_FF -> l_corr(s); phase-coverage histogram; the pre-committed dt_save decision rule; output = one report (figures + a (theory, measured, ratio) table for every quantity in theory doc §2 and §4).
2. `audit_resolution.py` — implements Audit B (theory doc §7.2–7.3 and §8/B): eta_phys derivation check, delta_eta classification per grid, wall-normal profiles, surface vorticity, theta_sep with ring-radius sensitivity, D_eff; (theory, measured) tables against §7.1; the 2-degree FLAG rule.
3. All audit and diagnostic code emits machine-readable summaries (yaml/npz) alongside figures, so results can be regression-compared on later reruns.

## 5. Choreography: checkpoints, autonomy, and emails

Between checkpoints the team operates FULLY AUTONOMOUSLY. Sanaa's involvement is exactly the checkpoints below; do not ask for input between them except via FLAG for genuine blockers.

- CP-1 (plan approval — code). After reading both committed docs, the global supervisor emails Sanaa its complete plan for the new audit codes AND the existing diagnostics stack: module list, per-module functional outline, which theoretical quantity each empirical measurement compares against, and the integration order with Phase A. WAIT for Sanaa's approval.
- CP-2 (plan approval — submission). The branch supervisor prepares the submission plan: exact job list (Gate 1 smokes; FPC-const; Gate D-1 Re=200; the 4 modulated runs held behind Audit A; the 7 convergence runs held behind the t=30 IC extract), resources per job (ibgpu.q, gpu=1), hold-chain structure, and what each submitting subagent will be dispatched to do. The global supervisor relays this plan to Sanaa. WAIT for approval.
- SUBMIT. After CP-2 approval and Gate 1 passing (Gate 1 remains a hard stop per the charter): submit per plan. Immediately email `[QG][SUBMIT][SGS-CLOSURE]` listing every job ID, case, config hash, save rates, and the hold chains.
- RUN COMPLETION. As runs land: Pi_FF per case (scales {2,4,8}), then ALL diagnostics (shedding_tracker, diagnostics_wake), ALL audits (A at the FPC-const gate before the 4 modulated runs are released; B on the convergence tier), and video rerendering — every one via batch or qlogin per §3. Then email `[QG][MILESTONE][SGS-CLOSURE] Cylinder pipeline complete` with: per-run diagnostic packages, both audit reports with (theory, measured, ratio) tables, Gate D-1 verdict, DATASET_MANIFEST.md per case, and the rerendered-video inventory.
- Sanaa reviews the milestone package; cape reopening is her call afterward.

## 6. Standing items unchanged

Gate 1 hard stop; Gate D-1 literature comparison allowed ONLY at Re=200; fixed-physical-eta rule for the convergence tier; qlogin-only rerendering with the existing rerender_videos.py; email enum per the approved merge; §8 authorship matrix; never delete DNS_FR files; PreToolUse guard rules.
