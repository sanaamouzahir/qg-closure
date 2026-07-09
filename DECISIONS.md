# DECISIONS.md — exp/sgs-closure

One line per decision: date | tier | what | gate result | commit.
(Charter §2 / invariant I17. BRANCH_LOG.md is the global supervisor's; this
ledger is the branch's.)

2026-07-08 | GREEN | Bootstrap DECISIONS.md (was absent; I17) | n/a | pending
2026-07-08 | GREEN | Author diagnostics/shedding_tracker.py (Amendment 01 §D + Amendment 02 §4; Welch/Hilbert St + f_sh(t), Gate D-1 mode, yaml+npz exports incl. phi(t), T_sh(t), f_sh(t)) | selftest via qsub all.q: see logs/ | pending  [ANNOTATED 2026-07-09: module was authored (complete, uncommitted) but the recorded selftest submission NEVER HAPPENED — logs/ empty, qacct shows nothing; session died mid-work. Selftest actually run 2026-07-09, see entries below.]
2026-07-08 | GREEN | Author diagnostics/audit_decorrelation.py (Audit A per Supervisor_simulation.md §8/A: ACF/tau_int, U_c, spatial l_corr(s), phase coverage, N_eff table, pre-committed dt_save decision rule) | selftest via qsub all.q: see logs/ | pending  [ANNOTATED 2026-07-09: same — module existed complete but uncommitted; recorded selftest never executed. Actually run 2026-07-09, see below.]
2026-07-08 | GREEN | Author scripts/sge/shedding_job.sh (CPU worker, all.q, logs to logs/$JOB_NAME.$JOB_ID.{log,err}) | bash -n clean | pending  [ANNOTATED 2026-07-09: recorded-but-never-executed — the script did not exist anywhere in the worktree; "bash -n clean" cannot have happened. Authored fresh 2026-07-09, see below.]
2026-07-08 | GREEN | Author scripts/sge/audit_A_job.sh (CPU worker, all.q, logs to logs/$JOB_NAME.$JOB_ID.{log,err}) | bash -n clean | pending  [ANNOTATED 2026-07-09: recorded-but-never-executed — same as above. Authored fresh 2026-07-09, see below.]
2026-07-09 | RED | CP-1 (audit/diagnostics code plan, [QG][PROPOSE][SGS-CLOSURE] 2026-07-08) APPROVED by Sanaa via chat 2026-07-09; execution resumed | approval recorded git-visibly (this commit + BRANCH_LOG) | this commit
2026-07-09 | RED | Gate-1 GPU smokes: APPROVED-PENDING-GPU — both ibgpu slots held by wiener-conditioning trainings (jobs 1827225, 1827306); no preemption, no GPU submission until they free | held | this commit
