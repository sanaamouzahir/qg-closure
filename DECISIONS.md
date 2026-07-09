# DECISIONS.md — new-file justifications (CHARTER v1.2 I17)

One entry per new file created on this branch after 2026-07-08. A new file needs a reason;
extending an existing document is the default.

- 2026-07-08 `DECISIONS.md` — bootstrap: I17 requires this ledger; no existing doc holds
  file-creation rationale.
- 2026-07-08 `diagnostics/diagnose_head_sign.py` — STEP-1b discriminator (signed correlation
  per NN head); no existing probe reports SIGNED correlation (diagnose_one_sample.py is
  stage-rms only) and the sign question is exactly what rel-L2 cannot answer.
- 2026-07-08 `scripts/sge/apost_ladder_job.sh` — I16 ladder runner (early-stop sequencing
  logic; not expressible as flags to an existing job script).
- 2026-07-08 `diagnostics/RESULTS_2026-07-08_smoke3.md` — pre-dates I17 by hours; kept
  (RESULTS_*.md session logs are the established diagnostics convention). Future sessions:
  extend the day's RESULTS file rather than opening parallel ones.
- 2026-07-08 `diagnostics/diagnose_condlocal_triage.py` (+ evidence CSV
  `diagnostics/triage_condlocal_D1.csv`) — ORDER 1 D1/D2/D3 triage of incident 1827034 in ONE
  GPU pass: no existing probe combines best-vs-zero-init eval, raw-vs-floored median/mean per
  root, the ch0/ch1 target-alignment recompute (D2), and grid-split init medians (D3);
  diagnose_error_distribution.py covers only the D1 slice on one model.
2026-07-09 | GREEN | new file diagnostics/consolidate_apost_cases.py (I17 reason: Sanaa's explicit 2026-07-09 output-discipline order -- one npz per (ckpt,variant,dT) case, intermediates deleted; no existing doc covers post-run consolidation) | py_compile PASS | pending-commit
2026-07-09 | GREEN | frozen eval copy training_runs/deriv7_cond_local_v2/frozen_eval_20260709/best.pt (epoch 63, val 0.2139) -- job 1827306 still writes best.pt; eval runs on the frozen copy | load-smoke PASS (cond_local, 10564 params) | n/a
2026-07-09 | YELLOW | --drop-nddot ablation flag in rollout_aposteriori.py (variant B of Sanaa's 07-09 matrix); + apost_matrix_job.sh (2x2x3 matrix, gamma=1, no remediation) | compile+ckpt-load PASS; sge-checker pending | pending-commit
2026-07-09 | GREEN | t=0 LTE regression row post-hoc (job 1828240, scratch outputs, not kept) -- COND ep63 rel_Nddot(t=0)=0.136 outside acceptance -> COND matrix rows labeled MID-TRAINING | protocol order violation acknowledged in RESULTS md | n/a
2026-07-09 | GREEN | new file diagnostics/RESULTS_2026-07-09_apost_matrix.md (established RESULTS_* session-ledger family) | n/a | pending-commit
2026-07-09 | YELLOW | --nn-project-radius (alias-safe 2/3 radial projection of the R3 correction ONLY; solver mask untouched per Sanaa RED ruling) + r3anal sigma logging + consolidate _proj variant + apost_matrix_p170_job.sh (NN matrix w/ projection + analytic r3anal arms std/proj) | compile+argparse PASS; sge-checker pending | pending-commit
2026-07-09 | GREEN | COND ckpt for p170 rerun = existing frozen_eval_20260709 (ep63): job 1827306 still running, best.pt UNCHANGED since freeze (best_val plateau 0.21389 through ep74) | qstat + mtime evidence | n/a
