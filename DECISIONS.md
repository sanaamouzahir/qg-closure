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
