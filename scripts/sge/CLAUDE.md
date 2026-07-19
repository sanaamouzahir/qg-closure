# scripts/sge/ — sub-agent brief
Pattern: submit_X.sh (qsub wrapper) → X_job.sh (worker: sources $QG_ROOT/qg-env, cd's,
forwards args verbatim). HARD RULES (never violate):
- NEVER -q ibamd.q. NEVER -l h_vmem=...G. GPU jobs: -q ibgpu.q -l gpu=1 ONLY.
- Match conventions of submit_qg.sh when writing new scripts.
- YAML overrides: write 5.0e-3, never 5e-3.

## NaN policy (Sanaa mandate 2026-07-19 — PROJECT-WIDE, any code)
NO long-running job ships without a NaN reaction (STOP > CHECK > FIX > RESUBMIT):
- Trainers: in-process hard abort — 2 consecutive non-finite epochs => save state,
  write NAN_ABORT.txt, exit 9 (train_piff.py / train_deriv_rollout.py are the pattern).
- Sims: NaN-guard sidecar killing on non-finite scalars (phaseB_A_job.sh pattern).
- Submit scripts wire their LIVE/FINALIZE monitor sidecars THEMSELVES — monitor
  wiring is never a separate manual step (root cause of the 2026-07-19 wallv2 burn).
- sge-checker: treat a submission without these as a FAIL finding.
