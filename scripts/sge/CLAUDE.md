# scripts/sge/ — sub-agent brief
Pattern: submit_X.sh (qsub wrapper) → X_job.sh (worker: sources $QG_ROOT/qg-env, cd's,
forwards args verbatim). HARD RULES (never violate):
- NEVER -q ibamd.q. NEVER -l h_vmem=...G. GPU jobs: -q ibgpu.q -l gpu=1 ONLY.
- Match conventions of submit_qg.sh when writing new scripts.
- YAML overrides: write 5.0e-3, never 5e-3.
