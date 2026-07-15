2026-07-14 | GREEN | Upstream-reflection force recheck LANDED (1833450/51, upstream_effect_on_forces.py): CYLINDER St 0.174 (inlet-norm) -> 0.203 with the body-local velocity (U_cyl/U_in = 0.859) ~ the 0.21 reference => SANAA CORRECT, the St deficit was the contaminated reference velocity; CAPE 0.076 -> 0.251 (U_local/U_in 0.30, mostly geometric blockage). Cd NOT explained by upstream (local norm moves 1.69 -> 2.29 vs ref 0.99; the excess is the standard 2D-at-Re-3900 artifact, charter "context never targets"). ubar/vbar single-frame in ALL DNS_LES_s4* variants of const members (445 expected) — field-based profile needs DNS_FR, on request | yamls+txt in upstream_reflection_effect_on_forces/ | this commit

## 2026-07-15 | GREEN (diagnostics) + Sanaa ruling | mean-prediction diagnostic suite:
diagnose_mean_prediction.py (per-member R2/RMSE/signed extrema, Re-error, error-location,
masked-FFT spatial ACFs of error/truth/pred + cross, temporal ACF, hexbin; filtered ylp75
targets, val split; summaries pushed to reports/mean_prediction_diag_*). G4 PASS (5 LOW
fixed), G5 PASS. RULING RECORDED: diagnostics NEVER on the GPU queue -> jobs run on all.q
(--device cpu, thread caps); sge-checker brief updated on main. Submitter:
scripts/sge/submit_mean_prediction_diag.sh; QG_DIGEST_RUN start/fail hook in piff_tool_job.sh.
