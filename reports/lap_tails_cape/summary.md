# error-tails diagnostics -- piff_cape_gjs_lap (val, filtered gaussian_jonly_ylp75)

| member | rmse | rel_rmse | n_px | n_gt_10x | frac_gt_10x | worst0.1pct_SS_share | r2_active_q90 | r2_quiet_q50 | rmse_active | rmse_quiet | near_zero_px_share | extremes_near_body | Re_corr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FPCape-const | 0.08045 | 0.125 | 6918032 | 12858 | 1.86e-03 | 64.5% | 0.984 | -256.059 | 0.2535 | 0.001 | 91.1% | 78% | undef(const) |
| FPCape-sine | 0.28781 | 0.285 | 6918032 | 4467 | 6.46e-04 | 90.6% | 0.919 | -543.24 | 0.9092 | 0.0023 | 90.7% | 100% | -0.422 |
| FPCape-ramp | 0.0975 | 0.139 | 6918032 | 12092 | 1.75e-03 | 66.7% | 0.981 | -190.067 | 0.3074 | 0.0012 | 90.5% | 80% | undef(const) |
| FPCape-ou | 0.09781 | 0.13 | 6918032 | 11500 | 1.66e-03 | 70.9% | 0.983 | -263.63 | 0.3085 | 0.0011 | 91.4% | 90% | -0.008 |
| FPCape-tel | 0.09517 | 0.134 | 6918032 | 12794 | 1.85e-03 | 66.3% | 0.982 | -284.428 | 0.3002 | 0.0012 | 91.3% | 68% | undef(const) |

figures: /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-sgs-closure/ml_closure/pngs/error_tails_diag/piff_cape_gjs_lap
per-member yaml/csv: runs_piff/piff_cape_gjs_lap/error_tails_diag
