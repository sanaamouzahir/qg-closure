# error-tails diagnostics -- piff_cape_gjs_ylp75 (val, filtered gaussian_jonly_ylp75)

| member | rmse | rel_rmse | n_px | n_gt_10x | frac_gt_10x | worst0.1pct_SS_share | r2_active_q90 | r2_quiet_q50 | rmse_active | rmse_quiet | near_zero_px_share | extremes_near_body | Re_corr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FPCape-const | 0.08066 | 0.125 | 6918032 | 12543 | 1.81e-03 | 64.7% | 0.984 | -232.559 | 0.2542 | 0.0009 | 91.1% | 78% | undef(const) |
| FPCape-sine | 0.33667 | 0.334 | 6918032 | 3703 | 5.35e-04 | 92.6% | 0.889 | -344.518 | 1.0637 | 0.0018 | 90.7% | 100% | -0.426 |
| FPCape-ramp | 0.09788 | 0.139 | 6918032 | 11902 | 1.72e-03 | 66.5% | 0.981 | -148.259 | 0.3085 | 0.0011 | 90.5% | 78% | undef(const) |
| FPCape-ou | 0.09965 | 0.133 | 6918032 | 11275 | 1.63e-03 | 71.4% | 0.982 | -283.404 | 0.3143 | 0.0011 | 91.4% | 88% | -0.033 |
| FPCape-tel | 0.0961 | 0.136 | 6918032 | 12466 | 1.80e-03 | 67.0% | 0.982 | -232.609 | 0.3031 | 0.001 | 91.3% | 68% | undef(const) |

figures: /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-sgs-closure/ml_closure/pngs/error_tails_diag/piff_cape_gjs_ylp75
per-member yaml/csv: runs_piff/piff_cape_gjs_ylp75/error_tails_diag
