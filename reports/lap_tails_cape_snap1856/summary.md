# error-tails diagnostics -- piff_cape_gjs_lap (val, filtered gaussian_jonly_ylp75)

| member | rmse | rel_rmse | n_px | n_gt_10x | frac_gt_10x | worst0.1pct_SS_share | r2_active_q90 | r2_quiet_q50 | rmse_active | rmse_quiet | near_zero_px_share | extremes_near_body | Re_corr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FPCape-const | 0.07839 | 0.122 | 6918032 | 12564 | 1.82e-03 | 65.8% | 0.985 | -363.07 | 0.2471 | 0.0011 | 91.1% | 76% | undef(const) |
| FPCape-sine | 0.31036 | 0.308 | 6918032 | 4117 | 5.95e-04 | 91.8% | 0.905 | -307.944 | 0.9805 | 0.0017 | 90.7% | 100% | -0.43 |
| FPCape-ramp | 0.09578 | 0.136 | 6918032 | 11914 | 1.72e-03 | 67.0% | 0.981 | -164.771 | 0.3019 | 0.0011 | 90.5% | 82% | undef(const) |
| FPCape-ou | 0.09593 | 0.128 | 6918032 | 11407 | 1.65e-03 | 71.6% | 0.984 | -275.862 | 0.3026 | 0.0011 | 91.4% | 94% | -0.058 |
| FPCape-tel | 0.09366 | 0.132 | 6918032 | 12628 | 1.83e-03 | 67.1% | 0.983 | -235.677 | 0.2954 | 0.0011 | 91.3% | 72% | undef(const) |

figures: pngs/error_tails_diag/piff_cape_gjs_lap_snap1856
per-member yaml/csv: runs_piff/piff_cape_gjs_lap/error_tails_diag_snap1856
