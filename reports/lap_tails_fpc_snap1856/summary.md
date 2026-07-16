# error-tails diagnostics -- piff_fpc_gjs_lap (val, filtered gaussian_jonly_ylp75)

| member | rmse | rel_rmse | n_px | n_gt_10x | frac_gt_10x | worst0.1pct_SS_share | r2_active_q90 | r2_quiet_q50 | rmse_active | rmse_quiet | near_zero_px_share | extremes_near_body | Re_corr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FPC-const | 0.19001 | 0.296 | 9990888 | 3857 | 3.86e-04 | 94.9% | 0.912 | -4423.758 | 0.6008 | 0.0014 | 94.9% | 100% | undef(const) |
| FPC-sine | 0.23585 | 0.342 | 5940528 | 2606 | 4.39e-04 | 93.4% | 0.883 | -4235.439 | 0.7457 | 0.0014 | 94.8% | 100% | -0.282 |
| FPC-ramp | 0.28856 | 0.386 | 5940528 | 2389 | 4.02e-04 | 95.9% | 0.851 | -1682.758 | 0.9124 | 0.0012 | 94.6% | 100% | undef(const) |
| FPC-ou | 0.22555 | 0.322 | 5940528 | 2812 | 4.73e-04 | 94.2% | 0.897 | -3241.68 | 0.7132 | 0.0013 | 95.1% | 100% | 0.117 |
| FPC-telS-A | 0.32029 | 0.39 | 5940528 | 2424 | 4.08e-04 | 96.1% | 0.848 | -1642.622 | 1.0128 | 0.0012 | 95.4% | 100% | undef(const) |

figures: pngs/error_tails_diag/piff_fpc_gjs_lap_snap1856
per-member yaml/csv: runs_piff/piff_fpc_gjs_lap/error_tails_diag_snap1856
