# mean-prediction diagnostics -- piff_cape_gjs_wallv2 (val, filtered variant gaussian_jonly_ylp75)

| member | modulation | n_frames | r2 | rmse | bias | max_err | min_err | pearson_Re_rmse | corr_ae_sdf | corr_ae_y | L_error | L_truth | L_pred | pred_truth_corr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FPCape-const | constant_inflow | 44 | 0.9832 | 0.08336 | -2.830e-03 | 7.4079 | -9.1036 | nan | -0.167 | -0.12 | 0.0452 | 0.0582 | 0.0586 | 0.9917 |
| FPCape-sine | sine_modulation | 44 | 0.9309 | 0.26523 | +6.383e-04 | 125.7205 | -108.8754 | -0.402 | -0.1 | -0.075 | 0.0346 | 0.0569 | 0.0577 | 0.9648 |
| FPCape-ramp | ramp_modulation | 44 | 0.9757 | 0.10947 | -1.444e-03 | 16.2216 | -12.723 | nan | -0.175 | -0.129 | 0.0442 | 0.0571 | 0.0572 | 0.9882 |
| FPCape-ou | ornstein_uhlenbeck_modulation | 44 | 0.9795 | 0.1075 | -2.412e-03 | 12.0588 | -10.8485 | 0.013 | -0.159 | -0.109 | 0.0335 | 0.0553 | 0.0558 | 0.9897 |
| FPCape-tel | telegraph_modulation | 44 | 0.9768 | 0.10783 | -1.442e-03 | 10.8795 | -16.4907 | nan | -0.159 | -0.116 | 0.0449 | 0.0574 | 0.0574 | 0.9893 |

figures: /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-sgs-closure/ml_closure/pngs/mean_prediction_diag/piff_cape_gjs_wallv2
per-member yaml/csv: runs_piff/piff_cape_gjs_wallv2/mean_prediction_diag
