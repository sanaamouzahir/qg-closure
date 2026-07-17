# mean-prediction diagnostics -- piff_cape_gjs_lap (val, filtered variant gaussian_jonly_ylp75)

| member | n_frames | r2 | rmse | bias | max_err | min_err | pearson_Re_rmse | corr_ae_sdf | corr_ae_y | L_error | L_truth | L_pred | pred_truth_corr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FPCape-const | 44 | 0.9844 | 0.08045 | +6.939e-04 | 7.9891 | -12.0574 | nan | -0.154 | -0.123 | 0.0453 | 0.0582 | 0.0586 | 0.9922 |
| FPCape-sine | 44 | 0.9186 | 0.28781 | -7.492e-05 | 160.3526 | -110.1101 | -0.422 | -0.079 | -0.064 | 0.0407 | 0.0569 | 0.0569 | 0.9587 |
| FPCape-ramp | 44 | 0.9807 | 0.0975 | -6.103e-06 | 13.8001 | -12.5946 | nan | -0.158 | -0.123 | 0.0437 | 0.0571 | 0.0575 | 0.9903 |
| FPCape-ou | 44 | 0.983 | 0.09781 | +5.150e-04 | 10.6832 | -14.5535 | -0.008 | -0.149 | -0.111 | 0.0337 | 0.0553 | 0.0559 | 0.9916 |
| FPCape-tel | 44 | 0.982 | 0.09517 | -2.118e-04 | 10.8209 | -13.1936 | nan | -0.147 | -0.112 | 0.0341 | 0.0574 | 0.0577 | 0.9909 |

figures: /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-sgs-closure/ml_closure/pngs/mean_prediction_diag/piff_cape_gjs_lap
per-member yaml/csv: runs_piff/piff_cape_gjs_lap/mean_prediction_diag
