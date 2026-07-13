CSV COLUMN DICTIONARY

1) *_per_member_diagnostics_table.csv  (one row per simulation member)
   member            which simulation member (inlet type)
   n_val_frames      validation frames evaluated
   r2_global         R^2 over all valid pixels (1 = perfect)
   rmse_global       root-mean-square error, physical units
   nll_global        negative log-likelihood (lower = better; skill AND
                     honesty of the uncertainty together)
   cov_1s/2s/3s      fraction of pixels whose true value falls within
                     1/2/3 predicted sigma (ideal: 0.68 / 0.95 / 0.997)
   cov1_wake, cov1_freestream   1-sigma coverage split into wake region vs
                     free stream
   r2_wake, r2_freestream       R^2 split the same way
   residual_kurtosis heavy-tailedness of the errors (3 = Gaussian)
   sign_acc          fraction of pixels where the PREDICTED SIGN of the
                     sub-grid forcing is correct
   sign_acc_wake     same, wake pixels only
   sign_acc_strong   same, only pixels with |truth| above the median
                     (the pixels that matter)
   sign_acc_wake_strong  both restrictions
   backscatter_frac_true fraction of pixels where the TRUE forcing is
                     negative (energy flowing back to resolved scales) --
                     how much backscatter there is to get right
   spearman_sigma_grad   rank correlation of predicted sigma with the local
                     vorticity gradient (does uncertainty track sharpness)
   spearman_abserr_grad  rank correlation of |error| with the gradient
                     (what sigma SHOULD look like)
   sigma_dyn_range   dynamic range of predicted sigma (max/min-like);
                     compare with...
   abserr_dyn_range  dynamic range of the actual errors. If abserr range is
                     much bigger than sigma range, the uncertainty is too
                     flat -- the known per-pixel SHAPE miscalibration.

2) *_per_epoch_validation_metrics_and_inlet_speed_lengthscale.csv
   (one row per training epoch, parsed from the trainer log)
   epoch      epoch number
   val_nll    validation negative log-likelihood
   val_rmse   validation RMSE
   val_r2     validation R^2
   val_sigma  mean predicted uncertainty
   zeta_ls    learned lengthscale of the inlet-speed feature (0.6931 =
              still at initialization = feature not being used yet)
