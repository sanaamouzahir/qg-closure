==========================================================================
YOUR DIAGNOSTICS LIBRARY -- ONE-PAGE MAP            (reorganized 2026-07-13)
==========================================================================

START IN pngs/.  Every folder there is ONE QUESTION we asked about the
models. Open the folder, look at the pictures, and read the single .txt
file inside -- it explains every figure in plain English: what it shows,
what went in, what it compares, the formula if there is one, and why we
made it. No code reading required.

THE QUESTIONS (pngs/, one line each):
  training_curves/                    did each training run learn healthily?
  field_panels_truth_vs_prediction/   does the prediction LOOK like the truth?
  field_panels_truth_vs_prediction_symlog/  same frames, log-like color scale
                                      (the wake is invisible on linear)
  calibration_reliability_and_spread_skill/  when the model says "I'm sure",
                                      is it telling the truth?
  sigma_recalibration_before_after/   what one global scale factor fixes
  cross_member_generalization_of_single_member_model/  the steady-inlet
                                      model thrown at inlets it never saw
  coverage_by_distance_from_obstacle/ is it honest near the body?
  skill_binned_by_inlet_speed/        is it good at all speeds?
  model_uncertainty_vs_flow_sharpness/ does sigma know where errors live?
  residual_kurtosis_and_spectra/      at which scales is the error?
  prediction_drift_across_shedding_cycle/  do errors travel with vortices?
  inlet_speed_lengthscale_training_trajectories/  did it learn to USE the
                                      inlet speed?
  reynolds_number_trace_of_evaluation_snapshots/  which regimes the field
                                      panels sample
  pi_filter_sharp_vs_gaussian_streak_check/   (triage) are the streaks
                                      filter ringing or a bug?
  valid_pixel_mask_sponge_audit/      (triage) which pixels we exclude, why
  uncertainty_decomposition_noise_vs_gp_variance/  (triage) what sigma is
                                      made of

yamls/ MIRRORS pngs/ WITH THE NUMBERS: evaluation summaries, training
summaries, run manifests, recalibration records, deep-diagnostic dumps.
Every file has a plain-English header at the top -- open any yaml and the
first thing you read is what it is.

csvs_and_npz/ = the tables and raw arrays under full descriptive names.
Every .npz has a sibling .txt naming its arrays.
README_what_the_csv_columns_mean.txt decodes every csv column.

train_models/ = EVERY MODEL WE HAVE TRAINED, one folder each, plain names:
  cylinder_steady_inlet_only_150epochs_PRODUCTION   <- the promoted one
  cape_five_inlet_types_baseline_100epochs
  cylinder_five_inlet_types_conditioned_IN_TRAINING  (lands tonight)
  cape_five_inlet_types_conditioned_IN_TRAINING      (lands tonight)
  cape_leave_one_inlet_out_folds_IN_TRAINING         (lands tonight)
  hyperparameter_grid_and_t6_ladder_archive  <- how we chose the settings,
        with the full verdict table in its README
Each folder: checkpoints/ (link to the weights) + code_and_config_used/
(EXACTLY the code and config that produced that model, config explained in
English at the top) + a README with the verdict.

codes/ = every script, with a README saying in one line what each does.

CONVENTION.md = the rule that keeps it this way: every future diagnostic
writes directly into this tree, plots first, English first.

ONE TEMPORARY NOTE: three models are still training today; their folders
say IN_TRAINING and their partial files carry an _in_training suffix.
Tonight, after the last job lands, the finals drop in and the code files
physically move into codes/ (they are symlinks until then because the
running jobs pin the old paths).
==========================================================================
