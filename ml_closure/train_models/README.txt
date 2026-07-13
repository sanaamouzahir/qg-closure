TRAINED MODELS -- one folder per model, plain-English names.

Inside each folder:
  checkpoints/            -> symlink to the live run directory under runs_piff/
                             (best.pt = best checkpoint, last.pt = latest;
                             plus the run's metrics/curves/eval artifacts)
  code_and_config_used/   REAL COPIES of the training config (with an English
                          header explaining it) and of train_piff.py /
                          model_piff.py / dataset_piff.py as of the commit
                          the model was trained from (stated in each README).

Models:
  cylinder_steady_inlet_only_150epochs_PRODUCTION  the promoted production model
  cape_five_inlet_types_baseline_100epochs         cape baseline, 5 inlet types
  cylinder_five_inlet_types_conditioned_IN_TRAINING  conditioned cylinder ensemble
  cape_five_inlet_types_conditioned_IN_TRAINING      conditioned cape ensemble
  cape_leave_one_inlet_out_folds_IN_TRAINING         generalization folds
  hyperparameter_grid_and_t6_ladder_archive          grid rungs, t6 arms, smokes
