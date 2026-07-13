CAPE LEAVE-ONE-INLET-OUT FOLDS -- IN TRAINING (run codenames: cape_lomo_*)

What it is: five training folds; each holds out ONE cape inlet type
entirely (train on the other four, validate on the held-out one). Measures
true generalization to unseen inlet behaviour -- the honest version of the
pooled ensemble numbers.

Fold pattern: runs_piff/cape_lomo_{const,ou,ramp,sine,tel}. The LOMO job
(1832241) launches folds sequentially; a checkpoints_fold_<name> symlink is
added here as each fold's directory appears.

Status 2026-07-13: const fold running (checkpoints_fold_const/); the other
four folds have not started yet.

code_and_config_used/: all five fold configs (headered copies) + current
working-copy code (commit ddd811b / 017bd52, the launch state).
