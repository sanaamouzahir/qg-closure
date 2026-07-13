CODES -- every script involved in the Pi_FF machine-learning closure.

NOTE ON SYMLINKS: everything here is currently a SYMLINK to the file's real
location (ml_closure/ root for python, scripts/sge/ for job scripts). The
running and queued SGE jobs (1832221/1832231/1832241 + the queued fleet)
address the files by their old paths, so the physical files CANNOT move
until the fleet lands. The physical move into this folder happens TONIGHT
after the last job finishes; the links already give the final layout.

PYTHON (the pipeline, in order of use)
  make_dataset_manifest.py  builds the canonical dataset manifest from the raw
                            simulation outputs (which frames, masks, inlet-speed
                            tables) -- step 0 before any training
  dataset_piff.py           loads the data: filtered vorticity/velocity/distance
                            fields as inputs, sub-grid forcing Pi as target;
                            applies the valid-pixel mask and wake-biased crops
  model_piff.py             the model itself: CNN feature extractor (FiLM-
                            conditioned on inlet speed) + sparse Gaussian
                            process head -> per-pixel prediction AND uncertainty
  train_piff.py             trains the model; writes checkpoints, training
                            curves, per-epoch metrics
  eval_piff.py              evaluates a checkpoint on full frames: field panels,
                            calibration figure, summary numbers
  replot_eval_fields.py     re-draws the field panels with a log-like color
                            scale (the linear panels hid the wake structure)
  calibrate_piff.py         post-hoc uncertainty recalibration: fits one scale
                            factor on half the validation window, tests on the
                            other half; before/after figure
  diagnose_piff.py          the deep diagnostics suite: per-member reliability,
                            skill by inlet speed, uncertainty vs flow sharpness,
                            residual spectra, coverage by distance from the
                            obstacle, error drift over the shedding cycle,
                            per-member table
  t6_arm.py                 one arm of the T6 learning-capability ladder
                            (likelihood/capacity/noise-model probes)
  tests_piff.py             pytest suite for the pipeline (T1-T6 test tiers)
  triage_ab_filter.py       2026-07-13 triage: recomputes Pi with a Gaussian
                            filter instead of the sharp cutoff to test whether
                            the vertical streaks are benign filter ringing
  triage_mask_audit.py      2026-07-13 triage: maps which pixels are excluded
                            from training and why (sponge strips / body)
  triage_sigma_decomp.py    2026-07-13 triage: decomposes predicted uncertainty
                            into its model components
  triage_streak_quant.py    2026-07-13 triage: quantifies streak amplitude in
                            the Pi fields
  triage_compile_check.py   2026-07-13 triage: quick import/compile sanity check

SGE JOB SCRIPTS (cluster wrappers; submit_* are the launchers)
  piff_step0_job.sh         runs make_dataset_manifest.py on the cluster
  piff_train_job.sh         GPU trainer wrapper around train_piff.py
  piff_eval_job.sh          GPU eval wrapper around eval_piff.py
  piff_lomo_job.sh          runs the leave-one-inlet-out folds back to back
  piff_t6arm_job.sh         runs one T6 ladder arm
  piff_smoke_job.sh         2-epoch end-to-end smoke test (train + eval)
  piff_tests_job.sh         runs the pytest suite (cpu and gpu tiers)
  piff_monitor_job.sh       live/final training-log monitor
  piff_tool_job.sh          generic wrapper to run any ml_closure script on a
                            compute node (no python on the login node)
  submit_piff_grid.sh       launcher: the 6-rung learning-rate x weight-decay grid
  submit_piff_ens.sh        launcher: the two conditioned ensembles + evals +
                            recalibrations + monitors (the current fleet)
  submit_piff_capeA.sh      launcher: the cape Pi_FF data-generation chain
  submit_piff_telSA.sh      launcher: the smoothed-telegraph member data chain
  submit_piff_wave2.sh      launcher: the 4 modulated-inlet members data wave
