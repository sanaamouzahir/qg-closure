CONDITIONED CYLINDER ENSEMBLE -- IN TRAINING (run codename: piff_fpc_ens)

What it is: one model for all five cylinder inlet types (constant, sine,
ramp, Ornstein-Uhlenbeck, smoothed-telegraph), CONDITIONED on inlet motion:
extra features zeta_dot (rate of change of inlet speed, via FiLM + ARD) and
|grad omega_bar| (local vorticity-gradient magnitude, via ARD). 150 epochs.

Status 2026-07-13: TRAINING NOW (SGE job 1832221 fleet). checkpoints/ holds
the partial artifacts (best.pt/last.pt so far, metrics.npz, run_info.yaml);
eval + diagnostics land tonight.

Question it answers: can one conditioned model serve every inlet type as
well as the single-inlet production model serves its own.

code_and_config_used/: config + code = current working copies (commit
ddd811b for train_piff.py, 017bd52 for model/dataset -- the state this run
launched from).
