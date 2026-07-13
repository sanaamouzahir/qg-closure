CONDITIONED CAPE ENSEMBLE -- IN TRAINING (run codename: piff_cape_cond)

What it is: cape counterpart of the conditioned cylinder ensemble -- all
five cape inlet types, with the zeta_dot (inlet-speed rate of change) and
|grad omega_bar| conditioning features on. 150 epochs.

Status 2026-07-13: TRAINING NOW (SGE fleet). checkpoints/ holds partial
artifacts; finals land tonight.

Question it answers: do the inlet-motion features lift the 0.800 R^2 of the
unconditioned cape baseline.

code_and_config_used/: config + code = current working copies (commit
ddd811b for train_piff.py, 017bd52 for model/dataset -- the state this run
launched from).
