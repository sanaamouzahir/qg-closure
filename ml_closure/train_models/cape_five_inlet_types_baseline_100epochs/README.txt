CAPE BASELINE MODEL (run codename: cape_base_100ep)

What it is: the unconditioned cape baseline. CNN + SVGP predicting
sub-grid-scale forcing Pi with per-pixel uncertainty, trained on ALL FIVE
cape inlet types pooled (constant, sine, ramp, Ornstein-Uhlenbeck,
telegraph), 100 epochs, lr 1.0e-3, wd 1.0e-5.

Verdict: validation R^2 0.800, NLL 6.25. First run where the inlet-speed
feature (zeta) became identifiable (ARD lengthscale 2.016, off its 0.6931
init) -- prerequisite for the leave-one-inlet-out study. Cape residuals are
~5x heavier-tailed than cylinder.

checkpoints/ -> ../../runs_piff/cape_base_100ep (best.pt, last.pt, eval/,
diagnostics/).

code_and_config_used/: config + train/model/dataset code extracted at commit
bb2dfc1 (the 2026-07-12 night-burst state this model was trained from).
