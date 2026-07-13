PRODUCTION MODEL (run codename: prod_ext150)

What it is: the promoted cylinder model. CNN + SVGP predicting sub-grid-scale
forcing Pi with per-pixel uncertainty, trained ONLY on the steady-inlet
cylinder member (FPC-const), 150 epochs at the grid-winner settings
(lr 1.0e-3, wd 1.0e-5).

Verdict: PASSED the promotion bar. Full-frame validation R^2 0.8584
(bar 0.85), NLL 5.16; after sigma recalibration (scale 0.521) NLL 4.91.
Known limitation: sigma miscalibration is per-pixel SHAPE, not overall scale
(coverage stuck near 0.95 after recalibration).

checkpoints/ -> ../../runs_piff/prod_ext150 (best.pt, last.pt, eval/,
diagnostics/, xeval/ = cross-member generalization tests).

code_and_config_used/: config + train/model/dataset code extracted at commit
bb2dfc1 (the 2026-07-12 night-burst state this model was trained from).
