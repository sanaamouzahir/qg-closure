# Acceptance predictions — ORDER-3 ensemble training (written BEFORE submission)

Recorded 2026-07-13, per Sanaa's ORDER 3d (I16-style: predictions first, then
the run; a prediction that fails is a FINDING to report, not a failure to hide).

Runs covered: `piff_fpc_ens` (conf_piff_fpc_ens.yaml — 5 FPC members, conditioned)
and `piff_cape_cond` (conf_piff_cape_cond.yaml — 5 cape members, conditioned).
Baseline rows: prod_ext150 cross-member eval (FPC, before-picture) and
cape_base_100ep (cape, unconditioned 5-member baseline).

## P1 — zeta identifiability on FPC multi-member

zeta ARD lengthscale comes OFF the 0.6931 init on `piff_fpc_ens`
(cape precedent: 2.016 at best-ckpt, 2.31 at ep99; Order-2 diagnostics
confirmed frozen-at-ln2 is a DATA property of const-only training, not a
model defect). **If zeta_ls stays at 0.6931 through the run: conditioning is
inert on FPC — FLAG, investigate before any downstream claim.**

## P2 — zeta_dot lengthscale finite = lag physics is real

zeta_dot ARD lengthscale stays FINITE (does not run to the >1e2 prune regime)
on at least one geometry. Mechanism: ramp-up vs sine-down at equal zeta are
different wake states; only zeta_dot separates the modulation classes.
**If it prunes to inf on FPC but not cape (or vice versa): report as a
finding about which geometry's wake carries usable lag memory at scale s4.**
Secondary: grad-feature lengthscale expected to stay finite everywhere
(Order-2 diagnostics: error varies 39.6x across |grad omega| deciles while
sigma is flat 1.0002x — the kernel has 40x of structure to absorb).

## P3 — per-member R2 spread shrinks vs the single-member ckpt

The cross-member eval of prod_ext150 (trained FPC-const only) will show a
WIDE per-member R2 spread (const ~0.86 in-dist; modulated members degraded,
OOD per I7). `piff_fpc_ens` per-member R2 spread (max-min over the 5 members)
SHRINKS vs that baseline row, at const R2 within ~0.03 of prod_ext150's
0.8584 (joint training may cost the control a little; more than ~0.03 is a
capacity finding). Numbers to fill at landing: baseline spread ____ ->
ensemble spread ____.

## Guard predictions (from the arm ladder + Order-2 diagnostics)

G-a: no ELBO collapse family (feature-spread PLAN-B symptom stays silent;
     R2 curve monotone-ish as in every healthy arm).
G-b: cape conditioned run's val NLL <= cape_base_100ep's 6.493 (the two new
     ARD dims cannot hurt a well-conditioned kernel; if NLL worsens >0.1,
     suspect the grad-feature normalization).
G-c: T6 re-gated bar (R2 >= 0.85 in 100 ep) passes on piff_fpc_ens; if it
     lands in [0.80, 0.85) the pool is harder than FPC-const alone — report
     against the bar with per-member breakdown, no silent re-gate.

## ADDENDUM (14:45 EDT) — Gaussian-target redo

Sanaa ruling: sharp filter abandoned; the runs above (piff_fpc_ens /
piff_cape_cond, killed at ~ep 15/16) are replaced by piff_fpc_ens_gauss /
piff_cape_gauss on DNS_LES_s4_gaussian.npz targets. P1-P3 and guards carry
over UNCHANGED (they are about conditioning, not the filter), with two notes:
(a) cross-convention metric comparisons are not apples-to-apples (different
target variance) — P3's spread comparison uses the gaussian-model xevals
against the prod_ext150 xevals as a GENERALIZATION-SHAPE comparison, stated
with that caveat; (b) NEW prediction P4-gauss: freestream coverage moves
toward nominal vs the sharp-trained twins at matched epoch (the ringing was
inflating the noise floor); wake coverage roughly unchanged. Early sharp-run
evidence preserved in the killed runs' logs: all three conditioning
lengthscales were alive by ep 13 (zeta 1.9, zeta_dot 2.2, grad 2.7).
