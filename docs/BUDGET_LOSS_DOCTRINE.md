# The budget-loss doctrine (EnsCon: why it works, where it's valid, what to watch)
*Recorded 2026-07-23 (global supervisor session, with Sanaa). Companion to
LOCALITY_DOCTRINE.md. Binding context for the V3CNN launch and every later
budget-term decision on the spatial track.*

**The question:** why does a scalar penalty on the interscale enstrophy
transfer, ens = mean[(T-hat - T*)^2] with T = <omega_bar * Pi> per crop, give a
closure the right enstrophy budget — and does the argument survive outside
forced homogeneous turbulence?

**1. The mechanism.** Decompose the prediction error eps = Pi_hat - Pi* into
eps_par (along omega_bar) + eps_perp. The budget error IS the parallel
component: T_hat - T* = <omega_bar eps>. Online, eps_perp dithers the budget
around zero; eps_par is SECULAR — same sign every step, integrates over a
rollout. Blowup and drift come from the coherent component, not the large-but-
incoherent one.

**2. MSE cannot fix this and actively causes it.** MSE charges ||eps||^2,
blind to direction — equal-MSE near-minimizers differ freely in where eps
points. Worse: the MSE-optimal map is the conditional mean, which under
irreducible variance SHRINKS (regression dilution; measured on
piff_fpc_cnn_v1: amplitude ratio 0.92-0.95). With Pi_hat ~ lambda Pi*,
lambda < 1:  <omega_bar eps> = (lambda-1) T*  — sign-definite, systematically
under-dissipative, everywhere, every step. Pattern-optimal and budget-correct
genuinely disagree; shrinkage is the optimizer doing its job.

**3. What the penalty does.** d(ens)/d(Pi_hat(x)) = 2(T_hat - T*) *
omega_bar(x) * mask / N_px — a rank-one constraint per crop, gradient along
the omega_bar pattern scaled by the budget mismatch. It cannot sculpt spatial
detail (MSE's job); it re-prices exactly the one direction that accumulates in
time. One scalar against thousands of orthogonal DOF: the network keeps its
variance-hedging in eps_perp (free, harmless) and restores amplitude along
omega_bar — near-zero MSE cost. The penalty adds no information; it removes
MSE's structural incentive to spend error in the secular direction.

**4. Validity for flow-past-obstacle (the FHIT objection, adjudicated).**
The budget mechanism is local and flow-agnostic: over any control area,
d/dt<omega_bar^2/2> picks up <omega_bar Pi> plus boundary fluxes. The per-crop
implementation makes the constraint regional (wake crops balance in the wake,
shear-layer crops in the shear layer; far-field crops degenerate to a
don't-inject guard) — it never relied on homogeneity or cascade arguments.
What does NOT transfer from the FHIT literature: Jakhar's Table-6 priority
ranking ("energy transfer is THE stability-critical budget in 2D" — doubly-
periodic evidence; in wakes the large-scale energy budget is dominated by
shedding/body/sponge terms), and inverse-cascade framing generally.

**5. The energy term (-<psi_bar Pi>): demoted, decision deferred to data.**
Two technical faults: per-crop <psi_bar Pi> is gauge-sensitive (psi defined up
to a constant; <Pi> != 0 on a crop — psi must be crop-demeaned) and demeaning
caps its 1/k^2 low-k supervision at the crop scale. Decision rule: run the
budget-error diagnostic on existing v1/v2 eval residuals — signed
<omega_bar eps> and <psi'_bar eps> (crop-demeaned) per (member x region:
near-wall / wake / far). The energy term earns a flag ONLY if its budget error
is large next to the enstrophy one in the wake band. Do not add it on FHIT
priors.

**6. Implementation facts + launch checklist (train_cnn.py:276-285).**
loss = (1-beta)*MSE + beta*ens, frozen physics target, --enscon-beta (0.1 in
the V3 chain), standardized units, valid-pixel mask, per-crop.
- beta is dimensionally arbitrary: read ens/((1-beta)*MSE) in the FIRST
  epochs — if << 1 the term is inert (raise beta), if >~ 1 it eats the
  accuracy gradient. Same lesson as the vn/anchor scaling law.
- ens_log is an optimizer signal, NOT physics: recompute transfer ratios
  T_hat/T* in PHYSICAL units, per member, for the reporting table (per-member
  rule applies — never pooled).
- Tri-objective: transfer ratio -> 1 must arrive WITH unchanged-or-better
  R2-all and near/wake/far per-pixel medians, same table.
- V3 confound: psi-input and EnsCon turn on together. If the low-k coherence
  gap closes, attribution needs one ablation (psi on, --enscon-beta 0) after
  the main run; a scalar-per-crop budget term is unlikely to manufacture low-k
  coherence, so the prior is psi gets the credit.

**7. Staged upgrades (in order).** (a) Sign-split: match Sum(omega_bar Pi)_+
and Sum(omega_bar Pi)_- separately — two rank-one constraints, kills the
diffusion/backscatter cancellation channel. (b) Energy term, only per the
Section-5 decision rule. Neither edits the fire-ready V3 chain.

**Honesty clause.** The penalty enforces the budget statistically, on training
crops, in distribution. It removes the one known, measured, sign-definite
drift mechanism a priori; online sufficiency stays empirical — that is what
the GO/NO-GO run decides.
