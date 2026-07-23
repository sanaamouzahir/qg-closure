# The Pi-omega locality doctrine (and its one escape)
*Recorded 2026-07-23 (global supervisor session, pre-station-move). The lens for
reading every SGS-CNN architecture paper in the phase-4+ improvement campaign.*

**The locality question:** how far away does the coarse field influence the SGS
forcing at a point? It is the single number that decides how deep a
convolutional closure needs to be.

**1. The definition.** Pi(x) is a functional of the resolved fields. If, to good
approximation, Pi(x) depends only on omega_bar, u_bar within a radius r of x,
then a CNN whose receptive field (RF) covers r captures everything learnable —
and extra RF buys nothing but parameters, overfitting surface, and cost. RF for
a plain conv stack = n_layers x (kernel-1) + 1. Ours: 3 x 4 + 1 = **13 coarse
points**. Guan et al. (arXiv:2201.07347): 10 x 4 + 1 = **41**.

**2. Why Pi should be local, physically.** Pi is a filter-nonlinearity
commutator: filter[J(psi,omega)] - J(psi_bar,omega_bar). Two ingredients, both
short-ranged: the filter kernel is compact (Gaussian, width ~2 Delta_LES), and J
is a local differential operator. The energy in Pi comes overwhelmingly from
interactions near the filter cutoff — small scales strained by slightly-larger
scales, all living within a few coarse cells. The measurement behind our number
(Srinivasan et al. 2024; the FiLMCNN docstring's "measured Pi-omega locality
~7 coarse points") is exactly this: the statistical dependence of Pi(x) on
omega_bar(x') decays to noise within ~7 grid points. RF 13 covers +-6 — sized to
that measurement, with the standing rule "do NOT deepen without a B-item".

**3. The one escape from locality: psi.** psi_bar = inv-Laplacian(omega_bar) is
a GLOBAL integral of the vorticity. Anything in Pi that depends on the
large-scale flow configuration (wake envelope position, ambient strain from
distant vortices) enters through psi, and no finite RF on omega_bar can
reconstruct it. This is precisely why the spectral diagnostic found the v1/v2
models blind at k <~ 15 (coherence ~0, hallucinated energy at scales larger
than the RF) while excellent in the mid-band. The locality measurement and the
low-k blindness are two views of one fact: **Pi is local GIVEN psi, not local
in omega_bar alone.**

**4. What the two experiment lines say.** Guan's sensitivity (<8 layers loses
correlation, >12 overfits) argues RF ~30-40 — but in homogeneous turbulence,
with {psi_bar, omega_bar} inputs, no walls. Our v1->v2 record argues most of Pi
is local: R2 0.95 (cylinder) / 0.99 (cape) at RF 13 once capacity + the
laplacian input were present. The residual disagreement concentrates exactly
where theory predicts: large-scale / low-k content.

**5. Why V3CNN is the clean test.** Three ways to buy nonlocality:
(a) grow RF — brute force; pays Guan's overfitting tax and online cost;
(b) feed the nonlocal information as an INPUT — psi_bar as a channel delivers
global content as a local value, RF untouched;  (c) architectural global paths
(U-Net, Fourier layers, attention). V3 runs (b). If psi_bar closes the low-k
coherence gap at RF 13, the doctrine survives amended ("local given psi") and
we never pay for depth. If low-k blindness persists WITH psi_bar, it is
genuinely an architecture problem, and the depth B-item (or a U-Net-style
path) earns its phase-4 slot.

**The reading lens for new papers:** every "deeper / wider / global-attention"
choice in the literature is either compensating for missing psi-type inputs, or
evidence about what remains once you have them — and only the second kind
should cost us receptive field.
