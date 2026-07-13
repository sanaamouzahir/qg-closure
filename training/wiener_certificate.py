"""
wiener_certificate.py -- frozen-coefficient von Neumann amplification of the
CLOSED AB2CN2 scheme, differentiable in the model's stencil taps (P1(b) of
Sanaa's 2026-07-13 plan; design: P1_DESIGN_2026-07-13.md).

Per sample and per radial shell kappa, assemble

    I(k) = L_hat(k) + cR * L_hat(k)^3                      [implicit: CN + R^-1 fold]
    E(k) = i*sig(k) * (1 + cS * L_hat(k)^2)                [frozen advection + S^-1 L^2N fold]
         + cN * (1/12) * (L_hat(k) * T1(k) - 5 * T2(k))    [learned closure transfer]

with cR = cS = cN = DT^3/... reduced to the ROLLOUT convention: the closure
enters the step as an explicit term of magnitude DT^3*(1/12)(L Ndot - 5 Nddot)
(rollout_aposteriori closure arm; NO (1-1/K^2) — RK4-truth world, I4), so in
G_eff the fold constants are cR = cS = DT^2/12 (the DT^3 term divided by the
DT the scheme multiplies E by) and cN likewise absorbs DT^2/12.

T_c(k) is the LINEARIZED transfer of the learned channel c in the Wiener
freeze (error_analysis_full.tex sec. 7): time-orders frozen to the exact
relation w^(j) = D^j q with D = L_hat + i sig; psi-side backgrounds to D^i;
every Jacobian product to i*sig; and the SPATIAL STENCILS entering as their
trig-polynomial symbols evaluated isotropically (theta = k*dx/sqrt(2) per
direction):

    T_c(k) = sum_{ij} mix[c, ij] * i*sig(k) * rho(k) * D(k)^(i+j)
    rho(k) = What_mod+base(theta) / What_exact(theta)   [stencil symbol ratio,
             carries BOTH the base taps and the per-sample modulated deltas --
             the differentiable path to the TapModulator]

CALIBRATION ANCHOR: with exact stencils and zero modulation, rho == 1 and
T reduces to the frozen ANALYTIC closure transfer -- the certificate then
reproduces the analytic-arm stability, which the ladder showed is genuinely
stable. Deviations of |G| above 1 are therefore attributable to the learned
taps. HONEST CAVEATS (documented in the SUBMIT email): linearized (r2/r3
neglected), isotropic-shell evaluation, background factors O(1)-normalized.
P0's dissipative projection stays the unconditional inference backstop.

AB2CN2 companion per shell (frozen E):
    r = 1 / (1 - DT/2 * I);  a = r*(1 + DT/2 * I);  b = 1.5*DT*r*E;  c2 = -0.5*DT*r*E
    G_eff(k) = spectral radius of [[a+b, c2], [1, 0]]
             = max |((a+b) +- sqrt((a+b)^2 + 4*c2)) / 2|

Loss:  L_stab = lam * mean_k relu(|G_eff| - (1-eps))^2 ,  eps = 0.02.
"""

import math

import torch

EPS_MARGIN = 0.02


def tap_symbol(taps: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Trig polynomial of centered 1D taps. taps (..., W), theta (n_sh,)
    -> complex (..., n_sh). Convention matches F.conv2d cross-correlation
    with circular pad: sum_j w_j e^{+i (j - pad) theta}."""
    W = taps.shape[-1]
    j = torch.arange(W, dtype=theta.dtype, device=theta.device) - (W // 2)
    ph = torch.exp(1j * j.view(-1, 1) * theta.view(1, -1))     # (W, n_sh)
    return torch.einsum('...w,ws->...s', taps.to(ph.real.dtype) + 0j, ph)


def assemble_geff(model, sig: torch.Tensor, dt: torch.Tensor,
                  L_hat_sh: torch.Tensor, kappa_sh: torch.Tensor,
                  dx: torch.Tensor, delta_taps: torch.Tensor = None):
    """|G_eff| (B, n_sh), differentiable through model.grad taps and
    delta_taps (the per-sample TapModulator output, ALREADY amp-scaled).

    sig      (B, n_sh)  raw sigma-hat per shell (sigma_hat_spec output)
    dt       (B,)       per-sample Delta_T
    L_hat_sh (n_sh,)    complex L_hat at shell-representative |k| (radial mean)
    kappa_sh (n_sh,)    physical |k| per shell
    dx       (B,)       per-sample grid spacing (square domains)
    """
    B, n_sh = sig.shape
    dtypec = torch.complex128
    dev = sig.device
    no = model.n_ord
    Lh = L_hat_sh.to(dev).to(dtypec).view(1, -1)               # (1, n_sh)
    sg = sig.to(dev).to(torch.float64)
    D = Lh + 1j * sg                                            # (B, n_sh)

    # ---- stencil symbol ratio rho(k): (base + delta) / exact ------------- #
    theta = (kappa_sh.to(dev).to(torch.float64)
             * dx.to(dev).to(torch.float64).view(-1, 1) / math.sqrt(2.0))  # (B, n_sh)
    # base taps: SpatialGrad dimensionless x-taps, channel-shared row 0.
    # NOT detached (G4 C2): the certificate is differentiable through the
    # base stencils too -- they train in the FT and must feel the penalty.
    base_x = model.grad.wx.to(dev)
    base_x = base_x.reshape(-1, base_x.shape[-1])[0]            # (W,)
    W = base_x.shape[-1]
    j = (torch.arange(W, dtype=torch.float64, device=dev) - W // 2).view(1, 1, -1)
    ph = torch.exp(1j * j * theta.unsqueeze(-1))                # (B, n_sh, W)
    sym_base = torch.einsum('w,bsw->bs', base_x.to(dtypec),
                            ph.to(dtypec))                      # (B, n_sh)
    exact = 1j * theta.to(dtypec)                               # ik in tap units
    if delta_taps is not None:
        # delta (B, C=2no, 2, W) -> per-channel x/y symbols; isotropic mean
        d = delta_taps.to(dev).to(dtypec)                       # (B, C, 2, W)
        sym_d = torch.einsum('bcdw,bsw->bcds', d, ph.to(dtypec)).mean(dim=2)
    else:
        sym_d = torch.zeros(B, 2 * no, n_sh, dtype=dtypec, device=dev)
    denom = exact + (exact.abs() < 1e-30) * 1.0
    # channel-resolved rho: omega block rows 0..no-1, psi block no..2no-1
    rho = (sym_base.unsqueeze(1) + sym_d) / denom.unsqueeze(1)  # (B, 2no, n_sh)

    # ---- per-order learned transfer T_c ---------------------------------- #
    mixw = model.mix.weight.squeeze(-1).squeeze(-1).to(dev)     # (3, n_jac)
    T = []
    for c in range(mixw.shape[0]):
        acc = torch.zeros(B, n_sh, dtype=dtypec, device=dev)
        f = 0
        for i in range(no):          # psi order (background)
            for jj in range(no):     # omega order (perturbation)
                rho_ij = 0.5 * (rho[:, no + i] + rho[:, jj])    # both sides' taps
                acc = acc + mixw[c, f].to(dtypec) * (1j * sg.to(dtypec)) \
                    * rho_ij * D ** (i + jj)
                f += 1
        T.append(acc)

    # ---- fold constants (rollout convention: coef = DT^3, no 1-1/K^2) ---- #
    dtv = dt.to(dev).to(torch.float64).view(-1, 1)
    cfold = (dtv ** 2) / 12.0                                   # DT^3/12 / DT
    E_sym = (1j * sg.to(dtypec)) * (1 + cfold.to(dtypec) * Lh ** 2) \
        + cfold.to(dtypec) * (Lh * T[0] - 5.0 * T[1])

    # ---- AB2CN2 companion spectral radius --------------------------------- #
    # Implicit side EXACTLY as the scheme folds it (G4 C3; rollout_
    # aposteriori denom_clos): denominator 1 - dt/2 L + dt^3/12 L^3, L^3
    # FULLY implicit (plus sign, no numerator counterpart).
    half = 0.5 * dtv.to(dtypec)
    r = 1.0 / (1.0 - half * Lh + (dtv.to(dtypec) ** 3 / 12.0) * Lh ** 3)
    a = r * (1.0 + half * Lh)
    b = 1.5 * dtv.to(dtypec) * r * E_sym
    c2 = -0.5 * dtv.to(dtypec) * r * E_sym
    tr = a + b
    disc = torch.sqrt(tr * tr + 4.0 * c2)
    g = torch.maximum(((tr + disc) / 2).abs(), ((tr - disc) / 2).abs())
    return g                                                    # (B, n_sh) real


def vn_penalty(g: torch.Tensor, lam: float, eps: float = EPS_MARGIN):
    """L_stab = lam * mean_k relu(|G|-(1-eps))^2, per-batch mean. Returns
    (loss_scalar, max_g_detached_per_sample) -- the max for the certificate
    histogram (bar 3)."""
    excess = torch.relu(g - (1.0 - eps))
    return lam * (excess ** 2).mean(), g.detach().amax(dim=1)
