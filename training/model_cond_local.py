"""
model_cond_local.py -- cond_local: the DELIVERABLE conditioned closure.

Physics-conditioned LOCAL closure: the cheap_deriv control pipeline with the
conditioning applied as TAP MODULATION of the width-15 learned local stencils.
This supersedes the spectral SpectralCondGrad design (cond_deriv), which is
dead as a deliverable: ~24 FFTs per inference step is the Jacobian budget the
closure exists to avoid. cond_grad.py stays in the repo as a ceiling-
measurement instrument only.

COST (the headline; measure it in benchmark_walltime.py):
    inference = control (cheap_deriv) + 2 rFFTs/step.
The ONLY spectral touch is the once-per-forward sigma-hat context
(rfft2(omega_0), rfft2(omega_0 - omega_m1)); gradients, Jacobians and the mix
are exactly the control's local ops. There is no other FFT anywhere in this
model -- asserted at runtime (self._ctx_calls == 1 per forward).
At ROLLOUT even those 2 rFFTs vanish: a spectral stepper already holds
qh_curr/qh_minus, so the driver computes the context via
context_feats_from_spectral (shell reduction only) and passes it as
forward(..., cond_feats=...) -- the cond_local NN is then conv-only,
zero transforms per step.

Design (binding, from the 2026-07-07 work order; theory: F_loc family +
sigma-hat estimator, error_analysis_full.tex sec. 7 -- the arcsin-debiased
estimator is reused VERBATIM from cond_grad.sigma_hat):

  * Control pipeline unchanged: frozen Vandermonde TimeFD (per-sample
    W_unit/dt^k path), ORDER CLIP (only time-orders 0..out_orders emitted),
    binomial physics-init 1x1 mix, width-15 learned local base stencils
    (dimensionless unit-spacing Parameters, per-sample 1/dx,1/dy at forward).
  * Conditioning = tap modulation, per channel k (time-order), per direction d:

        taps^(k)_d = taps_base^(k)_d + dT^(S-k) * Delta_taps_theta(features)

    with S = n_time = 7 and the dT^(S-k) EXACT per-channel scaling applied
    analytically (k = the time-order of the differentiated field; the Wiener
    analysis gives delta* ~ -ik C_m (dT sigma(kappa))^(S-k), so the analytic
    dT power is factored out and the network only learns the sigma-dependent
    dimensionless profile). Delta_taps lives in the same DIMENSIONLESS
    unit-spacing space as the base stencils (the per-sample 1/dx scaling is
    applied after the sum, exactly as for the base).
  * Delta_taps_theta: shared 2-layer tanh trunk + one zero-init linear head
    emitting 15 taps per (channel, direction) -- 2*n_ord channels x 2
    directions x width taps. Zero-init final layer => at init the modulation
    path contributes EXACT zeros and the model is BIT-IDENTICAL to the
    unconditioned control (conv with an all-zero kernel is exactly 0.0, and
    x + 0.0 == x in IEEE754). ~6.9k new parameters (< 10k budget).
  * Features (dimensionless, per sample): debiased sigma-hat_omega(kappa) from
    the two newest omega marks (arcsin formula), sampled at fixed RELATIVE
    shells kappa/kappa_max in {0.15, 0.3, 0.5, 0.7, 0.85}, each times dT:
        x_i = dT * sigma-hat(kappa_i),   feats = [x_i, log(1+x_i)]  (10-dim).
    Exactly ONE context computation per forward (2 rFFTs) -- asserted.
  * The modulation is a 1D tap row: the x-direction delta modulates the
    central row (the 15 taps along x) and the y-direction delta the central
    column, i.e. the same 1D subspace the FD init lives in. The base 15x15
    Parameter remains free to learn 2D structure exactly as in the control.

Selection: build_model('cond_local', ...) in model_deriv_closure.py, or
train_deriv.py --model cond_local. Same CLI surface as cheap_deriv
(--grad-kernel sets the base width).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_deriv_closure import CheapDerivClosureNet
from cond_grad import sigma_hat, sigma_hat_spec


REL_SHELLS = (0.15, 0.30, 0.50, 0.70, 0.85)


class TapModulator(nn.Module):
    """feats (B, 2*n_shells) -> Delta_taps (B, n_channels, 2, width).

    Shared trunk + one zero-init linear head (equivalent to per-(channel,
    direction) heads with shared trunk; single matrix keeps it simple and the
    parameter count identical). Zero-init head => exact zeros at init.
    """

    def __init__(self, n_feats: int, n_channels: int, width: int,
                 hidden: int = 24):
        super().__init__()
        self.n_channels = n_channels
        self.width = width
        self.trunk = nn.Sequential(
            nn.Linear(n_feats, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.head = nn.Linear(hidden, n_channels * 2 * width)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        d = self.head(self.trunk(feats))
        return d.view(-1, self.n_channels, 2, self.width)


class CondLocalDerivClosureNet(CheapDerivClosureNet):
    """cheap_deriv + sigma-hat-conditioned tap modulation (the F_loc model).

    Inherits the ENTIRE control pipeline; adds one parallel, zero-init 1D
    modulation conv per direction whose taps are predicted per sample from the
    sigma-hat features and scaled by the analytic per-channel dT^(S-k).
    """

    def __init__(self, in_channels: int = 14, out_orders: int = 3,
                 n_time: int = 7, refine_channels: int = 0,
                 learnable_stencils: bool = True, kernel: int = 3,
                 grad_kernel: int = 15,
                 dt: float = 1e-3, dx: float = 1.0, dy: float = 1.0,
                 physics_init: bool = True,
                 hidden_channels: int = 0, depth: int = 0,
                 cond_hidden: int = 24):
        super().__init__(in_channels=in_channels, out_orders=out_orders,
                         n_time=n_time, refine_channels=refine_channels,
                         learnable_stencils=learnable_stencils, kernel=kernel,
                         grad_kernel=grad_kernel, dt=dt, dx=dx, dy=dy,
                         physics_init=physics_init,
                         hidden_channels=hidden_channels, depth=depth)
        self.S = n_time                      # dT^(S-k) analytic power, S = 7
        self.width = grad_kernel
        # channels of the grad stage: [omega^(0..n_ord-1), psi^(0..n_ord-1)]
        self.cond = TapModulator(n_feats=2 * len(REL_SHELLS),
                                 n_channels=2 * self.n_ord,
                                 width=grad_kernel, hidden=cond_hidden)
        # per-channel time-order k (omega block then psi block) for dT^(S-k)
        k_of_channel = torch.arange(self.n_ord).repeat(2).to(torch.float64)
        self.register_buffer('k_of_channel', k_of_channel)
        self.register_buffer('dt0_cond', torch.tensor(float(dt)))
        self._ctx_calls = 0                  # per-forward context counter

    # ------------------------------------------------------------------ #
    @staticmethod
    def _feats_from_sig(sig, dt_vec):
        """Shell select + feature build shared by both context paths."""
        n_sh = sig.shape[1]
        idx = torch.tensor([min(int(round(r * (n_sh - 1))), n_sh - 1)
                            for r in REL_SHELLS],
                           device=sig.device, dtype=torch.long)
        x = sig.index_select(1, idx) * dt_vec.view(-1, 1)  # dimensionless
        return torch.cat([x, torch.log1p(x)], dim=1)

    def _context_feats(self, om0, om1, dt_vec, Lx, Ly):
        """The ONE spectral touch: sigma-hat at fixed relative shells.

        2 rFFTs total (inside sigma_hat: rfft2(om0), rfft2(om0-om1)).
        Returns (B, 2*n_shells) float feats [dT*sig_i, log1p(dT*sig_i)].
        """
        self._ctx_calls += 1
        sig, _ = sigma_hat(om0, om1, dt_vec, Lx, Ly)      # (B, n_sh)
        return self._feats_from_sig(sig, dt_vec)

    def context_feats_from_spectral(self, qh0, qh_diff, dt_vec, Lx, Ly, Ny, Nx):
        """Rollout-side context: sigma-hat from a stepper's OWN spectral
        states (rfft half-plane layout, any global norm -- it cancels in the
        shell ratio). Shell reduction only, ZERO FFTs. qh_diff = qh0 - qh_m1
        (== rfft2(om0 - om1) by linearity, to round-off). Pass the result as
        forward(..., cond_feats=...) to skip the internal 2-rFFT context."""
        sig, _ = sigma_hat_spec(qh0, qh_diff, dt_vec, Lx, Ly, Ny, Nx)
        return self._feats_from_sig(sig, dt_vec)

    @staticmethod
    def _uniform_spacing(v, default: torch.Tensor, name: str) -> float:
        if v is None:
            return float(default)
        if torch.is_tensor(v):
            v = v.reshape(-1)
            if v.numel() > 1 and float(v.max() - v.min()) >= 1e-12:
                raise ValueError(
                    f"cond_local needs a grid-uniform batch: {name} spans "
                    f"[{float(v.min()):.6e}, {float(v.max()):.6e}]")
            return float(v[0])
        return float(v)

    def _mod_grads(self, allf, delta_taps, dx, dy):
        """Per-sample 1D modulation convs (grouped conv over B*C).

        delta_taps (B, C, 2, W) DIMENSIONLESS; output scaled by per-sample
        1/dx (x-direction) and 1/dy (y-direction), mirroring SpatialGrad.
        """
        B, C, H, Wd = allf.shape
        pad = self.width // 2
        flat = allf.reshape(1, B * C, H, Wd)
        wx = delta_taps[:, :, 0].reshape(B * C, 1, 1, self.width)
        wy = delta_taps[:, :, 1].reshape(B * C, 1, self.width, 1)
        gx = F.conv2d(F.pad(flat, (pad, pad, 0, 0), mode='circular'),
                      wx.to(allf.dtype), groups=B * C).reshape(B, C, H, Wd)
        gy = F.conv2d(F.pad(flat, (0, 0, pad, pad), mode='circular'),
                      wy.to(allf.dtype), groups=B * C).reshape(B, C, H, Wd)
        sx = (self.grad.dx0 if dx is None else dx).reshape(-1, 1, 1, 1).to(gx.dtype)
        sy = (self.grad.dy0 if dy is None else dy).reshape(-1, 1, 1, 1).to(gy.dtype)
        return gx / sx, gy / sy

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, dt: torch.Tensor = None,
                dx: torch.Tensor = None, dy: torch.Tensor = None,
                cond_feats: torch.Tensor = None) -> torch.Tensor:
        nt = self.n_time
        no = self.n_ord
        omega_stack = x[:, :nt]
        psi_stack = x[:, nt:2 * nt]

        omega_ord = self.time_fd(omega_stack, dt)[:, :no]   # ORDER CLIP
        psi_ord = self.time_fd(psi_stack, dt)[:, :no]

        cdtype = self.grad.wx.dtype
        if omega_ord.dtype != cdtype:
            omega_ord = omega_ord.to(cdtype)
            psi_ord = psi_ord.to(cdtype)

        # ---- conditioning context: EXACTLY ONE computation per forward ---- #
        self._ctx_calls = 0
        B, Ny, Nx = x.shape[0], x.shape[-2], x.shape[-1]
        if dt is None:
            dt_vec = self.dt0_cond.to(x.device, x.dtype).expand(B)
        elif torch.is_tensor(dt):
            dt_vec = dt.reshape(-1).to(device=x.device, dtype=x.dtype)
            if dt_vec.numel() == 1:
                dt_vec = dt_vec.expand(B)
        else:
            dt_vec = torch.full((B,), float(dt), device=x.device, dtype=x.dtype)
        if cond_feats is not None:
            # rollout path: context precomputed by context_feats_from_spectral
            # from the stepper's spectral qh states (zero FFTs this forward).
            feats = cond_feats
        else:
            Lx = Nx * self._uniform_spacing(dx, self.grad.dx0, 'dx')
            Ly = Ny * self._uniform_spacing(dy, self.grad.dy0, 'dy')
            # sigma-hat reads the two newest PHYSICAL omega marks (raw channels).
            feats = self._context_feats(x[:, 0], x[:, 1], dt_vec, Lx, Ly)
            assert self._ctx_calls == 1, \
                f"context computed {self._ctx_calls}x per forward (must be exactly 1)"

        # ---- tap deltas, analytic per-channel dT^(S-k) scaling ---- #
        delta = self.cond(feats.to(self.cond.head.weight.dtype))  # (B,2no,2,W)
        amp = dt_vec.view(B, 1).to(delta.dtype) ** \
            (self.S - self.k_of_channel.to(delta.dtype)).view(1, -1)  # (B,2no)
        delta = delta * amp.view(B, 2 * no, 1, 1)

        # ---- gradients: control base path + zero-init modulation path ---- #
        allf = torch.cat([omega_ord, psi_ord], dim=1)       # (B, 2no, H, W)
        dxg, dyg = self.grad(allf, dx=dx, dy=dy)            # control, unchanged
        mx, my = self._mod_grads(allf, delta, dx, dy)       # + conditioned taps
        dxg = dxg + mx
        dyg = dyg + my

        wx, px = dxg[:, :no], dxg[:, no:]
        wy, py = dyg[:, :no], dyg[:, no:]
        jac = []
        for i in range(no):              # psi order
            for j in range(no):          # omega order
                jac.append(px[:, i:i + 1] * wy[:, j:j + 1]
                           - py[:, i:i + 1] * wx[:, j:j + 1])
        jac = torch.cat(jac, dim=1)

        if self.refine is not None:
            jac = jac + self.refine(jac)
        out = self.mix(jac)
        if self.corrector is not None:
            feat = torch.cat([jac, omega_ord, psi_ord], dim=1)
            out = out + self.corrector(feat)
        return out


if __name__ == '__main__':
    # smoke: zero-init bit-identity vs the control + param count
    torch.manual_seed(0)
    B, nt, Ny, Nx = 2, 7, 64, 64
    x = torch.randn(B, 2 * nt, Ny, Nx, dtype=torch.float64)
    dt = torch.full((B,), 5e-3, dtype=torch.float64)
    dxs = torch.full((B,), 4 * 3.141592653589793 / Nx, dtype=torch.float64)
    ctrl = CheapDerivClosureNet(in_channels=2 * nt, out_orders=3, n_time=nt,
                                grad_kernel=15).double()
    cond = CondLocalDerivClosureNet(in_channels=2 * nt, out_orders=3,
                                    n_time=nt, grad_kernel=15).double()
    cond.load_state_dict(ctrl.state_dict(), strict=False)
    with torch.no_grad():
        y0 = ctrl(x, dt=dt, dx=dxs, dy=dxs)
        y1 = cond(x, dt=dt, dx=dxs, dy=dxs)
    d = float((y0 - y1).abs().max())
    new = sum(p.numel() for p in cond.cond.parameters())
    print(f"zero-init |ctrl - cond_local|_max = {d:.3e} (must be exactly 0.0)")
    print(f"new conditioning params = {new} (< 10k budget)")
    assert d == 0.0

    # spectral-context path == internal 2-rFFT path (rollout equivalence).
    # Kick the head off zero-init so the feats actually influence the output,
    # and feed a solver-style norm='forward' spectrum (the global scale must
    # cancel in the shell ratio).
    with torch.no_grad():
        cond.cond.head.weight.normal_(0.0, 1.0)
        cond.cond.head.bias.normal_(0.0, 1.0)
        Lx = float(dxs[0]) * Nx
        qh0 = torch.fft.rfftn(x[:, 0], dim=(-2, -1), norm='forward')
        qh1 = torch.fft.rfftn(x[:, 1], dim=(-2, -1), norm='forward')
        feats_spec = cond.context_feats_from_spectral(qh0, qh0 - qh1, dt,
                                                      Lx, Lx, Ny, Nx)
        y_int = cond(x, dt=dt, dx=dxs, dy=dxs)
        y_ext = cond(x, dt=dt, dx=dxs, dy=dxs, cond_feats=feats_spec)
    r = float((y_int - y_ext).abs().max() / y_int.abs().max().clamp_min(1e-30))
    print(f"spectral-context vs internal-context forward: max rel = {r:.3e}")
    assert r < 1e-10
