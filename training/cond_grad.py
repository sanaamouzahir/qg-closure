"""
cond_grad.py -- conditioned spectral gradient layer for exp/wiener-conditioning.

Implements the step-2 design (see conditioned_parameterization_note.md):

    D_hat_d^(k)(kvec) = i k_d [1 + dT^(S-k) g^A_theta(x(kappa), kappa~)]
                      +   k_d  dT^(S-k) g^B_theta(x(kappa), kappa~)

with x(kappa) = dT * sigma_hat_omega(kappa) computed per sample from its own
two newest marks (arcsin-debiased), and kappa~ = kappa / kappa_max.

Base operator = EXACT spectral ik (solver Derivative multipliers).
Init: last layers zero -> exact gradients -> the model must reproduce the
[spec] floors (0.0003 / 0.008 / 0.10 at 5e-3) at initialization. That is the
integration acceptance test.

Integration contract (for the branch supervisor / Fable):
  1. build_model('cond_deriv', ...) constructs the same TimeFD + binomial-mix
     pipeline as 'cheap_deriv' but replaces SpatialGrad with SpectralCondGrad
     (one instance shared; it holds per-channel MLPs internally).
  2. forward(x, dt, dx, dy): compute sigma-hat once per sample from channels
     (omega_0, omega_-1) = x[:, 0], x[:, 1]; pass (fields, channel_order k,
     dt) into grad_x/grad_y calls inside the Jacobian assembly.
  3. S (stencil depth used for the dT power) and n_orders come from the model
     config; kappa binning derives from the per-sample grid shape (per-shape
     cache, same pattern as the dealias projections).

Everything float64-safe; no learned local stencils anywhere in this branch.
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
#  per-shape spectral cache
# --------------------------------------------------------------------------- #
class _ShapeCache:
    """kx, ky multipliers, |k|, integer shell index, kappa_max per (Ny, Nx, L)."""

    def __init__(self):
        self._c = {}

    def get(self, Ny, Nx, Lx, Ly, device, dtype=torch.float64):
        key = (Ny, Nx, round(float(Lx), 9), round(float(Ly), 9), str(device))
        if key not in self._c:
            kx = 2 * math.pi * torch.fft.rfftfreq(Nx, d=Lx / Nx,
                                                  device=device, dtype=dtype) * 1.0
            ky = 2 * math.pi * torch.fft.fftfreq(Ny, d=Ly / Ny,
                                                 device=device, dtype=dtype) * 1.0
            KX = kx[None, :].expand(Ny, kx.shape[0])
            KY = ky[:, None].expand(Ny, kx.shape[0])
            kmag = torch.sqrt(KX ** 2 + KY ** 2)
            shell = torch.round(kmag * (Lx / (2 * math.pi))).to(torch.int64)
            n_sh = int(shell.max().item()) + 1
            self._c[key] = dict(KX=KX, KY=KY, kmag=kmag, shell=shell,
                                n_sh=n_sh,
                                kmax=float(kmag.max().item()))
        return self._c[key]


_CACHE = _ShapeCache()


# --------------------------------------------------------------------------- #
#  sigma-hat: per-sample, per-shell rate estimate (arcsin-debiased 2-mark FD)
# --------------------------------------------------------------------------- #
def sigma_hat(om0: torch.Tensor, om1: torch.Tensor, dt: torch.Tensor,
              Lx: float, Ly: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    om0, om1: (B, Ny, Nx) newest and second-newest omega marks (physical).
    dt:       (B,) per-sample lag spacing.
    Returns (sig_shell, shell_index): sig_shell (B, n_sh) debiased
    sigma_hat(kappa); shell_index (Ny, Nx_r) for broadcasting back to modes.
    """
    B, Ny, Nx = om0.shape
    c = _CACHE.get(Ny, Nx, Lx, Ly, om0.device, om0.dtype)
    sh, n_sh = c['shell'], c['n_sh']
    f0 = torch.fft.rfft2(om0)
    fd = torch.fft.rfft2(om0 - om1)
    e0 = (f0.real ** 2 + f0.imag ** 2).reshape(B, -1)
    ed = (fd.real ** 2 + fd.imag ** 2).reshape(B, -1)
    idx = sh.reshape(-1).expand(B, -1)
    E0 = torch.zeros(B, n_sh, dtype=om0.dtype, device=om0.device)
    Ed = torch.zeros_like(E0)
    E0.scatter_add_(1, idx, e0)
    Ed.scatter_add_(1, idx, ed)
    raw = torch.sqrt(Ed / E0.clamp_min(1e-300)) / dt.view(B, 1)
    # arcsin de-bias of the 2-mark FD response: raw = (2/dt) sin(sig dt / 2)
    arg = (raw * dt.view(B, 1) / 2.0).clamp(max=1.0 - 1e-12)
    sig = (2.0 / dt.view(B, 1)) * torch.arcsin(arg)
    return sig, sh


# --------------------------------------------------------------------------- #
#  the conditioned gradient layer
# --------------------------------------------------------------------------- #
class _ChannelMLP(nn.Module):
    """g_theta for one channel: (x, kappa~) -> (g_A, g_B). Zero-init output."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feats):                      # feats (..., 2)
        return self.net(feats)                     # (..., 2) = (g_A, g_B)


class SpectralCondGrad(nn.Module):
    """
    Exact spectral gradients with a per-channel conditioned correction.

    Usage inside the Jacobian assembly (channel order k = time-derivative
    order of the field being differentiated):

        gx, gy = layer.grad(field, k_order, sample_ctx)

    where sample_ctx = layer.context(x_stack, dt, Lx, Ly) is computed ONCE per
    forward from the two newest omega channels.
    """

    def __init__(self, S: int = 7, n_channels: int = 4, hidden: int = 16):
        super().__init__()
        self.S = S
        # channels 0..n_channels-1 for omega^(k) and psi^(k) SEPARATELY:
        # index c = k for omega, c = n_channels + k for psi
        self.mlps = nn.ModuleList([_ChannelMLP(hidden)
                                   for _ in range(2 * n_channels)])
        self.n_channels = n_channels

    # ---- per-forward context ------------------------------------------------
    def context(self, om0, om1, dt, Lx, Ly):
        """om0, om1 (B,Ny,Nx); dt (B,). Returns dict used by grad()."""
        B, Ny, Nx = om0.shape
        c = _CACHE.get(Ny, Nx, Lx, Ly, om0.device, om0.dtype)
        sig, sh = sigma_hat(om0, om1, dt, Lx, Ly)          # (B,n_sh), (Ny,Nxr)
        x = sig * dt.view(B, 1)                            # dT*sigma per shell
        kap = torch.arange(sig.shape[1], device=om0.device,
                           dtype=om0.dtype) / max(sig.shape[1] - 1, 1)
        feats = torch.stack([x, kap[None, :].expand_as(x)], dim=-1)  # (B,n_sh,2)
        return dict(cache=c, feats=feats, dt=dt, B=B)

    # ---- the conditioned gradient -------------------------------------------
    def grad(self, f, k_order, is_psi, ctx):
        """
        f: (B, Ny, Nx) physical field of channel (k_order, is_psi).
        Returns (df/dx, df/dy) physical, with the conditioned correction.
        """
        c = ctx['cache']; B = ctx['B']
        fh = torch.fft.rfft2(f)                            # (B, Ny, Nxr)
        cidx = (self.n_channels if is_psi else 0) + k_order
        g = self.mlps[cidx](ctx['feats'])                  # (B, n_sh, 2)
        # broadcast per-shell g back to modes
        sh_flat = c['shell'].reshape(-1)
        gA = g[..., 0].index_select(1, sh_flat).reshape(B, *c['shell'].shape)
        gB = g[..., 1].index_select(1, sh_flat).reshape(B, *c['shell'].shape)
        amp = (ctx['dt'].view(B, 1, 1) ** (self.S - k_order))
        one_pA = 1.0 + amp * gA
        qB = amp * gB
        KX, KY = c['KX'], c['KY']
        # D_d = i k_d (1 + amp gA) + k_d amp gB   (complex multiply, per mode)
        gx = torch.fft.irfft2(fh * (1j * KX * one_pA + KX * qB),
                              s=f.shape[-2:])
        gy = torch.fft.irfft2(fh * (1j * KY * one_pA + KY * qB),
                              s=f.shape[-2:])
        return gx, gy

    def extra_repr(self):
        n = sum(p.numel() for p in self.parameters())
        return f"S={self.S}, channels={2*self.n_channels}, params={n}"


# --------------------------------------------------------------------------- #
#  smoke test (shapes + zero-init exactness)
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    torch.manual_seed(0)
    B, Ny, Nx = 2, 64, 64
    L = 4 * math.pi
    om0 = torch.randn(B, Ny, Nx, dtype=torch.float64)
    om1 = om0 + 1e-3 * torch.randn_like(om0)
    dt = torch.full((B,), 5e-3, dtype=torch.float64)
    layer = SpectralCondGrad(S=7, n_channels=4).double()
    ctx = layer.context(om0, om1, dt, L, L)
    gx, gy = layer.grad(om0, k_order=2, is_psi=False, ctx=ctx)
    # zero-init => must equal exact spectral gradient
    c = _CACHE.get(Ny, Nx, L, L, om0.device, om0.dtype)
    fh = torch.fft.rfft2(om0)
    gx_ref = torch.fft.irfft2(1j * c['KX'] * fh, s=(Ny, Nx))
    err = float(torch.norm(gx - gx_ref) / torch.norm(gx_ref))
    print(f"zero-init exactness: rel err = {err:.3e} (must be ~1e-16)")
    n = sum(p.numel() for p in layer.parameters())
    print(f"params = {n} (8 channel-MLPs)")
    sig, _ = sigma_hat(om0, om1, dt, L, L)
    print(f"sigma_hat shape = {tuple(sig.shape)}, finite = "
          f"{bool(torch.isfinite(sig).all())}")
