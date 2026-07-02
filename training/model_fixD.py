"""
model_fixD.py - Minimal residual CNN for Fix D v2 closure NN.

Architecture rationale (see derivation):
------------------------------------------------------------
The closure target is f_NN_target = (1/12)*[L*Ndot - 5*Nddot]. With 6 inputs
{omega_0, omega_m1, omega_m2, psi_0, psi_m1, psi_m2}, the network needs only
LOCAL operations:

  1. Time derivatives via finite differences (1x1 channel mix)
        omega_dot   = (omega_0 - omega_m1)/dT          (centered at -dT/2)
        omega_ddot  = (omega_0 - 2*omega_m1 + omega_m2)/dT^2  (centered at -dT)
        same for psi

  2. Spatial gradients (3x3 conv)
        u  = -d_y psi,  v = +d_x psi  for each psi-order
        d_x omega, d_y omega  for each omega-order

  3. Bilinear Jacobian products (GLU)
        J(psi, omega) = u * d_x omega + v * d_y omega
        5 such terms: J(psi,dot omega), J(dot psi, omega), J(dot psi, dot omega),
                      J(ddot psi, omega), J(psi, ddot omega)

  4. Apply L = nu*Lap for the L*Ndot piece (3x3 conv)

  5. Final linear combination (1x1 conv)

No global operator (inv-Laplacian) is needed because psi is given as input.
This means the receptive field is small and the network is correspondingly
lean -- about 5 conv layers, ~150K parameters.

Architecture:
                                                          receptive field
  Stem    : Conv2d(6 -> C, 1x1)                                  1
  Block A : Conv2d(C -> C, 1x1)              # time derivs       1
  Block B : Conv2d(C -> 4C, 3x3) + GLU       # gradients +       3
                                                bilinear products
  Block C : Conv2d(2C -> 2C, 3x3, residual)  # refine            5
  Block D : Conv2d(2C -> C, 3x3)             # apply L           7
  Output  : Conv2d(C -> 1, 1x1)                                  7
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _periodic_pad(x: torch.Tensor, p: int) -> torch.Tensor:
    """Circular pad by p on the last two dims (Ny, Nx)."""
    return F.pad(x, (p, p, p, p), mode='circular')


class PeriodicConv(nn.Module):
    """Conv2d with circular padding."""
    def __init__(self, in_c: int, out_c: int, kernel: int = 3, bias: bool = True):
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel, padding=0, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel > 1:
            p = (self.kernel - 1) // 2
            x = _periodic_pad(x, p)
        return self.conv(x)


class GLU2d(nn.Module):
    """Channel-wise GLU: split last-2-dim tensor along channel into halves
    (a, b), return a * sigmoid(b). The conv before this layer should
    output 2*C channels; GLU returns C channels.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * torch.sigmoid(b)


class BilinearClosureNet(nn.Module):
    """
    Minimal residual CNN matching the closure operator structure.

    Args:
      in_channels: number of inputs (default 6: omega + psi at 3 time levels)
      hidden:      base channel width (default 64)
      kernel:      conv kernel size (default 3)
    """
    def __init__(self, in_channels: int = 6, hidden: int = 64, kernel: int = 3):
        super().__init__()
        C = hidden

        # Stem: 1x1 channel mix from raw inputs to hidden width.
        self.stem = PeriodicConv(in_channels, C, kernel=1)

        # Block A: 1x1 channel mix (lets the network learn finite-difference
        # weights for time derivatives without spatial coupling).
        self.block_a = PeriodicConv(C, C, kernel=1)

        # Block B: 3x3 conv producing 4C channels, then GLU -> 2C channels.
        # The 3x3 part computes spatial gradients; GLU's multiplicative gating
        # produces the bilinear Jacobian-like products.
        self.block_b_conv = PeriodicConv(C, 4 * C, kernel=kernel)
        self.block_b_glu  = GLU2d()  # 4C -> 2C

        # Block C: refinement at 2C channels with residual connection.
        # Helps gradient flow during training; mathematically optional.
        self.block_c_conv = PeriodicConv(2 * C, 2 * C, kernel=kernel)
        self.block_c_act  = nn.GELU()

        # Block D: 3x3 conv to apply L (Laplacian) for the L*Ndot piece,
        # mixing back down to C channels.
        self.block_d = PeriodicConv(2 * C, C, kernel=kernel)
        self.block_d_act = nn.GELU()

        # Output: 1x1 to single channel (linear, no activation).
        self.head = PeriodicConv(C, 1, kernel=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, Ny, Nx)
        h = self.stem(x)              # (B, C, Ny, Nx)
        h = self.block_a(h)           # 1x1 mix for time derivatives

        h = self.block_b_conv(h)      # gradients via 3x3 conv (4C channels)
        h = self.block_b_glu(h)       # GLU -> 2C; bilinear products

        # Block C with residual
        c = self.block_c_conv(h)
        c = self.block_c_act(c)
        h = h + c

        # Block D: apply L
        h = self.block_d(h)
        h = self.block_d_act(h)

        # Output
        return self.head(h)


def build_model(name: str, in_channels: int = 6, **kwargs) -> nn.Module:
    """Factory: 'bilinear_closure' (Fix D v2 minimal CNN)."""
    name = name.lower()
    if name in ('bilinear_closure', 'bilin', 'fixd_v2'):
        # Allow either 'hidden' or 'hidden_channels' from train.py
        if 'hidden_channels' in kwargs and 'hidden' not in kwargs:
            kwargs['hidden'] = kwargs.pop('hidden_channels')
        # Strip args this model doesn't accept
        accepted = {'hidden', 'kernel'}
        kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        return BilinearClosureNet(in_channels=in_channels, **kwargs)
    raise ValueError(f"unknown model name '{name}'")


if __name__ == '__main__':
    print("BilinearClosureNet:")
    m = build_model('bilinear_closure', in_channels=6, hidden=64, kernel=3)
    x = torch.randn(2, 6, 256, 256)
    y = m(x)
    print(f"  in:  {tuple(x.shape)}")
    print(f"  out: {tuple(y.shape)}")
    n_params = sum(p.numel() for p in m.parameters())
    print(f"  params: {n_params:,}")
    print()
    # Print per-layer parameter count
    for name, p in m.named_parameters():
        print(f"  {name:40s} {tuple(p.shape)} = {p.numel():>9,}")
