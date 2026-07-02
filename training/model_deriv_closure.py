"""
model_deriv_closure.py

Cheap, physics-structured closure network for the QG temporal closure.

It predicts the LOCAL N-time-derivatives  [N^(1), N^(2), ..., N^(M)]
( = Ndot, Nddot, N3dot, ... )  from the six-channel snapshot input

    [omega_0, omega_m1, omega_m2,  psi_0, psi_m1, psi_m2]

so the closure brackets (R3, R4, ...) are assembled at inference by applying
the spectral L^k operators (including the nonlocal beta term) analytically --
exactly the split used in build_training_data_fixD_v2.py:

    "N-derivatives are the local, learnable quantities; the L^k weighting
     (incl. the nonlocal beta term) is applied spectrally at inference,
     never learned."

Why this is cheap and well-motivated
------------------------------------
By the chain rule + bilinearity of the Jacobian, each N-derivative is a
binomial-weighted sum of Jacobians of the streamfunction-family and the
vorticity-family time-derivatives:

    N^(m) = - sum_{j=0}^m C(m,j) J( psi^(m-j), omega^(j) )
    J(a, b) = (d_x a)(d_y b) - (d_y a)(d_x b)

The time-derivatives omega^(k), psi^(k) are recovered from the three provided
snapshots by a finite-difference-in-time stencil; the Jacobian is bilinear in
spatial first derivatives. Four cheap stages:

    (1) time-FD          : 3 snapshots          -> {f, f_dot, f_ddot}
    (2) spatial grads    : depthwise 3x3 conv    -> d_x, d_y of each
    (3) Jacobian features : elementwise products  -> J(psi^(i), omega^(j))
    (4) learned mixing   : 1x1 conv              -> [Ndot, Nddot, N3dot, ...]

PHYSICAL UNITS (critical)
-------------------------
The FD stencils MUST carry their grid scalings or the features come out off by
dx*dy*dt^(i+j) (~1e-7 .. 1e-13 here), which would force the 1x1 mix to learn
weights of ~1e6 .. 1e13 -- unreachable from a small init, so the loss sits at
rel == 1.0 and crawls. We therefore divide the time stencils by dt^k and the
spatial stencils by dx, dy at construction. The features are then the true
physical Jacobians and the ideal mix weights are the O(1) chain-rule binomials.

The full (scaled) stencil is the learnable Parameter -- Adam steps the physical
operator directly, which is well conditioned. We deliberately do NOT factor the
stencil into (dt^-k or 1/dx) * dimensionless-correction: under Adam that steps
the dimensionless part ~lr while the scaling post-multiply amplifies the
effective step on the operator by the scaling (up to dt^-2 ~ 1e6 for the order-2
time row), which destabilises training. dt/dx portability is instead applied at
LOAD time by rescaling the loaded stencil -- that does not touch the optimiser.

With physics_init=True the mix is initialised directly to those binomials, so
epoch 0 already predicts Ndot/Nddot up to the FD truncation error of the
stencils (the learnable stencils then refine it). N3dot's order-3 terms are not
representable from 3 snapshots and are only approximated (use refine_channels or
a 4th snapshot if it underfits).

Caveat on order
---------------
Three snapshots give clean access to time-orders 0,1,2, so N^(1)=Ndot and
N^(2)=Nddot are *exactly* in the span. N^(3) (R4 bracket) is *approximated*.

A note for the MAC counter
--------------------------
The Conv2d/Linear MAC hook does NOT see the time-FD einsum, the depthwise
F.conv2d gradients, or the elementwise Jacobian products. The true cost is
~170 MACs/grid-point (mix ~27 + grads ~108 + time-FD ~18 + products ~18); the
hook reports only the ~27 of the 1x1 mix.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _deriv1d_weights(width: int, h: float) -> np.ndarray:
    """Central 1st-derivative weights on nodes [-(w//2)..w//2]*h (includes 1/h).

    width=3 -> [-1,0,1]/(2h); width=5 -> [1,-8,0,8,-1]/(12h); etc. (Vandermonde).
    """
    r = width // 2
    x = np.array([(i - r) * h for i in range(width)], dtype=np.float64)
    A = np.array([[x[j] ** m / math.factorial(m) for j in range(width)]
                  for m in range(width)], dtype=np.float64)
    return np.linalg.inv(A)[:, 1]          # column 1 = 1st-derivative weights


def _central_diff_kernel(axis: str, width: int = 3, h: float = 1.0) -> torch.Tensor:
    """width x width kernel: 1D central-difference d/dx or d/dy (spacing h)."""
    w1 = torch.from_numpy(_deriv1d_weights(width, h)).to(torch.float32)
    c = width // 2
    k = torch.zeros(width, width)
    if axis == 'x':
        k[c, :] = w1               # derivative along columns (x)
    elif axis == 'y':
        k[:, c] = w1               # derivative along rows (y)
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    return k


class TimeFD(nn.Module):
    """Mix n_time snapshots [t0, t-1, ..., t-(nt-1)] -> derivative orders 0..nt-1.

    Weights are backward finite-difference stencils on nodes x_j = -j*dt,
    divided by dt^k for the k-th derivative (so outputs are TRUE time
    derivatives). Solved from the Vandermonde system A w = e_k with
    A[m,j] = x_j^m / m!, exact for any n_time. For n_time=3 this reproduces
    [1,0,0], [1.5,-2,0.5]/dt, [1,-2,1]/dt^2.

    The full (scaled) stencil is the learnable Parameter, so Adam steps the
    physical operator directly -- well conditioned. (Do NOT factor this into
    dt^-k * dimensionless: that makes Adam step the dimensionless part ~lr while
    the dt^-k post-multiply amplifies the effective step on the operator by up to
    dt^-k ~ 1e6 for the order-2 row, which blows training up. dt portability is
    handled at LOAD time by rescaling the loaded stencil, not in the parametrization.)
    """

    def __init__(self, n_time: int, dt: float, learnable: bool = True):
        super().__init__()
        x = np.array([-j * dt for j in range(n_time)], dtype=np.float64)
        A = np.array([[x[j] ** m / math.factorial(m) for j in range(n_time)]
                      for m in range(n_time)], dtype=np.float64)
        W = np.linalg.inv(A).T            # (nt, nt): row k = order-k stencil at this dt
        self.weight = nn.Parameter(torch.from_numpy(W).to(torch.float32),
                                   requires_grad=learnable)
        # Unit-spacing reference (dt=1): row k = order-k backward-diff stencil. For a
        # per-sample dt the order-k weights are W_unit[k] / dt^k -- exact FD at any dt,
        # which is what the Delta_T SWEEP needs (one model, many dt). Kept as a buffer
        # (frozen): rescaling W_unit per sample is exact, so there is nothing to learn,
        # and it sidesteps the cross-dt gradient imbalance a single learnable stencil
        # would suffer (the 1/dt^k weights small-dt samples ~1e9x heavier).
        xu = np.array([-j for j in range(n_time)], dtype=np.float64)
        Au = np.array([[xu[j] ** m / math.factorial(m) for j in range(n_time)]
                       for m in range(n_time)], dtype=np.float64)
        self.register_buffer('W_unit', torch.from_numpy(np.linalg.inv(Au).T).to(torch.float32))
        self.n_time = n_time

    def forward(self, stack: torch.Tensor, dt: torch.Tensor = None) -> torch.Tensor:
        # stack: (B, nt, H, W) ordered [t0, t-1, ...]
        if dt is None:                                   # fixed-dt path (single dataset)
            return torch.einsum('oi,bihw->bohw', self.weight.to(stack.dtype), stack)
        # per-sample dt path (the Delta_T sweep): order-k weights = W_unit[k] / dt^k.
        nt = self.n_time
        orders = torch.arange(nt, device=stack.device, dtype=stack.dtype)
        scale = dt.reshape(-1, 1).to(stack.dtype) ** (-orders).reshape(1, nt)  # (B, nt)
        Wb = self.W_unit.to(stack.dtype)[None] * scale[:, :, None]             # (B, nt, nt)
        return torch.einsum('boi,bihw->bohw', Wb, stack)


class SpatialGrad(nn.Module):
    """Depthwise d/dx, d/dy via a (width x width) central-difference conv.

    *** dx-INDEPENDENT (factored) version ***
    The learnable Parameter is the DIMENSIONLESS unit-spacing (h=1) stencil; the
    physical 1/dx, 1/dy scaling is applied per-sample at FORWARD, exactly mirroring
    how TimeFD applies W_unit/dt^k. Consequences:
      * one model serves ANY grid in the pool -- the learned SHAPE is grid-
        independent, the analytic 1/dx scaling never touches the optimizer;
      * dx, dy are threaded per-sample (B,) so a grid-homogeneous batch at 256^2
        and one at 512^2 get their own scaling from the same shared stencil;
      * dx != dy (Lx != Ly) is handled by construction -- x uses 1/dx, y uses 1/dy,
        separately -- so the anisotropic-grid step needs NO change here.
    Because conv is linear, conv(x, S_unit)/dx == conv(x, S_unit/dx), so applying
    1/dx to the OUTPUT is exact and lets a single stencil serve all spacings.

    width=3 is 2nd-order, 5/7 are 4th/6th order (narrowing the FD-vs-spectral gap
    on high-k fields). Learnable kernels refine the dimensionless shape from there.
    Adam steps the DIMENSIONLESS operator -- well conditioned, no 1/dx amplification.
    """

    def __init__(self, channels: int, dx: float = 1.0, dy: float = 1.0,
                 width: int = 3, learnable: bool = True):
        super().__init__()
        if width % 2 == 0:
            raise ValueError(f"grad kernel width must be odd, got {width}")
        self.channels = channels
        self.pad = width // 2
        # DIMENSIONLESS unit-spacing stencils (h=1), built in FLOAT64 so the factored
        # 1/dx path is exact in double (the .to(torch.float32) inside
        # _central_diff_kernel would otherwise inject ~1e-7 roundoff before the
        # per-sample /dx). Module .float()/.double() then casts once, cleanly.
        kx = _central_diff_kernel('x', width, 1.0).double().repeat(channels, 1, 1, 1)
        ky = _central_diff_kernel('y', width, 1.0).double().repeat(channels, 1, 1, 1)
        self.wx = nn.Parameter(kx, requires_grad=learnable)
        self.wy = nn.Parameter(ky, requires_grad=learnable)
        # Default spacing (fixed-grid fallback when dx/dy not passed at forward),
        # kept as buffers so .to(device)/dtype follow the module. Mirrors TimeFD's
        # dt=None fixed path.
        self.register_buffer('dx0', torch.tensor(float(dx)))
        self.register_buffer('dy0', torch.tensor(float(dy)))

    def forward(self, x: torch.Tensor, dx: torch.Tensor = None,
                dy: torch.Tensor = None):
        xp = F.pad(x, (self.pad,) * 4, mode='circular')
        gx = F.conv2d(xp, self.wx.to(x.dtype), groups=self.channels)
        gy = F.conv2d(xp, self.wy.to(x.dtype), groups=self.channels)
        B = x.shape[0]
        # per-sample 1/dx, 1/dy (analytic, exact). dx/dy None -> fixed-grid default.
        sx = (self.dx0 if dx is None else dx).reshape(-1, 1, 1, 1).to(gx.dtype)
        sy = (self.dy0 if dy is None else dy).reshape(-1, 1, 1, 1).to(gy.dtype)
        # broadcast scalar-buffer (numel 1) or per-sample (B,) alike
        return gx / sx, gy / sy


class _Corrector(nn.Module):
    """Zero-initialised residual CNN that learns the FD-truncation correction on
    top of the physics prediction.

    Motivation: the physics path predicts N^(m) by the chain-rule binomial sum of
    Jacobians built from FINITE-DIFFERENCE time-derivatives. Each FD derivative
    carries a truncation error (with 4 snapshots: omega_ddot 2nd-order, omega_3dot
    1st-order), so the physics output sits a fixed bias away from the EXACT
    analytic target -- which a linear mix cannot remove. This block learns that
    bias from the local features.

    Two design choices make it safe:
      * the OUTPUT conv is zero-initialised, so at epoch 0 the corrector is exactly
        0 and the network reproduces the physics_init prediction; training only
        ADDS the correction (no destabilising of the well-conditioned physics).
      * the INPUT is per-sample, per-channel normalised (instance norm). The
        physical features span ~5 orders of magnitude (omega ~1 ... omega_3dot
        ~1e5 at dt=1e-3), which a plain conv cannot condition -- the same scaling
        trap the physics path avoids by construction. Normalising only this branch
        leaves the physics mix on the raw physical features untouched.
    """

    def __init__(self, in_ch: int, out_ch: int, hidden: int, depth: int,
                 kernel: int):
        super().__init__()
        pad = kernel // 2
        # GroupNorm with one group per channel == instance norm: per-sample,
        # per-channel, batch-independent (safe for the batch=1 autoregressive
        # rollout, unlike BatchNorm running stats).
        self.norm = nn.GroupNorm(in_ch, in_ch, affine=False)
        layers, c = [], in_ch
        for _ in range(max(depth - 1, 1)):
            layers += [nn.Conv2d(c, hidden, kernel, padding=pad,
                                 padding_mode='circular'), nn.GELU()]
            c = hidden
        last = nn.Conv2d(c, out_ch, kernel, padding=pad, padding_mode='circular')
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)   # start as no-op
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x))


class CheapDerivClosureNet(nn.Module):
    """Predict local N-time-derivatives from snapshot input.

    Input : (B, 2*n_time, H, W)
            = [omega_0, ..., omega_{-(n_time-1)},  psi_0, ..., psi_{-(n_time-1)}]
    Output: (B, out_orders, H, W) = [Ndot, Nddot, N3dot, ...]
    """

    def __init__(self, in_channels: int = 6, out_orders: int = 3,
                 n_time: int = 3, refine_channels: int = 0,
                 learnable_stencils: bool = True, kernel: int = 3,
                 grad_kernel: int = 3,
                 dt: float = 1e-3, dx: float = 1.0, dy: float = 1.0,
                 physics_init: bool = True,
                 hidden_channels: int = 0, depth: int = 0):
        super().__init__()
        if in_channels != 2 * n_time:
            raise ValueError(
                f"in_channels ({in_channels}) must equal 2*n_time "
                f"({2 * n_time}): [omega x{n_time}, psi x{n_time}]")
        self.n_time = n_time
        self.out_orders = out_orders

        self.time_fd = TimeFD(n_time, dt, learnable=learnable_stencils)
        # spatial grads applied to (n_time omega-orders + n_time psi-orders)
        self.grad = SpatialGrad(2 * n_time, dx, dy, width=grad_kernel,
                                learnable=learnable_stencils)

        self.n_jac = n_time * n_time  # J(psi^i, omega^j), i,j in 0..n_time-1

        self.refine = None
        if refine_channels > 0:
            pad = kernel // 2
            self.refine = nn.Sequential(
                nn.Conv2d(self.n_jac, refine_channels, kernel, padding=pad,
                          padding_mode='circular'),
                nn.GELU(),
                nn.Conv2d(refine_channels, self.n_jac, kernel, padding=pad,
                          padding_mode='circular'),
            )

        # learned mixing: Jacobian features -> N-derivative outputs (1x1 conv)
        self.mix = nn.Conv2d(self.n_jac, out_orders, kernel_size=1)
        nn.init.zeros_(self.mix.bias)
        if physics_init:
            self._physics_init_mix()
        else:
            nn.init.normal_(self.mix.weight, std=0.1)

        # Zero-init residual corrector: capacity to learn the FD-truncation bias
        # the linear physics mix cannot. Fed [jac, omega-orders, psi-orders]; the
        # conv re-derives any spatial structure it needs. Off when hidden==0.
        self.corrector = None
        if hidden_channels > 0 and depth >= 1:
            self.corrector = _Corrector(self.n_jac + 2 * n_time, out_orders,
                                        hidden_channels, depth, kernel)

    def _physics_init_mix(self):
        """Init the 1x1 mix to the analytic chain-rule binomials, for any n_time.

        Output o predicts N^(m), m = o+1, and
            N^(m) = -sum_{j=0}^m C(m,j) J(psi^(m-j), omega^(j)).
        The feature J(psi^i, omega^j) lives at index i*n_time + j and exists only
        if i < n_time and j < n_time; terms needing an unavailable order are
        dropped (so the highest order is approximate, the rest exact).
        """
        nt = self.n_time
        w = self.mix.weight.data
        w.zero_()
        for o in range(self.out_orders):
            m = o + 1
            for j in range(0, m + 1):
                i = m - j
                if i < nt and j < nt:
                    w[o, i * nt + j, 0, 0] = -float(math.comb(m, j))

    def forward(self, x: torch.Tensor, dt: torch.Tensor = None,
                dx: torch.Tensor = None, dy: torch.Tensor = None) -> torch.Tensor:
        nt = self.n_time
        omega_stack = x[:, :nt]          # [omega_0, omega_m1, ...]
        psi_stack   = x[:, nt:2 * nt]    # [psi_0,   psi_m1,   ...]

        omega_ord = self.time_fd(omega_stack, dt)   # (B, nt, H, W) [w, wdot, wddot]
        psi_ord   = self.time_fd(psi_stack, dt)      # (B, nt, H, W) [p, pdot, pddot]

        # Mixed precision: the TimeFD differencing above runs at the INPUT dtype. At
        # inference we feed float64 so the high-order differences (omega_3dot ~ dt^-3,
        # a ~1e-9 signal that sits under ~1e-7 float32 eps) are cancellation-clean.
        # The expensive spatial convs below then run in the PARAMETER dtype (float32
        # at inference -> ~30x faster on the A6000), where ~1e-7 round-off is far below
        # the NN's own error. In float32 training input==param dtype so this is a no-op.
        cdtype = self.grad.wx.dtype
        if omega_ord.dtype != cdtype:
            omega_ord = omega_ord.to(cdtype)
            psi_ord   = psi_ord.to(cdtype)

        allf = torch.cat([omega_ord, psi_ord], dim=1)   # (B, 2*nt, H, W)
        dxg, dyg = self.grad(allf, dx=dx, dy=dy)
        wx, px = dxg[:, :nt], dxg[:, nt:]   # omega-x grads, psi-x grads
        wy, py = dyg[:, :nt], dyg[:, nt:]

        # J(psi^i, omega^j) = (d_x psi^i)(d_y omega^j) - (d_y psi^i)(d_x omega^j)
        jac = []
        for i in range(nt):              # psi order
            for j in range(nt):          # omega order
                jac.append(px[:, i:i + 1] * wy[:, j:j + 1]
                           - py[:, i:i + 1] * wx[:, j:j + 1])
        jac = torch.cat(jac, dim=1)      # (B, n_jac, H, W)

        if self.refine is not None:
            jac = jac + self.refine(jac)

        out = self.mix(jac)              # (B, out_orders, H, W) -- physics
        if self.corrector is not None:
            feat = torch.cat([jac, omega_ord, psi_ord], dim=1)
            out = out + self.corrector(feat)   # + learned truncation correction
        return out


def build_model(name: str = 'cheap_deriv', in_channels: int = 6, **kw):
    """Factory matching the train.py build_model(...) convention.

    n_time defaults to in_channels // 2 (half omega, half psi), so 6->3 and
    8->4 without an explicit flag.
    """
    n_time = kw.get('n_time', in_channels // 2)
    return CheapDerivClosureNet(
        in_channels=in_channels,
        out_orders=kw.get('out_orders', 3),
        n_time=n_time,
        refine_channels=kw.get('refine_channels', 0),
        learnable_stencils=kw.get('learnable_stencils', True),
        kernel=kw.get('kernel', 3),
        grad_kernel=kw.get('grad_kernel', 3),
        dt=kw.get('dt', 1e-3), dx=kw.get('dx', 1.0), dy=kw.get('dy', 1.0),
        physics_init=kw.get('physics_init', True),
        hidden_channels=kw.get('hidden_channels', 0),
        depth=kw.get('depth', 0),
    )