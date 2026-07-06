"""[free-time-fd] Init-reproduction check.

The learnable-time-FD model (--learn-time-fd) at init must reproduce the CONTROL model's
outputs to float64 round-off: W_learn is initialised to the EXACT Vandermonde rows, so the
overlaid unit stencil W_eff == W_unit at epoch 0. This proves the learnable change starts
from the control and can only refine from there (clean attributable delta).

Run on a COMPUTE NODE (SGE), never the front end.
"""
import numpy as np
import torch
from model_deriv_closure import build_model

torch.manual_seed(0)
nt, out_orders, B, H, W = 7, 3, 2, 32, 32
dt0 = 1e-2
dx0 = dy0 = 4.0 * np.pi / 512.0
common = dict(in_channels=2 * nt, out_orders=out_orders, n_time=nt,
              grad_kernel=15, dt=dt0, dx=dx0, dy=dy0, physics_init=True)

ctrl = build_model('cheap_deriv', learn_time_fd=False, **common).double()
free = build_model('cheap_deriv', learn_time_fd=True, **common).double()

x = torch.randn(B, 2 * nt, H, W, dtype=torch.float64)
dt = torch.full((B,), dt0, dtype=torch.float64)
dx = torch.full((B,), dx0, dtype=torch.float64)
dy = torch.full((B,), dy0, dtype=torch.float64)

with torch.no_grad():
    yc = ctrl(x, dt, dx, dy)
    yf = free(x, dt, dx, dy)

mad = (yc - yf).abs().max().item()
nc = sum(p.numel() for p in ctrl.parameters() if p.requires_grad)
nf = sum(p.numel() for p in free.parameters() if p.requires_grad)
print(f"[init-repro] max|control - freeW| = {mad:.3e}  (float64 round-off ~1e-12)")
print(f"[init-repro] trainable params: control={nc}  freeW={nf}  delta=+{nf - nc}")
print(f"[init-repro] W_learn shape: {tuple(free.time_fd.W_learn.shape)} "
      f"(rows 1..{out_orders} of the {nt}-node unit stencil)")

# also confirm grad actually flows into W_learn (the rows are live, not dead like self.weight)
yf2 = free(x, dt, dx, dy)
yf2.sum().backward()
g = free.time_fd.W_learn.grad
gnorm = float(g.abs().sum()) if g is not None else 0.0
print(f"[init-repro] W_learn grad L1 = {gnorm:.3e}  (must be > 0 -> rows are trainable)")

assert mad < 1e-10, f"INIT REPRODUCTION FAILED: max diff {mad:.3e} >= 1e-10"
assert gnorm > 0.0, "W_learn received no gradient -> the learnable rows are dead"
print("[init-repro] PASS: freeW at init == control to float64 round-off, and W_learn is live")
