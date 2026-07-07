#!/usr/bin/env python
r"""diagnose_cond_init_sanity.py -- zero-init exactness check for cond_deriv.

The SpectralCondGrad layer (Theoretical_guarantees/cond_grad.py) initialises its
per-channel correction MLPs to zero, so at init its gradient operator is the
EXACT spectral ik multiplier. This script asserts, on several (member, dT)
samples, that a freshly built cond_deriv model reproduces an INDEPENDENT
exact-spectral-advective assembly of its N-derivative outputs to fft round-off.

What it compares (per sample, float64, on the model's OWN TimeFD fields so the
shared float32-stencil TimeFD is not under test here -- only the spatial layer):

  M   = cond_deriv(x)              -- model output, end-dealiased
  Aref= Sum_j -C(m,j) J_adv( psi^(m-j), omega^(j) ),  end-dealiased,
        where J_adv is built from the SOLVER's spectral gradients (qg
        Derivative.dx/.dy), i.e. an operator INDEPENDENT of cond_grad.py.

  ASSERT rel(M, Aref) < TOL (default 1e-10) on N1dot, N2dot, N3dot.

This is the true "zero-init exactness of SpectralCondGrad": the conditioned
spectral operator, at init, equals the exact spectral operator to round-off, so
the whole cond_deriv assembly equals the exact-spectral floor.

It ALSO reports, for context (NOT asserted):
  * rel(M, [spec] flux) -- the advective-vs-flux FORM gap (a deferred design
    item shared with cheap_deriv; NOT a wiring defect). This is why cond_deriv
    does not match the flux-form [spec] floors of diagnose_one_sample to 1e-12.
  * rel(model TimeFD, float64 Vandermonde TimeFD) -- the float32 W_unit stencil
    contribution (pre-existing cheap_deriv behaviour; ~1e-4 on N2dot, far below
    the science floor).

Any cond_deriv variant should pass this before training -- run it as the
pre-training gate.

Usage (from the worktree root or anywhere; paths resolve to training/data):
    python diagnostics/diagnose_cond_init_sanity.py
    python diagnostics/diagnose_cond_init_sanity.py --tol 1e-10 --device cpu
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

# --- make the flat training/ modules importable from anywhere ---
_TRAIN = Path(__file__).resolve().parent.parent / 'training'
if str(_TRAIN) not in sys.path:
    sys.path.insert(0, str(_TRAIN))

_DATA = _TRAIN / 'data' / 'ensemble_N5_7lag'

# Default probe set: span both grids (256^2, 512^2) and both dT tiers, incl. a
# near-wall member (Re25k). (root relative to training/data/ensemble_N5_7lag,
# sample index.)
DEFAULT_PROBES = [
    ('FRC-256/sweep_dT_5em3', 500),
    ('FRC-kf4/sweep_dT_1em2', 300),
    ('FRC-combo/sweep_dT_5em3', 700),
    ('FRC-Re25k/sweep_dT_1em2', 200),
]


def vandermonde_rows(S: int) -> np.ndarray:
    x = np.arange(0, -S, -1, dtype=np.float64)
    A = np.array([[x[j] ** m / math.factorial(m) for j in range(S)]
                  for m in range(S)])
    return np.linalg.inv(A).T


def rel(a, b):
    return float(torch.norm(a - b) / torch.norm(b).clamp_min(1e-30))


def check_sample(root: Path, sample: int, dev: str, tol: float):
    man = json.loads((root / 'manifest.json').read_text())
    S = int(man['n_snapshots_per_sample'])
    dt = float(man['Delta_T'])
    Nx, Ny = int(man['Nx']), int(man['Ny'])
    Lx, Ly = float(man['Lx']), float(man['Ly'])
    dx_, dy_ = Lx / Nx, Ly / Ny

    inp = np.load(root / 'packed' / 'inputs.npy', mmap_mode='r')
    N = inp.shape[0]
    s = min(sample, N - 1)
    om = torch.tensor(np.asarray(inp[s, :S], np.float64), device=dev)
    ps = torch.tensor(np.asarray(inp[s, S:2 * S], np.float64), device=dev)

    from qg.solver.grid.cartesian import CartesianGrid
    from qg.solver.opt.derivative import Derivative
    from qg.solver.opt.basis import to_spectral, to_physical
    grid = CartesianGrid(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, device=dev, precision='float64')
    der = Derivative(grid)
    keep = (~der.alias_mask).to(dev)

    def proj(p):
        return to_physical(to_spectral(p) * keep.to(dtype=p.dtype))

    def grad_solver(f):
        fh = to_spectral(f.unsqueeze(0))
        return (to_physical(der.dx * fh).squeeze(0),
                to_physical(der.dy * fh).squeeze(0))

    def jac_flux(psi_f, om_f):
        px, py = grad_solver(psi_f)
        u, v = -py, px
        uq = to_spectral((u * om_f).unsqueeze(0))
        vq = to_spectral((v * om_f).unsqueeze(0))
        der.dealias(uq); der.dealias(vq)
        return to_physical(der.dx * uq + der.dy * vq).squeeze(0)

    # ---- build the model (zero-init cond MLPs) ----
    from model_deriv_closure import build_model
    mdl = build_model('cond_deriv', in_channels=2 * S, out_orders=3, n_time=S,
                      dt=dt, dx=dx_, dy=dy_, physics_init=True).to(dev).double().eval()

    dtv = torch.tensor([dt], device=dev, dtype=torch.float64)
    # model's own TimeFD fields (float32 W_unit stencil, shared with cheap_deriv)
    om_k = mdl.time_fd(om[None], dtv)[0, :4]
    ps_k = mdl.time_fd(ps[None], dtv)[0, :4]

    # ---- Aref: INDEPENDENT solver-spectral advective assembly on those fields ----
    def jac_adv_solver(psi_f, om_f):
        px, py = grad_solver(psi_f)
        wx, wy = grad_solver(om_f)
        return px * wy - py * wx

    Aref = []
    for m in (1, 2, 3):
        acc = 0
        for j in range(m + 1):
            acc = acc - math.comb(m, j) * jac_adv_solver(ps_k[m - j], om_k[j])
        Aref.append(proj(acc))
    Aref = torch.stack(Aref)

    # ---- [spec] flux reference (context only) ----
    Fref = []
    for m in (1, 2, 3):
        acc = 0
        for j in range(m + 1):
            acc = acc + math.comb(m, j) * jac_flux(ps_k[m - j], om_k[j])
        Fref.append(acc)
    Fref = torch.stack(Fref)

    # ---- model output ----
    x = torch.cat([om, ps], 0)[None].double()
    with torch.no_grad():
        M = proj(mdl(x, dt=dtv,
                     dx=torch.tensor([dx_], device=dev, dtype=torch.float64),
                     dy=torch.tensor([dy_], device=dev, dtype=torch.float64))[0])

    # ---- float32-TimeFD contribution (context only) ----
    W = torch.tensor(vandermonde_rows(S), device=dev)
    sc = torch.tensor([dt ** k for k in range(S)], device=dev).view(-1, 1, 1)
    ps_k64 = (torch.einsum('ks,shw->khw', W, ps) / sc)[:4]

    exact = [rel(M[i], Aref[i]) for i in range(3)]
    form = [min(rel(M[i], Fref[i]), rel(M[i], -Fref[i])) for i in range(3)]
    tfd = rel(ps_k[2], ps_k64[2])  # order-2 psi, the worst float32-stencil channel

    ok = all(e < tol for e in exact)
    tag = 'PASS' if ok else 'FAIL'
    print(f"[{tag}] {root.parent.name}/{root.name}  s={s}  {Ny}x{Nx} dt={dt}")
    print("   zero-init exactness rel(M, spectral-advective):  "
          + '  '.join(f"N{m}dot={e:.2e}" for m, e in zip((1, 2, 3), exact))
          + f"   (tol {tol:.0e})")
    print("   [ctx] advective-vs-flux FORM gap rel(M,[spec]):   "
          + '  '.join(f"N{m}dot={e:.2e}" for m, e in zip((1, 2, 3), form)))
    print(f"   [ctx] float32-W_unit TimeFD gap (psi order-2):    {tfd:.2e}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tol', type=float, default=1e-10,
                    help='max allowed rel(M, spectral-advective) per order')
    ap.add_argument('--device', default='cpu',
                    help='cpu is fine and device-independent for this f64 check')
    ap.add_argument('--probes', nargs='*', default=None,
                    help="override probes as 'root:sample' (root under "
                         "training/data/ensemble_N5_7lag)")
    args = ap.parse_args()

    if args.probes:
        probes = []
        for p in args.probes:
            r, _, s = p.rpartition(':')
            probes.append((r, int(s)))
    else:
        probes = DEFAULT_PROBES

    results = []
    for rel_root, sample in probes:
        root = _DATA / rel_root
        if not (root / 'manifest.json').exists() or \
           not (root / 'packed' / 'inputs.npy').exists():
            print(f"[SKIP] {rel_root} -- missing manifest/inputs")
            continue
        results.append(check_sample(root, sample, args.device, args.tol))

    if not results:
        raise SystemExit("no probes ran (no valid roots found)")
    n_ok = sum(results)
    print(f"\n{n_ok}/{len(results)} probes passed zero-init exactness "
          f"(tol {args.tol:.0e}).")
    if n_ok != len(results):
        raise SystemExit(1)
    print("cond_deriv zero-init exactness: CONFIRMED. "
          "Residual to [spec] is the advective-vs-flux form gap (deferred), "
          "not a wiring defect.")


if __name__ == '__main__':
    main()
