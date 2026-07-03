#!/usr/bin/env python
r"""
diagnose_one_sample.py -- stage-by-stage audit of ONE sliced sample.

Builds, in float64, outside the model:
  (1) omega^(k), psi^(k) via a textbook float64 Vandermonde FD on the 7 snapshots
  (2) N^(m) assembled with EXACT SPECTRAL gradients + product dealias
      ("FD-time + exact-space" -- the physics floor of this sample)
  (3) N^(m) assembled with the width-15 INIT central-difference stencils
      (what the physics-init model SHOULD output)
then runs the actual model (init ckpt) on the same sample and compares everything
against the stored f64 target.

Interpretation:
  (2) vs target SMALL, model vs target HUGE      -> model/eval code bug
  (2) vs target HUGE  (same ~50x as init eval)   -> target/bookkeeping bug
  (3) ~ model but (2) << (3)                     -> spatial-stencil pathology
Norm ratios ||pred||/||target|| separate mis-scaling (ratio ~50) from
decorrelation (ratio ~1, error ~1.4).

Usage (from $QG_DIR/training):
    python diagnose_one_sample.py \
        --sliced data/ensemble_N5_7lag/FRC-256/sweep_dT_5em3 \
        --sample 500 --grad-kernel 15 \
        --ckpt data/ensemble_N5_7lag/training_runs/deriv7_equalw_R3R4/init.pt
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


def vandermonde_rows(S: int) -> np.ndarray:
    x = np.arange(0, -S, -1, dtype=np.float64)
    A = np.array([[x[j] ** m / math.factorial(m) for j in range(S)]
                  for m in range(S)])
    return np.linalg.inv(A).T                      # row k = order-k, unit spacing


def central_1d(width: int) -> np.ndarray:
    h = width // 2
    x = np.arange(-h, h + 1, dtype=np.float64)
    A = np.array([[x[j] ** m / math.factorial(m) for j in range(width)]
                  for m in range(width)])
    return np.linalg.inv(A).T[1]                   # first-derivative row, unit dx


def rel(a, b):
    return float(torch.norm(a - b) / torch.norm(b).clamp_min(1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sliced', type=Path, required=True)
    ap.add_argument('--sample', type=int, default=500)
    ap.add_argument('--grad-kernel', type=int, default=15)
    ap.add_argument('--ckpt', type=Path, default=None,
                    help='init.pt (or best.pt) to run the actual model too')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    dev = args.device

    man = json.loads((args.sliced / 'manifest.json').read_text())
    S = int(man['n_snapshots_per_sample'])
    dt = float(man['Delta_T'])
    Nx, Ny = int(man['Nx']), int(man['Ny'])
    Lx, Ly = float(man['Lx']), float(man['Ly'])
    inp = np.load(args.sliced / 'packed' / 'inputs.npy', mmap_mode='r')
    tgt = np.load(args.sliced / 'packed' / 'deriv_anal_f64.npy', mmap_mode='r')
    s = args.sample
    om = torch.tensor(np.asarray(inp[s, :S], np.float64), device=dev)     # (S,Ny,Nx)
    ps = torch.tensor(np.asarray(inp[s, S:2 * S], np.float64), device=dev)
    y = torch.tensor(np.asarray(tgt[s], np.float64), device=dev)          # (3,Ny,Nx)
    print(f"[sample {s}] dt={dt} grid={Ny}x{Nx}  target norms: "
          + '  '.join(f"N{m}dot={torch.norm(y[m-1]):.4e}" for m in (1, 2, 3)))

    # ---- (1) time-FD, float64 textbook ----
    W = torch.tensor(vandermonde_rows(S), device=dev)                     # (S,S)
    om_k = torch.einsum('ks,shw->khw', W, om) / torch.tensor(
        [dt ** k for k in range(S)], device=dev).view(-1, 1, 1)
    ps_k = torch.einsum('ks,shw->khw', W, ps) / torch.tensor(
        [dt ** k for k in range(S)], device=dev).view(-1, 1, 1)
    print("[fd] ||omega^(k)||: " + '  '.join(f"k{k}={torch.norm(om_k[k]):.3e}"
                                             for k in range(4)))

    # ---- solver spectral ops ----
    from qg.solver.grid.cartesian import CartesianGrid
    from qg.solver.opt.derivative import Derivative
    from qg.solver.opt.basis import to_spectral, to_physical
    grid = CartesianGrid(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, device=dev, precision='float64')
    der = Derivative(grid)

    def grad_spec(f):
        fh = to_spectral(f.unsqueeze(0))
        return (to_physical(der.dx * fh).squeeze(0),
                to_physical(der.dy * fh).squeeze(0))

    def jac_flux(psi_f, om_f):
        """Solver-form Jacobian: u=-psi_y, v=+psi_x; J = dx(u q)+dy(v q),
        products dealiased. Returns the SIGNED object the solver adds; the
        N-derivative assembly below tries both signs and reports the better."""
        px, py = grad_spec(psi_f)
        u, v = -py, px
        uq = to_spectral((u * om_f).unsqueeze(0))
        vq = to_spectral((v * om_f).unsqueeze(0))
        der.dealias(uq); der.dealias(vq)
        return to_physical(der.dx * uq + der.dy * vq).squeeze(0)

    def assemble(jfun):
        out = []
        for m in (1, 2, 3):
            acc = 0
            for j in range(m + 1):
                acc = acc + math.comb(m, j) * jfun(ps_k[m - j], om_k[j])
            out.append(acc)
        return torch.stack(out)

    # ---- (2) FD-time + exact spectral space ----
    n_spec = assemble(jac_flux)
    for sign, tag in ((+1, '+'), (-1, '-')):
        errs = [rel(sign * n_spec[m - 1], y[m - 1]) for m in (1, 2, 3)]
        print(f"[spec {tag}] rel vs target: " +
              '  '.join(f"N{m}dot={e:.4f}" for m, e in zip((1, 2, 3), errs)))
    ratios = [float(torch.norm(n_spec[m - 1]) / torch.norm(y[m - 1]).clamp_min(1e-30))
              for m in (1, 2, 3)]
    print("[spec] norm ratio pred/target: " +
          '  '.join(f"N{m}dot={r:.3f}" for m, r in zip((1, 2, 3), ratios)))

    # ---- (3) FD-time + width-K central FD space (init stencils) ----
    K = args.grad_kernel
    w1 = torch.tensor(central_1d(K), device=dev)
    dx_, dy_ = Lx / Nx, Ly / Ny

    def grad_fd(f):
        pad = K // 2
        fp = torch.nn.functional.pad(f[None, None], (pad, pad, pad, pad),
                                     mode='circular')[0, 0]
        gx = sum(w1[i] * torch.roll(f, shifts=-(i - pad), dims=1) for i in range(K)) / dx_
        gy = sum(w1[i] * torch.roll(f, shifts=-(i - pad), dims=0) for i in range(K)) / dy_
        return gx, gy

    def jac_adv_fd(psi_f, om_f):
        px, py = grad_fd(psi_f)
        wx, wy = grad_fd(om_f)
        return px * wy - py * wx

    n_fd = assemble(jac_adv_fd)
    # project like the trainer (dealias prediction)
    keep = (~der.alias_mask).to(dev)
    n_fd_p = to_physical(to_spectral(n_fd) * keep)
    for sign, tag in ((+1, '+'), (-1, '-')):
        errs = [rel(sign * n_fd_p[m - 1], y[m - 1]) for m in (1, 2, 3)]
        print(f"[fdgrad {tag}] rel vs target: " +
              '  '.join(f"N{m}dot={e:.4f}" for m, e in zip((1, 2, 3), errs)))

    # ---- (4) the actual model, if ckpt given ----
    if args.ckpt is not None:
        from model_deriv_closure import build_model
        ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
        mdl = build_model('cheap_deriv', in_channels=2 * S, out_orders=3,
                          grad_kernel=K, dt=dt, dx=dx_, dy=dy_,
                          physics_init=True, learnable_stencils=True
                          ).to(dev).double()
        mdl.load_state_dict(ck['model']); mdl.eval()
        x = torch.cat([om, ps], 0)[None].double()
        with torch.no_grad():
            nd = mdl(x, dt=torch.tensor([dt], device=dev, dtype=torch.float64),
                     dx=torch.tensor([dx_], device=dev, dtype=torch.float64),
                     dy=torch.tensor([dy_], device=dev, dtype=torch.float64))[0]
        nd_p = to_physical(to_spectral(nd) * keep)
        errs = [rel(nd_p[m - 1], y[m - 1]) for m in (1, 2, 3)]
        print("[model] rel vs target: " +
              '  '.join(f"N{m}dot={e:.4f}" for m, e in zip((1, 2, 3), errs)))
        errs2 = [rel(nd_p[m - 1], n_fd_p[m - 1]) for m in (1, 2, 3)]
        print("[model] rel vs hand-FD variant (should be ~0): " +
              '  '.join(f"N{m}dot={e:.2e}" for m, e in zip((1, 2, 3), errs2)))
        ratios = [float(torch.norm(nd_p[m - 1]) / torch.norm(y[m - 1]).clamp_min(1e-30))
                  for m in (1, 2, 3)]
        print("[model] norm ratio pred/target: " +
              '  '.join(f"N{m}dot={r:.3f}" for m, r in zip((1, 2, 3), ratios)))


if __name__ == '__main__':
    main()
