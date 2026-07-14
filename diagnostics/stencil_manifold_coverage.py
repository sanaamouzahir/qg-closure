#!/usr/bin/env python
r"""stencil_manifold_coverage.py -- how much of the derivative task projects
onto the manifold spanned by a width-w 1D stencil (Sanaa mandate 2026-07-14:
price the width-31 wall between the width-15 wall ~0.031 and the spectral
time-FD floor ~0.008 BEFORE the w31 run finishes).

Two levels, all float64, all a-priori (no training):

  OPERATOR level: for width w, solve the spectrum-weighted least-squares
      (Wiener) projection of the spectral symbol i*k onto the antisymmetric
      w-tap stencil manifold,
          min_s  sum_theta W(theta) | 2 sum_j s_j sin(j theta) - theta |^2 ,
      weight W = pooled |f-hat|^2 marginal of the actual fields the model
      differentiates (om^(k), ps^(k), k=0..3, each order normalized to unit
      norm = the equal-weight loss convention). Reported: E_op(w) =
      relative weighted L2 residual, per direction. This IS "the part of the
      operator that does not project onto the w-tap manifold".

  PIPELINE level: assemble N^(m) (m=1..3) with exact textbook time-FD and
      width-w spatial gradients (advective Jacobian, end dealias-projection,
      = the model's structure at physics init), stencils either
        fd    : central-difference max-order taps (init),
        opt   : per-root Wiener-optimal taps (the conditioned ceiling of
                the manifold for the SPATIAL task),
        pool  : one pooled-optimal tap set across all roots (the
                unconditioned ceiling; gap to 'opt' = value of conditioning
                at width w),
      plus the two w->inf references: spectral advective and spectral flux
      (the [spec] floor of diagnose_one_sample). Reported per (root, w,
      variant, order): MEDIAN over val samples of rel-L2 vs the stored f64
      target (rule 16).

Caveat (stated, load-bearing): 'opt' optimizes the SPATIAL approximation
only; the true training optimum also absorbs time-FD error (delta* of the
Wiener theory), so these are manifold-coverage bounds for the spatial task,
not the exact reachable loss. The N^(m) numbers from 'opt' decompose as
time-FD floor (+) spatial projection residual.

Usage (from training/, flat sibling imports; CPU is fine and is the rule
for anything that ends in a plot):
  python ../diagnostics/stencil_manifold_coverage.py \
      --roots data/ensemble_N5_7lag/FRC-{Re25k,combo,kf4}/sweep_dT_{5em3,1p5em2} \
      --widths 7 15 31 63 --n-samples 6 \
      --out ../diagnostics/Results/stencil_manifold_coverage
Writes ONE npz + ONE png under --out (no per-sample litter).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


def _find_training_dir():
    here = Path(__file__).resolve().parent
    for cand in [here.parent / 'training', here]:
        if (cand / 'dataset.py').exists() or (cand / 'deriv_dataset.py').exists():
            return cand
    return here


sys.path.insert(0, str(_find_training_dir()))

ORDERS = ('Ndot', 'Nddot', 'N3dot')


def vandermonde_rows(S: int) -> np.ndarray:
    x = np.arange(0, -S, -1, dtype=np.float64)
    A = np.array([[x[j] ** m / math.factorial(m) for j in range(S)]
                  for m in range(S)])
    return np.linalg.inv(A).T


def central_1d(width: int) -> np.ndarray:
    h = width // 2
    x = np.arange(-h, h + 1, dtype=np.float64)
    A = np.array([[x[j] ** m / math.factorial(m) for j in range(width)]
                  for m in range(width)])
    return np.linalg.inv(A).T[1]


def rel(a, b):
    return float(torch.norm(a - b) / torch.norm(b).clamp_min(1e-30))


def optimal_taps(width: int, Wtheta: np.ndarray) -> np.ndarray:
    """Antisymmetric w-tap Wiener fit to theta on [0,pi): full taps (len w)."""
    h = width // 2
    n = Wtheta.shape[0]                # one-sided modes 0..n-1, Nyquist at n-1
    theta = np.pi * np.arange(n) / (n - 1)
    basis = 2.0 * np.sin(np.outer(theta, np.arange(1, h + 1)))   # (n, h)
    sw = np.sqrt(np.maximum(Wtheta, 0.0))
    A = basis * sw[:, None]
    b = theta * sw
    s_pos, *_ = np.linalg.lstsq(A, b, rcond=None)
    taps = np.zeros(width)
    taps[h + 1:] = s_pos
    taps[:h] = -s_pos[::-1]
    resid = float(np.sqrt(np.sum((A @ s_pos - b) ** 2) /
                          max(np.sum(b ** 2), 1e-300)))
    return taps, resid


class RootData:
    """One (member, dt) sweep root: val samples, manifest, spectra weights."""

    def __init__(self, root: Path, n_samples: int, dev):
        self.root = root
        man = json.loads((root / 'manifest.json').read_text())
        self.S = int(man['n_snapshots_per_sample'])
        self.dt = float(man['Delta_T'])
        self.Nx, self.Ny = int(man['Nx']), int(man['Ny'])
        self.Lx, self.Ly = float(man['Lx']), float(man['Ly'])
        if abs(self.Lx - self.Ly) > 1e-12 or self.Nx != self.Ny:
            raise SystemExit(f"anisotropic root {root} -- not supported")
        self.dx, self.dy = self.Lx / self.Nx, self.Ly / self.Ny
        sp = np.load(root / 'split.npz')
        val = np.asarray(sp['val_idx'])
        stride = max(1, len(val) // n_samples)
        self.idx = val[::stride][:n_samples]
        self.inp = np.load(root / 'packed' / 'inputs.npy', mmap_mode='r')
        self.tgt = np.load(root / 'packed' / 'deriv_anal_f64.npy', mmap_mode='r')
        self.Wv = torch.tensor(vandermonde_rows(self.S), device=dev)
        self.dev = dev

    def fields(self, s: int):
        """Return om_k, ps_k (4,Ny,Nx each, k=0..3) and target (3,Ny,Nx)."""
        S = self.S
        om = torch.tensor(np.asarray(self.inp[s, :S], np.float64), device=self.dev)
        ps = torch.tensor(np.asarray(self.inp[s, S:2 * S], np.float64), device=self.dev)
        y = torch.tensor(np.asarray(self.tgt[s], np.float64), device=self.dev)
        sc = torch.tensor([self.dt ** k for k in range(S)],
                          device=self.dev).view(-1, 1, 1)
        om_k = (torch.einsum('ks,shw->khw', self.Wv, om) / sc)[:4]
        ps_k = (torch.einsum('ks,shw->khw', self.Wv, ps) / sc)[:4]
        return om_k, ps_k, y

    def spectrum_weights(self):
        """Pooled 1D marginals W(theta_x), W(theta_y) over val samples.

        Weight convention (load-bearing): each field's marginal power is
        normalized by that field's own DERIVATIVE norm along the direction,
        so the LS objective sum_theta W |shat - theta|^2 is the SUM OF
        RELATIVE derivative errors over fields -- one shared stencil, every
        field's rel error counting equally (red psi fields demand low-theta
        exactness, white omega^(k) fields demand the high-theta band; the
        naive field-norm weighting lets the fit destroy psi_x, observed
        0.31 rel vs FD 2e-4 -- do not revert).
        Interior one-sided columns double-counted (Hermitian factor 2)."""
        n = self.Nx // 2 + 1
        m_idx = np.arange(n, dtype=np.float64)
        dbl = np.where((m_idx > 0) & (m_idx < n - 1), 2.0, 1.0)
        Wx = np.zeros(n)
        Wy = np.zeros(n)
        for s in self.idx:
            om_k, ps_k, _ = self.fields(int(s))
            for f in list(om_k) + list(ps_k):
                fh = torch.fft.rfft2(f)                       # (Ny, Nx//2+1)
                p2 = (fh.real ** 2 + fh.imag ** 2).cpu().numpy().sum(axis=0)
                p2 *= dbl
                dn2 = float((m_idx ** 2 * p2).sum())          # prop ||d_x f||^2
                if dn2 > 0:
                    Wx += p2 / dn2
                fhy = torch.fft.rfft2(f.T)
                p2y = (fhy.real ** 2 + fhy.imag ** 2).cpu().numpy().sum(axis=0)
                p2y *= dbl
                dn2y = float((m_idx ** 2 * p2y).sum())
                if dn2y > 0:
                    Wy += p2y / dn2y
        return Wx, Wy


def make_ops(Nx, Ny, Lx, Ly, dev):
    from qg.solver.grid.cartesian import CartesianGrid
    from qg.solver.opt.derivative import Derivative
    from qg.solver.opt.basis import to_spectral, to_physical
    grid = CartesianGrid(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, device=dev,
                         precision='float64')
    der = Derivative(grid)
    return der, to_spectral, to_physical


def assemble(jfun, om_k, ps_k):
    out = []
    for m in (1, 2, 3):
        acc = 0
        for j in range(m + 1):
            acc = acc + math.comb(m, j) * jfun(ps_k[m - j], om_k[j])
        out.append(acc)
    return torch.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--roots', nargs='+', required=True, type=Path)
    ap.add_argument('--widths', nargs='+', type=int, default=[7, 15, 31, 63])
    ap.add_argument('--n-samples', type=int, default=6)
    ap.add_argument('--out', type=Path,
                    default=Path('../diagnostics/Results/stencil_manifold_coverage'))
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    dev = args.device
    args.out.mkdir(parents=True, exist_ok=True)

    roots = [RootData(r, args.n_samples, dev) for r in args.roots]

    # ---------- pass 1: spectra weights, per root + pooled ----------
    weights = {}
    for rd in roots:
        Wx, Wy = rd.spectrum_weights()
        weights[str(rd.root)] = (Wx, Wy)
        print(f"[weights] {rd.root.parent.name}/{rd.root.name}: "
              f"n_val_used={len(rd.idx)}", flush=True)
    shapes = {rd.Nx for rd in roots}
    if len(shapes) != 1:
        raise SystemExit("mixed grids -- run per shape (one theta convention)")
    Wx_pool = sum(w[0] for w in weights.values())
    Wy_pool = sum(w[1] for w in weights.values())

    # ---------- operator-level coverage ----------
    op_rows = []       # (root, width, dir, variant) -> E_op
    taps_all = {}      # (rootkey, w) -> (taps_x, taps_y); 'POOL' for pooled
    for w in args.widths:
        tx, ex = optimal_taps(w, Wx_pool)
        ty, ey = optimal_taps(w, Wy_pool)
        taps_all[('POOL', w)] = (tx, ty)
        op_rows.append(('POOL', w, 'x', 'opt', ex))
        op_rows.append(('POOL', w, 'y', 'opt', ey))
        for rd in roots:
            k = str(rd.root)
            tx, ex = optimal_taps(w, weights[k][0])
            ty, ey = optimal_taps(w, weights[k][1])
            taps_all[(k, w)] = (tx, ty)
            op_rows.append((k, w, 'x', 'opt', ex))
            op_rows.append((k, w, 'y', 'opt', ey))
        # FD-init taps residual under pooled weight, for the same curve
        fd = central_1d(w)
        n = Wx_pool.shape[0]
        theta = np.pi * np.arange(n) / (n - 1)
        h = w // 2
        sym = 2.0 * np.sin(np.outer(theta, np.arange(1, h + 1))) @ fd[h + 1:]
        for tag, Wt in (('x', Wx_pool), ('y', Wy_pool)):
            e = float(np.sqrt(np.sum(Wt * (sym - theta) ** 2) /
                              max(np.sum(Wt * theta ** 2), 1e-300)))
            op_rows.append(('POOL', w, tag, 'fd', e))
        print(f"[operator] w={w}: pooled-opt x={ex:.5f}  fd(x,pool)={e:.5f}",
              flush=True)

    # ---------- pass 2: pipeline eps per (root, width, variant, order) ----------
    pipe = {}          # key -> list over samples of (e1,e2,e3)

    for rd in roots:
        der, to_spectral, to_physical = make_ops(rd.Nx, rd.Ny, rd.Lx, rd.Ly, dev)
        keep = (~der.alias_mask).to(dev)

        def grad_spec(f):
            fh = to_spectral(f.unsqueeze(0))
            return (to_physical(der.dx * fh).squeeze(0),
                    to_physical(der.dy * fh).squeeze(0))

        def jac_flux(psi_f, om_f):
            px, py = grad_spec(psi_f)
            u, v = -py, px
            uq = to_spectral((u * om_f).unsqueeze(0))
            vq = to_spectral((v * om_f).unsqueeze(0))
            der.dealias(uq); der.dealias(vq)
            return to_physical(der.dx * uq + der.dy * vq).squeeze(0)

        def jac_adv_spec(psi_f, om_f):
            px, py = grad_spec(psi_f)
            wx, wy = grad_spec(om_f)
            return px * wy - py * wx

        def make_jac_taps(taps_x, taps_y):
            twx = torch.tensor(taps_x, device=dev)
            twy = torch.tensor(taps_y, device=dev)
            K = len(taps_x)
            pad = K // 2

            def grad_fd(f):
                gx = sum(twx[i] * torch.roll(f, shifts=-(i - pad), dims=1)
                         for i in range(K)) / rd.dx
                gy = sum(twy[i] * torch.roll(f, shifts=-(i - pad), dims=0)
                         for i in range(K)) / rd.dy
                return gx, gy

            def jac(psi_f, om_f):
                px, py = grad_fd(psi_f)
                wx, wy = grad_fd(om_f)
                return px * wy - py * wx
            return jac

        rk = str(rd.root)
        variants = {'spec_flux': jac_flux, 'spec_adv': jac_adv_spec}
        for w in args.widths:
            fd = central_1d(w)
            variants[f'fd_w{w}'] = make_jac_taps(fd, fd)
            variants[f'opt_w{w}'] = make_jac_taps(*taps_all[(rk, w)])
            variants[f'pool_w{w}'] = make_jac_taps(*taps_all[('POOL', w)])

        for s in rd.idx:
            om_k, ps_k, y = rd.fields(int(s))
            for name, jf in variants.items():
                n_m = assemble(jf, om_k, ps_k)
                # trainer convention: dealias-project prediction, then the
                # sign convention (target may be -assembled, solver-form):
                # take the better sign
                n_p = to_physical(to_spectral(n_m) * keep)
                ep = [rel(n_p[m], y[m]) for m in range(3)]
                em = [rel(-n_p[m], y[m]) for m in range(3)]
                errs = ep if sum(ep) <= sum(em) else em
                pipe.setdefault((rk, name), []).append(errs)
        print(f"[pipeline] {rd.root.parent.name}/{rd.root.name} done "
              f"({len(rd.idx)} samples x {len(variants)} variants)", flush=True)

    # ---------- aggregate + save ----------
    med = {k: np.median(np.array(v), axis=0) for k, v in pipe.items()}
    out_npz = {}
    print("\n===== MEDIAN rel-L2 vs f64 target (per root; Ndot/Nddot/N3dot) =====")
    for (rk, name), m in sorted(med.items()):
        tag = f"{Path(rk).parent.name}/{Path(rk).name}"
        print(f"{tag:34s} {name:12s} " +
              '  '.join(f"{ORDERS[i]}={m[i]:.4f}" for i in range(3)))
        out_npz[f"pipe|{tag}|{name}"] = m
    for row in op_rows:
        out_npz[f"op|{row[0]}|w{row[1]}|{row[2]}|{row[3]}"] = np.array(row[4])
    for (rk, w), (tx, ty) in taps_all.items():
        tag = 'POOL' if rk == 'POOL' else \
            f"{Path(rk).parent.name}/{Path(rk).name}"
        out_npz[f"taps|{tag}|w{w}|x"] = tx
        out_npz[f"taps|{tag}|w{w}|y"] = ty
    np.savez(args.out / 'stencil_manifold_coverage.npz', **out_npz)

    # ---------- one figure: eps_Nddot vs width ----------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    root_tags = sorted({f"{Path(k[0]).parent.name}/{Path(k[0]).name}"
                        for k in med})
    for ax, order_i in zip(axes, range(3)):
        for tag in root_tags:
            for var, ls in (('fd', ':'), ('pool', '--'), ('opt', '-')):
                ys = [med[(k, f'{var}_w{w}')][order_i]
                      for w in args.widths
                      for k in [next(kk for (kk, nn) in med
                                     if f"{Path(kk).parent.name}/{Path(kk).name}"
                                     == tag and nn == f'{var}_w{w}')]]
                ax.plot(args.widths, ys, ls, marker='o', ms=3, alpha=0.75,
                        label=f"{tag} {var}" if order_i == 1 else None)
            k0 = next(kk for (kk, nn) in med
                      if f"{Path(kk).parent.name}/{Path(kk).name}" == tag
                      and nn == 'spec_flux')
            ax.axhline(med[(k0, 'spec_flux')][order_i], color='k', lw=0.6,
                       alpha=0.4)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xticks(args.widths)
        ax.set_xticklabels([str(w) for w in args.widths])
        ax.set_xlabel('stencil width w')
        ax.set_title(rf'$\epsilon$ {ORDERS[order_i]} (median, val)')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('rel-L2 vs f64 target')
    axes[1].legend(fontsize=6, ncol=2, loc='upper right')
    fig.suptitle('Manifold coverage: FD-init (dotted) vs pooled-Wiener (dashed) '
                 'vs per-root-Wiener (solid); black = spectral-flux floor')
    fig.tight_layout()
    fig.savefig(args.out / 'stencil_manifold_coverage.png', dpi=160)
    print(f"\n[out] {args.out}/stencil_manifold_coverage.npz + .png")


if __name__ == '__main__':
    main()
