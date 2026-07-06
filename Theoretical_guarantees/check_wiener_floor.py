#!/usr/bin/env python
r"""
check_wiener_floor.py -- 1.b empirical floor decomposition, v2 (measured, no models).

Per (member, dt): build the ACTUAL depth-7 FD estimate of omega^(m) from the
deep marks, subtract the ANALYTIC truth -> the true time-error field e. Then:

  raw    = ||e|| / ||omega^(m)||          (must reproduce the k=7 console table:
                                           Re25k 0.020/0.411/1.510 for m=2 --
                                           built-in validation anchor)
  coh(kappa) = |<e, omega^(m)>_shell|^2 / (E_e E_m)   COMPLEX magnitude --
           the fraction of the error coherent with the field the stencil
           filters; the quadrature (odd S-m) relation is reachable by a real
           odd stencil (symbol ~ ik), so the complex modulus is the right
           coherence. (v1 used the real part, which vanishes structurally for
           odd S-m -- the coh~1e-4 artifact.)
  (i)    = sum_kappa E_e(kappa) (1 - coh(kappa)) / sum E_e   -- irreducible.
  r_j(kappa) = <e, omega^(m)>_shell / E_m(kappa)   -- the MEASURED per-config
           optimal transfer (complex). The family {r_j} over (member,dt) is
           what one shared stencil must compromise across.
  (ii)   = uniform-tier-weighted variance of r_j around the pooled mean,
           normalized by the weighted mean |r_j|^2. (v1's energy weighting let
           the largest tier dominate and collapsed this to 0.013.)
  (iii)  = width-W reachability residue of the pooled |r_bar(kappa)| * k symbol
           on the sin(jk) odd-stencil basis.

Everything float64, analytic recursion, dealiased flux Jacobians, manifest
forcing, post-filter windows only.

Run FROM $QG_DIR/training on a GPU node (qlogin first):
    python Theoretical_guarantees/check_wiener_floor.py \
        --members data/ensemble_N5_7lag/FRC-Re25k data/ensemble_N5_7lag/FRC-kf4 \
                  data/ensemble_N5_7lag/FRC-256 \
        --dts 5e-3 1e-2 1.5e-2 --m 2 --n-windows 12 --grad-kernel 15
Outputs -> Theoretical_guarantees/Results/wiener_floor/.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qg.solver.grid.cartesian import CartesianGrid
from qg.solver.opt.derivative import Derivative
from qg.solver.opt.basis import to_spectral, to_physical

OUT_DEFAULT = Path(__file__).resolve().parent / 'Results' / 'wiener_floor'


def build_L_hat(der, nu, mu, beta):
    L = nu * der.laplacian - mu
    if beta != 0.0:
        L = L - beta * der.dx * der.inv_laplacian
    return L


def build_F(man, grid, dev):
    fc = man.get('forcing', None)
    if not isinstance(fc, dict):
        return None
    A = float(fc.get('A', 0)); B = float(fc.get('B', 0))
    D = float(fc.get('D', 0)); E = float(fc.get('E', 0))
    if A == 0 and D == 0:
        return None
    x = torch.linspace(0, grid.Lx, grid.Nx, device=dev, dtype=torch.float64)
    y = torch.linspace(0, grid.Ly, grid.Ny, device=dev, dtype=torch.float64)
    return A * torch.cos(B * x[None, :]) + D * torch.cos(E * y[:, None])


def jac_flux(psi, om, der):
    ph = to_spectral(psi.unsqueeze(0))
    u = to_physical(-der.dy * ph).squeeze(0)
    v = to_physical(der.dx * ph).squeeze(0)
    uq = to_spectral((u * om).unsqueeze(0)); vq = to_spectral((v * om).unsqueeze(0))
    der.dealias(uq); der.dealias(vq)
    return to_physical(der.dx * uq + der.dy * vq).squeeze(0)


def recursion_omega(om, psi, der, L_hat, F, max_k):
    """omega^(0..max_k) via the exact spectral recursion."""
    oms = [om]; pss = [psi]
    for m in range(max_k):
        acc = None
        for j in range(m + 1):
            t = math.comb(m, j) * jac_flux(pss[m - j], oms[j], der)
            acc = t if acc is None else acc + t
        Nm = -acc
        if m == 0 and F is not None:
            Nm = Nm + F
        onext = to_physical(L_hat * to_spectral(oms[m].unsqueeze(0))).squeeze(0) + Nm
        oms.append(onext)
        pss.append(to_physical(der.inv_laplacian *
                               to_spectral(onext.unsqueeze(0))).squeeze(0))
    return oms


def fd_rows(S):
    x = np.arange(0, -S, -1, dtype=np.float64)
    A = np.array([[x[j] ** mm / math.factorial(mm) for j in range(S)]
                  for mm in range(S)])
    return np.linalg.inv(A).T


def shell_index(der, Ny, Nx, dev):
    kxg = der.dx.imag; kyg = der.dy.imag
    kmag = torch.sqrt((kxg ** 2 + kyg ** 2)).squeeze()
    probe = to_spectral(torch.zeros(1, Ny, Nx, dtype=torch.float64, device=dev))
    if kmag.shape != probe.shape[-2:]:
        kmag = torch.sqrt(kxg.squeeze()[None, :] ** 2 + kyg.squeeze()[:, None] ** 2)
    return torch.round(kmag).to(torch.int64)


def sadd(x_flat, sh_flat, n_sh):
    out = torch.zeros(n_sh, dtype=x_flat.dtype, device=x_flat.device)
    out.scatter_add_(0, sh_flat, x_flat)
    return out


def reachable_residual(g_mag, width):
    """(iii): project s(k) = g(|k|)*k (odd real filter target, grid units)
    onto span{sin(jk)}_{j=1..width//2}; return relative unreachable energy."""
    K = len(g_mag)
    kk = np.linspace(1e-3, math.pi, 256)
    kap = np.arange(K, dtype=np.float64) / max(K - 1, 1) * math.pi
    tgt = np.interp(kk, kap, g_mag) * kk
    H = width // 2
    B = np.stack([np.sin(j * kk) for j in range(1, H + 1)], axis=1)
    coef, *_ = np.linalg.lstsq(B, tgt, rcond=None)
    res = tgt - B @ coef
    return float((res ** 2).sum() / max((tgt ** 2).sum(), 1e-300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--members', nargs='+', type=Path, required=True)
    ap.add_argument('--dts', type=float, nargs='+', default=[5e-3, 1e-2, 1.5e-2])
    ap.add_argument('--m', type=int, default=2, help='target order (2 = Nddot proxy)')
    ap.add_argument('--S', type=int, default=7, help='stencil depth')
    ap.add_argument('--n-windows', type=int, default=12)
    ap.add_argument('--grad-kernel', type=int, default=15)
    ap.add_argument('--observed-floor', type=float, default=0.19)
    ap.add_argument('--skip-fullslot', action='store_true',
                    help='skip CHECK 3 (the full-slot regression) for speed')
    ap.add_argument('--out', type=Path, default=OUT_DEFAULT)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    dev = args.device
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    m, S = args.m, args.S
    W = fd_rows(S)
    wrow = torch.tensor(W[m], dtype=torch.float64, device=dev)

    rows = []
    r_family, coh_store = {}, {}

    for mem in sorted(args.members):
        deep = mem / 'forced_turbulence_dT_5em3'
        if not (deep / 'packed' / 'inputs.npy').exists():
            print(f"[skip] {mem.name}"); continue
        man = json.loads((deep / 'manifest.json').read_text())
        M = int(man['n_snapshots_per_sample'])
        dtf = float(man['Delta_T'])
        Nx, Ny = int(man['Nx']), int(man['Ny'])
        nu = float(man.get('nu', 0)); mu = float(man.get('mu', 0))
        beta = float(man.get('beta', man.get('B', 0)))
        inp = np.load(deep / 'packed' / 'inputs.npy', mmap_mode='r')

        surv = np.arange(inp.shape[0])
        sweep = mem / 'sweep_dT_5em3'
        if (sweep / 'split.npz').exists():
            sp = np.load(sweep / 'split.npz')
            na = int(json.loads((sweep / 'manifest.json').read_text()
                                ).get('n_anchors', 1))
            surv = np.unique(np.concatenate([sp[k] for k in sp.files]) // na)
        picks = surv[np.linspace(0, len(surv) - 1,
                                 min(args.n_windows, len(surv)), dtype=int)]

        grid = CartesianGrid(Nx=Nx, Ny=Ny, Lx=float(man['Lx']), Ly=float(man['Ly']),
                             device=dev, precision='float64')
        der = Derivative(grid)
        L_hat = build_L_hat(der, nu, mu, beta)
        F = build_F(man, grid, dev)
        sh = shell_index(der, Ny, Nx, dev)
        sh_flat = sh.reshape(-1)
        n_sh = int(sh.max().item()) + 1

        # ---- CHECK 2 accumulators: 2-mark sigma-hat vs analytic sigma (per member)
        E0 = torch.zeros(n_sh, dtype=torch.float64, device=dev)      # |w0|^2
        Ed_hat = torch.zeros(n_sh, dtype=torch.float64, device=dev)  # |FD2 wdot|^2
        Ed_true = torch.zeros(n_sh, dtype=torch.float64, device=dev) # |wdot true|^2

        for dt in args.dts:
            j = int(round(dt / dtf))
            if (S - 1) * j > M - 1:
                print(f"  [{mem.name}] dT={dt}: span exceeds marks, skip"); continue

            E_e = torch.zeros(n_sh, dtype=torch.float64, device=dev)
            E_m = torch.zeros(n_sh, dtype=torch.float64, device=dev)
            Xc = torch.zeros(n_sh, dtype=torch.complex128, device=dev)
            num2 = den2 = 0.0
            # CHECK 3 accumulators (full-slot regression at the N^(m) level):
            # regressors = response of the assembled N^(m) to a unit basis filter
            # B_q^d applied to channel c (c in psi^(0..m), omega^(0..m)).
            H = args.grad_kernel // 2
            n_reg = 2 * (m + 1) * 2 * H          # channels x {psi,omega} x dir x q
            G3 = torch.zeros(n_reg, n_reg, dtype=torch.float64, device=dev)
            b3 = torch.zeros(n_reg, dtype=torch.float64, device=dev)
            eN2 = 0.0

            for wdx in picks:
                marks = torch.tensor(
                    np.asarray(inp[wdx, [i * j for i in range(S)]], np.float64),
                    device=dev)                                   # (S, Ny, Nx)
                psi0 = torch.tensor(np.asarray(inp[wdx, M], np.float64), device=dev)
                est = torch.einsum('s,shw->hw', wrow, marks) / dt ** m
                truth = recursion_omega(marks[0], psi0, der, L_hat, F, m)[m]
                e = est - truth
                num2 += float(e.pow(2).sum()); den2 += float(truth.pow(2).sum())
                eh = to_spectral(e.unsqueeze(0)).squeeze(0)
                th = to_spectral(truth.unsqueeze(0)).squeeze(0)
                E_e += sadd((eh.real**2 + eh.imag**2).reshape(-1), sh_flat, n_sh)
                E_m += sadd((th.real**2 + th.imag**2).reshape(-1), sh_flat, n_sh)
                Xc += sadd((eh * th.conj()).reshape(-1), sh_flat, n_sh)
                # CHECK 2 (once per member: use the finest tier only)
                if dt == args.dts[0]:
                    w0h = to_spectral(marks[0].unsqueeze(0)).squeeze(0)
                    wd_hat = (marks[0] - marks[1]) / dtf          # 2-mark FD wdot
                    wd_true = recursion_omega(marks[0], psi0, der, L_hat, F, 1)[1]
                    dh = to_spectral(wd_hat.unsqueeze(0)).squeeze(0)
                    dth = to_spectral(wd_true.unsqueeze(0)).squeeze(0)
                    E0 += sadd((w0h.real**2 + w0h.imag**2).reshape(-1), sh_flat, n_sh)
                    Ed_hat += sadd((dh.real**2 + dh.imag**2).reshape(-1),
                                   sh_flat, n_sh)
                    Ed_true += sadd((dth.real**2 + dth.imag**2).reshape(-1),
                                    sh_flat, n_sh)

                # ------- CHECK 3: N-level error + full-slot response basis -------
                if not args.skip_fullslot:
                    psim = torch.tensor(
                        np.asarray(inp[wdx, [M + i * j for i in range(S)]],
                                   np.float64), device=dev)
                    Wall = torch.tensor(fd_rows(S)[:m + 1], dtype=torch.float64,
                                        device=dev)
                    scal = torch.tensor([dt ** kk for kk in range(m + 1)],
                                        device=dev).view(-1, 1, 1)
                    om_fd = torch.einsum('ks,shw->khw', Wall, marks) / scal
                    ps_fd = torch.einsum('ks,shw->khw', Wall, psim) / scal
                    oms_t = recursion_omega(marks[0], psi0, der, L_hat, F, m)
                    pss_t = [psi0] + [
                        to_physical(der.inv_laplacian *
                                    to_spectral(o.unsqueeze(0))).squeeze(0)
                        for o in oms_t[1:m + 1]]

                    def gspec(f):
                        fh = to_spectral(f.unsqueeze(0))
                        return (to_physical(der.dx * fh).squeeze(0),
                                to_physical(der.dy * fh).squeeze(0))

                    def asm(o_l, p_l):
                        acc = None
                        for jj in range(m + 1):
                            ax_, ay_ = gspec(p_l[m - jj])
                            bx_, by_ = gspec(o_l[jj])
                            t = math.comb(m, jj) * (ax_ * by_ - ay_ * bx_)
                            acc = t if acc is None else acc + t
                        return -acc

                    eN = asm(list(om_fd), list(ps_fd)) - asm(oms_t[:m + 1], pss_t)
                    eN2 += float(eN.pow(2).sum())

                    grads_p = [gspec(pss_t[kk]) for kk in range(m + 1)]
                    grads_o = [gspec(oms_t[kk]) for kk in range(m + 1)]

                    def bfilt(f, q, dimm):
                        return (torch.roll(f, -q, dims=dimm)
                                - torch.roll(f, q, dims=dimm)) / 2.0

                    Rlist = []
                    for kk in range(m + 1):
                        for is_psi in (True, False):
                            f_t = pss_t[kk] if is_psi else oms_t[kk]
                            for d, dimm in (('x', 1), ('y', 0)):
                                for q in range(1, H + 1):
                                    Bf = bfilt(f_t, q, dimm)
                                    acc = None
                                    for jj in range(m + 1):
                                        c = float(math.comb(m, jj))
                                        px_, py_ = grads_p[m - jj]
                                        wx_, wy_ = grads_o[jj]
                                        if is_psi and (m - jj) == kk:
                                            t = Bf * wy_ if d == 'x' else -Bf * wx_
                                        elif (not is_psi) and jj == kk:
                                            t = -py_ * Bf if d == 'x' else px_ * Bf
                                        else:
                                            continue
                                        acc = c * t if acc is None else acc + c * t
                                    Rlist.append(-acc if acc is not None
                                                 else torch.zeros_like(eN))
                    Rmat = torch.stack([r.reshape(-1) for r in Rlist])
                    G3 += Rmat @ Rmat.T
                    b3 += Rmat @ eN.reshape(-1)

            raw = math.sqrt(num2 / max(den2, 1e-300))
            coh = (Xc.abs() ** 2 / (E_e * E_m).clamp_min(1e-300)).cpu().numpy()
            E_e_np = E_e.cpu().numpy()
            irr_frac = float((E_e_np * (1 - coh)).sum() / max(E_e_np.sum(), 1e-300))
            r_j = (Xc / E_m.clamp_min(1e-300).to(torch.complex128)).cpu().numpy()
            r_family[(mem.name, dt)] = r_j
            coh_store[(mem.name, dt)] = (coh, E_e_np, E_m.cpu().numpy())

            # CHECK 3 solve: absorbed fraction under the FULL slot basis
            abs3 = float('nan')
            if not args.skip_fullslot and eN2 > 0:
                reg = 1e-12 * float(torch.diagonal(G3).mean()) + 1e-300
                sol = torch.linalg.solve(
                    G3 + reg * torch.eye(n_reg, dtype=torch.float64, device=dev),
                    b3)
                abs3 = float((b3 @ sol) / eN2)
                abs3 = min(max(abs3, 0.0), 1.0)
            rows.append(dict(member=mem.name, dt=dt, raw=raw,
                             irr_frac=irr_frac,
                             irreducible=raw * math.sqrt(irr_frac),
                             abs_fullslot=abs3,
                             ceiling_cond=(raw * math.sqrt(max(1 - abs3, 0.0))
                                           if abs3 == abs3 else float('nan'))))
            xtra = (f"  FULL-SLOT absorbable={100*abs3:.1f}%  "
                    f"conditioned-ceiling={raw*math.sqrt(max(1-abs3,0)):.4f}"
                    if abs3 == abs3 else "")
            print(f"  {mem.name:10s} dT={dt:<7g} raw={raw:.4f}  "
                  f"diag-absorbable={100*(1-irr_frac):.1f}%  "
                  f"(i)-slice={raw*math.sqrt(irr_frac):.4f}{xtra}")

        # ---- CHECK 2 plot: 2-mark sigma-hat(kappa) vs analytic sigma(kappa) ----
        sig_hat = torch.sqrt(Ed_hat / E0.clamp_min(1e-300)).cpu().numpy()
        sig_true = torch.sqrt(Ed_true / E0.clamp_min(1e-300)).cpu().numpy()
        band = E0.cpu().numpy() > 1e-12 * float(E0.max())
        rel = np.median(np.abs(sig_hat[band] - sig_true[band])
                        / np.maximum(sig_true[band], 1e-300))
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        ax.plot(np.arange(n_sh)[band], sig_true[band], 'k-',
                label=r'analytic $\sigma_\omega(\kappa)$')
        ax.plot(np.arange(n_sh)[band], sig_hat[band], 'r--',
                label=r'2-mark FD $\hat\sigma_\omega(\kappa)$')
        ax.set_xlabel(r'shell $\kappa$'); ax.set_ylabel(r'$\sigma_\omega(\kappa)$')
        ax.set_title(f'{mem.name}: conditioning input fidelity '
                     f'(median rel diff {100*rel:.1f}%)')
        ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(args.out / f'sigma_hat_fidelity_{mem.name}.png', dpi=160)
        plt.close(fig)
        print(f"  [{mem.name}] CHECK 2: 2-mark sigma-hat vs analytic, "
              f"median rel diff = {100*rel:.1f}%")

    # ---------------- pooled (ii)/(iii): VALID BAND + uniform tiers ----------------
    keys = list(r_family.keys())
    Kmin = min(len(r_family[k]) for k in keys)
    # valid band: every tier must have real field energy at the shell (inside its
    # OWN 2/3 cutoff) -- kills the E_m~0 division artifact beyond the coarsest
    # member's cutoff (the kappa>60 explosion for FRC-256).
    Em_all = np.stack([coh_store[k][2][:Kmin] for k in keys])
    Kband = int(min(np.nonzero(Em_all[jj] > 1e-10 * Em_all[jj].max())[0].max()
                    for jj in range(len(keys))))
    R = np.stack([r_family[k][:Kband] for k in keys])            # (J, Kband)
    Ew = np.stack([coh_store[k][1][:Kband] for k in keys]).mean(axis=0)
    r_bar = R.mean(axis=0)
    num_var = np.mean(np.abs(R - r_bar[None]) ** 2, axis=0)
    den_var = np.mean(np.abs(R) ** 2, axis=0)
    var_ii = float((Ew * num_var).sum() / max((Ew * den_var).sum(), 1e-300))
    res_iii = reachable_residual(np.abs(r_bar), args.grad_kernel)

    print(f"\npooled slices (m={m}, S={S}, width={args.grad_kernel}, "
          f"uniform tiers, valid band kappa<={Kband}):")
    print(f"  (ii) pooled-variance mismatch fraction = {var_ii:.3f}")
    print(f"  (iii) width-{args.grad_kernel} shape-deficit fraction = {res_iii:.3f}")
    print(f"  observed trained plateau (pooled Nddot) = {args.observed_floor}")

    # ---------------- CHECK 1: dT-collapse of r_j / dT^(S-m) ----------------
    members_seen = sorted({k[0] for k in keys})
    for name in members_seen:
        ks = [k for k in keys if k[0] == name]
        if len(ks) < 2:
            continue
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        curves = []
        for k in ks:
            g = np.abs(r_family[k][:Kband]) / (k[1] ** (S - m))
            curves.append(g)
            ax.plot(np.arange(Kband), g, label=f'dT={k[1]:g}')
        C = np.stack(curves)
        # collapse metric: median over shells of (max/min) across dts
        with np.errstate(divide='ignore', invalid='ignore'):
            spread = np.nanmedian(C.max(axis=0) / np.maximum(C.min(axis=0), 1e-300))
        ax.set_yscale('log'); ax.set_xlabel(r'shell $\kappa$')
        ax.set_ylabel(r'$|r_j(\kappa)| / \Delta T^{\,S-m}$')
        ax.set_title(f'{name}: dT-collapse (median spread {spread:.2f}x; '
                     f'1 = perfect factorization)')
        ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(args.out / f'dT_collapse_{name}.png', dpi=160); plt.close(fig)
        print(f"  CHECK 1 [{name}]: dT-collapse median spread = {spread:.2f}x "
              f"(1.0 = exact dT^(S-m) factorization; >>1 = h.o.t. matter)")

    # ---------------- plots ----------------
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for (name, dt), (coh, _) in coh_store.items():
        ax.plot(np.arange(len(coh)), coh, alpha=0.75, label=f'{name} dT={dt:g}')
    ax.set_xlabel(r'shell $\kappa$')
    ax.set_ylabel(r'$|\langle e,\,\omega^{(m)}\rangle|^2 / (E_e E_m)$')
    ax.set_title(rf'absorbable (coherent) fraction of the FD error, $m={m}$, $S={S}$')
    ax.legend(fontsize=6); fig.tight_layout()
    fig.savefig(args.out / 'coherence_per_shell.png', dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for k in keys:
        ax.plot(np.arange(Kmin), np.abs(r_family[k][:Kmin]), alpha=0.6,
                label=f'{k[0]} dT={k[1]:g}')
    ax.plot(np.arange(Kmin), np.abs(r_bar), 'k-', lw=2.5, label='pooled compromise')
    ax.set_yscale('log'); ax.set_xlabel(r'shell $\kappa$')
    ax.set_ylabel(r'$|r_j(\kappa)|$ (measured optimal transfer)')
    ax.set_title('per-(member, dT) optimal correction vs the one shared stencil')
    ax.legend(fontsize=6); fig.tight_layout()
    fig.savefig(args.out / 'r_family_vs_pooled.png', dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    mean_i = float(np.mean([r['irr_frac'] for r in rows]))
    ax.bar(['(i) incoherent', '(ii) pooled var', f'(iii) width-{args.grad_kernel}'],
           [mean_i, var_ii, res_iii])
    ax.set_ylabel('fraction'); ax.set_title('Wiener-floor decomposition (pooled)')
    fig.tight_layout(); fig.savefig(args.out / 'floor_decomposition.png', dpi=160)
    plt.close(fig)

    with open(args.out / 'wiener_floor_summary.csv', 'w', newline='') as f:
        wcsv = csv.writer(f)
        wcsv.writerow(['member', 'dt', 'raw_fd_floor', 'irr_frac',
                       'irreducible_i', 'abs_fullslot', 'ceiling_conditioned'])
        for r in rows:
            wcsv.writerow([r['member'], r['dt'], r['raw'], r['irr_frac'],
                           r['irreducible'], r['abs_fullslot'],
                           r['ceiling_cond']])
        wcsv.writerow([]); wcsv.writerow(['pooled_var_ii', var_ii])
        wcsv.writerow(['shape_deficit_iii', res_iii])
        wcsv.writerow(['valid_band_kappa_max', Kband])
        wcsv.writerow(['observed_floor', args.observed_floor])
    if not args.skip_fullslot:
        cc = [r['ceiling_cond'] for r in rows if r['ceiling_cond'] == r['ceiling_cond']]
        if cc:
            print(f"\nDECISION LINE: pooled conditioned ceiling (mean over tiers) "
                  f"= {float(np.mean(cc)):.4f} vs observed unconditioned plateau "
                  f"{args.observed_floor}. The gap is conditioning's measured prize.")
    print(f"[1b] plots + CSV in {args.out}")


if __name__ == '__main__':
    main()
