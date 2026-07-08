#!/usr/bin/env python
r"""
check_decorrelation_time.py -- empirical verification of 1.a (Theoretical_guarantees).

Verifies, on the FORCED deep 28-mark builds:

  (1) IDENTITY: tau_lambda from a curvature fit of the measured rho(tau)
      equals 1/sigma with sigma = ||X_dot||_rms / ||X||_rms, for BOTH
      processes X = N and X = omega. (Derivation: rho(tau) = 1 - tau^2 sigma^2/2
      + O(tau^4) under stationarity; first-order term vanishes because
      <X, X_dot> = (1/2) d/dt ||X||^2.)
  (2) 28x28 correlation-matrix heatmaps C_ij = corr(X(t_i), X(t_j)) per member
      (band width = tau_lambda; members ordered by sigma).
  (3) Per-shell rho_kappa(tau) heatmap and the sigma(kappa) profile per member
      -- sigma(kappa; Re, beta, mu) is THE conditioning target.
  (4) Per-order laddering: sigma_m = ||N^(m+1)||/||N^(m)||, m=0,1,2, vs bulk
      sigma -- the moment-ratio growth (fast-tail sampling).

All derivatives are ANALYTIC (spectral recursion, flux-form dealiased Jacobians,
forcing rebuilt from the manifest) -- no finite differences anywhere, so the
check is independent of the FD machinery it ultimately informs.

Windows: only those SURVIVING the quiescent filter of sweep_dT_5em3/split.npz
(stationarity assumption (S) requires developed flow).

Run FROM $QG_DIR/training (rule 2):
    python Theoretical_guarantees/check_decorrelation_time.py \
        --members data/ensemble_N5_7lag/FRC-* --n-windows 24 --device cuda
Outputs (plots + CSV) go to Theoretical_guarantees/Results/decorrelation/.
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

OUT_DEFAULT = Path(__file__).resolve().parent / 'Results' / 'decorrelation'


# ---------------- analytic operators (slicer conventions) ---------------- #
def build_L_hat(der, nu, mu, beta):
    L = nu * der.laplacian - mu
    if beta != 0.0:
        L = L - beta * der.dx * der.inv_laplacian
    return L


def build_F(man, grid, dev, dt):
    fc = man.get('forcing', None)
    if not isinstance(fc, dict):
        return None
    A = float(fc.get('A', 0)); B = float(fc.get('B', 0))
    D = float(fc.get('D', 0)); E = float(fc.get('E', 0))
    if A == 0 and D == 0:
        return None
    x = torch.linspace(0, grid.Lx, grid.Nx, device=dev, dtype=dt)
    y = torch.linspace(0, grid.Ly, grid.Ny, device=dev, dtype=dt)
    return A * torch.cos(B * x[None, :]) + D * torch.cos(E * y[:, None])


def jac_flux(psi, om, der):
    """Solver-form dealiased Jacobian J(psi, omega) (flux form)."""
    ph = to_spectral(psi.unsqueeze(0))
    u = to_physical(-der.dy * ph).squeeze(0)
    v = to_physical(der.dx * ph).squeeze(0)
    uq = to_spectral((u * om).unsqueeze(0)); vq = to_spectral((v * om).unsqueeze(0))
    der.dealias(uq); der.dealias(vq)
    return to_physical(der.dx * uq + der.dy * vq).squeeze(0)


def n_derivs(om, psi, der, L_hat, F, max_m=3):
    """Analytic recursion at one snapshot: returns lists omega^(k), N^(m)."""
    oms = [om]; pss = [psi]; Ns = []
    for m in range(max_m + 1):
        acc = None
        for j in range(m + 1):
            t = math.comb(m, j) * jac_flux(pss[m - j], oms[j], der)
            acc = t if acc is None else acc + t
        Nm = -acc
        if m == 0 and F is not None:
            Nm = Nm + F
        Ns.append(Nm)
        onext = to_physical(L_hat * to_spectral(oms[m].unsqueeze(0))).squeeze(0) + Nm
        oms.append(onext)
        pss.append(to_physical(der.inv_laplacian *
                               to_spectral(onext.unsqueeze(0))).squeeze(0))
    return oms, Ns


# ---------------- correlation machinery ---------------- #
def corr_matrix(stack):
    """(M, Ny, Nx) -> (M, M) correlation matrix of mean-removed fields."""
    M = stack.shape[0]
    f = stack.reshape(M, -1)
    f = f - f.mean(dim=1, keepdim=True)
    nrm = f.norm(dim=1).clamp_min(1e-300)
    g = f / nrm[:, None]
    return (g @ g.T).cpu().numpy()


def rho_from_corrs(Cs):
    """Average lag-correlation rho(l) from a list of (M,M) matrices."""
    M = Cs[0].shape[0]
    rho = np.zeros(M); cnt = np.zeros(M)
    for C in Cs:
        for l in range(M):
            d = np.diagonal(C, offset=l)
            rho[l] += d.sum(); cnt[l] += d.size
    return rho / cnt


def fit_tau_lambda(rho, dt, n_fit=3):
    """Curvature fit 1 - rho(tau) = a tau^2 (+ b tau^4) on the first n_fit lags."""
    tau = dt * np.arange(1, n_fit + 1)
    y = 1.0 - rho[1:n_fit + 1]
    A = np.stack([tau ** 2, tau ** 4], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a = max(coef[0], 1e-300)
    return 1.0 / math.sqrt(2.0 * a)


# ---------------- per-member analysis ---------------- #
def analyze(member_dir: Path, n_windows: int, dev: str, out: Path, n_lag_shell: int):
    deep = member_dir / 'forced_turbulence_dT_5em3'
    sweep = member_dir / 'sweep_dT_5em3'
    man = json.loads((deep / 'manifest.json').read_text())
    M = int(man['n_snapshots_per_sample'])
    dtf = float(man['Delta_T'])
    Nx, Ny = int(man['Nx']), int(man['Ny'])
    Lx, Ly = float(man['Lx']), float(man['Ly'])
    nu = float(man.get('nu', 0)); mu = float(man.get('mu', 0))
    beta = float(man.get('beta', man.get('B', 0)))
    name = member_dir.name

    inp = np.load(deep / 'packed' / 'inputs.npy', mmap_mode='r')  # (Nwin, 2M, Ny, Nx)
    Nwin = inp.shape[0]

    # surviving (post-filter) windows from the sliced split -- stationarity (S)
    surv = np.arange(Nwin)
    spf = sweep / 'split.npz'
    if spf.exists():
        sp = np.load(spf)
        na = int(json.loads((sweep / 'manifest.json').read_text()).get('n_anchors', 1))
        rows = np.concatenate([sp[k] for k in sp.files])
        surv = np.unique(rows // na)
    picks = surv[np.linspace(0, len(surv) - 1, min(n_windows, len(surv)), dtype=int)]

    grid = CartesianGrid(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, device=dev, precision='float64')
    der = Derivative(grid)
    L_hat = build_L_hat(der, nu, mu, beta)
    F = build_F(man, grid, dev, torch.float64)

    # integer-|k| shell index over the spectral layout
    kxg = der.dx.imag; kyg = der.dy.imag
    kmag = torch.sqrt((kxg ** 2 + kyg ** 2)).squeeze()
    probe = to_spectral(torch.zeros(1, Ny, Nx, dtype=torch.float64, device=dev))
    if kmag.shape != probe.shape[-2:]:
        kmag = torch.sqrt(kxg.squeeze()[None, :] ** 2 + kyg.squeeze()[:, None] ** 2)
    sh = torch.round(kmag).to(torch.int64)
    n_sh = int(sh.max().item()) + 1

    Cn_list, Cw_list = [], []
    rshell_num = torch.zeros(n_sh, n_lag_shell, dtype=torch.float64, device=dev)
    rshell_den = torch.zeros(n_sh, dtype=torch.float64, device=dev)
    sigN_num = sigN_den = 0.0
    sigW_num = sigW_den = 0.0
    lad = np.zeros(4)
    shell_sN_num = torch.zeros(n_sh, dtype=torch.float64, device=dev)
    shell_sN_den = torch.zeros(n_sh, dtype=torch.float64, device=dev)

    for w in picks:
        om_all = torch.tensor(np.asarray(inp[w, :M], np.float64), device=dev)
        ps_all = torch.tensor(np.asarray(inp[w, M:2 * M], np.float64), device=dev)
        # N at every mark (m=0 only)
        N_all = torch.stack([
            (-jac_flux(ps_all[i], om_all[i], der) + (F if F is not None else 0))
            for i in range(M)])
        Cn_list.append(corr_matrix(N_all))
        Cw_list.append(corr_matrix(om_all))

        # per-shell rho_kappa(l): N-process
        Nh = to_spectral(N_all)
        Eh = Nh.real ** 2 + Nh.imag ** 2
        for l in range(n_lag_shell):
            prod = (Nh[:M - l] * Nh[l:].conj()).real.sum(dim=0)
            num = torch.zeros(n_sh, dtype=torch.float64, device=dev)
            num.scatter_add_(0, sh.reshape(-1), prod.reshape(-1))
            rshell_num[:, l] += num
        den = torch.zeros(n_sh, dtype=torch.float64, device=dev)
        den.scatter_add_(0, sh.reshape(-1), Eh.sum(dim=0).reshape(-1))
        rshell_den += den

        # analytic derivatives at the ANCHOR (mark 0)
        oms, Ns = n_derivs(om_all[0], ps_all[0], der, L_hat, F, max_m=3)
        sigN_num += float(Ns[1].pow(2).sum()); sigN_den += float(Ns[0].pow(2).sum())
        sigW_num += float(oms[1].pow(2).sum()); sigW_den += float(om_all[0].pow(2).sum())
        for m in range(4):
            lad[m] += float(Ns[m].pow(2).sum())
        N0h = to_spectral(Ns[0].unsqueeze(0)).squeeze(0)
        N1h = to_spectral(Ns[1].unsqueeze(0)).squeeze(0)
        shell_sN_den.scatter_add_(0, sh.reshape(-1),
                                  (N0h.real ** 2 + N0h.imag ** 2).reshape(-1))
        shell_sN_num.scatter_add_(0, sh.reshape(-1),
                                  (N1h.real ** 2 + N1h.imag ** 2).reshape(-1))

    rhoN = rho_from_corrs(Cn_list)
    rhoW = rho_from_corrs(Cw_list)
    tauN = fit_tau_lambda(rhoN, dtf)
    tauW = fit_tau_lambda(rhoW, dtf)
    sigN = math.sqrt(sigN_num / max(sigN_den, 1e-300))
    sigW = math.sqrt(sigW_num / max(sigW_den, 1e-300))
    lam = np.sqrt(lad)
    sig_m = lam[1:] / np.maximum(lam[:-1], 1e-300)
    sig_kappa = torch.sqrt(shell_sN_num / shell_sN_den.clamp_min(1e-300)).cpu().numpy()
    rho_shell = (rshell_num / rshell_den.clamp_min(1e-300)[:, None]).cpu().numpy()

    # ---- plots ----
    mo = out / name; mo.mkdir(parents=True, exist_ok=True)
    for tag, C, tl in (('N', np.mean(Cn_list, axis=0), tauN),
                       ('omega', np.mean(Cw_list, axis=0), tauW)):
        fig, ax = plt.subplots(figsize=(6, 5.4))
        im = ax.imshow(C, cmap='seismic', vmin=-1, vmax=1, aspect='equal')
        ax.set_title(rf"{name}: corr({tag}$(t_i)$, {tag}$(t_j)$)   "
                     rf"$\tau_\lambda$={tl:.4f}")
        ax.set_xlabel('mark j'); ax.set_ylabel('mark i')
        fig.colorbar(im, ax=ax); fig.tight_layout()
        fig.savefig(mo / f'corr_matrix_{tag}.png', dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    tau = dtf * np.arange(len(rhoN))
    ax.plot(tau, rhoN, 'o-', label=r'measured $\rho_N(\tau)$')
    ax.plot(tau, rhoW, 's-', label=r'measured $\rho_\omega(\tau)$', alpha=0.7)
    tt = np.linspace(0, tau.max(), 200)
    ax.plot(tt, 1 - tt ** 2 * sigN ** 2 / 2, '--',
            label=rf'$1-\frac{{\tau^2\sigma_N^2}}{{2}}$,  $1/\sigma_N$={1/sigN:.4f}')
    ax.plot(tt, 1 - tt ** 2 * sigW ** 2 / 2, ':',
            label=rf'$1-\frac{{\tau^2\sigma_\omega^2}}{{2}}$,  $1/\sigma_\omega$={1/sigW:.4f}')
    ax.set_ylim(-0.3, 1.02); ax.set_xlabel(r'$\tau$'); ax.set_ylabel(r'$\rho$')
    ax.set_title(f'{name}: decorrelation, fit vs identity'); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(mo / 'rho_vs_identity.png', dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    im = ax.imshow(rho_shell, cmap='seismic', vmin=-1, vmax=1,
                   aspect='auto', origin='lower',
                   extent=[0, dtf * (n_lag_shell - 1), 0, n_sh])
    ax.set_xlabel(r'$\tau$'); ax.set_ylabel(r'shell $\kappa$')
    ax.set_title(rf'{name}: $\rho_\kappa(\tau)$')
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(mo / 'rho_per_shell.png', dpi=160); plt.close(fig)

    return dict(name=name, beta=beta, nu=nu, mu=mu,
                tauN=tauN, tauW=tauW, sigN=sigN, sigW=sigW,
                inv_sigN=1 / sigN, inv_sigW=1 / sigW,
                ratioN=tauN * sigN, ratioW=tauW * sigW,
                sig0=sig_m[0], sig1=sig_m[1], sig2=sig_m[2],
                sig_kappa=sig_kappa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--members', nargs='+', type=Path, required=True,
                    help='member dirs (FRC-*) under data/ensemble_N5_7lag/')
    ap.add_argument('--n-windows', type=int, default=24)
    ap.add_argument('--n-lag-shell', type=int, default=14)
    ap.add_argument('--out', type=Path, default=OUT_DEFAULT)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    members = [m for m in args.members
               if (m / 'forced_turbulence_dT_5em3' / 'packed' / 'inputs.npy').exists()]
    print(f"[1a] {len(members)} forced member(s); windows/member={args.n_windows}")

    rows = []
    for m in sorted(members):
        r = analyze(m, args.n_windows, args.device, args.out, args.n_lag_shell)
        rows.append(r)
        print(f"  {r['name']:12s} tau_l(N)={r['tauN']:.4f} 1/sig_N={r['inv_sigN']:.4f} "
              f"ratio={r['ratioN']:.3f} | tau_l(w)={r['tauW']:.4f} "
              f"1/sig_w={r['inv_sigW']:.4f} ratio={r['ratioW']:.3f} | "
              f"ladder sig0={r['sig0']:.1f} sig1={r['sig1']:.1f} sig2={r['sig2']:.1f}")

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for r in rows:
        ax.plot(np.arange(len(r['sig_kappa'])), r['sig_kappa'],
                label=rf"{r['name']} ($\beta$={r['beta']:g})")
    ax.set_xlabel(r'shell $\kappa$'); ax.set_ylabel(r'$\sigma_N(\kappa)$')
    ax.set_title(r'conditioning target: $\sigma(\kappa;\,Re,\beta,\mu)$')
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(args.out / 'sigma_kappa_all_members.png', dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    for r in rows:
        ax.plot([0, 1, 2], [r['sig0'], r['sig1'], r['sig2']], 'o-', label=r['name'])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([r'$\sigma_0=\frac{\|\dot N\|}{\|N\|}$',
                        r'$\sigma_1=\frac{\|\ddot N\|}{\|\dot N\|}$',
                        r'$\sigma_2=\frac{\|N^{(3)}\|}{\|\ddot N\|}$'])
    ax.set_ylabel('growth per order'); ax.set_title('per-order laddering (moment ratios)')
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(args.out / 'laddering_sigma_m.png', dpi=160); plt.close(fig)

    with open(args.out / 'decorrelation_summary.csv', 'w', newline='') as f:
        w = csv.writer(f)
        keys = ['name', 'beta', 'nu', 'mu', 'tauN', 'inv_sigN', 'ratioN',
                'tauW', 'inv_sigW', 'ratioW', 'sig0', 'sig1', 'sig2']
        w.writerow(keys)
        for r in rows:
            w.writerow([r[k] for k in keys])
    print(f"[1a] done. Plots + CSV in {args.out}")
    print("[1a] PASS criterion: ratio = tau_lambda * sigma ~ 1 per member (both "
          "processes); rho_kappa band narrowing with kappa; sig0 < sig1 < sig2.")


if __name__ == '__main__':
    main()
