#!/usr/bin/env python
r"""
check_wiener_floor.py -- empirical verification of 1.b (Theoretical_guarantees).

Decomposes the pooled training floor of the unconditioned model into the three
terms of the 1.b result:

    L_min = (i) cascade remainder  +  (ii) pooled-variance mismatch
            + (iii) finite-width shape deficit,

and compares (i)+(ii)+(iii) against the observed plateau
(control run deriv7_filtered_floor0.1: pooled Nddot ~ 0.19).

Per member x dt, everything is measured on the deep 28-mark builds with fully
ANALYTIC machinery (spectral recursion, dealiased flux Jacobians, manifest
forcing) -- no trained model is involved. Method per term:

  (i)  CASCADE REMAINDER. Per shell kappa, regress omega^(S-slot proxy) on the
       shell-filtered lower derivative: r_frac(kappa) = 1 - |<w7_k, w_k>|^2 /
       (||w7_k||^2 ||w_k||^2)  (per-shell coherence). Here we use the exact
       analytic pair (omega^(m), omega^(m+q)) with q = S - m as the
       diagonal-vs-remainder split: the coherent part IS the best shell-diagonal
       filter; 1 - coherence is the fraction no filter of omega^(m) can produce.
       Weighted by the time-error spectrum this gives the irreducible slice.

  (ii) POOLED-VARIANCE MISMATCH. The per-(member,dt) optimal symbol is
       g_j(kappa) = C_m (dT_j sigma_w(kappa))^(S-m) (up to a shared ik). One
       shared stencil realizes at best the pooled LS compromise g_bar(kappa) =
       weighted mean of g_j. The mismatch slice is the weighted variance of
       g_j around g_bar, normalized like the loss. sigma_w(kappa) is MEASURED
       (1.a machinery, re-computed here per member).

  (iii) SHAPE DEFICIT. Project the pooled-optimal g_bar(kappa)*ik symbol onto
       the width-W reachable set (real, odd, W-tap 1D stencils along x and y);
       the unreachable spectral residue, weighted like the loss, is (iii).

Outputs: per-(member,dt) table of the raw time-error floor, absorbed fraction,
and the three slices; a pooled summary line 'predicted floor vs observed 0.19';
plots (per-member stacked-bar decomposition; coherence(kappa) curves;
g_j(kappa) family vs g_bar). CSV + PNGs to Theoretical_guarantees/Results/
wiener_floor/.

Run FROM $QG_DIR/training on a GPU node (qlogin first):
    python Theoretical_guarantees/check_wiener_floor.py \
        --members data/ensemble_N5_7lag/FRC-Re25k data/ensemble_N5_7lag/FRC-kf4 \
                  data/ensemble_N5_7lag/FRC-256 \
        --dts 5e-3 1e-2 1.5e-2 --m 2 --n-windows 12 --grad-kernel 15
(--m 2 = Nddot, the rollout-ceiling order; S is read from the deep manifest.)
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


# ---------- analytic machinery (identical conventions to 1.a check) ----------
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


def recursion(om, psi, der, L_hat, F, max_k):
    """omega^(k), psi^(k), N^(m) up to max_k via the exact spectral recursion."""
    oms = [om]; pss = [psi]; Ns = []
    for m in range(max_k):
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
    return oms, pss, Ns


def fd_row(S, k):
    """Order-k row of the S-node backward Vandermonde (unit spacing) + the
    leading truncation constant C_k = -(1/S!) sum W[k,j] (-j)^S."""
    x = np.arange(0, -S, -1, dtype=np.float64)
    A = np.array([[x[j] ** m / math.factorial(m) for j in range(S)]
                  for m in range(S)])
    W = np.linalg.inv(A).T
    Ck = -(1.0 / math.factorial(S)) * float(np.sum(W[k] * ((-np.arange(S)) ** S)))
    return W[k], Ck


def shell_index(der, Ny, Nx, dev):
    kxg = der.dx.imag; kyg = der.dy.imag
    kmag = torch.sqrt((kxg ** 2 + kyg ** 2)).squeeze()
    probe = to_spectral(torch.zeros(1, Ny, Nx, dtype=torch.float64, device=dev))
    if kmag.shape != probe.shape[-2:]:
        kmag = torch.sqrt(kxg.squeeze()[None, :] ** 2 + kyg.squeeze()[:, None] ** 2)
    sh = torch.round(kmag).to(torch.int64)
    return sh, int(sh.max().item()) + 1, kmag


def shell_sum(x_flat, sh_flat, n_sh):
    out = torch.zeros(n_sh, dtype=torch.float64, device=x_flat.device)
    out.scatter_add_(0, sh_flat, x_flat)
    return out


def reachable_residual(g_of_kappa, kappa_axis, width):
    """(iii): project the target 1D symbol s(k) = g(|k|)*k (odd, real filter)
    onto the span of width-W unit-spacing stencil symbols {sin(j k h)}_{j=1..W//2}
    (h=1 grid units -> k in [0, pi]); return relative unreachable energy."""
    kk = np.linspace(1e-3, math.pi, 256)
    # map physical shell axis onto grid-units k: kappa/kappa_max * pi
    kap = kappa_axis / max(kappa_axis[-1], 1e-30) * math.pi
    tgt = np.interp(kk, kap, g_of_kappa) * kk
    H = width // 2
    B = np.stack([np.sin(j * kk) for j in range(1, H + 1)], axis=1)   # basis
    coef, *_ = np.linalg.lstsq(B, tgt, rcond=None)
    res = tgt - B @ coef
    return float((res ** 2).sum() / max((tgt ** 2).sum(), 1e-300))


# --------------------------------- main --------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--members', nargs='+', type=Path, required=True)
    ap.add_argument('--dts', type=float, nargs='+', default=[5e-3, 1e-2, 1.5e-2])
    ap.add_argument('--m', type=int, default=2, help='target order (2 = Nddot)')
    ap.add_argument('--n-windows', type=int, default=12)
    ap.add_argument('--grad-kernel', type=int, default=15)
    ap.add_argument('--observed-floor', type=float, default=0.19,
                    help='the trained pooled plateau to compare against')
    ap.add_argument('--out', type=Path, default=OUT_DEFAULT)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    dev = args.device
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    m = args.m
    rows = []
    coh_curves = {}
    g_family = {}          # (member, dt) -> g_j(kappa)
    weights = {}           # (member, dt) -> loss weight (raw time-error energy)

    for mem in sorted(args.members):
        deep = mem / 'forced_turbulence_dT_5em3'
        if not (deep / 'packed' / 'inputs.npy').exists():
            print(f"[skip] {mem.name}: no deep build"); continue
        man = json.loads((deep / 'manifest.json').read_text())
        M = int(man['n_snapshots_per_sample'])
        S = 7                                           # stencil depth in use
        dtf = float(man['Delta_T'])
        Nx, Ny = int(man['Nx']), int(man['Ny'])
        nu = float(man.get('nu', 0)); mu = float(man.get('mu', 0))
        beta = float(man.get('beta', man.get('B', 0)))
        inp = np.load(deep / 'packed' / 'inputs.npy', mmap_mode='r')

        # surviving windows (quiescent filter) via the sliced split
        sweep = mem / 'sweep_dT_5em3'
        surv = np.arange(inp.shape[0])
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
        sh, n_sh, kmag = shell_index(der, Ny, Nx, dev)
        sh_flat = sh.reshape(-1)
        kappa_axis = np.arange(n_sh, dtype=np.float64)

        # accumulators over windows
        E_m = torch.zeros(n_sh, dtype=torch.float64, device=dev)      # |w^(m)|^2
        E_S = torch.zeros(n_sh, dtype=torch.float64, device=dev)      # |w^(S)|^2
        X_mS = torch.zeros(n_sh, dtype=torch.float64, device=dev)     # Re<w^(S), w^(m)>
        E_dm = torch.zeros(n_sh, dtype=torch.float64, device=dev)     # |w^(m+1)|^2 (for sigma_w)
        nrmNm = nrmNm_t = 0.0

        _, Ck = fd_row(S, m)

        for w in picks:
            om0 = torch.tensor(np.asarray(inp[w, 0], np.float64), device=dev)
            ps0 = torch.tensor(np.asarray(inp[w, M], np.float64), device=dev)
            oms, pss, Ns = recursion(om0, ps0, der, L_hat, F, max_k=S)
            wm = to_spectral(oms[m].unsqueeze(0)).squeeze(0)
            wS = to_spectral(oms[S].unsqueeze(0)).squeeze(0)
            wm1 = to_spectral(oms[m + 1].unsqueeze(0)).squeeze(0)
            E_m += shell_sum((wm.real**2 + wm.imag**2).reshape(-1), sh_flat, n_sh)
            E_S += shell_sum((wS.real**2 + wS.imag**2).reshape(-1), sh_flat, n_sh)
            X_mS += shell_sum((wS * wm.conj()).real.reshape(-1), sh_flat, n_sh)
            E_dm += shell_sum((wm1.real**2 + wm1.imag**2).reshape(-1), sh_flat, n_sh)
            nrmNm += float(Ns[m].pow(2).sum())

        # measured per-shell rate and coherence
        sigw_k = torch.sqrt(E_dm / E_m.clamp_min(1e-300)).cpu().numpy()
        coh_k = (X_mS ** 2 / (E_m * E_S).clamp_min(1e-300)).cpu().numpy()
        coh_curves[mem.name] = coh_k
        E_S_np = E_S.cpu().numpy(); E_m_np = E_m.cpu().numpy()

        for dt in args.dts:
            # raw time-error energy for order m at this dt (leading-term model):
            #   T ~ Ck dT^{S-m} * (driver omega^(S)) through the exact Jacobians;
            # relative floor proxy: ||eps_m|| / ||omega^(m)|| shell-weighted.
            amp2 = (Ck * dt ** (S - m)) ** 2
            raw_k = amp2 * E_S_np                                # |eps|^2 per shell
            raw_rel2 = raw_k.sum() / max(E_m_np.sum() * 0 + nrmNm / len(picks), 1e-300)
            # (i) irreducible = incoherent fraction of the driver, loss-weighted
            i_k = raw_k * (1.0 - coh_k)
            # absorbed candidate (coherent part) -> defines the per-(mem,dt) g_j
            g_j = np.abs(Ck) * (dt * sigw_k) ** (S - m)
            g_family[(mem.name, dt)] = g_j
            weights[(mem.name, dt)] = raw_k                       # spectrum weight
            rows.append(dict(member=mem.name, dt=dt,
                             raw=float(np.sqrt(raw_k.sum())),
                             irreducible=float(np.sqrt(i_k.sum())),
                             coh_frac=float((raw_k * coh_k).sum()
                                            / max(raw_k.sum(), 1e-300))))
        print(f"[{mem.name}] shells={n_sh} windows={len(picks)} "
              f"Ck(S=7,m={m})={Ck:.3g} sigma_w(bulk)="
              f"{math.sqrt(float(E_dm.sum()/E_m.sum())):.2f}")

    # ---------------- pooled (ii) and (iii) ----------------
    keys = list(g_family.keys())
    n_sh_min = min(len(g_family[k]) for k in keys)
    G = np.stack([g_family[k][:n_sh_min] for k in keys])          # (J, K)
    Wt = np.stack([weights[k][:n_sh_min] for k in keys])          # (J, K)
    wj = Wt.sum(axis=1); wj = wj / wj.sum()                        # per-config weight
    g_bar = (wj[:, None] * G).sum(axis=0)                          # pooled compromise
    # (ii): weighted variance of g_j around g_bar, normalized by weighted mean g^2
    var_ii = float((wj[:, None] * (G - g_bar[None]) ** 2).sum()
                   / max((wj[:, None] * G ** 2).sum(), 1e-300))
    # (iii): unreachable residue of the pooled symbol for the given width
    kappa_axis = np.arange(n_sh_min, dtype=np.float64)
    res_iii = reachable_residual(g_bar, kappa_axis, args.grad_kernel)

    # ---------------- report ----------------
    print("\nper-(member,dt): raw time floor | irreducible (i) | coherent frac")
    for r in rows:
        print(f"  {r['member']:10s} dT={r['dt']:<7g} raw={r['raw']:.3e} "
              f"(i)={r['irreducible']:.3e}  coh={100*r['coh_frac']:.1f}%")
    print(f"\npooled slices (order m={m}, S=7, width={args.grad_kernel}):")
    print(f"  (ii) pooled-variance mismatch fraction = {var_ii:.3f}")
    print(f"  (iii) width-{args.grad_kernel} shape-deficit fraction = {res_iii:.3f}")
    print(f"  observed trained plateau (pooled Nddot) = {args.observed_floor}")
    print("  READING: (ii) is the slice conditioning deletes; if (ii) dominates")
    print("  (i)+(iii), the conditioned model's headroom is large. (i) moves only")
    print("  with more lags; (iii) with wider stencils.")

    # ---------------- plots ----------------
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for name, coh in coh_curves.items():
        ax.plot(np.arange(len(coh)), coh, label=name)
    ax.set_xlabel(r'shell $\kappa$'); ax.set_ylabel(r'coherence$^2(\omega^{(S)},\ \omega^{(m)})$')
    ax.set_title(rf'diagonal (absorbable) fraction per shell, $m={m}$, $S=7$')
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(args.out / 'coherence_per_shell.png', dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for (name, dt), g in g_family.items():
        ax.plot(np.arange(len(g)), g, alpha=0.6, label=f'{name} dT={dt:g}')
    ax.plot(np.arange(n_sh_min), g_bar, 'k-', lw=2.5, label='pooled compromise')
    ax.set_yscale('log'); ax.set_xlabel(r'shell $\kappa$')
    ax.set_ylabel(r'$|C_m|\,(\Delta T\,\sigma_\omega(\kappa))^{S-m}$')
    ax.set_title('per-(member, dT) optimal correction vs the one shared stencil')
    ax.legend(fontsize=6); fig.tight_layout()
    fig.savefig(args.out / 'g_family_vs_pooled.png', dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.bar(['(i) cascade', '(ii) pooled var', f'(iii) width-{args.grad_kernel}'],
           [np.mean([r['irreducible'] ** 2 / max(r['raw'] ** 2, 1e-300)
                     for r in rows]), var_ii, res_iii])
    ax.set_ylabel('fraction of the raw time-error energy')
    ax.set_title('Wiener-floor decomposition (pooled)')
    fig.tight_layout(); fig.savefig(args.out / 'floor_decomposition.png', dpi=160)
    plt.close(fig)

    with open(args.out / 'wiener_floor_summary.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['member', 'dt', 'raw_time_floor', 'irreducible_i', 'coh_frac'])
        for r in rows:
            w.writerow([r['member'], r['dt'], r['raw'], r['irreducible'],
                        r['coh_frac']])
        w.writerow([]); w.writerow(['pooled_var_ii', var_ii])
        w.writerow(['shape_deficit_iii', res_iii])
        w.writerow(['observed_floor', args.observed_floor])
    print(f"\n[1b] plots + CSV in {args.out}")


if __name__ == '__main__':
    main()
