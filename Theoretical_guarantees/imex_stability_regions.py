r"""
imex_stability_regimes.py
=========================
IMEX absolute-stability analysis using the ACTUAL physical parameters AND the real
characteristic velocity of the Step-size/resolution closure ensemble. For each
regime every resolved Fourier mode k is placed in the (zL, zN) plane and we ask
which schemes keep |g(zL(k), zN(k))| <= 1 at a given coarse step Delta_T.

Per-mode eigenvalues (coarse closure step Delta_T, NOT the fine IC dt):
  zL(k) = Delta_T ( -nu|k|^2 - mu + i beta kx/|k|^2 )     (CN, implicit; complex)
  zN(k) = i Delta_T u |k|                                 (AB advection; CFL surrogate)

Three scheme-sets (one output folder each), selected with --scheme-set:
  bare    : AB2CN2, AB4CN2, RK4                        -> Stability_figs/Bare_schemes
  ab2_nn  : + AB2CN2+E_NN1 (match RK4), +E_NN2 (exact) -> Stability_figs/AB2CN2_NN
  ab4_nn  : + AB4CN2+E_NN1 (match RK4), +E_NN2 (exact) -> Stability_figs/AB4CN2_NN
The NN schemes share the SAME analytic closure machinery; only the BASE truncation
the closure cancels changes (AB2CN2 vs AB4CN2 coefficients, verified in
closure_operators.py). E_NN2 cancels the base truncation -> matches the exact flow;
E_NN1 leaves RK4's own z^5/120 -> matches RK4.
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TWO_PI = 2 * np.pi
FOUR_PI = 4 * np.pi
DEC = dict(family='decaying', Nx=256, Ny=256, Lx=TWO_PI,  Ly=TWO_PI,  nu=1.025e-5, mu=0.0,  beta=0.0, kf=2)
FRC = dict(family='forced',   Nx=512, Ny=512, Lx=FOUR_PI, Ly=FOUR_PI, nu=1.025e-4, mu=0.02, beta=1.0, kf=2)

def _mk(base, **ov):
    d = dict(base); d.update(ov); return d

REGIMES = {
    "DEC-base":  _mk(DEC),
    "DEC-loRe":  _mk(DEC, nu=1.025e-4),
    "DEC-512":   _mk(DEC, Nx=512, Ny=512),
    "DEC-hiRe":  _mk(DEC, Nx=512, Ny=512, nu=2.5e-6),
    "FRC-b0":    _mk(FRC, beta=0.0),
    "FRC-b05":   _mk(FRC, beta=0.5),
    "FRC-b075":  _mk(FRC, beta=0.75),
    "FRC-b1":    _mk(FRC),
    "FRC-b2":    _mk(FRC, beta=2.0),
    "FRC-b25":   _mk(FRC, beta=2.5),
    "FRC-kf4":   _mk(FRC, kf=4),
    "FRC-Re25k": _mk(FRC, nu=4.0e-5),
    "FRC-combo": _mk(FRC, nu=4.0e-5, beta=0.5, kf=4),
    "FRC-256":   _mk(FRC, Nx=256, Ny=256, nu=2.0e-4),
}
DEFAULT_ENS = ("/gdata/projects/ml_scope/Closure_modeling/QG-closure/"
               "qg-simple-package-stable/src/qg/outputs/Step_size_resolution_closure_ensemble")
FIGS_ROOT = ("/gdata/projects/ml_scope/Closure_modeling/QG-closure/"
             "qg-simple-package-stable/src/qg/training/Stability_figs")


def _sci(x):
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x)))); m = x / 10**e
    ms = (f"{m:.2f}".rstrip('0').rstrip('.'))
    return rf"{ms}\times10^{{{e}}}"

def regime_label(reg):
    fam = "Decaying" if reg['family'] == 'decaying' else "Forced"
    parts = [rf"$\nu={_sci(reg['nu'])}$", rf"${reg['Nx']}^2$"]
    if reg['family'] == 'forced':
        parts.insert(0, rf"$\beta={reg['beta']:g}$")
        if reg['kf'] != 2:
            parts.append(rf"$k_f={reg['kf']:g}$")
    return f"{fam}: " + ", ".join(parts)


# --- closure operator coefficients (verified, from closure_operators.py) ----- #
# COEFFS[base][p] is the amplification-space R_p stencil; DENOM[base][p] its D_p.
COEFFS = {
    'ab2cn2': {3: [1, 1, 1, -5], 4: [2, 2, 2, -4, 1],
               5: [13, 13, 13, -17, 8, -7], 6: [43, 43, 43, -47, 28, -17, 4]},
    'ab4cn2': {3: [1, 1, 1, 0], 4: [2, 2, 2, 1, 0],
               5: [39, 39, 39, 24, 9, -251]},               # note: D5 = 720
}
DENOM = {
    'ab2cn2': {3: 12, 4: 24, 5: 240, 6: 1440},
    'ab4cn2': {3: 12, 4: 24, 5: 720},
}
NN_BASE = 'ab2cn2'                                # set per scheme-set in main()

def P_p(zL, zN, p, base):
    a = COEFFS[base][p]; z = zL + zN
    val = a[0] * zL**p
    for k in range(1, p + 1):
        val = val + a[k] * zL**(p - k) * zN * z**(k - 1)
    return val

def dhat(zL, zN, which, orders, base):
    d = np.zeros(np.broadcast(zL, zN).shape, dtype=complex)
    for p in orders:
        if p in COEFFS[base] and p in DENOM[base]:
            d = d - P_p(zL, zN, p, base) / DENOM[base][p]
    if which == 'E_NN1':                          # match RK4: leave RK4's z^5/120
        d = d - (zL + zN)**5 / 120.0
    return d

def spec_radius(coeffs_list):
    grids = [np.asarray(c, dtype=complex) for c in coeffs_list]
    n = len(grids); shp = grids[0].shape
    comp = np.zeros((grids[0].size, n, n), complex)
    comp[:, 0, :] = -np.stack([g.ravel() for g in grids], axis=1)
    idx = np.arange(n - 1); comp[:, idx + 1, idx] = 1.0
    return np.max(np.abs(np.linalg.eigvals(comp)), axis=1).reshape(shp)

def _quad_radius(c1, c0):
    c1 = np.asarray(c1, complex); c0 = np.asarray(c0, complex)
    disc = np.sqrt(c1*c1 - 4.0*c0)
    r1 = 0.5*(c1 + disc); r2 = 0.5*(c1 - disc)
    return np.maximum(np.abs(r1), np.abs(r2))

def _ab4_companion(r, rho, dh=0.0):
    """AB4CN2 char poly coeffs (g^4 - c1 g^3 + c2 g^2 - c3 g + c4); dh adds to c1."""
    return [-(r + (55/24)*rho + dh), (59/24)*rho, -(37/24)*rho, (9/24)*rho]

def amp(name, zL, zN, orders):
    zL = np.asarray(zL, complex); zN = np.asarray(zN, complex)
    if name == 'RK4':
        z = zL + zN
        return np.abs(1 + z + z**2/2 + z**3/6 + z**4/24)
    r = (1 + zL/2) / (1 - zL/2); rho = zN / (1 - zL/2)
    if name == 'AB4CN2':
        return spec_radius(_ab4_companion(r, rho))
    if name in ('E_NN1', 'E_NN2'):
        dh = dhat(zL, zN, name, orders, NN_BASE)
        if NN_BASE == 'ab2cn2':                   # quadratic base
            return _quad_radius(r + 1.5*rho + dh, 0.5*rho)
        return spec_radius(_ab4_companion(r, rho, dh))   # quartic base
    if name != 'AB2CN2':
        raise ValueError(name)
    return _quad_radius(r + 1.5*rho, 0.5*rho)


SCHEME_SETS = {
    'bare':   ["AB2CN2", "AB4CN2", "RK4"],
    'ab2_nn': ["AB2CN2", "AB4CN2", "E_NN1", "E_NN2", "RK4"],
    'ab4_nn': ["AB2CN2", "AB4CN2", "E_NN1", "E_NN2", "RK4"],
}
NN_BASE_OF = {'bare': 'ab2cn2', 'ab2_nn': 'ab2cn2', 'ab4_nn': 'ab4cn2'}
OUTDIR_OF = {'bare':   os.path.join(FIGS_ROOT, 'Bare_schemes'),
             'ab2_nn': os.path.join(FIGS_ROOT, 'AB2CN2_NN'),
             'ab4_nn': os.path.join(FIGS_ROOT, 'AB4CN2_NN')}
COLOR = {"AB2CN2": "C0", "AB4CN2": "C1", "E_NN1": "C3", "E_NN2": "C2", "RK4": "k"}
SCHEMES = SCHEME_SETS['ab2_nn']                   # default; overwritten in main()
LABEL = {}

def _build_labels(nn_base):
    bd = 'AB2CN2' if nn_base == 'ab2cn2' else 'AB4CN2'
    return {"AB2CN2": r"$\mathrm{AB2CN2}$", "AB4CN2": r"$\mathrm{AB4CN2}$",
            "E_NN1": rf"$\mathrm{{{bd}}}+E_{{NN1}}$ (match RK4)",
            "E_NN2": rf"$\mathrm{{{bd}}}+E_{{NN2}}$ (match exact flow)",
            "RK4": r"$\mathrm{RK4}$"}


# --- real characteristic velocity from a regime's DNS_FR.npz ---------------- #
def _load_omega(path):
    z = np.load(path)
    key = next((k for k in ('omega_FR', 'omega', 'q', 'q_FR', 'vorticity') if k in z.files), None)
    if key is None:
        raise KeyError(f"no omega key in {path}; have {z.files}")
    return np.asarray(z[key])

def velocity_stats_2d(omega2d, Lx, Ly):
    Ny, Nx = omega2d.shape
    kx = np.fft.fftfreq(Nx, d=Lx/Nx) * TWO_PI
    ky = np.fft.fftfreq(Ny, d=Ly/Ny) * TWO_PI
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2; K2[0, 0] = 1.0
    psih = -np.fft.fft2(omega2d) / K2
    u = np.real(np.fft.ifft2(-1j*KY*psih))
    v = np.real(np.fft.ifft2( 1j*KX*psih))
    s2 = u*u + v*v
    return np.sqrt(s2.mean()), np.sqrt(s2.max())

def u_char_from_npz(path, Lx, Ly, stat='rms', n_time=8):
    om = _load_omega(path)
    while om.ndim > 3:
        om = om.reshape(-1, om.shape[-2], om.shape[-1]); break
    if om.ndim == 2:
        frames = [om]
    else:
        idx = np.linspace(0, om.shape[0]-1, min(n_time, om.shape[0])).astype(int)
        frames = [om[i] for i in idx]
    urms = umax = 0.0
    for fr in frames:
        a, b = velocity_stats_2d(np.asarray(fr, float), Lx, Ly)
        urms = max(urms, a); umax = max(umax, b)
    return umax if stat == 'max' else urms, urms, umax


def regime_modes(reg, dt, u_char, dealias=True):
    Nx, Ny, Lx, Ly = reg['Nx'], reg['Ny'], reg['Lx'], reg['Ly']
    nu, mu, beta = reg['nu'], reg['mu'], reg['beta']
    kx = np.fft.fftfreq(Nx, d=Lx/Nx) * TWO_PI
    ky = np.fft.fftfreq(Ny, d=Ly/Ny) * TWO_PI
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2
    nz = K2 > 0
    K2s = np.where(nz, K2, 1.0)
    Kmag = np.sqrt(KX**2 + KY**2)
    zL = dt * (-nu*K2s - mu + 1j*beta*KX/K2s)
    zN = 1j * dt * u_char * Kmag
    keep = nz.copy()
    if dealias:
        kcx = (Nx/3) * (TWO_PI/Lx); kcy = (Ny/3) * (TWO_PI/Ly)
        keep &= (np.abs(KX) <= kcx) & (np.abs(KY) <= kcy)
    return zL[nz], zN[nz], Kmag[nz], keep[nz]

def max_stable_dt(reg, name, u_char, orders, dt_lo=1e-6, dt_hi=1.0, tol=1e-3):
    zL0, zN0, _, keep = regime_modes(reg, 1.0, u_char)
    lamL = zL0[keep]; aN = zN0[keep]
    nb = 220
    def _binned():
        if lamL.size <= nb:
            return np.arange(lamL.size)
        xr, xi, an = lamL.real, lamL.imag, aN.imag
        def q(a):
            lo, hi = a.min(), a.max()
            return np.zeros_like(a, int) if hi <= lo else np.round((a-lo)/(hi-lo)*nb).astype(int)
        key = q(xr)*(nb+1)**2 + q(xi)*(nb+1) + q(an)
        _, idx = np.unique(key, return_index=True)
        return idx
    idx = _binned()
    lamLb = lamL[idx]; aNb = aN[idx]
    def ok(dt):
        return np.all(amp(name, dt*lamLb, dt*aNb, orders) <= 1 + 1e-9)
    if not ok(dt_lo):
        return 0.0
    if ok(dt_hi):
        return dt_hi
    lo, hi = dt_lo, dt_hi
    while (hi - lo) / hi > tol:
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


def _safe_tag(reg, nm):
    fam = 'decaying' if reg['family'] == 'decaying' else 'forced'
    t = f"{fam}_nu{reg['nu']:.2e}_beta{reg['beta']:g}_N{reg['Nx']}"
    if reg['family'] == 'forced' and reg['kf'] != 2:
        t += f"_kf{reg['kf']:g}"
    return t.replace('+', '')


def _repr_modes(reg, u, nb=240):
    zL0, zN0, K0, keep = regime_modes(reg, 1.0, u)
    lamL = zL0[keep]; aN = zN0[keep]; K = K0[keep]
    xr, xi, an = lamL.real, lamL.imag, aN.imag
    def q(a):
        lo, hi = a.min(), a.max()
        return np.zeros_like(a, int) if hi <= lo else np.round((a - lo) / (hi - lo) * nb).astype(int)
    key = q(xr) * (nb + 1)**2 + q(xi) * (nb + 1) + q(an)
    _, idx = np.unique(key, return_index=True)
    return lamL[idx], aN[idx], K[idx]


def make_regime_figure(nm, reg, u, dt, orders, out_path):
    from matplotlib.colors import TwoSlopeNorm
    zL, zN, Kmag, keep = regime_modes(reg, dt, u)
    zLk = zL[keep]; zNk = zN[keep]; Kk = Kmag[keep]
    zL_med = complex(float(np.median(zLk.real)), float(np.median(zLk.imag)))
    fig, ax = plt.subplots(2, 2, figsize=(15.0, 12.6))

    xr = (-3.2, 0.6); yr = (-3.6, 3.6); n = 480
    xx = np.linspace(*xr, n); yy = np.linspace(*yr, n)
    XX, YY = np.meshgrid(xx, yy); zNgrid = XX + 1j * YY
    zLgrid = np.full_like(zNgrid, zL_med)
    for s in SCHEMES:
        rad = amp(s, zLgrid, zNgrid, orders)
        ax[0, 0].contour(xx, yy, rad, levels=[1.0], colors=COLOR[s], linewidths=2.2)
        ax[0, 0].plot([], [], color=COLOR[s], lw=2.2, label=LABEL[s])
    gmode = amp('AB2CN2', zLk, zNk, orders)
    norm = TwoSlopeNorm(vmin=min(0.95, float(gmode.min())), vcenter=1.0,
                        vmax=max(1.05, float(gmode.max())))
    sc = ax[0, 0].scatter(np.zeros(keep.sum()), np.abs(zNk), c=gmode, cmap='coolwarm',
                          norm=norm, s=10, zorder=5)
    ax[0, 0].scatter(np.zeros(keep.sum()), -np.abs(zNk), c=gmode, cmap='coolwarm',
                     norm=norm, s=10, zorder=5)
    plt.colorbar(sc, ax=ax[0, 0], label=r"true $|g|_{\mathrm{AB2CN2}}$ at each mode")
    ax[0, 0].axhline(0, color='0.6', lw=.5); ax[0, 0].axvline(0, color='0.6', lw=.5, ls='--')
    ax[0, 0].set_xlim(*xr); ax[0, 0].set_ylim(*yr)
    ax[0, 0].set_xlabel(r"$\mathrm{Re}\,z_N(k)$  (advection)", fontsize=12)
    ax[0, 0].set_ylabel(r"$\mathrm{Im}\,z_N(k)=\Delta T\,u\,|k|$  (advection)", fontsize=12)
    ax[0, 0].set_title(rf"stability regions ($|g|=1$) at median $z_L={zL_med.real:.2g}{zL_med.imag:+.2g}i$"
                       rf"  ($\Delta T={dt:g}$)", fontsize=11)
    ax[0, 0].grid(alpha=.3)

    sc2 = ax[0, 1].scatter(zLk.real, zLk.imag, s=7, c=np.log10(Kk), cmap='viridis')
    ax[0, 1].set_xlabel(r"$\mathrm{Re}\,z_L(k)=-\Delta T(\nu|k|^2+\mu)$  (diffusion)", fontsize=12)
    ax[0, 1].set_ylabel(r"$\mathrm{Im}\,z_L(k)=\Delta T\,\beta k_x/|k|^2$  ($\beta$ drift)", fontsize=12)
    ax[0, 1].set_title(r"implicit eigenvalues $z_L(k)$ (every mode's true $z_L$)", fontsize=11)
    ax[0, 1].axhline(0, color='0.6', lw=.5); ax[0, 1].axvline(0, color='0.6', lw=.5)
    ax[0, 1].grid(alpha=.3); plt.colorbar(sc2, ax=ax[0, 1], label=r"$\log_{10}|k|$")

    lamL, aN, Kr = _repr_modes(reg, u)
    ok = np.argsort(Kr); gR = amp('RK4', dt * lamL, dt * aN, orders); gmax = 1e-14
    for s in SCHEMES:
        if s == 'RK4':
            continue
        gap = np.abs(amp(s, dt * lamL, dt * aN, orders) - gR)
        gmax = max(gmax, float(gap.max()))
        ax[1, 0].plot(Kr[ok], gap[ok] + 1e-18, '.', ms=2.6, color=COLOR[s], label=LABEL[s])
    ax[1, 0].set_yscale('log')
    ax[1, 0].set_ylim(1e-15, max(1e-6, 5 * gmax))
    ax[1, 0].set_xlabel(r"$|k|$  (mode wavenumber)", fontsize=12)
    ax[1, 0].set_ylabel(r"$|\,|g_s(z_L,z_N)|-|g_{\mathrm{RK4}}|\,|$  (scheme$-$RK4 gap)", fontsize=12)
    ax[1, 0].set_title(rf"scheme$-$RK4 amplification gap at every mode  "
                       rf"($\Delta T={dt:g}$; RK4 $=$ reference)", fontsize=11)
    ax[1, 0].grid(alpha=.3, which='both')

    dts = np.logspace(-4, -1, 40)
    for s in SCHEMES:
        worst = [float(np.max(amp(s, d * lamL, d * aN, orders))) for d in dts]
        ax[1, 1].plot(dts, worst, '-', color=COLOR[s], lw=2, label=LABEL[s])
    ax[1, 1].axhline(1.0, color='0.4', ls='--', lw=1)
    ax[1, 1].axvline(dt, color='0.35', ls=':', lw=1.6, label=rf"operating $\Delta T={dt:g}$")
    ax[1, 1].set_xscale('log'); ax[1, 1].set_yscale('log')
    ax[1, 1].set_xlabel(r"$\Delta T$", fontsize=12)
    ax[1, 1].set_ylabel(r"$\max_k\,|g(z_L(k),z_N(k))|$", fontsize=12)
    ax[1, 1].set_title(r"worst-mode amplification vs $\Delta T$ (CFL limit)", fontsize=11)
    ax[1, 1].grid(alpha=.3, which='both')

    fig.suptitle(regime_label(reg), fontsize=13)
    handles, labels = ax[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def _base_for(family):
    return dict(DEC if family == 'decaying' else FRC)

SWEEP_GRID = {
    'beta': np.linspace(0.0, 3.0, 31),
    'nu':   np.logspace(-6, -3, 31),
    'N':    np.array([128, 192, 256, 384, 512, 768, 1024]),
    'kf':   np.arange(1, 9),
}
SWEEP_MARKS = {
    ('beta', 'forced'):   [0, 0.5, 0.75, 1, 2, 2.5],
    ('nu', 'forced'):     [1.025e-4, 4.0e-5, 2.0e-4],
    ('nu', 'decaying'):   [1.025e-5, 1.025e-4, 2.5e-6],
    ('N', 'forced'):      [256, 512],
    ('N', 'decaying'):    [256, 512],
    ('kf', 'forced'):     [2, 4],
}
SWEEP_XLABEL = {
    'beta': r'$\beta$  (rotation / Rossby drift)',
    'nu':   r'$\nu$  (viscosity; lower $=$ higher Re)',
    'N':    r'$N$  (grid points per side; higher $=$ larger $k_{\max}$)',
    'kf':   r'$k_f$  (forcing wavenumber)',
}

def _apply(reg, param, val):
    r = dict(reg)
    if param == 'beta': r['beta'] = float(val)
    elif param == 'nu': r['nu'] = float(val)
    elif param == 'N':  r['Nx'] = r['Ny'] = int(val)
    elif param == 'kf': r['kf'] = float(val)
    return r

def _crossing(grid, vals, dt):
    vals = np.asarray(vals)
    below = vals < dt
    for i in range(1, len(grid)):
        if below[i] and not below[i-1]:
            return grid[i]
        if below[i-1] and not below[i]:
            return grid[i-1]
    return None

def do_sweep(param, family, u, dt, orders, out_path, tol=1e-2):
    base = _base_for(family)
    grid = SWEEP_GRID[param]
    curves = {s: [] for s in SCHEMES}
    for val in grid:
        reg = _apply(base, param, val)
        for s in SCHEMES:
            curves[s].append(max_stable_dt(reg, s, u, orders, tol=tol))

    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    for mv in SWEEP_MARKS.get((param, family), []):
        ax.axvline(mv, color='0.85', lw=0.9, zorder=0)
    for s in SCHEMES:
        ax.plot(grid, curves[s], 'o-', color=COLOR[s], lw=2.0, ms=4, label=LABEL[s])
    ax.axhline(dt, color='0.35', ls='--', lw=1.4, label=rf'operating $\Delta T={dt:g}$')
    xc = _crossing(grid, curves['AB2CN2'], dt)
    if xc is not None:
        ax.annotate(rf'$\mathrm{{AB2CN2}}$ unstable here', xy=(xc, dt),
                    xytext=(0.04, 0.06), textcoords='axes fraction', fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='C0', lw=1.2), color='C0')

    ax.set_yscale('log')
    if param == 'nu':
        ax.set_xscale('log')
    fixed = []
    if param != 'nu':  fixed.append(rf"$\nu={_sci(base['nu'])}$")
    if param != 'beta' and family == 'forced': fixed.append(rf"$\beta={base['beta']:g}$")
    if param != 'N':   fixed.append(rf"${base['Nx']}^2$")
    fam = 'Decaying' if family == 'decaying' else 'Forced'
    ax.set_xlabel(SWEEP_XLABEL[param], fontsize=12)
    ax.set_ylabel(r'max stable $\Delta T$', fontsize=12)
    ax.set_title(rf"{fam} turbulence: CFL limit vs {param}"
                 rf"   ($u={u:.3g}$, " + ", ".join(fixed) + ")", fontsize=11)
    ax.grid(alpha=.3, which='both')
    ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def main():
    global SCHEMES, LABEL, NN_BASE
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scheme-set', choices=['bare', 'ab2_nn', 'ab4_nn'], default='ab2_nn',
                   help="which schemes to compare. bare=AB2CN2/AB4CN2/RK4; "
                        "ab2_nn=+AB2CN2 NN corrections; ab4_nn=+AB4CN2 NN corrections. "
                        "Sets the output folder unless --out-dir is given.")
    p.add_argument('--orders', type=str, default='3,4,5')
    p.add_argument('--dt', type=float, default=1.0e-3)
    p.add_argument('--ensemble-dir', type=str, default=DEFAULT_ENS)
    p.add_argument('--u-stat', choices=['rms', 'max'], default='rms')
    p.add_argument('--u-char', type=float, default=1.0)
    p.add_argument('--regimes', type=str, default='all')
    p.add_argument('--out-dir', type=str, default=None,
                   help='override the scheme-set default output dir')
    p.add_argument('--sweep', type=str, default=None)
    p.add_argument('--sweep-family', type=str, default='forced', choices=['forced', 'decaying'])
    p.add_argument('--regime-figs', action='store_true')
    args = p.parse_args()
    orders = tuple(int(s) for s in args.orders.split(','))

    SCHEMES = SCHEME_SETS[args.scheme_set]
    NN_BASE = NN_BASE_OF[args.scheme_set]
    LABEL = _build_labels(NN_BASE)
    out_dir = args.out_dir or OUTDIR_OF[args.scheme_set]
    print(f"[scheme-set={args.scheme_set}]  schemes={SCHEMES}  NN_base={NN_BASE}\n"
          f"  out_dir={out_dir}")

    if args.regimes == 'all':
        names = list(REGIMES)
    elif args.regimes in ('DEC', 'FRC'):
        names = [n for n in REGIMES if n.startswith(args.regimes)]
    else:
        names = [s.strip() for s in args.regimes.split(',')]
    use_ens = args.ensemble_dir.lower() != 'none'
    os.makedirs(out_dir, exist_ok=True)

    u_of = {}
    for nm in names:
        reg = REGIMES[nm]
        if use_ens:
            fpath = os.path.join(args.ensemble_dir, nm, 'DNS_FR.npz')
            try:
                uc, ur, um = u_char_from_npz(fpath, reg['Lx'], reg['Ly'], stat=args.u_stat)
                u_of[nm] = uc
                print(f"  [u] {nm:10s} u_rms={ur:.4f} u_max={um:.4f} -> u_{args.u_stat}={uc:.4f}")
            except Exception as e:                       # noqa: BLE001
                u_of[nm] = args.u_char
                print(f"  [u] {nm:10s} FALLBACK u={args.u_char} ({type(e).__name__}: {e})")
        else:
            u_of[nm] = args.u_char
    print()

    hdr = (f"{'family':9}{'N':>5}{'nu':>11}{'beta':>6}{'kf':>4}{'u':>7}  "
           + "".join(f"{s:>11}" for s in SCHEMES) + "   @dt")
    print(f"scheme-set {args.scheme_set}; closure orders {orders};  u = u_{args.u_stat}\n")
    print(hdr); print('-' * len(hdr))
    for nm in names:
        reg = REGIMES[nm]; u = u_of[nm]
        dts = {s: max_stable_dt(reg, s, u, orders) for s in SCHEMES}
        fails = [s for s in SCHEMES if dts[s] < args.dt]
        fam = 'decay' if reg['family'] == 'decaying' else 'forced'
        row = (f"{fam:9}{reg['Nx']:>5}{reg['nu']:>11.2e}{reg['beta']:>6.2f}{reg['kf']:>4}{u:>7.3f}  "
               + "".join(f"{dts[s]:>11.2e}" for s in SCHEMES)
               + (f"  FAIL:{','.join(fails)}" if fails else "  ok"))
        print(row)
    print(f"\n(entries = max Delta_T keeping every dealiased advecting mode |g|<=1; "
          f"operating dt={args.dt:g}.)\n")

    if args.sweep:
        base_id = 'FRC-b1' if args.sweep_family == 'forced' else 'DEC-base'
        if use_ens:
            reg = REGIMES[base_id]
            try:
                u_sweep, ur, um = u_char_from_npz(
                    os.path.join(args.ensemble_dir, base_id, 'DNS_FR.npz'),
                    reg['Lx'], reg['Ly'], stat=args.u_stat)
                print(f"  [sweep u] {base_id} u_{args.u_stat}={u_sweep:.4f}")
            except Exception as e:                       # noqa: BLE001
                u_sweep = args.u_char
                print(f"  [sweep u] FALLBACK u={args.u_char} ({type(e).__name__}: {e})")
        else:
            u_sweep = args.u_char
        for prm in [s.strip() for s in args.sweep.split(',')]:
            if prm not in SWEEP_GRID:
                print(f"  [sweep] unknown parameter '{prm}'"); continue
            fname = os.path.join(out_dir, f"sweep_{args.sweep_family}_{prm}.png")
            do_sweep(prm, args.sweep_family, u_sweep, args.dt, orders, fname)
            print(f"  saved {fname}")

    if (not args.sweep) or args.regime_figs:
        for nm in names:
            reg = REGIMES[nm]
            fname = os.path.join(out_dir, f"stability_{_safe_tag(reg, nm)}.png")
            make_regime_figure(nm, reg, u_of[nm], args.dt, orders, fname)
            print(f"  saved {fname}")


if __name__ == '__main__':
    main()