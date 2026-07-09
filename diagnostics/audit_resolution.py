r"""audit_resolution.py -- Audit B: resolution (SGS-closure).

Implements Supervisor_simulation.md S7.2-S7.3 and S8/B EXACTLY (binding theory
doc; do not re-derive), 5-GRID-AWARE per Sanaa's 2026-07-08 directive (commit
e9a2b2d): the convergence tier is {256, 512, 1024, 2048, 4096}^2, MOD-const,
shared IC, fixed physical eta. Rides the convergence-tier runs; no new runs.

Measured quantities and their theory targets:
  1. eta_phys derivation check (S7.2). The solver's discrete penalty update is
       du/dt |_penalty = -chi * (u - u_o) / tau_eta,   tau_eta = penalty * dt
     (the YAML's factor*dt convention, as implemented -- verified against
     qg/_output/scalars.py, whose meta stores eta = penalty*dt "as applied").
     TERMINOLOGY RECONCILIATION (documented, per S7.2's "the YAML's
     eta = factor*dt convention is not the physical rate"): the theory doc's
     eta_phys is the RATE 1/tau_eta [1/t.u.]; the YAML/scalars eta is the
     TIMESCALE tau_eta [t.u.]. Both are reported. The audit checks, per grid:
     tau_eta(config) == scalars-meta eta (ratio 1), and ACROSS grids the
     fixed-physical-eta rule: tau_eta identical on every tier member (dt
     changes per grid, so the YAML factor must co-vary; violations FLAG).
  2. delta_eta classification (S7.2): the Brinkman penalty layer
       delta_eta = sqrt(nu * tau_eta) = sqrt(nu / eta_phys_rate)
     with the O(delta_eta) Angot et al. (1999) model error; classified per
     grid: delta_eta <= dx  ->  'sharp body';  delta_eta > dx  ->  'mushy
     body'.
  3. Wall-normal tangential-velocity profiles (S7.3) at theta = 60/90/120 deg
     from the time-mean flow (usable window). theta is measured FROM THE FRONT
     (upstream) STAGNATION POINT over the UPPER surface: a point at angle
     theta sits at (xc - r cos(theta), yc + r sin(theta)); the tangential
     direction (sin(theta), cos(theta)) points front-to-rear, so attached-flow
     u_t > 0. Mean u includes U_inlet (re-added per snapshot; snapshots store
     zero-mean u -- solver finding, BRANCH_LOG 1b).
  4. Effective diameter D_eff (S7.3): 2 x the radius where the mean tangential
     velocity crosses zero through the smeared mask (outermost zero crossing
     below the profile maximum; threshold fallback recorded when the profile
     never goes non-positive). Primary value from the azimuthally averaged
     (upper-half) profile; per-theta values also reported.
     Reported as D_eff = D_nominal + Y dx with Y measured.
  5. Surface-vorticity distribution + separation angle theta_sep (S7.3): mean
     omega sampled on the ring r = R + 2 dx (sensitivity-checked at
     R + {1,2,3} dx); theta_sep = first sign change away from the attached
     sign (attached sign auto-detected over theta in [20, 50] deg and
     reported), scanning from 30 deg; sub-sample by linear interpolation.
  6. A priori pts/delta table (S7.1): delta = D Re^{-1/2}; (theory, measured)
     with measured pts/delta = delta_emp/dx, delta_emp = r(max u_t) - r_zero
     at theta = 90. The S7.1 reference table is reproduced and extended to
     the 256/512 coarse anchors, which are labeled UNDER-RESOLVED BY DESIGN
     (theory: pts/delta ~ 0.20 / 0.41 at Re_mid) -- their measured numbers
     are LOWER BOUNDS, not convergence anchors of the wall layer.
  7. Pre-committed 2-DEGREE RULE (S7.3, S8/B), fine pair ONLY (4096 vs 2048
     per the 5-grid directive): FLAG (not stop) if |theta_sep(4096) -
     theta_sep(2048)| > 2 deg; ruling pre-committed: accept-and-report, truth
     remains defined at the fine grid.
  8. Deliverable claim template (S7.3) auto-filled: "wall layer under-resolved
     by ~Nx at 2048^2; theta_sep and shedding change by X% from 2048 to 4096;
     D_eff = D_nominal + Y dx" with N, X, Y measured (shedding X% from
     shedding_summary.npz in the two run dirs when present).

5-grid awareness: --grids CLI (default 256 512 1024 2048 4096); PARTIAL-TIER
TOLERANT -- runs are keyed by the grid N inferred from the snapshot stack;
missing tier members are warned about and listed in the yaml, never fatal.

Outputs: audit_B_summary.yaml + audit_B_summary.npz (full profiles/rings) +
figures (cmap='seismic' for field-like panels, aspect-preserving). Precision:
float64 compute throughout; float32 storage inputs upcast on load (theory doc
S9); summaries stay float64 (tiny).

Usage:
    python audit_resolution.py --run-dirs DIR [DIR ...] [--name DNS]
        [--grids 256 512 1024 2048 4096] [--outdir DIR] [--t-min 30.0]
        [--t-max X] [--batch 0] [--thetas 60 90 120] [--ring-dx 1 2 3]
    python audit_resolution.py --selftest

Selftest (analytically known answers, no data deps): eta/delta_eta arithmetic
against hand-computed values + both classification branches; S7.1 table
reproduction (1.09/1.64/2.74); synthetic vorticity field with an exact
separation angle recovered on all three rings; synthetic swirl profile with
exact r_zero (D_eff) and exact delta_emp; fixed-eta check both branches;
2-degree rule both branches; claim-template arithmetic. Nonzero exit on any
failure; PASS table printed. Runs via qsub only (Amendment 02 S3).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from diagnostics_wake import (                    # flat sibling import (repo rule)
    stream_snapshots, npz_field_shape, uv_from_omega, load_scalars,
    chi_centroid)

# ---------------------------------------------------------------------------- #
# Theory targets (Supervisor_simulation.md S1/S7) -- comparison only.
THEORY = dict(
    D=1.256637,
    nu=6.4443e-4,
    Lx=8.0 * np.pi,
    Re_mid=3900.0,
    Re_rows=(2200.0, 3900.0, 5600.0),
    # S7.1 reference table: pts/delta at the three production/tier grids
    pts_per_delta_S71={
        2200.0: {1024: 1.09, 2048: 2.18, 4096: 4.37},
        3900.0: {1024: 0.82, 2048: 1.64, 4096: 3.28},
        5600.0: {1024: 0.69, 2048: 1.37, 4096: 2.74},
    },
    wall_resolved_pts=10.0,        # classical wall-resolution wish (S7.1)
    grids_full_tier=(256, 512, 1024, 2048, 4096),
    coarse_anchors=(256, 512),     # under-resolved BY DESIGN (e9a2b2d)
    coarse_ptsdelta_Remid={256: 0.20, 512: 0.41},
    two_degree_limit=2.0,          # S7.3 FLAG rule, fine pair only
    fine_pair=(2048, 4096),
    T_wait=30.0,
)

_WARNINGS: list[str] = []


def warn(msg: str) -> None:
    _WARNINGS.append(msg)
    print(f'WARNING: {msg}', flush=True)


# ---------------------------------------------------------------------------- #
# eta / delta_eta (S7.2)

def derive_eta(cfg):
    """eta from the discrete update as implemented: tau_eta = penalty * dt
    (timescale, == scalars-meta eta); eta_phys_rate = 1/tau_eta (the theory
    doc's physical rate); delta_eta = sqrt(nu * tau_eta)."""
    dt = float(cfg['time']['dt'])
    penalty = float(cfg['pde']['penalty'])
    nu = float(cfg['pde']['nu'])
    tau_eta = penalty * dt
    return dict(dt=dt, penalty_factor=penalty, nu=nu,
                tau_eta=tau_eta, eta_phys_rate=1.0 / tau_eta,
                delta_eta=float(np.sqrt(nu * tau_eta)),
                derivation='du/dt|_pen = -chi (u - u_o)/tau_eta, '
                           'tau_eta = penalty*dt (YAML factor*dt as applied); '
                           'eta_phys (theory-doc rate) = 1/tau_eta; '
                           'delta_eta = sqrt(nu tau_eta) = sqrt(nu/eta_phys)')


def classify_delta_eta(delta_eta, dx):
    """S7.2: sharp (delta_eta <= dx) vs mushy (> dx) body."""
    return 'sharp' if delta_eta <= dx else 'mushy'


def check_fixed_eta(tau_by_grid, rtol=1.0e-6):
    """Fixed-physical-eta rule across the tier (charter; S7.2)."""
    vals = {int(k): float(v) for k, v in tau_by_grid.items()}
    if len(vals) < 2:
        return dict(status='single-grid (nothing to compare)', tau_eta=vals,
                    passed=None)
    arr = np.array(list(vals.values()))
    spread = float(arr.max() / arr.min() - 1.0)
    ok = bool(spread <= rtol)
    return dict(passed=ok, spread=spread, rtol=rtol, tau_eta=vals,
                status=('PASS: physical eta frozen across the tier' if ok else
                        'FAIL: tau_eta differs across grids -- '
                        'fixed-physical-eta rule violated (FLAG)'))


# ---------------------------------------------------------------------------- #
# a priori wall layer (S7.1)

def delta_laminar(Re, D=THEORY['D']):
    return float(D / np.sqrt(Re))


def pts_per_delta(Re, N, Lx=THEORY['Lx'], D=THEORY['D']):
    return float(delta_laminar(Re, D) / (Lx / N))


# ---------------------------------------------------------------------------- #
# sampling on the mean fields

def bilinear(field, x, y, Lx, Ly):
    """Periodic bilinear interpolation of a (Ny, Nx) field at physical
    points (x, y); nodes at i*dx (float64)."""
    field = np.asarray(field, dtype=np.float64)
    Ny, Nx = field.shape
    dx, dy = Lx / Nx, Ly / Ny
    fx = np.asarray(x, dtype=np.float64) / dx
    fy = np.asarray(y, dtype=np.float64) / dy
    ix0 = np.floor(fx).astype(np.int64)
    iy0 = np.floor(fy).astype(np.int64)
    wx = fx - ix0
    wy = fy - iy0
    ix0 %= Nx
    iy0 %= Ny
    ix1 = (ix0 + 1) % Nx
    iy1 = (iy0 + 1) % Ny
    return ((1 - wy) * ((1 - wx) * field[iy0, ix0] + wx * field[iy0, ix1])
            + wy * ((1 - wx) * field[iy1, ix0] + wx * field[iy1, ix1]))


def ray_points(xc, yc, theta_deg, r):
    """Points at wall-angle theta (deg, from the FRONT stagnation point over
    the UPPER surface) and radii r: (xc - r cos t, yc + r sin t)."""
    t = np.deg2rad(theta_deg)
    return xc - r * np.cos(t), yc + r * np.sin(t)


def tangential_profile(u_mean, v_mean, xc, yc, theta_deg, r, Lx, Ly):
    """Mean tangential velocity u_t(r) along the theta ray; the tangential
    unit vector (sin t, cos t) points front-to-rear (attached flow > 0)."""
    xs, ys = ray_points(xc, yc, theta_deg, r)
    u = bilinear(u_mean, xs, ys, Lx, Ly)
    v = bilinear(v_mean, xs, ys, Lx, Ly)
    t = np.deg2rad(theta_deg)
    return u * np.sin(t) + v * np.cos(t)


def profile_metrics(r, ut):
    """r_zero (outermost non-positive -> positive crossing below the profile
    max; linear interp), delta_emp = r(max u_t) - r_zero, u_t max."""
    r = np.asarray(r, dtype=np.float64)
    ut = np.asarray(ut, dtype=np.float64)
    imax = int(np.argmax(ut))
    out = dict(u_t_max=float(ut[imax]), r_at_max=float(r[imax]),
               max_at_boundary=bool(imax == len(r) - 1))
    if ut[imax] <= 0.0:
        out.update(found=False, r_zero=None, delta_emp=None,
                   method='no positive tangential flow on ray')
        return out
    seg = ut[:imax + 1]
    cross = np.flatnonzero((seg[:-1] <= 0.0) & (seg[1:] > 0.0))
    if len(cross):
        k = int(cross[-1])
        frac = (0.0 - ut[k]) / (ut[k + 1] - ut[k])
        rz = float(r[k] + frac * (r[k + 1] - r[k]))
        method = 'zero-crossing'
    else:
        thr = 0.01 * ut[imax]
        above = np.flatnonzero(seg > thr)
        k = int(above[0])
        if k == 0:
            rz = float(r[0])
        else:
            frac = (thr - ut[k - 1]) / (ut[k] - ut[k - 1])
            rz = float(r[k - 1] + frac * (r[k] - r[k - 1]))
        method = 'threshold-fallback (u_t never non-positive; 1% of max)'
    out.update(found=True, r_zero=rz, delta_emp=float(r[imax] - rz),
               method=method)
    return out


def theta_sep_from_ring(theta_deg, om_ring, scan_from=30.0,
                        attached_band=(20.0, 50.0)):
    """Separation angle from the mean surface vorticity on a ring: first sign
    change away from the attached sign (auto-detected over attached_band),
    scanning from scan_from deg; linear interpolation."""
    theta_deg = np.asarray(theta_deg, dtype=np.float64)
    om = np.asarray(om_ring, dtype=np.float64)
    m0 = (theta_deg >= attached_band[0]) & (theta_deg <= attached_band[1])
    med = float(np.median(om[m0])) if np.any(m0) else 0.0
    s0 = float(np.sign(med))
    if s0 == 0.0:
        return dict(found=False, theta_sep=None, attached_sign=0.0,
                    note='attached-band vorticity has no definite sign')
    sig = s0 * om
    idx = np.flatnonzero(theta_deg >= scan_from)
    for k in idx[1:]:
        if sig[k - 1] > 0.0 >= sig[k]:
            frac = sig[k - 1] / (sig[k - 1] - sig[k])
            th = float(theta_deg[k - 1]
                       + frac * (theta_deg[k] - theta_deg[k - 1]))
            return dict(found=True, theta_sep=th, attached_sign=s0)
    return dict(found=False, theta_sep=None, attached_sign=s0,
                note='no sign change on ring within scan range')


def two_degree_rule(th_coarse, th_fine, limit=THEORY['two_degree_limit']):
    """Pre-committed S7.3 rule on the fine pair only."""
    if th_coarse is None or th_fine is None:
        return dict(applicable=False, verdict='fine pair incomplete')
    d = float(abs(th_fine - th_coarse))
    flagged = bool(d > limit)
    return dict(applicable=True, delta_deg=d, limit_deg=float(limit),
                flagged=flagged,
                verdict=('FLAG (accept-and-report; truth remains defined at '
                         'the fine grid)' if flagged else 'PASS'))


def claim_template(pts_2048, th_2048, th_4096, fsh_change_pct, D_eff_fine,
                   D_nom, dx_fine):
    """S7.3 deliverable claim with N, X, Y measured."""
    N_under = (THEORY['wall_resolved_pts'] / pts_2048
               if pts_2048 and pts_2048 > 0 else None)
    X_sep = (100.0 * (th_4096 - th_2048) / th_2048
             if th_2048 not in (None, 0) and th_4096 is not None else None)
    Y = ((D_eff_fine - D_nom) / dx_fine
         if D_eff_fine is not None and dx_fine else None)
    txt = ('wall layer under-resolved by ~{}x at 2048^2; theta_sep changes '
           'by {}% (shedding by {}%) from 2048 to 4096; '
           'D_eff = D_nominal + {} dx'.format(
               'N/A' if N_under is None else f'{N_under:.1f}',
               'N/A' if X_sep is None else f'{X_sep:+.2f}',
               'N/A' if fsh_change_pct is None else f'{fsh_change_pct:+.2f}',
               'N/A' if Y is None else f'{Y:.2f}'))
    return dict(N_underresolution_2048=N_under, X_theta_sep_pct=X_sep,
                X_shedding_pct=fsh_change_pct, Y_Deff_in_dx=Y, statement=txt)


# ---------------------------------------------------------------------------- #
# per-run processing

def load_run_config(run_dir, name):
    p = os.path.join(run_dir, f'{name}_FR_params.yaml')
    if os.path.exists(p):
        with open(p) as f:
            return yaml.safe_load(f)
    warn(f'{p} absent; trying config.yaml')
    with open(os.path.join(run_dir, 'config.yaml')) as f:
        return yaml.safe_load(f).get('qg', {})


def mean_fields(fr_npz, keep, Lx, Ly, U_add, batch=0):
    """Streamed time-mean omega, u, v over the usable window. u from omega is
    zero-mean; the mean inlet flow is added analytically afterwards (linear)."""
    n = 0
    om_sum = u_sum = v_sum = None
    for i, sl in stream_snapshots(fr_npz, 'omega_FR', batch_index=batch):
        if not keep[i]:
            continue
        if om_sum is None:
            om_sum = np.zeros_like(sl)
            u_sum = np.zeros_like(sl)
            v_sum = np.zeros_like(sl)
        u, v = uv_from_omega(sl, Lx, Ly)
        om_sum += sl
        u_sum += u
        v_sum += v
        n += 1
    if n == 0:
        raise RuntimeError(f'{fr_npz}: no snapshots in the usable window')
    U_mean = float(np.mean(U_add[keep])) if U_add is not None else 0.0
    return om_sum / n, u_sum / n + U_mean, v_sum / n, n, U_mean


def analyze_run(run_dir, name, t_min, t_max, batch, thetas, ring_ks,
                theta_ring):
    fr_npz = os.path.join(run_dir, f'{name}_FR.npz')
    if not os.path.exists(fr_npz):
        raise FileNotFoundError(fr_npz)
    cfg = load_run_config(run_dir, name)
    Lx = float(cfg['grid']['Lx'])
    Ly = float(cfg['grid']['Ly'])
    R_nom = float(cfg['mask']['r'])
    D_nom = 2.0 * R_nom
    eta = derive_eta(cfg)

    z = np.load(fr_npz)
    times = np.asarray(z['times'], dtype=np.float64)
    chi = (np.asarray(z['chi_obs'], dtype=np.float64)
           if 'chi_obs' in z.files else None)
    shape, _ = npz_field_shape(fr_npz, 'omega_FR')
    Ny, Nx = shape[-2], shape[-1]
    dx, dy = Lx / Nx, Ly / Ny

    tm = float(times[-1]) if t_max is None else float(t_max)
    keep = (times >= t_min) & (times <= tm)
    if not keep.any():
        warn(f'{run_dir}: no snapshots at t >= {t_min}; using ALL '
             f'{len(times)} snapshots (restart-clock run?)')
        keep = np.ones(len(times), dtype=bool)

    scal = load_scalars(os.path.join(run_dir, 'scalars.npz'))
    U_cfg = float(cfg.get('bc', {}).get('inlet_velocity', np.nan)) \
        if cfg.get('bc') else np.nan
    if scal is not None and 'U_inlet' in scal:
        Ui = scal['U_inlet']
        Ui = Ui[:, batch] if Ui.ndim == 2 else Ui
        U_add = np.interp(times, scal['t'], Ui)
    elif np.isfinite(U_cfg):
        U_add = np.full(len(times), U_cfg)
        warn(f'{run_dir}: scalars.npz absent/short -- config '
             f'inlet_velocity={U_cfg} used for the mean flow')
    else:
        U_add = np.zeros(len(times))
        warn(f'{run_dir}: no inlet speed available; mean u EXCLUDES the '
             'mean flow')

    # eta cross-check vs the recorder's as-applied value
    eta_meta_ratio = None
    if scal is not None and 'eta' in scal.get('meta', {}):
        eta_meta_ratio = float(eta['tau_eta'] / float(scal['meta']['eta']))

    om_mean, u_mean, v_mean, n_used, U_mean = mean_fields(
        fr_npz, keep, Lx, Ly, U_add, batch=batch)

    if chi is not None:
        xc, yc = chi_centroid(chi, dx, dy)
    elif scal is not None and 'obstacle_centroid_xy' in scal.get('meta', {}):
        xc, yc = map(float, scal['meta']['obstacle_centroid_xy'])
        warn(f'{run_dir}: chi_obs absent; centroid from scalars meta')
    else:
        xc, yc = Lx / 2.0, Ly / 2.0
        warn(f'{run_dir}: centroid assumed at domain center')

    # Re of the run (MOD-const expectation: Re_mid)
    if scal is not None and 'Re_inlet' in scal:
        Re = scal['Re_inlet']
        Re = Re[:, batch] if Re.ndim == 2 else Re
        Re_run = float(np.nanmedian(Re))
    elif np.isfinite(U_cfg):
        Re_run = float(U_cfg * D_nom / eta['nu'])
    else:
        Re_run = THEORY['Re_mid']
        warn(f'{run_dir}: Re unknown; assuming Re_mid = {Re_run}')

    # ---- wall-normal profiles (S7.3) ------------------------------------- #
    r_lo, r_hi = 0.5 * R_nom, R_nom + 0.6 * D_nom
    r = np.arange(r_lo, r_hi, 0.5 * dx)
    profiles, prof_metrics = {}, {}
    for th in thetas:
        ut = tangential_profile(u_mean, v_mean, xc, yc, th, r, Lx, Ly)
        profiles[th] = ut
        prof_metrics[th] = profile_metrics(r, ut)

    # azimuthally averaged (upper half) -> D_eff
    th_az = np.arange(10.0, 170.5, 5.0)
    ut_az = np.mean(np.stack([
        tangential_profile(u_mean, v_mean, xc, yc, th, r, Lx, Ly)
        for th in th_az]), axis=0)
    az = profile_metrics(r, ut_az)
    D_eff = 2.0 * az['r_zero'] if az.get('found') and az['r_zero'] else None

    # ---- surface vorticity rings + theta_sep (S7.3) ----------------------- #
    rings, seps = {}, {}
    for k in ring_ks:
        xs, ys = ray_points(xc, yc, theta_ring, R_nom + k * dx)
        om_ring = bilinear(om_mean, xs, ys, Lx, Ly)
        rings[k] = om_ring
        seps[k] = theta_sep_from_ring(theta_ring, om_ring)

    # shedding (for the fine-pair X%), if the tracker already ran here
    f_sh = None
    shed_p = os.path.join(run_dir, 'shedding_summary.npz')
    if os.path.exists(shed_p):
        zs = np.load(shed_p)
        if 'f_sh_median' in zs.files:
            f_sh = float(zs['f_sh_median'])

    delta_th = delta_laminar(Re_run, D_nom)
    m90 = prof_metrics.get(90, next(iter(prof_metrics.values())))
    return dict(
        run_dir=os.path.abspath(run_dir), N=int(Nx), Ny=int(Ny),
        Lx=Lx, dx=dx, n_snapshots_used=int(n_used),
        n_snapshots_total=int(len(times)), U_mean_added=U_mean,
        R_nominal=R_nom, D_nominal=D_nom, centroid=[float(xc), float(yc)],
        Re_run=Re_run, eta=eta, eta_vs_scalars_meta_ratio=eta_meta_ratio,
        delta_eta_over_dx=float(eta['delta_eta'] / dx),
        body_class=classify_delta_eta(eta['delta_eta'], dx),
        delta_theory=delta_th,
        pts_per_delta_theory=float(delta_th / dx),
        delta_emp_90=m90.get('delta_emp'),
        pts_per_delta_measured=(float(m90['delta_emp'] / dx)
                                if m90.get('delta_emp') else None),
        profile_metrics={int(t): prof_metrics[t] for t in prof_metrics},
        azimuthal=az, D_eff=D_eff,
        Y_Deff_in_dx=(float((D_eff - D_nom) / dx)
                      if D_eff is not None else None),
        theta_sep={int(k): seps[k] for k in seps},
        f_sh_median=f_sh,
        _arrays=dict(r=r, profiles=profiles, ut_az=ut_az, rings=rings),
    )


# ---------------------------------------------------------------------------- #
def run_audit(args):
    os.makedirs(args.outdir, exist_ok=True)
    theta_ring = np.arange(2.0, 178.01, 0.5)
    ring_ks = [int(k) for k in args.ring_dx]
    k_primary = 2 if 2 in ring_ks else ring_ks[0]

    runs = {}
    for d in args.run_dirs:
        res = analyze_run(d, args.name, args.t_min, args.t_max, args.batch,
                          [int(t) for t in args.thetas], ring_ks, theta_ring)
        N = res['N']
        if N in runs:
            warn(f'duplicate grid {N}: keeping {runs[N]["run_dir"]}, '
                 f'ignoring {res["run_dir"]}')
            continue
        runs[N] = res
        print(f'[auditB] grid {N}^2: {res["n_snapshots_used"]} snapshots, '
              f'tau_eta={res["eta"]["tau_eta"]:.6g}, '
              f'delta_eta/dx={res["delta_eta_over_dx"]:.3g} '
              f'({res["body_class"]}), '
              f'theta_sep(k={k_primary})='
              f'{runs[N]["theta_sep"][k_primary].get("theta_sep")}')

    expected = [int(g) for g in args.grids]
    missing = sorted(set(expected) - set(runs))
    if missing:
        warn(f'partial tier: grids {missing} have no run dir (tolerated)')

    # ---- cross-grid ------------------------------------------------------- #
    fixed_eta = check_fixed_eta({N: runs[N]['eta']['tau_eta'] for N in runs})
    if fixed_eta.get('passed') is False:
        warn(fixed_eta['status'])

    cN, fN = THEORY['fine_pair']
    th_c = (runs[cN]['theta_sep'][k_primary].get('theta_sep')
            if cN in runs else None)
    th_f = (runs[fN]['theta_sep'][k_primary].get('theta_sep')
            if fN in runs else None)
    rule = two_degree_rule(th_c, th_f)
    if rule.get('flagged'):
        warn(f'2-degree rule FLAG: |theta_sep({fN}) - theta_sep({cN})| = '
             f'{rule["delta_deg"]:.2f} deg > {rule["limit_deg"]} deg')

    fsh_pct = None
    if cN in runs and fN in runs and runs[cN]['f_sh_median'] \
            and runs[fN]['f_sh_median']:
        fsh_pct = float(100.0 * (runs[fN]['f_sh_median']
                                 - runs[cN]['f_sh_median'])
                        / runs[cN]['f_sh_median'])

    claim = claim_template(
        pts_2048=(runs[cN]['pts_per_delta_measured'] if cN in runs else None),
        th_2048=th_c, th_4096=th_f, fsh_change_pct=fsh_pct,
        D_eff_fine=(runs[fN]['D_eff'] if fN in runs else None),
        D_nom=(runs[fN]['D_nominal'] if fN in runs
               else next(iter(runs.values()))['D_nominal']),
        dx_fine=(runs[fN]['dx'] if fN in runs else None))

    # S7.1 comparison table (theory formula vs the doc's numbers + measured)
    s71 = []
    for Re in THEORY['Re_rows']:
        row = dict(Re=float(Re), delta=delta_laminar(Re))
        for N in sorted(set(list(runs) + list(THEORY['grids_full_tier']))):
            e = dict(formula=pts_per_delta(Re, N))
            doc = THEORY['pts_per_delta_S71'].get(Re, {}).get(N)
            if doc is not None:
                e['S71_table'] = doc
            if N in THEORY['coarse_anchors']:
                e['note'] = 'under-resolved coarse anchor (lower bound)'
            s71.append(dict(row, N=int(N), **e))

    per_grid = {}
    for N in sorted(runs):
        r = {k: v for k, v in runs[N].items() if k != '_arrays'}
        if N in THEORY['coarse_anchors']:
            r['resolution_class'] = (
                'UNDER-RESOLVED COARSE ANCHOR (BY DESIGN, e9a2b2d): '
                'pts/delta ~ '
                f'{THEORY["coarse_ptsdelta_Remid"][N]} at Re_mid; measured '
                'wall-layer numbers are LOWER BOUNDS, not convergence anchors')
        per_grid[f'N{N}'] = r

    summary = dict(
        inputs=dict(run_dirs=[os.path.abspath(d) for d in args.run_dirs],
                    name=args.name, grids_expected=expected,
                    grids_found=sorted(runs), grids_missing=missing,
                    t_min=float(args.t_min),
                    t_max=(None if args.t_max is None else float(args.t_max)),
                    batch=int(args.batch), thetas=[int(t) for t in args.thetas],
                    ring_dx=ring_ks, ring_primary_k=k_primary),
        fixed_eta_check=fixed_eta,
        per_grid=per_grid,
        S71_pts_per_delta=s71,
        theta_sep_fine_pair=dict(coarse_N=cN, fine_N=fN,
                                 theta_sep_coarse=th_c, theta_sep_fine=th_f,
                                 two_degree_rule=rule),
        claim=claim,
        warnings=list(_WARNINGS),
    )
    with open(os.path.join(args.outdir, 'audit_B_summary.yaml'), 'w') as fh:
        yaml.safe_dump(json.loads(json.dumps(summary, default=float)), fh,
                       sort_keys=False)

    npz = dict(theta_ring=theta_ring,
               meta=json.dumps(summary, default=float))
    for N in sorted(runs):
        a = runs[N]['_arrays']
        npz[f'r_N{N}'] = a['r']
        npz[f'ut_az_N{N}'] = a['ut_az']
        for th, ut in a['profiles'].items():
            npz[f'ut_N{N}_th{int(th)}'] = ut
        for k, om in a['rings'].items():
            npz[f'ring_om_N{N}_k{int(k)}'] = om
    np.savez(os.path.join(args.outdir, 'audit_B_summary.npz'), **npz)

    _figures(args.outdir, runs, theta_ring, k_primary,
             [int(t) for t in args.thetas], rule)
    print(yaml.safe_dump(json.loads(json.dumps(
        dict(fixed_eta_check=fixed_eta, claim=claim,
             theta_sep_fine_pair=summary['theta_sep_fine_pair']),
        default=float)), sort_keys=False))
    return summary


def _figures(outdir, runs, theta_ring, k_primary, thetas, rule):
    if not runs:
        return
    grids = sorted(runs)
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(grids)))

    # wall-normal profiles
    fig, axs = plt.subplots(1, len(thetas), figsize=(4.2 * len(thetas), 4.0),
                            sharey=True)
    axs = np.atleast_1d(axs)
    for ax, th in zip(axs, thetas):
        for c, N in zip(colors, grids):
            a = runs[N]['_arrays']
            R = runs[N]['R_nominal']
            D = runs[N]['D_nominal']
            ax.plot((a['r'] - R) / D, a['profiles'][th], lw=1.1, color=c,
                    label=f'{N}^2')
            mz = runs[N]['profile_metrics'][th]
            if mz.get('r_zero'):
                ax.axvline((mz['r_zero'] - R) / D, color=c, ls=':', lw=0.7)
        ax.axhline(0.0, color='k', lw=0.6)
        ax.set_xlabel(r'$(r - R)/D$')
        ax.set_title(rf'$\theta = {th}^\circ$')
    axs[0].set_ylabel(r'$\overline{u}_t$')
    axs[0].legend(fontsize=8)
    fig.suptitle('wall-normal mean tangential velocity (dotted: r_zero)')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig_auditB_profiles.png'), dpi=150)
    plt.close(fig)

    # surface vorticity + theta_sep
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for c, N in zip(colors, grids):
        a = runs[N]['_arrays']
        om = a['rings'][k_primary]
        ax.plot(theta_ring, om / max(np.max(np.abs(om)), 1e-30), lw=1.1,
                color=c, label=f'{N}^2')
        sp = runs[N]['theta_sep'][k_primary]
        if sp.get('found'):
            ax.axvline(sp['theta_sep'], color=c, ls='--', lw=0.8)
    ax.axhline(0.0, color='k', lw=0.6)
    ax.set_xlabel(r'$\theta$ [deg from front stagnation]')
    ax.set_ylabel(rf'$\overline{{\omega}}$ on $r=R+{k_primary}dx$ (normalized)')
    ax.set_title(r'mean surface vorticity; dashed: $\theta_{sep}$')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig_auditB_surfvort.png'), dpi=150)
    plt.close(fig)

    # tier convergence panels
    fig, axs = plt.subplots(1, 4, figsize=(16.0, 3.8))
    Ns = np.array(grids, dtype=float)
    th_sep = [runs[N]['theta_sep'][k_primary].get('theta_sep') for N in grids]
    axs[0].plot(Ns, [np.nan if t is None else t for t in th_sep], 'o-')
    axs[0].set_ylabel(r'$\theta_{sep}$ [deg]')
    if rule.get('applicable'):
        axs[0].set_title(f"2-deg rule ({THEORY['fine_pair'][0]} vs "
                         f"{THEORY['fine_pair'][1]}): {rule['verdict']}",
                         fontsize=9)
    ydef = [np.nan if runs[N]['Y_Deff_in_dx'] is None
            else runs[N]['Y_Deff_in_dx'] for N in grids]
    axs[1].plot(Ns, ydef, 'o-')
    axs[1].axhline(0.0, color='k', lw=0.6)
    axs[1].set_ylabel(r'$(D_{eff} - D_{nom})/dx$')
    ptsm = [np.nan if runs[N]['pts_per_delta_measured'] is None
            else runs[N]['pts_per_delta_measured'] for N in grids]
    axs[2].plot(Ns, ptsm, 'o-', label='measured (delta_emp/dx)')
    axs[2].plot(Ns, [runs[N]['pts_per_delta_theory'] for N in grids], 's--',
                label='a priori (S7.1)')
    axs[2].axhline(THEORY['wall_resolved_pts'], color='r', ls=':', lw=0.8,
                   label='resolved (~10)')
    axs[2].set_ylabel(r'pts/$\delta$')
    axs[2].set_yscale('log')
    axs[2].legend(fontsize=7)
    axs[3].plot(Ns, [runs[N]['delta_eta_over_dx'] for N in grids], 'o-')
    axs[3].axhline(1.0, color='r', ls=':', lw=0.8, label='sharp/mushy')
    axs[3].set_ylabel(r'$\delta_\eta / dx$')
    axs[3].set_yscale('log')
    axs[3].legend(fontsize=7)
    for ax in axs:
        ax.set_xscale('log', base=2)
        ax.set_xticks(Ns)
        ax.set_xticklabels([f'{int(n)}' for n in Ns])
        ax.set_xlabel('N')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig_auditB_tier.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------- #
# Selftest -- synthetic, analytically known answers (Amendment 02 S3: qsub only)

def selftest():
    checks = []

    def add(name, target, got, tol_rel):
        ok = (got is not None and np.isfinite(got)
              and abs(got - target) <= tol_rel * max(abs(target), 1e-30))
        checks.append((name, target, got, tol_rel, bool(ok)))

    def add_bool(name, ok):
        checks.append((name, True, ok, 0.0, bool(ok)))

    # 1. eta derivation + delta_eta arithmetic (hand-computed targets)
    cfg = dict(pde=dict(penalty=1.25, nu=6.4443e-4), time=dict(dt=2.5e-4))
    eta = derive_eta(cfg)
    add('tau_eta = penalty*dt', 3.125e-4, eta['tau_eta'], 1e-12)
    add('eta_phys rate = 1/tau_eta', 3200.0, eta['eta_phys_rate'], 1e-12)
    add('delta_eta = sqrt(nu tau_eta)', 4.48758e-4, eta['delta_eta'], 1e-4)
    dx2048 = THEORY['Lx'] / 2048
    add_bool('classification sharp branch',
             classify_delta_eta(eta['delta_eta'], dx2048) == 'sharp')
    add_bool('classification mushy branch',
             classify_delta_eta(0.02, dx2048) == 'mushy')

    # 2. fixed-eta check, both branches
    fe = check_fixed_eta({2048: 3.125e-4, 4096: 3.125e-4})
    add_bool('fixed-eta PASS branch', fe['passed'] is True)
    fe = check_fixed_eta({2048: 3.125e-4, 4096: 1.25 * 1.25e-4})
    add_bool('fixed-eta FAIL branch', fe['passed'] is False)

    # 3. S7.1 table reproduction (doc rounds to 2-3 figures; 1% tol)
    add('pts/delta Re2200 @1024 (S7.1: 1.09)', 1.09,
        pts_per_delta(2200.0, 1024), 0.01)
    add('pts/delta Re3900 @2048 (S7.1: 1.64)', 1.64,
        pts_per_delta(3900.0, 2048), 0.01)
    add('pts/delta Re5600 @4096 (S7.1: 2.74)', 2.74,
        pts_per_delta(5600.0, 4096), 0.01)

    # 4. synthetic vorticity: omega = theta(x,y) - theta_sep_true, so the
    #    ring sign change sits at EXACTLY theta_sep_true at every radius.
    N, Lx = 512, THEORY['Lx']
    dx = Lx / N
    xc = yc = Lx / 2.0
    xs = np.arange(N) * dx
    X, Y = np.meshgrid(xs, xs)
    phi = np.arctan2(Y - yc, X - xc)
    theta_pt = np.pi - np.abs(phi)
    th_true = 95.0
    om_syn = theta_pt - np.deg2rad(th_true)
    theta_ring = np.arange(2.0, 178.01, 0.5)
    R_nom = 0.9
    for k in (1, 2, 3):
        rx, ry = ray_points(xc, yc, theta_ring, R_nom + k * dx)
        sp = theta_sep_from_ring(theta_ring, bilinear(om_syn, rx, ry, Lx, Lx))
        add(f'theta_sep ring k={k}', th_true, sp.get('theta_sep'), 0.005)
        add_bool(f'attached sign detected (k={k})',
                 sp.get('attached_sign') == -1.0)

    # 5. synthetic swirl: u_t(r) = z e^{1-z}, z = (r - r0)/w -- zero crossing
    #    at r0 exactly (D_eff = 2 r0), max at r0 + w (delta_emp = w exactly).
    r0, w = 1.0, 0.35
    rr = np.sqrt((X - xc) ** 2 + (Y - yc) ** 2)
    zz = (rr - r0) / w
    ut_field = zz * np.exp(1.0 - zz)
    sgn = np.where(Y >= yc, 1.0, -1.0)
    u_syn = ut_field * np.sin(theta_pt)
    v_syn = ut_field * np.cos(theta_pt) * sgn
    r = np.arange(0.5 * R_nom, R_nom + 0.6 * (2 * R_nom), 0.5 * dx)
    ut90 = tangential_profile(u_syn, v_syn, xc, yc, 90.0, r, Lx, Lx)
    m = profile_metrics(r, ut90)
    add('r_zero (theta=90)', r0, m['r_zero'], 0.01)
    add('delta_emp = w', w, m['delta_emp'], 0.08)
    add('u_t max = 1', 1.0, m['u_t_max'], 0.02)
    add_bool('crossing method used', m['method'] == 'zero-crossing')
    # azimuthal D_eff over the upper half
    th_az = np.arange(10.0, 170.5, 5.0)
    ut_az = np.mean(np.stack([
        tangential_profile(u_syn, v_syn, xc, yc, th, r, Lx, Lx)
        for th in th_az]), axis=0)
    maz = profile_metrics(r, ut_az)
    add('azimuthal D_eff = 2 r0', 2.0 * r0, 2.0 * maz['r_zero'], 0.01)
    # threshold fallback branch: strictly positive profile
    mfb = profile_metrics(r, ut90 - ut90.min() + 0.05)
    add_bool('threshold fallback engages',
             mfb['method'].startswith('threshold-fallback'))

    # 6. 2-degree rule, both branches + inapplicable
    add_bool('2-deg PASS branch',
             two_degree_rule(95.0, 96.5)['flagged'] is False)
    add_bool('2-deg FLAG branch',
             two_degree_rule(95.0, 98.0)['flagged'] is True)
    add_bool('2-deg inapplicable on partial pair',
             two_degree_rule(None, 96.0)['applicable'] is False)

    # 7. claim-template arithmetic
    cl = claim_template(pts_2048=1.64, th_2048=90.0, th_4096=94.5,
                        fsh_change_pct=1.2, D_eff_fine=1.35, D_nom=1.256637,
                        dx_fine=THEORY['Lx'] / 4096)
    add('claim N = 10/1.64', 10.0 / 1.64, cl['N_underresolution_2048'], 1e-9)
    add('claim X = 5%', 5.0, cl['X_theta_sep_pct'], 1e-9)
    add('claim Y in dx', (1.35 - 1.256637) / (THEORY['Lx'] / 4096),
        cl['Y_Deff_in_dx'], 1e-9)

    # ---- PASS table -------------------------------------------------------- #
    print('\n===== audit_resolution selftest =====')
    print(f"{'check':<44}{'target':>12}{'got':>14}{'tol':>8}  verdict")
    all_ok = True
    for name, tgt, got, tol, ok in checks:
        all_ok &= ok
        gs = f'{got:.6g}' if isinstance(got, float) else str(got)
        print(f'{name:<44}{tgt!s:>12.12}{gs:>14.14}{tol!s:>8}  '
              f"{'PASS' if ok else 'FAIL'}")
    print(f"===== overall: {'PASS' if all_ok else 'FAIL'} "
          f'({sum(c[-1] for c in checks)}/{len(checks)}) =====')
    return 0 if all_ok else 1


# ---------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--run-dirs', nargs='+', default=None,
                   help='convergence-tier run dirs (each with '
                        '{name}_FR.npz, {name}_FR_params.yaml, scalars.npz)')
    p.add_argument('--name', default='DNS')
    p.add_argument('--grids', type=int, nargs='+',
                   default=list(THEORY['grids_full_tier']),
                   help='expected tier (partial-tier tolerant)')
    p.add_argument('--outdir', default=None,
                   help='output dir (default: <first run dir>/audit_B)')
    p.add_argument('--t-min', type=float, default=30.0,
                   help='usable-window start (T_wait, theory doc S1)')
    p.add_argument('--t-max', type=float, default=None)
    p.add_argument('--batch', type=int, default=0)
    p.add_argument('--thetas', type=int, nargs='+', default=[60, 90, 120],
                   help='wall-normal profile angles [deg from front '
                        'stagnation] (S7.3)')
    p.add_argument('--ring-dx', type=int, nargs='+', default=[1, 2, 3],
                   help='surface-vorticity ring offsets in dx (S7.3; '
                        'primary = 2)')
    p.add_argument('--selftest', action='store_true')
    args = p.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if not args.run_dirs:
        p.error('--run-dirs required (or --selftest)')
    if args.outdir is None:
        args.outdir = os.path.join(os.path.abspath(args.run_dirs[0]),
                                   'audit_B')
    run_audit(args)


if __name__ == '__main__':
    main()
