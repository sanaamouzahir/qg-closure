r"""audit_decorrelation.py -- Audit A: decorrelation and sampling (SGS-closure).

Implements Supervisor_simulation.md S8/Audit-A EXACTLY (binding theory doc;
comparison targets from S2/S4/S5, all thresholds pre-committed -- do not
re-derive). Runs at the FPC-const gate, BEFORE the 4 modulated runs are
submitted.

Measured quantities and their theory targets:
  1. ACF rho_hat(tau) + integral timescale tau_int (trapezoid to FIRST ZERO
     CROSSING) of:
       - C_L(t)            [dense scalars, Cl_mid]        vs tau_E_wake =
       - probe v(t), xc+1D [dense scalars, probe index 0]     D/(0.85 U):
                                                              [0.52, 1.31],
                                                              0.74 at U_mid
       - Pi_FF(t) at the 5 recorder probe points, per scale s in {2,4,8}
         [Pi_FF snapshot series at dt_save cadence]        vs tau_E_s =
                                                              {0.014, 0.029,
                                                               0.058}
     Reported as (theory, measured, ratio). NOTE (pre-stated): at dt_save =
     0.25 the filter-scale ACF is under-resolved BY CONSTRUCTION
     (2 tau_int < dt_save); such entries report status
     "tau_int < dt_save/2 (unresolved, consistent with theory)" instead of a
     bogus number. The dense-scalar ACFs are the resolved ones.
  2. Convection speed U_c: peak-lag cross-correlation of probe_v at xc+1D vs
     xc+2D (indices 0, 1), sub-sample lag via parabolic interpolation of the
     correlation peak; U_c = D/lag, vs 0.85 U (0.85*U_mid = 1.7 and
     0.85*mean U_inlet).
  3. Spatial ACF of Pi_FF, radially averaged over the Pi-active wake box
     (~3D x 0.7 Lx downstream, S4) -> l_corr(s) = first 1/e crossing; the S4
     patch counts are recomputed with measured l_corr replacing the
     [Delta_s, D] bracket.
  4. Phase-coverage histogram of snapshot times mod measured T_sh (from
     shedding_summary.npz), 8 and 24 bins; uniformity = max/min bin count.
     Validates the S5 dt_save = 0.27 commensurability fix on MOD-const.
  5. Recomputed N_eff table (S4 format: scale | 2 tau_int | regime | N_eff)
     with measured values; T_u = actual series end minus T_wait = 30.
  6. DECISION RULE (pre-committed, S8/A): if measured 2 tau_int(s=4) > 0.5 ->
     "RELAX: modulated runs dt_save=0.5; re-check n_pp = T_sh_measured/0.5
     >= 8"; else "HOLD-0.25".

Outputs: audit_A_summary.yaml + audit_A_summary.npz (full curves) + figures.
Precision: float64 compute; Pi_FF/snapshot inputs may arrive float32 (theory
doc S9 storage policy) and are upcast on load.

Usage:
    python audit_decorrelation.py --scalars scalars.npz \
        --shedding-summary <dir>/shedding_summary.npz \
        --piff-dir <dir with *LES*.npz Pi_FF files (compute_pi_ff.py layout:
                    keys pi_ff (B,T,Ny,Nx), times, _scale)> \
        [--outdir DIR] [--t-min 30.0] [--batch 0] [--Lx 25.132741] [--T-sh X]
    python audit_decorrelation.py --selftest

Selftest (analytically known answers, no data): OU process with known theta
(ACF e^{-tau/theta} -> tau_int = theta), fractionally shifted signal pair for
the U_c lag, 2D Gaussian-correlated field with known l_corr, uniform vs
commensurate-clustered snapshot times for the phase histogram, and the
decision rule exercised on both branches. Nonzero exit on any failure; PASS
table printed. Runs via qsub only (Amendment 02 S3).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from shedding_tracker import _load_scalars      # flat sibling import (repo rule)

# ---------------------------------------------------------------------------- #
# Theory targets (Supervisor_simulation.md S1/S2/S4) -- comparison only.
THEORY = dict(
    U_mid=2.0,
    Uc_factor=0.85,
    tau_E_wake_mid=0.74,
    tau_E_wake_range=(0.52, 1.31),
    tau_E_s={2: 0.014, 4: 0.029, 8: 0.058},
    dx_FR=0.012272,
    Delta_s={2: 0.0245, 4: 0.0491, 8: 0.0982},
    D=1.256637,
    A_wake=66.0,                       # ~3D x 0.7 Lx (S4)
    T_wait=30.0,
    dt_save=0.25,
    two_tau_wake=1.48, Neff_wake=61, Neff_s=360, T_u=90.0,
    decision_threshold=0.5,            # on 2 tau_int(s=4), S8/A
    n_pp_min=8,                        # phase-binning floor (S3.2)
)
UNRESOLVED_MSG = 'tau_int < dt_save/2 (unresolved, consistent with theory)'


# ---------------------------------------------------------------------------- #
# Estimators (float64 throughout)

def acf(x, dt, max_frac=0.5):
    """Biased (1/n) autocorrelation estimate via zero-padded FFT.
    Returns (lags, rho) with rho[0] = 1, up to max_frac of the series."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    X = np.fft.rfft(x, nfft)
    c = np.fft.irfft(X * np.conj(X))[:n] / n
    if c[0] <= 0:
        raise ValueError('zero-variance series')
    K = max(2, int(n * max_frac))
    return np.arange(K) * dt, (c / c[0])[:K]


def tau_int_first_zero(lags, rho):
    """Integral timescale: trapezoid of rho_hat to its FIRST ZERO CROSSING
    (linearly interpolated). Returns dict(tau_int, tau_cross, crossed,
    k_cross, resolved). resolved is DETERMINISTIC under estimator noise:
    rho_hat(dt) >= 1/e <=> the true decorrelation time is at least the
    sampling interval; below that the ACF is unresolved at this cadence
    (rho_hat(dt) ~ 0 +- O(1/sqrt(n)), far from 1/e)."""
    dt = lags[1] - lags[0]
    resolved = bool(rho[1] >= 1.0 / np.e)
    neg = np.flatnonzero(rho <= 0.0)
    if len(neg) == 0:                      # no crossing inside the window
        tau = float(np.trapz(rho, lags))
        return dict(tau_int=tau, tau_cross=float(lags[-1]), crossed=False,
                    k_cross=int(len(rho)), resolved=resolved)
    k = int(neg[0])
    if k == 0:
        raise ValueError('rho[0] must be 1')
    frac = rho[k - 1] / (rho[k - 1] - rho[k])
    tau_cross = float(lags[k - 1] + frac * dt)
    tau = float(np.trapz(rho[:k], lags[:k])
                + 0.5 * rho[k - 1] * (tau_cross - lags[k - 1]))
    return dict(tau_int=tau, tau_cross=tau_cross, crossed=True, k_cross=k,
                resolved=resolved)


def xcorr_peak_lag(a, b, dt, max_lag):
    """Positive-lag cross-correlation C(tau) = <a(t) b(t+tau)>; the delay of b
    relative to a is the peak lag, refined SUB-SAMPLE by parabolic
    interpolation of (C[k-1], C[k], C[k+1]). Returns (lag, lags, c)."""
    a = np.asarray(a, np.float64) - np.mean(a)
    b = np.asarray(b, np.float64) - np.mean(b)
    n = len(a)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    C = np.fft.irfft(np.conj(np.fft.rfft(a, nfft)) * np.fft.rfft(b, nfft))
    K = min(int(round(max_lag / dt)), n - 2)
    c = C[:K]
    k = int(np.argmax(c[1:K - 1])) + 1
    denom = c[k - 1] - 2.0 * c[k] + c[k + 1]
    delta = 0.5 * (c[k - 1] - c[k + 1]) / denom if denom < 0 else 0.0
    lag = (k + float(delta)) * dt
    return float(lag), np.arange(K) * dt, c


def spatial_acf(fields, dx):
    """Mean 2D spatial ACF of a stack of (ny, nx) boxes: zero-padded FFT
    autocorrelation, overlap-count (unbiased) normalization, normalized to 1
    at zero lag. Returns (rho2d fftshifted, center_index (cy, cx))."""
    fields = np.asarray(fields, dtype=np.float64)
    if fields.ndim == 2:
        fields = fields[None]
    M, ny, nx = fields.shape
    py, px = 2 * ny, 2 * nx
    ones = np.zeros((py, px))
    ones[:ny, :nx] = 1.0
    O = np.fft.rfft2(ones)
    cnt = np.fft.fftshift(np.fft.irfft2(O * np.conj(O), s=(py, px)))
    acc = np.zeros((py, px))
    for w in fields:
        w = w - w.mean()
        W = np.fft.rfft2(w, s=(py, px))
        acc += np.fft.irfft2(W * np.conj(W), s=(py, px))
    acc = np.fft.fftshift(acc)
    Cn = acc / np.maximum(cnt, 1.0)
    cy, cx = ny, nx                      # zero lag after fftshift (even sizes)
    rho2d = Cn / Cn[cy, cx]
    return rho2d, (cy, cx)


def radial_profile(rho2d, center, dx, r_max):
    """Radially averaged profile of a centered 2D ACF, bin width dx.
    Returns (r, rho_r) with r = mean radius per bin."""
    cy, cx = center
    ny, nx = rho2d.shape
    yy = (np.arange(ny) - cy)[:, None]
    xx = (np.arange(nx) - cx)[None, :]
    r = np.hypot(yy, xx) * dx
    nb = int(r_max / dx)
    ib = (r / dx).astype(np.int64)
    m = ib < nb
    cnt = np.bincount(ib[m], minlength=nb).astype(np.float64)
    s_rho = np.bincount(ib[m], weights=rho2d[m], minlength=nb)
    s_r = np.bincount(ib[m], weights=r[m], minlength=nb)
    good = cnt > 0
    return (s_r[good] / cnt[good], s_rho[good] / cnt[good])


def l_corr_1e(r, rho_r):
    """First 1/e crossing of the radial ACF (linear interpolation)."""
    e1 = 1.0 / np.e
    below = np.flatnonzero(rho_r < e1)
    if len(below) == 0:
        return np.nan
    k = int(below[0])
    if k == 0:
        return float(r[0])
    frac = (rho_r[k - 1] - e1) / (rho_r[k - 1] - rho_r[k])
    return float(r[k - 1] + frac * (r[k] - r[k - 1]))


def phase_histogram(times, T_sh, nbins):
    """Histogram of times mod T_sh; uniformity = max/min bin count
    (inf when a bin is empty). Returns dict."""
    ph = np.mod(np.asarray(times, np.float64), T_sh) / T_sh
    counts, _ = np.histogram(ph, bins=nbins, range=(0.0, 1.0))
    mn, mx = int(counts.min()), int(counts.max())
    uni = float('inf') if mn == 0 else float(mx) / mn
    return dict(counts=counts, nbins=nbins, uniformity=uni,
                n_empty=int((counts == 0).sum()), n_times=int(len(ph)))


def decision_rule(two_tau_s4, T_sh_measured,
                  threshold=THEORY['decision_threshold'],
                  dt_relaxed=0.5, n_pp_min=THEORY['n_pp_min']):
    """Pre-committed dt_save decision (theory doc S8/A)."""
    out = dict(two_tau_int_s4=float(two_tau_s4), threshold=float(threshold),
               T_sh_measured=float(T_sh_measured))
    if not np.isfinite(two_tau_s4):
        out.update(verdict='INDETERMINATE: 2 tau_int(s=4) unavailable',
                   n_pp_at_relaxed=None, n_pp_ok=None)
        return out
    if two_tau_s4 > threshold:
        n_pp = T_sh_measured / dt_relaxed
        ok = bool(n_pp >= n_pp_min)
        v = (f'RELAX: modulated runs dt_save={dt_relaxed}; re-check '
             f'n_pp = T_sh_measured/{dt_relaxed} = {n_pp:.2f} '
             f'{">= 8: OK" if ok else "< 8: PHASE-BINNING VIOLATION -- FLAG"}')
        out.update(verdict=v, n_pp_at_relaxed=float(n_pp), n_pp_ok=ok)
    else:
        out.update(verdict='HOLD-0.25', n_pp_at_relaxed=None, n_pp_ok=None)
    return out


# ---------------------------------------------------------------------------- #
# Pi_FF directory handling (compute_pi_ff.py conventions, S9 float32 storage)

def load_piff_dir(piff_dir):
    """Collect {scale: dict(path, times, pi_ff_lazy npz handle)} from a
    directory of compute_pi_ff.py outputs (*LES*.npz with keys pi_ff, times,
    _scale). float32 arrays are upcast only on the slices actually used."""
    files = sorted(glob.glob(os.path.join(piff_dir, '*LES*.npz')))
    out = {}
    for fp in files:
        z = np.load(fp)
        if 'pi_ff' not in z.files or '_scale' not in z.files:
            continue
        s = int(np.asarray(z['_scale']).ravel()[0])
        out[s] = dict(path=fp, z=z, times=np.asarray(z['times'],
                                                     dtype=np.float64))
    return out


def piff_probe_series(pi_ff, tw_mask, batch, probes_xy, Lx):
    """Extract Pi_FF(t) at the 5 recorder probe points (physical coords from
    the scalars meta) on the LES grid. Returns (n_t, 5) float64."""
    ny, nx = pi_ff.shape[-2], pi_ff.shape[-1]
    dx = Lx / nx
    dy = Lx / nx                      # isotropic square cells (branch guard)
    cols = []
    for (x, y) in probes_xy:
        iy = int(round(y / dy)) % ny
        ix = int(round(x / dx)) % nx
        cols.append(np.asarray(pi_ff[batch, tw_mask, iy, ix],
                               dtype=np.float64))
    return np.stack(cols, axis=1)


# ---------------------------------------------------------------------------- #
def run_audit(args):
    os.makedirs(args.outdir, exist_ok=True)
    T = THEORY
    b = int(args.batch)

    # ---- dense scalars ----------------------------------------------------- #
    data = _load_scalars(args.scalars)
    meta = data['meta']
    D = float(meta['length'])
    t_all = data['t']
    cl_all = data['Cl_mid'][:, b] if data['Cl_mid'].ndim == 2 else data['Cl_mid']
    win = (t_all >= args.t_min) & np.isfinite(cl_all)
    t = t_all[win]
    dt_sc = float(np.median(np.diff(t)))
    cl = cl_all[win]
    pv = data['probe_v']
    v0 = pv[win, b, 0]
    v1 = pv[win, b, 1]
    U_inlet = data['U_inlet'][win, b] if data['U_inlet'].ndim == 2 \
        else data['U_inlet'][win]
    U_mean = float(np.nanmean(U_inlet))
    T_u = float(t_all[np.isfinite(t_all)][-1] - T['T_wait'])

    # T_sh measured, from shedding_summary (upstream export)
    if args.T_sh is not None:
        T_sh_meas = float(args.T_sh)
    else:
        zs = np.load(args.shedding_summary)
        T_sh_meas = float(zs['T_sh_median'])

    # ---- 1a. dense ACFs ----------------------------------------------------- #
    lag_cl, rho_cl = acf(cl, dt_sc, max_frac=0.25)
    ti_cl = tau_int_first_zero(lag_cl, rho_cl)
    lag_v0, rho_v0 = acf(v0, dt_sc, max_frac=0.25)
    ti_v0 = tau_int_first_zero(lag_v0, rho_v0)

    def _cmp(theory, measured):
        return dict(theory=float(theory), measured=float(measured),
                    ratio=float(measured / theory))

    tau_cmp = {
        'CL': dict(**_cmp(T['tau_E_wake_mid'], ti_cl['tau_int']),
                   theory_range=list(T['tau_E_wake_range']),
                   tau_cross=ti_cl['tau_cross'], status='resolved'),
        'probe_v_1D': dict(**_cmp(T['tau_E_wake_mid'], ti_v0['tau_int']),
                           theory_range=list(T['tau_E_wake_range']),
                           tau_cross=ti_v0['tau_cross'], status='resolved'),
    }

    # ---- 2. convection speed U_c ------------------------------------------- #
    lag_uc, lags_x, cxc = xcorr_peak_lag(v0, v1, dt_sc, max_lag=5.0)
    U_c = D / lag_uc
    uc_cmp = dict(theory=float(T['Uc_factor'] * T['U_mid']),
                  theory_def='0.85*U_mid',
                  measured=float(U_c),
                  ratio=float(U_c / (T['Uc_factor'] * T['U_mid'])),
                  lag=float(lag_uc),
                  vs_0p85_mean_U_inlet=float(U_c / (T['Uc_factor'] * U_mean)))

    # ---- Pi_FF: temporal ACFs at probes + spatial ACF ----------------------- #
    piff = load_piff_dir(args.piff_dir) if args.piff_dir else {}
    probes_xy = meta.get('probes_xy_requested')
    xc, yc = meta.get('obstacle_centroid_xy', (np.nan, np.nan))
    piff_tau = {}
    piff_curves = {}
    lcorr = {}
    patch = {}
    spatial_npz = {}
    dt_save = T['dt_save']
    snap_times = None

    for s in sorted(piff.keys()):
        entry = piff[s]
        times = entry['times']
        tw = times >= args.t_min
        tt = times[tw]
        if snap_times is None:
            snap_times = tt
        dt_save = float(np.median(np.diff(tt)))
        pi_ff = entry['z']['pi_ff']              # float32, (B, T, ny, nx)

        # 1b. temporal ACF at the 5 probe points, dt_save cadence
        if probes_xy is None:
            raise ValueError('scalars meta lacks probes_xy_requested; cannot '
                             'place Pi_FF probes')
        series = piff_probe_series(pi_ff, tw, b, probes_xy, args.Lx)
        taus, statuses, med_curve = [], [], None
        rho_stack = []
        for j in range(series.shape[1]):
            lg, rh = acf(series[:, j], dt_save, max_frac=0.5)
            ti = tau_int_first_zero(lg, rh)
            taus.append(ti['tau_int'])
            statuses.append('resolved' if ti['resolved'] else 'unresolved')
            rho_stack.append(rh)
        nmin = min(len(r) for r in rho_stack)
        med_curve = np.median(np.stack([r[:nmin] for r in rho_stack]), axis=0)
        tau_med = float(np.median(taus))
        unres = statuses.count('unresolved') >= 3
        piff_tau[s] = dict(
            **_cmp(T['tau_E_s'][s], tau_med),
            tau_int_per_probe=[float(x) for x in taus],
            status=(UNRESOLVED_MSG if unres else 'resolved'),
            dt_save=dt_save)
        piff_curves[s] = (np.arange(nmin) * dt_save, med_curve)

        # 3. spatial ACF over the Pi-active wake box (~3D x 0.7 Lx downstream)
        ny, nx = pi_ff.shape[-2], pi_ff.shape[-1]
        dx = args.Lx / nx
        x_lo = xc + 0.5 * D
        x_hi = min(x_lo + 0.7 * args.Lx, args.Lx - dx)
        ix0, ix1 = int(np.ceil(x_lo / dx)), int(np.floor(x_hi / dx))
        iy0 = max(0, int(np.ceil((yc - 1.5 * D) / dx)))
        iy1 = min(ny, int(np.floor((yc + 1.5 * D) / dx)) + 1)
        t_idx = np.flatnonzero(tw)
        stride = max(1, len(t_idx) // args.n_spatial_snaps)
        sel = t_idx[::stride]
        boxes = np.asarray(pi_ff[b, sel, iy0:iy1, ix0:ix1], dtype=np.float64)
        rho2d, ctr = spatial_acf(boxes, dx)
        r_max = 0.5 * min(iy1 - iy0, ix1 - ix0) * dx
        r, rho_r = radial_profile(rho2d, ctr, dx, r_max)
        lc = l_corr_1e(r, rho_r)
        A_box = (iy1 - iy0) * (ix1 - ix0) * dx * dx
        lcorr[s] = dict(measured=float(lc),
                        theory_bracket=[float(T['Delta_s'][s]), float(D)],
                        n_snapshots=int(len(sel)), box_area=float(A_box),
                        box=dict(ix=[ix0, ix1], iy=[iy0, iy1], dx=float(dx)))
        # S4 patch counts, measured l_corr replacing the [Delta_s, D] bracket
        patch[s] = dict(
            patches_per_snapshot_measured=float(A_box / lc ** 2),
            theory_bracket_S4=[float(T['A_wake'] / D ** 2),
                               float(T['A_wake'] / T['Delta_s'][4] ** 2)],
            A_wake_theory=float(T['A_wake']), A_box_used=float(A_box))
        spatial_npz[f'r_s{s}'] = r
        spatial_npz[f'rho_r_s{s}'] = rho_r
        if s == args.spatial_map_scale or f'rho2d_s{args.spatial_map_scale}' \
                not in spatial_npz:
            spatial_npz[f'rho2d_s{s}'] = rho2d.astype(np.float64)
        del pi_ff

    # ---- 4. phase-coverage histogram ---------------------------------------- #
    phase = {}
    if snap_times is not None:
        for nb in (8, 24):
            h = phase_histogram(snap_times, T_sh_meas, nb)
            phase[f'{nb}_bins'] = dict(uniformity=h['uniformity'],
                                       n_empty=h['n_empty'],
                                       counts=[int(c) for c in h['counts']],
                                       n_times=h['n_times'])
        phase['T_sh_used'] = float(T_sh_meas)
        phase['validates'] = ('S5 commensurability fix: dt_save=0.27 on '
                              'MOD-const should give uniformity ~ O(1), no '
                              'empty bins')

    # ---- 5. recomputed N_eff table (S4 format) ------------------------------ #
    def _neff_row(name, two_tau, theory_two_tau, theory_neff):
        if not np.isfinite(two_tau):
            return dict(scale=name, two_tau_int=None, regime='unavailable',
                        N_eff=None)
        if two_tau <= dt_save:
            regime, neff = 'independent', T_u / dt_save
        else:
            regime = f'oversampled {two_tau / dt_save:.1f}x'
            neff = T_u / two_tau
        return dict(scale=name, two_tau_int=float(two_tau),
                    regime=regime, N_eff=float(neff),
                    theory=dict(two_tau_int=theory_two_tau,
                                N_eff=theory_neff))

    neff_table = [
        _neff_row('wake (l = D, from C_L)', 2.0 * ti_cl['tau_int'],
                  T['two_tau_wake'], T['Neff_wake'])]
    for s in (8, 4, 2):
        tt2 = 2.0 * piff_tau[s]['measured'] if s in piff_tau else np.nan
        neff_table.append(_neff_row(f's = {s}', tt2,
                                    2.0 * T['tau_E_s'][s], T['Neff_s']))
    neff_note = (f'T_u = t_end - T_wait = {T_u:.1f} (actual series); '
                 f'dt_save = {dt_save}; N_eff = T_u/(2 tau_int) if '
                 'dt_save << 2 tau_int else T_u/dt_save (S4)')

    # ---- 6. decision rule ---------------------------------------------------- #
    two_tau_s4 = 2.0 * piff_tau[4]['measured'] if 4 in piff_tau else np.nan
    decision = decision_rule(two_tau_s4, T_sh_meas)
    if 4 in piff_tau and piff_tau[4]['status'] == UNRESOLVED_MSG:
        decision['note'] = ('s=4 ACF unresolved at dt_save (' + UNRESOLVED_MSG
                            + '); 2 tau_int(s=4) <= dt_save << 0.5 '
                            '=> HOLD is the theory-consistent branch')

    # ---- assemble + write ----------------------------------------------------- #
    summary = dict(
        inputs=dict(scalars=os.path.abspath(args.scalars),
                    piff_dir=(os.path.abspath(args.piff_dir)
                              if args.piff_dir else None),
                    shedding_summary=(os.path.abspath(args.shedding_summary)
                                      if args.shedding_summary else None),
                    batch=b, t_min=float(args.t_min), Lx=float(args.Lx),
                    dt_scalars=dt_sc, dt_save=float(dt_save),
                    T_u=float(T_u), T_sh_measured=float(T_sh_meas),
                    scales_found=sorted(piff.keys())),
        tau_int=dict(dense=tau_cmp,
                     piff={f's{s}': piff_tau[s] for s in sorted(piff_tau)}),
        U_c=uc_cmp,
        l_corr={f's{s}': lcorr[s] for s in sorted(lcorr)},
        patch_counts={f's{s}': patch[s] for s in sorted(patch)},
        phase_coverage=phase,
        N_eff_table=neff_table,
        N_eff_note=neff_note,
        decision=decision,
    )
    with open(os.path.join(args.outdir, 'audit_A_summary.yaml'), 'w') as fh:
        yaml.safe_dump(json.loads(json.dumps(summary, default=float)), fh,
                       sort_keys=False)

    npz = dict(lag_CL=lag_cl, rho_CL=rho_cl, lag_probe_v=lag_v0,
               rho_probe_v=rho_v0, xcorr_lags=lags_x, xcorr=cxc,
               tau_int_CL=np.float64(ti_cl['tau_int']),
               tau_int_probe_v=np.float64(ti_v0['tau_int']),
               U_c=np.float64(U_c), lag_Uc=np.float64(lag_uc),
               T_sh_measured=np.float64(T_sh_meas),
               meta=json.dumps(summary, default=float))
    for s, (lg, rh) in piff_curves.items():
        npz[f'piff_acf_lags_s{s}'] = lg
        npz[f'piff_acf_rho_s{s}'] = rh
    npz.update(spatial_npz)
    if snap_times is not None:
        npz['snapshot_times'] = snap_times
        for nb in (8, 24):
            npz[f'phase_counts_{nb}'] = np.array(
                summary['phase_coverage'][f'{nb}_bins']['counts'])
    np.savez(os.path.join(args.outdir, 'audit_A_summary.npz'), **npz)

    _figures(args.outdir, lag_cl, rho_cl, ti_cl, lag_v0, rho_v0, ti_v0,
             piff_curves, piff_tau, lags_x, cxc, lag_uc, spatial_npz, lcorr,
             summary, T_sh_meas)
    print(yaml.safe_dump(json.loads(json.dumps(
        dict(decision=decision, N_eff_table=neff_table), default=float)),
        sort_keys=False))
    return summary


def _figures(outdir, lag_cl, rho_cl, ti_cl, lag_v0, rho_v0, ti_v0,
             piff_curves, piff_tau, lags_x, cxc, lag_uc, spatial_npz, lcorr,
             summary, T_sh_meas):
    # temporal ACFs
    fig, axs = plt.subplots(1, 2, figsize=(10.0, 4.0))
    m = lag_cl <= 5.0
    axs[0].plot(lag_cl[m], rho_cl[m], lw=1.2, label=r'$C_L$')
    m = lag_v0 <= 5.0
    axs[0].plot(lag_v0[m], rho_v0[m], lw=1.2, label=r'$v$ at $x_c+1D$')
    axs[0].axvline(ti_cl['tau_cross'], color='C0', ls=':', lw=0.8)
    axs[0].axvline(ti_v0['tau_cross'], color='C1', ls=':', lw=0.8)
    axs[0].axhline(0.0, color='k', lw=0.6)
    axs[0].set_xlabel(r'$\tau$  [t.u.]')
    axs[0].set_ylabel(r'$\hat{\rho}(\tau)$')
    axs[0].set_title(r'dense-scalar ACFs; $\tau_{int}$ to first zero')
    axs[0].legend(fontsize=8)
    for s, (lg, rh) in sorted(piff_curves.items()):
        mm = lg <= 2.5
        axs[1].plot(lg[mm], rh[mm], marker='o', ms=3, lw=1.0,
                    label=(f's={s} '
                           f"({'unres.' if piff_tau[s]['status'] != 'resolved' else 'res.'})"))
    axs[1].axhline(0.0, color='k', lw=0.6)
    axs[1].set_xlabel(r'$\tau$  [t.u.]')
    axs[1].set_title(r'$\Pi_{FF}$ probe ACFs at $dt_{save}$ cadence')
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig_audit_acf.png'), dpi=150)
    plt.close(fig)

    # U_c cross-correlation
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(lags_x, cxc / np.max(np.abs(cxc)), lw=1.0)
    ax.axvline(lag_uc, color='r', ls='--', lw=1.0,
               label=f'peak lag = {lag_uc:.3f}')
    ax.set_xlabel('lag  [t.u.]')
    ax.set_ylabel(r'$C_{v_0 v_1}$ (normalized)')
    ax.set_title(r'probe $v$: $x_c+1D$ vs $x_c+2D$ (parabolic sub-sample peak)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig_audit_uc.png'), dpi=150)
    plt.close(fig)

    # spatial ACF
    if lcorr:
        fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2))
        map_key = [k for k in spatial_npz if k.startswith('rho2d_s')]
        if map_key:
            k = ('rho2d_s4' if 'rho2d_s4' in map_key else sorted(map_key)[0])
            s_map = k.split('_s')[-1]
            rho2d = spatial_npz[k]
            dxm = lcorr[int(s_map)]['box']['dx']
            ny, nx = rho2d.shape
            ext = [-0.5 * nx * dxm, 0.5 * nx * dxm,
                   -0.5 * ny * dxm, 0.5 * ny * dxm]
            im = axs[0].imshow(rho2d, cmap='seismic', vmin=-1, vmax=1,
                               origin='lower', extent=ext, aspect='equal')
            w = 12.0 * lcorr[int(s_map)]['measured']
            axs[0].set_xlim(-w, w)
            axs[0].set_ylim(-w, w)
            axs[0].set_title(rf'2D ACF of $\Pi_{{FF}}$, s={s_map}')
            axs[0].set_xlabel(r'$\delta x$')
            axs[0].set_ylabel(r'$\delta y$')
            fig.colorbar(im, ax=axs[0], shrink=0.85)
        for s in sorted(lcorr):
            r = spatial_npz[f'r_s{s}']
            rr = spatial_npz[f'rho_r_s{s}']
            mm = r <= 10.0 * max(lcorr[s]['measured'], 1e-9)
            axs[1].plot(r[mm], rr[mm], lw=1.2,
                        label=rf's={s}: $l_{{corr}}$={lcorr[s]["measured"]:.4f}')
            axs[1].axvline(lcorr[s]['measured'], ls=':', lw=0.8)
        axs[1].axhline(1.0 / np.e, color='k', ls='--', lw=0.8, label='1/e')
        axs[1].set_xlabel('r')
        axs[1].set_ylabel(r'$\hat{\rho}(r)$ (radial avg)')
        axs[1].set_title('radial spatial ACF, wake box')
        axs[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, 'fig_audit_spatial.png'), dpi=150)
        plt.close(fig)

    # phase coverage
    pc = summary.get('phase_coverage', {})
    if '8_bins' in pc:
        fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.6))
        for ax, nb in zip(axs, (8, 24)):
            cnts = pc[f'{nb}_bins']['counts']
            ax.bar(np.arange(nb) + 0.5, cnts, width=0.9)
            uni = pc[f'{nb}_bins']['uniformity']
            ax.set_title(f'{nb} bins, max/min = '
                         f'{"inf" if not np.isfinite(uni) else f"{uni:.2f}"}')
            ax.set_xlabel(rf'snapshot phase bin (t mod $T_{{sh}}$='
                          rf'{T_sh_meas:.3f})')
            ax.set_ylabel('count')
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, 'fig_audit_phase.png'), dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------- #
# Selftest -- synthetic, analytically known answers (Amendment 02 S3: qsub only)

def selftest():
    rng = np.random.default_rng(0)
    checks = []

    def add(name, target, got, tol_rel):
        ok = (np.isfinite(got)
              and abs(got - target) <= tol_rel * max(abs(target), 1e-30))
        checks.append((name, target, got, tol_rel, bool(ok)))

    def add_bool(name, ok):
        checks.append((name, True, ok, 0.0, bool(ok)))

    # 1. OU process: ACF = exp(-tau/theta) => tau_int = theta analytically
    theta, dt, n = 0.5, 0.01, 400_000
    a = np.exp(-dt / theta)
    sig = np.sqrt(1.0 - a * a)
    from scipy.signal import lfilter
    x = lfilter([sig], [1.0, -a], rng.standard_normal(n))
    x = x[int(20 * theta / dt):]                    # discard warm-up
    lg, rh = acf(x, dt, max_frac=0.25)
    ti = tau_int_first_zero(lg, rh)
    add('OU tau_int = theta', theta, ti['tau_int'], 0.15)
    add_bool('OU ACF resolved flag', ti['resolved'])
    # unresolved branch: same OU subsampled at 10*theta -> rho(dt) ~ e^-10 ~ 0
    step = int(10 * theta / dt)
    lg2, rh2 = acf(x[::step], 10 * theta, max_frac=0.5)
    ti2 = tau_int_first_zero(lg2, rh2)
    add_bool('OU subsampled flagged UNRESOLVED', not ti2['resolved'])

    # 2. U_c lag: fractionally shifted band-limited pair, parabolic sub-sample
    n2, dt2 = 100_000, 2.5e-3
    lag_true = 0.09338                              # 37.35 samples
    w = rng.standard_normal(n2)
    k = np.arange(-100, 101)
    g = np.exp(-0.5 * (k / 20.0) ** 2)
    s0 = np.convolve(w, g / g.sum(), mode='same')
    f = np.fft.rfftfreq(n2, d=dt2)
    s1 = np.fft.irfft(np.fft.rfft(s0)
                      * np.exp(-2j * np.pi * f * lag_true), n=n2)
    lag_est, _, _ = xcorr_peak_lag(s0, s1, dt2, max_lag=1.0)
    add('U_c peak lag (sub-sample)', lag_true, lag_est, 0.005)
    D = THEORY['D']
    add('U_c = D/lag consistency', D / lag_true, D / lag_est, 0.005)

    # 3. 2D Gaussian-correlated field: kernel exp(-r^2/(2a^2)) => field ACF
    #    exp(-r^2/(4a^2)) => 1/e crossing at l_corr = 2a exactly
    N, a2 = 512, 6.0
    kx = np.fft.fftfreq(N) * N
    KX, KY = np.meshgrid(kx, kx)
    # FFT of the Gaussian kernel (periodic, unit spacing)
    Gh = np.exp(-2.0 * (np.pi * a2 / N) ** 2 * (KX ** 2 + KY ** 2))
    flds = []
    for _ in range(4):
        wn = rng.standard_normal((N, N))
        flds.append(np.real(np.fft.ifft2(np.fft.fft2(wn) * Gh)))
    rho2d, ctr = spatial_acf(np.stack(flds), dx=1.0)
    r, rho_r = radial_profile(rho2d, ctr, dx=1.0, r_max=N / 2.0)
    lc = l_corr_1e(r, rho_r)
    add('2D Gaussian field l_corr = 2a', 2.0 * a2, lc, 0.10)

    # 4. phase-coverage histogram: uniform vs commensurate-clustered times
    T_sh = 2.992
    t_uni = 30.0 + 0.263 * np.arange(1000)          # incommensurate cadence
    h8 = phase_histogram(t_uni, T_sh, 8)
    add_bool('phase hist uniform: max/min < 2 (8 bins)',
             np.isfinite(h8['uniformity']) and h8['uniformity'] < 2.0)
    # S5 commensurate trap; +T_sh/48 keeps the 12 frozen phases mid-bin
    # (they would otherwise sit exactly ON 24-bin edges, a float coin-flip)
    t_clu = 30.0 + T_sh / 48.0 + (T_sh / 12.0) * np.arange(1000)
    h24 = phase_histogram(t_clu, T_sh, 24)
    add_bool('phase hist clustered: >= 8 empty of 24 bins',
             h24['n_empty'] >= 8)
    add_bool('phase hist clustered: uniformity -> inf',
             not np.isfinite(h24['uniformity']))

    # 5. decision rule, both branches + the n_pp re-check
    d = decision_rule(0.6, 8.0)
    add_bool('decision RELAX branch', d['verdict'].startswith('RELAX')
             and d['n_pp_ok'] is True)
    d = decision_rule(0.1, 2.992)
    add_bool('decision HOLD branch', d['verdict'] == 'HOLD-0.25')
    d = decision_rule(0.6, 2.992)
    add_bool('decision RELAX + n_pp violation flagged',
             d['verdict'].startswith('RELAX') and d['n_pp_ok'] is False
             and 'FLAG' in d['verdict'])

    # ---- PASS table ---------------------------------------------------------- #
    print('\n===== audit_decorrelation selftest =====')
    print(f"{'check':<46}{'target':>12}{'got':>14}{'tol':>8}  verdict")
    all_ok = True
    for name, tgt, got, tol, ok in checks:
        all_ok &= ok
        gs = f'{got:.6g}' if isinstance(got, float) else str(got)
        print(f'{name:<46}{tgt!s:>12.12}{gs:>14.14}{tol!s:>8}  '
              f"{'PASS' if ok else 'FAIL'}")
    print(f"===== overall: {'PASS' if all_ok else 'FAIL'} "
          f'({sum(c[-1] for c in checks)}/{len(checks)}) =====')
    return 0 if all_ok else 1


# ---------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--scalars', help='path to scalars.npz')
    p.add_argument('--shedding-summary', default=None,
                   help='shedding_summary.npz from shedding_tracker.py '
                        '(source of measured T_sh)')
    p.add_argument('--piff-dir', default=None,
                   help='directory of Pi_FF *LES*.npz files (scales 2,4,8)')
    p.add_argument('--outdir', default=None,
                   help='output dir (default: alongside scalars.npz)')
    p.add_argument('--batch', type=int, default=0)
    p.add_argument('--t-min', type=float, default=30.0,
                   help='usable-window start (T_wait, theory doc S1)')
    p.add_argument('--Lx', type=float, default=8.0 * np.pi,
                   help='domain size (FPC: 8*pi)')
    p.add_argument('--T-sh', type=float, default=None,
                   help='override measured T_sh (else read from '
                        '--shedding-summary)')
    p.add_argument('--n-spatial-snaps', type=int, default=30,
                   help='max snapshots used for the spatial ACF')
    p.add_argument('--spatial-map-scale', type=int, default=4)
    p.add_argument('--selftest', action='store_true')
    args = p.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if not args.scalars:
        p.error('--scalars required (or --selftest)')
    if args.T_sh is None and not args.shedding_summary:
        p.error('need --shedding-summary (or --T-sh) for the measured T_sh')
    if args.outdir is None:
        args.outdir = os.path.dirname(os.path.abspath(args.scalars))
    run_audit(args)


if __name__ == '__main__':
    main()
