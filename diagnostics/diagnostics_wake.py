#!/usr/bin/env python
r"""diagnostics_wake.py -- per-run wake diagnostic package (SGS-closure branch).

Implements task A of the diagnostics authoring order, per the binding theory
doc docs/briefs/Supervisor_simulation.md (SECTION references below) and
AMENDMENT_01/_02:

  1. Phase-conditioned averages of omega and Pi_FF (per scale s in {2,4,8})
     in EXACTLY 8 phase bins of 45 deg (theory doc SEC 3.2 -- the bin count is
     PRE-COMMITTED; do not change). Phase per snapshot = instantaneous shedding
     phase phi_t interpolated to snapshot times from shedding_summary.npz
     (sibling-authored shedding_tracker.py output; documented interface:
     keys phi_t, T_sh_t, f_sh_t, t). If the file is absent the module DEGRADES
     GRACEFULLY: uniform placeholder phases phi_j = 2*pi*j/n are assigned, a
     WARNING is printed, and the yaml records phase_source =
     'uniform-fallback (NOT physical)'.
  2. Mean and rms wake maps (omega, u, v, Pi_FF per scale); recirculation-zone
     extent from the mean-u zero crossing on the centerline behind the body;
     probe statistics table (mean/rms/min/max of every recorder scalar,
     rms = std about the mean).
  3. Phase-wheel coverage plot (snapshot count per 45-deg bin) + histogram of
     snapshot times mod T_sh -- the visual SEC 5 commensurability verification
     for MOD-const runs. A pre-committed flag fires when the emptiest bin holds
     < 50% of the uniform expectation.
  4. Outputs: figures (cmap='seismic', aspect-preserving 'equal', mathtext uses
     \frac only), wake_summary.yaml, wake_summary.npz (all conditional means
     as arrays, float32 at write per theory doc SEC 9).

SNAPSHOT FORMAT CODED AGAINST (verified on
qg-simple-package-stable/src/qg/outputs/flow_past_cylinder_re1000, written by
qg/_output/dataset.py::build_dataset_npz):
    <run-dir>/{name}_FR.npz : omega_FR (B,T,Ny,Nx) float32 physical vorticity,
                              times (T,) float64 = arange(T)*save_rate*dt,
                              chi_obs (1,Ny,Nx) float32 (smoothed, as applied),
                              chi_sponge_ramp (1,Ny,Nx) float32 [optional]
    <run-dir>/{name}_FR_params.yaml : raw config snapshot (grid.Lx, grid.Nx,
                              pde.nu, pde.penalty, time.dt, mask.r,
                              bc.inlet_velocity, ...)
    <run-dir>/scalars.npz   : per qg/_output/scalars.py -- t, step, and (n,B)
                              U_inlet, Re_inlet, Fx, Fy, Cd_inst, Cl_inst,
                              Cd_mid, Cl_mid, U_cyl, Re_cyl, E, Z,
                              probe_u/probe_v (n,B,n_probes), meta (json str;
                              contains eta = penalty*dt as applied, the
                              obstacle centroid, and the probe coordinates).
    Pi_FF files (compute_pi_ff.py conventions): {name}_LES*.npz with
                              omega_bar / pi_ff (B,T,ny,nx) float32 on the
                              coarse grid, times, chi_obs_bar, _scale, _alpha.
                              Scale is read from the '_scale' key (fallback:
                              a _s<scale> filename token).

VELOCITY RECONSTRUCTION: solver convention (qg/solver/opt/basis.py::puv):
    ph = -qh/k^2 (k=0 mode zeroed), uh = -i*ky*ph, vh = +i*kx*ph,
    wavenumbers k = 2*pi*m/L (qg/solver/grid/cartesian.py).
omega does NOT carry the mean inlet flow (uniform flow is irrotational), so
U_inlet(t_snap) -- interpolated from scalars.npz, falling back to the config
bc.inlet_velocity -- is added to u per snapshot before accumulating moments.

MEMORY: snapshot stacks are STREAMED one 2-D slice at a time straight out of
the (possibly deflated) npz zip member -- a 4096^2 x 480-snapshot stack never
materialises. All accumulation in float64 (float32 inputs upcast at load,
theory doc SEC 9); npz outputs cast to float32 at write.

EXECUTION: batch only (AMENDMENT_02 SEC 3). Submit via
scripts/sge/wake_diag_job.sh. --selftest is synthetic, analytically known,
zero data deps, and also runs via qsub.

Usage:
    python diagnostics_wake.py --run-dir <dir> [--name DNS] [--pi-dir <dir>]
        [--shedding <path>] [--out-dir <dir>] [--t-min 30] [--t-max T]
        [--scales 2 4 8] [--mod-const] [--selftest]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import tempfile
import zipfile

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N_BINS = 8                                   # SEC 3.2 -- pre-committed
BIN_EDGES_DEG = np.arange(N_BINS + 1) * 45.0
BIN_EDGES = np.deg2rad(BIN_EDGES_DEG)
COVERAGE_MIN_FRACTION = 0.5                  # emptiest-bin flag threshold

_WARNINGS: list[str] = []


def warn(msg: str) -> None:
    _WARNINGS.append(msg)
    print(f"WARNING: {msg}", flush=True)


# --------------------------------------------------------------------------- #
# npz streaming (slice-at-a-time out of the zip member; no full-stack loads)
# --------------------------------------------------------------------------- #
def _read_exact(f, n):
    chunks, got = [], 0
    while got < n:
        b = f.read(n - got)
        if not b:
            raise EOFError(f"npz member truncated: wanted {n}, got {got}")
        chunks.append(b)
        got += len(b)
    return b''.join(chunks)


def _open_member(path, key):
    """Open one .npy member of an npz for sequential reading.

    Returns (zipfile, fileobj, shape, dtype). Caller closes the zipfile."""
    zf = zipfile.ZipFile(path)
    f = zf.open(key + '.npy')
    version = np.lib.format.read_magic(f)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
    else:
        raise ValueError(f"unsupported .npy format version {version} in {path}")
    if fortran:
        raise ValueError(f"{key} in {path} is Fortran-ordered; unsupported")
    return zf, f, shape, dtype


def npz_field_shape(path, key):
    zf, f, shape, dtype = _open_member(path, key)
    f.close()
    zf.close()
    return shape, dtype


def stream_snapshots(path, key, batch_index=0):
    """Yield (t_index, 2-D float64 slice) for each snapshot of key.

    Accepts (B,T,Ny,Nx) or (T,Ny,Nx) stacks; B>1 selects batch_index with a
    warning (cylinder runs are single-member)."""
    zf, f, shape, dtype = _open_member(path, key)
    try:
        if len(shape) == 4:
            B, T, Ny, Nx = shape
        elif len(shape) == 3:
            B, (T, Ny, Nx) = 1, shape
        else:
            raise ValueError(f"{key} in {path}: unexpected shape {shape}")
        if B > 1 and batch_index == 0:
            warn(f"{os.path.basename(path)}:{key} has B={B}; using batch 0")
        nbytes = Ny * Nx * dtype.itemsize
        skip = batch_index * T * nbytes
        while skip > 0:
            got = f.read(min(skip, 1 << 24))
            if not got:
                raise EOFError("batch skip ran past end of member")
            skip -= len(got)
        for i in range(T):
            buf = _read_exact(f, nbytes)
            yield i, np.frombuffer(buf, dtype=dtype).reshape(Ny, Nx).astype(np.float64)
    finally:
        f.close()
        zf.close()


# --------------------------------------------------------------------------- #
# spectral velocity reconstruction (solver convention, see module docstring)
# --------------------------------------------------------------------------- #
def uv_from_omega(omega, Lx, Ly):
    """u, v from physical vorticity via psi = inv_laplacian(omega).

    Matches qg/solver/opt/basis.py::puv: ph = -qh/k^2, uh = -i ky ph,
    vh = +i kx ph. Mean (k=0) mode of psi is zeroed -- the mean inlet flow is
    NOT included and must be added by the caller."""
    omega = np.asarray(omega, dtype=np.float64)
    Ny, Nx = omega.shape
    qh = np.fft.rfft2(omega)
    ky = np.fft.fftfreq(Ny, Ly / (Ny * 2.0 * np.pi)).reshape(Ny, 1)
    kx = np.fft.rfftfreq(Nx, Lx / (Nx * 2.0 * np.pi)).reshape(1, -1)
    k2 = kx ** 2 + ky ** 2
    inv_lap = np.zeros_like(k2)
    nz = k2 > 0
    inv_lap[nz] = -1.0 / k2[nz]
    ph = inv_lap * qh
    uh = -1j * ky * ph
    vh = 1j * kx * ph
    u = np.fft.irfft2(uh, s=omega.shape)
    v = np.fft.irfft2(vh, s=omega.shape)
    return u, v


# --------------------------------------------------------------------------- #
# shedding phase
# --------------------------------------------------------------------------- #
def load_shedding(path):
    """Load shedding_summary.npz (documented sibling interface: phi_t, T_sh_t,
    f_sh_t, t). Returns dict of float64 arrays or None."""
    if not path or not os.path.exists(path):
        return None
    z = np.load(path)
    out = {}
    # shedding_tracker.py exports 'phi' (and f_sh_t/T_sh_t/t); accept both the
    # documented 'phi_t' and the actual 'phi' (interface drift found 2026-07-09)
    aliases = {'phi_t': ('phi_t', 'phi'), 'T_sh_t': ('T_sh_t',),
               'f_sh_t': ('f_sh_t',), 't': ('t',)}
    for k, names in aliases.items():
        for name in names:
            if name in z.files:
                out[k] = np.asarray(z[name], dtype=np.float64).ravel()
                break
    if 'phi_t' not in out or 't' not in out:
        warn(f"{path}: missing phi_t/phi/t keys; treating as absent")
        return None
    order = np.argsort(out['t'])
    for k in out:
        out[k] = out[k][order] if out[k].shape == out['t'].shape else out[k]
    return out


def snapshot_phases(times, shed):
    """Phase in [0, 2*pi) per snapshot time.

    Interpolates the UNWRAPPED phi_t (np.unwrap handles a wrapped series;
    a no-op on an already-unwrapped one), then re-wraps. Falls back to
    uniform placeholder phases with a WARNING when shed is None."""
    times = np.asarray(times, dtype=np.float64)
    if shed is None:
        warn("shedding_summary.npz absent -- UNIFORM-PHASE FALLBACK: phase "
             "bins are placeholders, NOT physical shedding phases")
        n = len(times)
        return np.mod(2.0 * np.pi * np.arange(n) / max(n, 1), 2.0 * np.pi), \
            'uniform-fallback (NOT physical)'
    phi_u = np.unwrap(shed['phi_t'])
    n_out = int(np.sum((times < shed['t'][0]) | (times > shed['t'][-1])))
    if n_out:
        warn(f"{n_out} snapshot time(s) outside the shedding series range "
             f"[{shed['t'][0]:.3f}, {shed['t'][-1]:.3f}]; np.interp clamps")
    phi = np.interp(times, shed['t'], phi_u)
    return np.mod(phi, 2.0 * np.pi), 'shedding_summary'


def bin_assign(phi):
    """45-deg bin index in 0..7 (SEC 3.2)."""
    idx = np.floor(np.mod(phi, 2.0 * np.pi) / (2.0 * np.pi / N_BINS)).astype(int)
    return np.clip(idx, 0, N_BINS - 1)


def coverage_metrics(bin_counts):
    """SEC 5 commensurability check on the per-bin snapshot counts."""
    bin_counts = np.asarray(bin_counts, dtype=np.float64)
    n = bin_counts.sum()
    expected = n / N_BINS if n else 0.0
    min_frac = float(bin_counts.min() / expected) if expected > 0 else 0.0
    flag = bool(expected > 0 and min_frac < COVERAGE_MIN_FRACTION)
    return {
        'n_snapshots': int(n),
        'bin_counts': [int(c) for c in bin_counts],
        'expected_per_bin': float(expected),
        'min_bin_fraction_of_uniform': min_frac,
        'commensurability_flag': flag,
        'flag_rule': f'min bin count < {COVERAGE_MIN_FRACTION} * (n/{N_BINS})',
    }


# --------------------------------------------------------------------------- #
# streaming accumulation of moments + phase-conditional means
# --------------------------------------------------------------------------- #
def accumulate(stream, keep_mask, bin_idx, do_uv=False, Lx=None, Ly=None,
               U_add=None):
    """One pass over a snapshot stream.

    stream    : iterator of (t_index, 2-D float64)
    keep_mask : bool (T,) usable-window selector on t_index
    bin_idx   : int (T,) phase-bin index per snapshot (only kept ones used)
    do_uv     : also reconstruct u,v per snapshot (u gets U_add[t_index])
    Returns dict of float64 arrays: <f>_mean, <f>_rms per field, plus
    phase_mean (N_BINS,Ny,Nx) and bin counts for the primary field."""
    sums = None
    for i, sl in stream:
        if not keep_mask[i]:
            continue
        if sums is None:
            shp = sl.shape
            sums = {
                'n': 0,
                'f_sum': np.zeros(shp), 'f_sq': np.zeros(shp),
                'bin_sum': np.zeros((N_BINS,) + shp),
                'bin_n': np.zeros(N_BINS, dtype=np.int64),
            }
            if do_uv:
                for k in ('u_sum', 'u_sq', 'v_sum', 'v_sq'):
                    sums[k] = np.zeros(shp)
        sums['n'] += 1
        sums['f_sum'] += sl
        sums['f_sq'] += sl * sl
        b = bin_idx[i]
        sums['bin_sum'][b] += sl
        sums['bin_n'][b] += 1
        if do_uv:
            u, v = uv_from_omega(sl, Lx, Ly)
            u = u + (U_add[i] if U_add is not None else 0.0)
            sums['u_sum'] += u
            sums['u_sq'] += u * u
            sums['v_sum'] += v
            sums['v_sq'] += v * v
    if sums is None or sums['n'] == 0:
        return None
    n = sums['n']
    out = {
        'n': n,
        'mean': sums['f_sum'] / n,
        'rms': np.sqrt(np.maximum(sums['f_sq'] / n - (sums['f_sum'] / n) ** 2, 0.0)),
        'bin_n': sums['bin_n'],
        'phase_mean': np.where(sums['bin_n'][:, None, None] > 0,
                               sums['bin_sum'] / np.maximum(sums['bin_n'], 1)[:, None, None],
                               np.nan),
    }
    if do_uv:
        for tag in ('u', 'v'):
            m = sums[f'{tag}_sum'] / n
            out[f'{tag}_mean'] = m
            out[f'{tag}_rms'] = np.sqrt(np.maximum(sums[f'{tag}_sq'] / n - m * m, 0.0))
    return out


# --------------------------------------------------------------------------- #
# geometry + recirculation
# --------------------------------------------------------------------------- #
def chi_centroid(chi, dx, dy):
    """chi-weighted centroid in index*spacing coordinates (matches
    scalars.py::_setup_geometry)."""
    c2 = chi[0] if chi.ndim == 3 else chi
    c2 = np.asarray(c2, dtype=np.float64)
    tot = c2.sum()
    ys = np.arange(c2.shape[0]) * dy
    xs = np.arange(c2.shape[1]) * dx
    xc = float((c2.sum(axis=0) * xs).sum() / tot)
    yc = float((c2.sum(axis=1) * ys).sum() / tot)
    return xc, yc


def recirculation_extent(u_mean, dx, dy, xc, yc, R, Lx):
    """Mean-u zero crossing on the centerline behind the body.

    Scans x in (xc+R, 0.92*Lx]; L_r is measured from the REAR EDGE xc+R.
    Returns dict; found=False when the centerline mean u never goes negative
    behind the body."""
    Ny, Nx = u_mean.shape
    iy = int(round(yc / dy)) % Ny
    row = u_mean[iy]
    x = np.arange(Nx) * dx
    sel = (x > xc + R) & (x <= 0.92 * Lx)
    idx = np.where(sel)[0]
    out = {'centerline_iy': iy, 'rear_edge_x': float(xc + R)}
    neg = idx[row[idx] < 0.0]
    if len(neg) == 0:
        out.update(found=False, L_r=0.0, L_r_over_D=0.0, x_zero=None,
                   note='mean centerline u never negative behind body')
        return out
    i_last = int(neg[-1])
    if i_last + 1 >= Nx or row[i_last + 1] <= 0:
        out.update(found=False, L_r=None, L_r_over_D=None, x_zero=None,
                   note='negative-u region extends to scan limit; no crossing')
        return out
    # linear interp of the zero between i_last and i_last+1
    x0 = x[i_last] + dx * (0.0 - row[i_last]) / (row[i_last + 1] - row[i_last])
    L_r = x0 - (xc + R)
    out.update(found=True, x_zero=float(x0), L_r=float(L_r),
               L_r_over_D=float(L_r / (2.0 * R)))
    return out


# --------------------------------------------------------------------------- #
# scalars.npz helpers
# --------------------------------------------------------------------------- #
def load_scalars(path):
    if not path or not os.path.exists(path):
        return None
    z = np.load(path)
    out = {k: np.asarray(z[k], dtype=np.float64) if z[k].dtype != np.int64
           else np.asarray(z[k]) for k in z.files if k != 'meta'}
    out['meta'] = json.loads(str(z['meta'])) if 'meta' in z.files else {}
    return out


def probe_statistics(scal, t_min, t_max):
    """mean / rms (= std about mean) / min / max of every recorder scalar over
    the usable window; per-probe rows for probe_u / probe_v."""
    m = (scal['t'] >= t_min) & (scal['t'] <= t_max)
    stats = {}

    def _row(a):
        a = a[np.isfinite(a)]
        if a.size == 0:
            return {'mean': None, 'rms': None, 'min': None, 'max': None, 'n': 0}
        return {'mean': float(a.mean()), 'rms': float(a.std()),
                'min': float(a.min()), 'max': float(a.max()), 'n': int(a.size)}

    for k, arr in scal.items():
        if k in ('t', 'step', 'meta'):
            continue
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 2:                     # (n, B)
            stats[k] = _row(arr[m, 0])
        elif arr.ndim == 3:                   # (n, B, n_probes)
            for p in range(arr.shape[2]):
                stats[f'{k}[{p}]'] = _row(arr[m, 0, p])
        elif arr.ndim == 1:
            stats[k] = _row(arr[m])
    return stats


# --------------------------------------------------------------------------- #
# Pi_FF file discovery
# --------------------------------------------------------------------------- #
def find_pi_files(pi_dir, name):
    """Map scale -> LES npz path. Scale from the '_scale' key (compute_pi_ff
    convention), falling back to an _s<scale> filename token."""
    out = {}
    for p in sorted(glob.glob(os.path.join(pi_dir, f'{name}*LES*.npz'))):
        try:
            z = np.load(p)
            if 'pi_ff' not in z.files:
                continue
            if '_scale' in z.files:
                s = int(np.asarray(z['_scale']).ravel()[0])
            else:
                mt = re.search(r'_s(\d+)', os.path.basename(p))
                if not mt:
                    warn(f"{p}: no _scale key or _s<N> token; skipped")
                    continue
                s = int(mt.group(1))
            z.close()
        except (OSError, zipfile.BadZipFile) as e:
            warn(f"{p}: unreadable ({e}); skipped")
            continue
        if s in out:
            warn(f"duplicate Pi_FF file for scale {s}: keeping {out[s]}, "
                 f"ignoring {p}")
        else:
            out[s] = p
    return out


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _sym_clim(*fields, pct=99.5):
    vals = np.concatenate([np.abs(f[np.isfinite(f)]).ravel() for f in fields])
    v = float(np.percentile(vals, pct)) if vals.size else 1.0
    return v if v > 0 else 1.0


def _trace_panel(ax, scal, t_min, t_max, title=True):
    """AMENDMENT_01 SEC E.5: every multi-panel figure carries the
    Re_inlet(t) / Re_cyl(t) forcing-history trace."""
    if scal is None:
        ax.text(0.5, 0.5, 'scalars.npz absent -- no Re trace',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    t = scal['t']
    ax.plot(t, scal['Re_inlet'][:, 0], lw=0.8, label=r'$Re_{inlet}$')
    if 'Re_cyl' in scal:
        ax.plot(t, scal['Re_cyl'][:, 0], lw=0.8, label=r'$Re_{cyl}$')
    ax.axvspan(t_min, min(t_max, t[-1]), alpha=0.15, color='green')
    ax.set_xlabel('t')
    if title:
        ax.set_ylabel('Re')
    ax.legend(loc='upper right', fontsize=7)


def fig_mean_rms(mean, rms, tag, extent, out_png, scal, t_min, t_max):
    fig = plt.figure(figsize=(11, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[4.2, 1.0])
    for j, (fld, sub) in enumerate([(mean, 'mean'), (rms, 'rms')]):
        ax = fig.add_subplot(gs[0, j])
        if sub == 'mean':
            v = _sym_clim(fld)
            im = ax.imshow(fld, origin='lower', extent=extent, cmap='seismic',
                           vmin=-v, vmax=v, aspect='equal')
        else:
            im = ax.imshow(fld, origin='lower', extent=extent, cmap='seismic',
                           vmin=0, vmax=_sym_clim(fld), aspect='equal')
        ax.set_title(f'{tag} {sub}')
        fig.colorbar(im, ax=ax, shrink=0.85)
    ax_tr = fig.add_subplot(gs[1, :])
    _trace_panel(ax_tr, scal, t_min, t_max)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def fig_phase_bins(phase_mean, bin_n, tag, extent, out_png, scal, t_min, t_max):
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 4, height_ratios=[3.6, 3.6, 1.0])
    v = _sym_clim(*[phase_mean[b] for b in range(N_BINS) if bin_n[b] > 0])
    for b in range(N_BINS):
        ax = fig.add_subplot(gs[b // 4, b % 4])
        if bin_n[b] > 0:
            ax.imshow(phase_mean[b], origin='lower', extent=extent,
                      cmap='seismic', vmin=-v, vmax=v, aspect='equal')
        else:
            ax.text(0.5, 0.5, 'empty bin', ha='center', va='center',
                    transform=ax.transAxes)
        ax.set_title(f'{tag}  $\\phi \\in [{BIN_EDGES_DEG[b]:.0f}^\\circ,'
                     f'{BIN_EDGES_DEG[b + 1]:.0f}^\\circ)$  n={int(bin_n[b])}',
                     fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    ax_tr = fig.add_subplot(gs[2, :])
    _trace_panel(ax_tr, scal, t_min, t_max)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def fig_coverage(bin_n, phi_snap, times, T_sh_mean, out_png, scal, t_min, t_max,
                 phase_source, mod_const=False):
    fig = plt.figure(figsize=(11, 6.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[4.0, 1.0])
    axp = fig.add_subplot(gs[0, 0], projection='polar')
    centers = np.deg2rad(BIN_EDGES_DEG[:-1] + 22.5)
    axp.bar(centers, bin_n, width=np.deg2rad(45) * 0.92, alpha=0.75)
    lbl = 'phase-wheel coverage' + (' (MOD-const)' if mod_const else '')
    axp.set_title(f'{lbl}\nsource: {phase_source}', fontsize=9)
    axh = fig.add_subplot(gs[0, 1])
    if T_sh_mean and np.isfinite(T_sh_mean):
        frac = np.mod(np.asarray(times), T_sh_mean) / T_sh_mean
        axh.hist(frac * 360.0, bins=48, range=(0, 360))
        axh.set_xlabel(r'snapshot time mod $T_{sh}$ [deg of cycle]')
        axh.set_title(f'times mod T_sh (T_sh={T_sh_mean:.4f})', fontsize=9)
    else:
        axh.text(0.5, 0.5, 'no T_sh available', ha='center', va='center',
                 transform=axh.transAxes)
    ax_tr = fig.add_subplot(gs[1, :])
    _trace_panel(ax_tr, scal, t_min, t_max)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def fig_centerline(u_mean, dx, dy, recirc, xc, yc, R, out_png, scal,
                   t_min, t_max):
    Ny, Nx = u_mean.shape
    iy = recirc['centerline_iy']
    x = np.arange(Nx) * dx
    fig = plt.figure(figsize=(10, 5.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(x, u_mean[iy], lw=1.0)
    ax.axhline(0.0, color='k', lw=0.6)
    ax.axvline(xc + R, color='gray', ls='--', lw=0.8, label='rear edge')
    if recirc.get('found'):
        ax.axvline(recirc['x_zero'], color='red', ls=':', lw=1.0,
                   label=f"u=0 crossing (L_r/D={recirc['L_r_over_D']:.3f})")
    ax.set_xlabel('x')
    ax.set_ylabel(r'$\overline{u}$ (centerline)')
    ax.legend(fontsize=8)
    ax.set_title('mean centerline streamwise velocity')
    ax_tr = fig.add_subplot(gs[1, 0])
    _trace_panel(ax_tr, scal, t_min, t_max)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# yaml sanitiser
# --------------------------------------------------------------------------- #
def _yamlify(o):
    if isinstance(o, dict):
        return {str(k): _yamlify(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_yamlify(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return _yamlify(o.tolist())
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    return o


# --------------------------------------------------------------------------- #
# main pipeline
# --------------------------------------------------------------------------- #
def run(args):
    run_dir = os.path.abspath(args.run_dir)
    out_dir = os.path.abspath(args.out_dir or os.path.join(run_dir, 'wake_diag'))
    os.makedirs(out_dir, exist_ok=True)
    pi_dir = os.path.abspath(args.pi_dir or run_dir)
    name = args.name

    fr_npz = os.path.join(run_dir, f'{name}_FR.npz')
    if not os.path.exists(fr_npz):
        raise FileNotFoundError(fr_npz)
    params_yaml = os.path.join(run_dir, f'{name}_FR_params.yaml')
    if os.path.exists(params_yaml):
        with open(params_yaml) as f:
            cfg = yaml.safe_load(f)
    else:
        warn(f'{params_yaml} absent; trying config.yaml')
        with open(os.path.join(run_dir, 'config.yaml')) as f:
            cfg = yaml.safe_load(f).get('qg', {})

    Lx = float(cfg['grid']['Lx'])
    Ly = float(cfg['grid']['Ly'])
    R_nom = float(cfg.get('mask', {}).get('r', np.nan)) if cfg.get('mask') else np.nan
    U_cfg = float(cfg.get('bc', {}).get('inlet_velocity', np.nan)) if cfg.get('bc') else np.nan

    z_small = np.load(fr_npz)
    times = np.asarray(z_small['times'], dtype=np.float64)
    chi = np.asarray(z_small['chi_obs'], dtype=np.float64) if 'chi_obs' in z_small.files else None
    shape, _ = npz_field_shape(fr_npz, 'omega_FR')
    n_frames = shape[-3] if len(shape) >= 3 else len(times)
    if n_frames == len(times) + 1:
        # solver packaging: field stack carries the IC frame at t=0, the
        # times array only the saved marks (audit_A postmortem 2026-07-09)
        times = np.concatenate(([0.0], times))
        print(f'[wake] times aligned: prepended t=0 IC mark '
              f'({n_frames} frames vs {len(times) - 1} saved times)')
    elif n_frames != len(times):
        raise ValueError(f'omega_FR frames={n_frames} vs times={len(times)}: '
                         'unrecognized layout')
    Ny, Nx = shape[-2], shape[-1]
    dx, dy = Lx / Nx, Ly / Ny
    extent = [0, Lx, 0, Ly]

    t_max = args.t_max if args.t_max is not None else float(times[-1])
    keep = (times >= args.t_min) & (times <= t_max)
    print(f'[wake] run_dir={run_dir}')
    print(f'[wake] grid {Ny}x{Nx}, Lx={Lx:.6f}, window t in [{args.t_min}, '
          f'{t_max}] -> {int(keep.sum())}/{len(times)} snapshots')

    scal = load_scalars(os.path.join(run_dir, 'scalars.npz'))
    if scal is None:
        warn('scalars.npz absent: probe table skipped, U_inlet from config, '
             'no Re trace panels')

    shed_path = args.shedding or os.path.join(run_dir, 'shedding_summary.npz')
    shed = load_shedding(shed_path)
    phi_snap, phase_source = snapshot_phases(times, shed)
    bins = bin_assign(phi_snap)
    T_sh_mean = None
    if shed is not None and 'T_sh_t' in shed:
        m_sh = (shed['t'] >= args.t_min) & (shed['t'] <= t_max)
        vals = shed['T_sh_t'][m_sh] if m_sh.any() else shed['T_sh_t']
        T_sh_mean = float(np.nanmean(vals))

    # per-snapshot inlet speed for u reconstruction
    if scal is not None and 'U_inlet' in scal:
        U_add = np.interp(times, scal['t'], scal['U_inlet'][:, 0])
    elif np.isfinite(U_cfg):
        U_add = np.full(len(times), U_cfg)
        warn(f'using config inlet_velocity={U_cfg} for all snapshots')
    else:
        U_add = np.zeros(len(times))
        warn('no inlet speed available; u maps EXCLUDE the mean flow')

    # ---- pass 1: omega (FR grid) + u,v ----------------------------------- #
    acc = accumulate(stream_snapshots(fr_npz, 'omega_FR'), keep, bins,
                     do_uv=True, Lx=Lx, Ly=Ly, U_add=U_add)
    if acc is None:
        raise RuntimeError('no snapshots in the usable window')
    cov = coverage_metrics(acc['bin_n'])
    if cov['commensurability_flag']:
        warn(f"phase coverage NON-UNIFORM (min bin = "
             f"{cov['min_bin_fraction_of_uniform']:.2f} of uniform) -- "
             f"SEC 5 commensurability suspected")

    # geometry + recirculation
    if chi is not None:
        xc, yc = chi_centroid(chi, dx, dy)
    elif scal is not None and 'obstacle_centroid_xy' in scal['meta']:
        xc, yc = map(float, scal['meta']['obstacle_centroid_xy'])
        warn('chi_obs absent; centroid from scalars meta')
    else:
        xc, yc = Lx / 2.0, Ly / 2.0
        warn('no chi_obs / scalars meta; centroid assumed at domain center')
    R_eff = R_nom if np.isfinite(R_nom) else 0.5
    recirc = recirculation_extent(acc['u_mean'], dx, dy, xc, yc, R_eff, Lx)

    # ---- Pi_FF per scale --------------------------------------------------#
    pi_files = find_pi_files(pi_dir, name)
    wanted = {int(s) for s in args.scales}
    missing = sorted(wanted - set(pi_files))
    if missing:
        warn(f'Pi_FF files missing for scale(s) {missing} in {pi_dir}')
    pi_results = {}
    for s in sorted(set(pi_files) & wanted):
        p = pi_files[s]
        zp = np.load(p)
        pt = np.asarray(zp['times'], dtype=np.float64)
        zp.close()
        pshape, _ = npz_field_shape(p, 'pi_ff')
        p_frames = pshape[-3] if len(pshape) >= 3 else len(pt)
        if p_frames == len(pt) + 1:
            # same IC-frame-vs-times layout as omega_FR (audit_A postmortem)
            pt = np.concatenate(([0.0], pt))
        elif p_frames != len(pt):
            raise ValueError(f'pi_ff frames={p_frames} vs times={len(pt)} '
                             f'in {p}: unrecognized layout')
        keep_p = (pt >= args.t_min) & (pt <= t_max)
        phi_p, _ = snapshot_phases(pt, shed)
        acc_p = accumulate(stream_snapshots(p, 'pi_ff'), keep_p,
                           bin_assign(phi_p))
        if acc_p is None:
            warn(f'scale {s}: no Pi_FF snapshots in window; skipped')
            continue
        pi_results[s] = {'path': p, 'acc': acc_p, 'times': pt}
        print(f'[wake] Pi_FF scale {s}: {acc_p["n"]} snapshots from '
              f'{os.path.basename(p)}')

    # ---- probe statistics ------------------------------------------------ #
    probe_stats = probe_statistics(scal, args.t_min, t_max) if scal else {}

    # ---- figures ----------------------------------------------------------#
    figs = []

    def _save(fn, *a, **k):
        path = os.path.join(out_dir, fn)
        figs.append(fn)
        return path

    fig_mean_rms(acc['mean'], acc['rms'], r'$\omega$', extent,
                 _save('fig_mean_rms_omega.png'), scal, args.t_min, t_max)
    fig_mean_rms(acc['u_mean'], acc['u_rms'], r'$u$', extent,
                 _save('fig_mean_rms_u.png'), scal, args.t_min, t_max)
    fig_mean_rms(acc['v_mean'], acc['v_rms'], r'$v$', extent,
                 _save('fig_mean_rms_v.png'), scal, args.t_min, t_max)
    fig_phase_bins(acc['phase_mean'], acc['bin_n'], r'$\omega$', extent,
                   _save('fig_phase_omega.png'), scal, args.t_min, t_max)
    for s, r in pi_results.items():
        a = r['acc']
        fig_mean_rms(a['mean'], a['rms'], rf'$\Pi_{{FF}}$ (s={s})', extent,
                     _save(f'fig_mean_rms_pi_s{s}.png'), scal, args.t_min, t_max)
        fig_phase_bins(a['phase_mean'], a['bin_n'], rf'$\Pi_{{FF}}$ (s={s})',
                       extent, _save(f'fig_phase_pi_s{s}.png'), scal,
                       args.t_min, t_max)
    fig_coverage(acc['bin_n'], phi_snap[keep], times[keep], T_sh_mean,
                 _save('fig_phase_coverage.png'), scal, args.t_min, t_max,
                 phase_source, mod_const=args.mod_const)
    fig_centerline(acc['u_mean'], dx, dy, recirc, xc, yc, R_eff,
                   _save('fig_centerline_u.png'), scal, args.t_min, t_max)

    # ---- summaries ---------------------------------------------------------#
    npz_out = {
        'bin_edges_deg': BIN_EDGES_DEG.astype(np.float32),
        'phi_snapshots': phi_snap.astype(np.float32),
        'snapshot_times': times.astype(np.float64),
        'window_mask': keep,
        'omega_bin_counts': acc['bin_n'],
        'omega_phase_mean': acc['phase_mean'].astype(np.float32),
        'omega_mean': acc['mean'].astype(np.float32),
        'omega_rms': acc['rms'].astype(np.float32),
        'u_mean': acc['u_mean'].astype(np.float32),
        'u_rms': acc['u_rms'].astype(np.float32),
        'v_mean': acc['v_mean'].astype(np.float32),
        'v_rms': acc['v_rms'].astype(np.float32),
    }
    for s, r in pi_results.items():
        a = r['acc']
        npz_out[f'pi_s{s}_phase_mean'] = a['phase_mean'].astype(np.float32)
        npz_out[f'pi_s{s}_bin_counts'] = a['bin_n']
        npz_out[f'pi_s{s}_mean'] = a['mean'].astype(np.float32)
        npz_out[f'pi_s{s}_rms'] = a['rms'].astype(np.float32)
    npz_path = os.path.join(out_dir, 'wake_summary.npz')
    np.savez_compressed(npz_path, **npz_out)

    summary = {
        'run_dir': run_dir,
        'name': name,
        'grid': {'Ny': Ny, 'Nx': Nx, 'Lx': Lx, 'Ly': Ly, 'dx': dx, 'dy': dy},
        'window': {'t_min': float(args.t_min), 't_max': float(t_max),
                   'n_snapshots_used': int(acc['n']),
                   'n_snapshots_total': int(len(times))},
        'phase': {'source': phase_source, 'n_bins': N_BINS,
                  'bin_edges_deg': BIN_EDGES_DEG.tolist(),
                  'T_sh_mean': T_sh_mean,
                  'coverage': cov},
        'geometry': {'obstacle_centroid_xy': [xc, yc], 'R_nominal': R_nom,
                     'D_nominal': 2.0 * R_nom if np.isfinite(R_nom) else None},
        'recirculation': recirc,
        'pi_ff': {str(s): {'path': r['path'],
                           'n_snapshots_used': int(r['acc']['n']),
                           'bin_counts': r['acc']['bin_n'].tolist()}
                  for s, r in pi_results.items()},
        'pi_scales_missing': missing,
        'probe_statistics': probe_stats,
        'figures': figs,
        'npz': os.path.basename(npz_path),
        'mod_const': bool(args.mod_const),
        'warnings': list(_WARNINGS),
    }
    yaml_path = os.path.join(out_dir, 'wake_summary.yaml')
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(_yamlify(summary), f, sort_keys=False,
                       default_flow_style=False)
    print(f'[wake] wrote {yaml_path}')
    print(f'[wake] wrote {npz_path}')
    print(f'[wake] figures: {", ".join(figs)}')
    return summary


# --------------------------------------------------------------------------- #
# selftest -- synthetic, analytically known, zero data deps
# --------------------------------------------------------------------------- #
def selftest():
    rows = []

    def check(nm, err, tol):
        ok = bool(err < tol)
        rows.append((nm, err, tol, 'PASS' if ok else 'FAIL'))
        return ok

    tmp = tempfile.mkdtemp(prefix='wake_selftest_')
    Ny = Nx = 64
    Lx = Ly = 2.0 * np.pi
    x = np.arange(Nx) * (Lx / Nx)
    y = np.arange(Ny) * (Ly / Ny)
    X, Y = np.meshgrid(x, y)

    # ---- 1. rotating pattern: phase-conditional means recover the pattern --#
    # omega(x,y;phi) = cos(3x - phi) + 0.5*sin(2y); phases: 128 distinct
    # uniformly spaced values (16 per bin), traversed over 16 shedding periods
    # (tests unwrap over many wraps + the npz streaming reader round trip).
    T_sh = 3.0
    Nsnap = 2048
    M = 16
    tt = (np.arange(Nsnap) + 0.5) / Nsnap * M * T_sh + 30.0
    phi_true = np.mod(2.0 * np.pi * (tt - 30.0) / T_sh, 2.0 * np.pi)
    stack = np.empty((1, Nsnap, Ny, Nx), dtype=np.float32)
    for j in range(Nsnap):
        stack[0, j] = (np.cos(3.0 * X - phi_true[j]) + 0.5 * np.sin(2.0 * Y)
                       ).astype(np.float32)
    fr = os.path.join(tmp, 'SYN_FR.npz')
    np.savez_compressed(fr, omega_FR=stack, times=tt)
    shed_npz = os.path.join(tmp, 'shedding_summary.npz')
    np.savez(shed_npz, t=tt, phi_t=phi_true,
             T_sh_t=np.full(Nsnap, T_sh), f_sh_t=np.full(Nsnap, 1.0 / T_sh))

    shed = load_shedding(shed_npz)
    phi_rec, src = snapshot_phases(tt, shed)
    check('phase interp+unwrap max|dphi|',
          float(np.max(np.abs(np.mod(phi_rec - phi_true + np.pi, 2 * np.pi)
                              - np.pi))), 1e-9)

    keep = np.ones(Nsnap, dtype=bool)
    bins = bin_assign(phi_rec)
    acc = accumulate(stream_snapshots(fr, 'omega_FR'), keep, bins)

    # exact discrete reference from the KNOWN phases and the stored f32 data
    stack64 = stack[0].astype(np.float64)
    err_disc = 0.0
    for b in range(N_BINS):
        ref = stack64[bins == b].mean(axis=0)
        err_disc = max(err_disc, float(np.max(np.abs(acc['phase_mean'][b] - ref))))
    check('phase-bin means vs exact discrete ref', err_disc, 1e-12)

    # analytic: bin mean = a*cos(3x - phi_c) + 0.5 sin(2y),
    # a = (2/Delta)*sin(Delta/2) for a uniform phase distribution in the bin
    delta = 2.0 * np.pi / N_BINS
    amp = np.sin(delta / 2.0) / (delta / 2.0)
    err_ana = 0.0
    for b in range(N_BINS):
        phi_c = (b + 0.5) * delta
        ana = amp * np.cos(3.0 * X - phi_c) + 0.5 * np.sin(2.0 * Y)
        err_ana = max(err_ana, float(np.max(np.abs(acc['phase_mean'][b] - ana))))
    check('phase-bin means vs analytic sinc pattern', err_ana, 1e-2)

    counts_ok = float(np.max(np.abs(acc['bin_n'] - Nsnap / N_BINS)))
    check('uniform-phase bin counts (max dev)', counts_ok, 0.51)

    # ---- 2. coverage flag: commensurate (clustered) vs incommensurate ------#
    # commensurate: dt_save = T_sh/4 -> only 4 distinct phases -> 4 empty bins
    t_com = 30.0 + np.arange(200) * (T_sh / 4.0)
    phi_com = np.mod(2.0 * np.pi * (t_com - 30.0) / T_sh, 2.0 * np.pi) + 0.01
    cov_com = coverage_metrics(np.bincount(bin_assign(phi_com), minlength=N_BINS))
    rows.append(('coverage flags commensurate times',
                 float(cov_com['commensurability_flag']), 'True',
                 'PASS' if cov_com['commensurability_flag'] else 'FAIL'))
    cov_uni = coverage_metrics(acc['bin_n'])
    rows.append(('coverage passes uniform times',
                 float(cov_uni['commensurability_flag']), 'False',
                 'PASS' if not cov_uni['commensurability_flag'] else 'FAIL'))

    # ---- 3. uniform-phase fallback path ------------------------------------#
    phi_fb, src_fb = snapshot_phases(tt[:16], None)
    rows.append(('fallback engages + warns', 0.0, 'uniform-fallback',
                 'PASS' if src_fb.startswith('uniform-fallback') else 'FAIL'))

    # ---- 4. u,v reconstruction vs analytic streamfunction ------------------#
    # psi = sin(x) sin(y): omega = lap psi = -2 psi, u = -dpsi/dy, v = dpsi/dx
    psi = np.sin(X) * np.sin(Y)
    om = -2.0 * psi
    u, v = uv_from_omega(om, Lx, Ly)
    err_u = float(np.max(np.abs(u - (-np.sin(X) * np.cos(Y)))))
    err_v = float(np.max(np.abs(v - (np.cos(X) * np.sin(Y)))))
    check('uv_from_omega u error', err_u, 1e-10)
    check('uv_from_omega v error', err_v, 1e-10)

    # ---- 5. recirculation zero crossing ------------------------------------#
    # manufactured centerline: u(x) = tanh((x - x0)/w), x0 off-node
    Nx2, Ny2 = 512, 64
    dx2 = Lx / Nx2
    x2 = np.arange(Nx2) * dx2
    x0 = 4.1234
    u2 = np.tile(np.tanh((x2 - x0) / 0.3), (Ny2, 1))
    rec = recirculation_extent(u2, dx2, Ly / Ny2, xc=2.0, yc=(Ly / Ny2) * 7,
                               R=0.5, Lx=Lx)
    check('recirculation u=0 crossing error',
          abs(rec['x_zero'] - x0) if rec['found'] else np.inf, 1e-3)

    # ---- table ------------------------------------------------------------ #
    print('\n===== diagnostics_wake SELFTEST =====')
    print(f"{'test':50s} {'value':>12s} {'tol/expect':>12s} {'verdict':>8s}")
    n_fail = 0
    for nm, err, tol, verdict in rows:
        n_fail += (verdict == 'FAIL')
        print(f'{nm:50s} {err:12.3e} {str(tol):>12s} {verdict:>8s}')
    print(f"===== {'PASS' if n_fail == 0 else f'FAIL ({n_fail})'} "
          f"({len(rows)} checks) =====")
    return 0 if n_fail == 0 else 1


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--run-dir', help='run directory with {name}_FR.npz etc.')
    p.add_argument('--name', default='DNS')
    p.add_argument('--pi-dir', default=None,
                   help='directory holding Pi_FF LES npz files (default: run dir)')
    p.add_argument('--shedding', default=None,
                   help='shedding_summary.npz (default: <run-dir>/shedding_summary.npz)')
    p.add_argument('--out-dir', default=None,
                   help='output directory (default: <run-dir>/wake_diag)')
    p.add_argument('--t-min', type=float, default=30.0,
                   help='usable-window start (theory doc: T_wait = 30)')
    p.add_argument('--t-max', type=float, default=None)
    p.add_argument('--scales', type=int, nargs='+', default=[2, 4, 8])
    p.add_argument('--mod-const', action='store_true',
                   help='annotate the coverage plot as the SEC 5 MOD-const check')
    p.add_argument('--selftest', action='store_true')
    args = p.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if not args.run_dir:
        p.error('--run-dir is required (or use --selftest)')
    run(args)


if __name__ == '__main__':
    main()
