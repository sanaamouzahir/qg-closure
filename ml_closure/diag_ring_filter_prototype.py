#!/usr/bin/env python3
"""diag_ring_filter_prototype.py -- PROTOTYPE: filter the high-frequency
"ringing" column OUT of the Pi_FF training TARGET, keeping the whole domain
(Sanaa 2026-07-14: filter the noise, do NOT mask/exclude any region).

Background (diag_truth_ringing_isolation.py, commit 7c0d86f): the Pi_J truth
carries a narrow, high-frequency oscillatory COLUMN at the obstacle x-station,
~83x the quiescent background, full domain height, flat to the domain edge --
a numerical artefact living where no real physics is. Today the strip is
handled by loss-masking; this prototype instead REPAIRS the target so every
pixel stays in play.

What it does, per geometry (one cylinder member + one cape member, val frames):
  1. localizes the artefact x-band AUTOMATICALLY: per-column RMS of the
     high-pass-filtered (along x) Pi, measured only in the quiescent band far
     from the body (sdf > 4D, sponge/upstream excluded) -- no hardcoded index;
  2. applies candidate repairs, DOMAIN KEPT EVERYWHERE:
       inpaint -- per y-row cubic fit through flanking columns replaces the
                  band values (cosine-blended edges);
       notch   -- inside the band only, low-pass along x (Gaussian, sigma ~
                  band width), same edge blend;
       ynotch  -- inside the band only, remove the y-GRID-NYQUIST component
                  ([1,2,1]/4 smoother in y). Added after measuring the
                  artefact: it alternates sign row-to-row in y (lag-1
                  autocorr ~ -0.99, a 2*dy checkerboard) while real Pi is
                  smooth in y -- so this notch targets exactly the artefact's
                  signature and can pass THROUGH the wake rows;
  3. figures: field panels (raw / both filters / removed component, symlog --
     the target is heavy-tailed), x-cross-sections at 3 y-stations, and the
     x-wavenumber spectrum of quiescent rows before/after;
  4. one summary.yaml: band location/width, artefact-vs-background amplitude,
     fraction of Pi^2 each filter removes globally / in the wake (sdf < 3D,
     the collateral-damage number) / in the quiescent band.

CPU only, no training, no existing file touched. Reuses RunData/load_conf
from dataset_piff.py (grad feature switched off in a COPY of the conf -- the
loader itself is untouched). Outputs per convention:
  pngs/ring_filter_prototype/<member>/*.png + README.txt per geometry,
  pngs/ring_filter_prototype/summary.yaml (+ explainer txt).
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from scipy.ndimage import gaussian_filter1d

from dataset_piff import RunData, load_conf

HERE = Path(__file__).resolve().parent
OUTD = HERE / 'pngs' / 'ring_filter_prototype'

GEOMETRIES = [
    ('cylinder', HERE / 'conf_piff_fpc_gjs.yaml'),
    ('cape', HERE / 'conf_piff_cape_gjs.yaml'),
]

HP_SIGMA_PX = 6.0        # high-pass scale for LOCALIZATION only (px along x)
BAND_MARGIN = 2          # px added to the detected band on each side
RAMP = 3                 # px cosine blend outside the band
N_FLANK = 6              # flanking columns per side for the cubic fit
QUIESC_Y_D = 4.0         # quiescent band = |y - y_c| > 4D (vertical distance from
                         # the body CENTERLINE, like the isolation script -- sdf
                         # alone would keep the downstream wake, which is far
                         # from the body but not quiescent)
WAKE_SDF_D = 3.0         # wake region for the energy budget = sdf < 3D

EXPLAINER = """ring_filter_prototype -- what these figures show

The Pi_J training target carries a numerical artefact: a narrow
high-frequency oscillatory COLUMN at the obstacle x-station, ~83x the
quiescent background, full domain height (isolated by
diag_truth_ringing_isolation.py). Sanaa's ask: FILTER it out of the target
instead of masking that strip out of the loss -- keep every pixel.

This prototype localizes the column automatically (per-column RMS of
high-pass-filtered Pi in the quiescent band |y - y_c| > 4D, vertical distance
from the body centerline -- NOT sdf, which would keep the wake) and repairs
the target three ways, everywhere in the domain:

  inpaint -- each y-row's values inside the band are replaced by a cubic
             fit through 6 flanking columns per side (cosine-blended edges);
  notch   -- inside the band only, Pi is low-passed along x (Gaussian,
             sigma = band width), same edge blend;
  ynotch  -- inside the band only, the y-grid-Nyquist component is removed
             ([1,2,1]/4 smoother along y). This one was added after
             MEASURING the artefact: it is a 2*dy CHECKERBOARD in y (lag-1
             autocorrelation ~ -0.99 between adjacent rows), organized as a
             ~0.5D-wavelength packet in x spanning the obstacle footprint
             (~1.3-1.6 D wide, NOT a few-px line). Real Pi is smooth in y at
             the grid scale, so a y-Nyquist notch separates artefact from
             physics far better than any x-interpolation can.

Measured consequence (see summary.yaml): the artefact band coincides with the
body's x-footprint, where the REAL Pi (the near-body cross) is orders of
magnitude larger than the artefact -- full-height inpaint/x-notch therefore
destroy real wake signal (removed energy ~ the wake's own energy). The
ynotch removes the checkerboard everywhere at ~zero collateral. A secondary,
weaker packet of the same kind sits at the outlet-sponge edge (not treated
by this prototype; same filter would apply).

Per geometry folder (<member>/README.txt describes each file):
  fields_t*.png     raw target vs both repairs vs what was removed (symlog)
  cross_sections.png  x-profiles at 3 y-stations: wake / mid-height / near edge
  spectrum.png      x-wavenumber spectrum of quiescent rows, before/after

Numbers: summary.yaml here. The judgement call is wake collateral damage:
frac_wake_pi2_removed (share of REAL wake signal each filter deletes) vs
removed_energy_share_quiescent (how much of what it removes is pure artefact).
"""

GEOM_README = """{member} ({geom}) -- ring-filter prototype figures

fields_t{t:.2f}.png
  Two rows, shared symlog scale (heavy-tailed field, eval-style linthresh =
  99th pct |Pi|). Top: raw Pi_J target | inpaint-repaired | x-notch-repaired
  | y-Nyquist-notch-repaired. Bottom: the quiescent-band column-RMS profile
  used to find the band (log-y, band shaded) | removed by inpaint | removed
  by x-notch | removed by ynotch (all raw - filtered). A removed panel
  should show ONLY the artefact column; wake structure appearing there =
  collateral damage.

cross_sections.png
  Pi vs x at three y-stations (through the wake, mid-height, near the top
  edge), raw vs the three filters, symlog y. Grey shading = the detected
  artefact band. Far from the body the raw curve oscillates inside the band
  and the filtered curves cut through it; through the wake the filter should
  hug the raw curve (real physics untouched) -- only ynotch does.

spectrum.png
  x-wavenumber power spectrum averaged over quiescent rows (|y - y_c| >
  {qd:.0f}D), before/after each filter. The artefact = the mid/high-k bump in
  the raw curve; a good filter drops it by orders of magnitude while leaving
  low k unchanged.

Band detected automatically: columns {j0}..{j1} (width {w_px} px = {w_D:.3f} D),
x-station {x_st:.3f} (obstacle x_c = {x_c:.3f}), artefact/background column-RMS
amplitude ratio {amp:.1f}x in the quiescent band. In-band adjacent-row
correlation (quiescent rows): {ylag:+.3f} -- near -1 means the artefact is a
pure y-grid-Nyquist checkerboard and ynotch removes it surgically; far from
-1 (the cape) means part of the artefact is smooth in y and ynotch only
captures the checkerboard share.
"""


def pick_val_frames(r, n, t_lo=100.0, t_hi=120.0):
    idx = r.frames_in(t_lo, t_hi)
    if len(idx) == 0:
        raise ValueError(f'{r.name}: no frames in [{t_lo},{t_hi})')
    sel = np.unique(np.linspace(0, len(idx) - 1, n).round().astype(int))
    return idx[sel]


def localize_band(frames_pi, quiesc):
    """Detect the artefact x-band from quiescent-band column statistics.
    Peak LOCATION from high-pass-filtered column energy (robust to smooth
    background); band WIDTH from the RAW column RMS -- the Gaussian high-pass
    smears a narrow column over ~3*sigma px, so thresholding the HP profile
    over-widens the band (observed: 40 px for a few-px column). Width
    threshold = geometric mean of peak and background (well above background,
    well below peak). Returns (j0, j1, colrms_raw, bg, peak) -- band inclusive
    [j0, j1], margin applied; amplitude ratio = peak/bg on the RAW profile
    (comparable to the isolation script's 83x)."""
    nx = frames_pi[0].shape[1]
    e_hp = np.zeros(nx)
    e_raw = np.zeros(nx)
    cnt = quiesc.sum(axis=0).astype(np.float64)
    for pi in frames_pi:
        hp = pi - gaussian_filter1d(pi, HP_SIGMA_PX, axis=1, mode='wrap')
        e_hp += np.where(quiesc, hp * hp, 0.0).sum(axis=0)
        e_raw += np.where(quiesc, pi * pi, 0.0).sum(axis=0)
    ok = cnt > 100
    rms_hp = np.full(nx, np.nan)
    rms_raw = np.full(nx, np.nan)
    rms_hp[ok] = np.sqrt(e_hp[ok] / (cnt[ok] * len(frames_pi)))
    rms_raw[ok] = np.sqrt(e_raw[ok] / (cnt[ok] * len(frames_pi)))
    jpk = int(np.nanargmax(rms_hp))
    bg = float(np.nanmedian(rms_raw))
    peak = float(rms_raw[jpk])
    thr = np.sqrt(max(peak, 1e-300) * max(bg, 1e-300))
    j0 = jpk
    while j0 - 1 >= 0 and np.isfinite(rms_raw[j0 - 1]) and rms_raw[j0 - 1] > thr:
        j0 -= 1
    j1 = jpk
    while j1 + 1 < nx and np.isfinite(rms_raw[j1 + 1]) and rms_raw[j1 + 1] > thr:
        j1 += 1
    j0 = max(j0 - BAND_MARGIN, 0)
    j1 = min(j1 + BAND_MARGIN, nx - 1)
    return j0, j1, rms_raw, bg, peak


def blend_weights(nx, j0, j1):
    """w(x): 1 inside [j0,j1], cosine ramp to 0 over RAMP px outside."""
    j = np.arange(nx)
    d = np.maximum(np.maximum(j0 - j, j - j1), 0)
    w = np.where(d >= RAMP, 0.0, 0.5 * (1.0 + np.cos(np.pi * d / RAMP)))
    w[(j >= j0) & (j <= j1)] = 1.0
    return w


def filter_inpaint(pi, j0, j1, w):
    """Per-row cubic fit through N_FLANK flanking columns each side, replacing
    the band (edge-blended). Vectorized over rows via one least-squares."""
    nx = pi.shape[1]
    lo, hi = j0 - RAMP, j1 + RAMP           # extended (blended) band
    fl = np.arange(max(lo - N_FLANK, 0), lo)
    fr = np.arange(hi + 1, min(hi + 1 + N_FLANK, nx))
    flank = np.concatenate([fl, fr])
    ctr, hw = 0.5 * (lo + hi), max(0.5 * (hi - lo), 1.0)
    A = np.vander((flank - ctr) / hw, 4)
    coef, *_ = np.linalg.lstsq(A, pi[:, flank].T, rcond=None)   # (4, Ny)
    cols = np.arange(max(lo, 0), min(hi + 1, nx))
    V = np.vander((cols - ctr) / hw, 4)
    fit = (V @ coef).T                                          # (Ny, ncols)
    out = pi.copy()
    out[:, cols] = (1.0 - w[cols])[None, :] * pi[:, cols] + w[cols][None, :] * fit
    return out


def filter_notch(pi, j0, j1, w):
    """Low-pass along x (Gaussian, sigma = band width) applied inside the band
    only, edge-blended -- the rest of the domain is bit-identical."""
    sigma = float(j1 - j0 + 1)
    sm = gaussian_filter1d(pi, sigma, axis=1, mode='wrap')
    return pi + w[None, :] * (sm - pi)


def filter_ynotch(pi, w):
    """Remove the y-grid-Nyquist (2*dy checkerboard) component inside the
    band: the [1,2,1]/4 smoother along y annihilates exactly the k_y-Nyquist
    mode and leaves smooth-in-y structure O(dy^2)-untouched. The checkerboard
    component is c = -(1/4) Lap_y pi (eigenfunction identity); filtered =
    pi - w(x) * c. Periodic in y like the domain."""
    c = 0.25 * (2.0 * pi - np.roll(pi, 1, axis=0) - np.roll(pi, -1, axis=0))
    return pi - w[None, :] * c


def energy_budget(frames_raw, frames_filt, wake, quiesc):
    """Summed over frames: fraction of Pi^2 removed globally / in the wake /
    in the quiescent band, and where the removed energy lives."""
    tot = wk = qu = rtot = rwk = rqu = 0.0
    for raw, filt in zip(frames_raw, frames_filt):
        p2 = raw.astype(np.float64) ** 2
        d2 = (raw - filt).astype(np.float64) ** 2
        tot += p2.sum(); wk += p2[wake].sum(); qu += p2[quiesc].sum()
        rtot += d2.sum(); rwk += d2[wake].sum(); rqu += d2[quiesc].sum()
    return {
        'frac_global_pi2_removed': float(rtot / tot),
        'frac_wake_pi2_removed': float(rwk / max(wk, 1e-300)),
        'frac_quiescent_pi2_removed': float(rqu / max(qu, 1e-300)),
        'removed_energy_share_quiescent': float(rqu / max(rtot, 1e-300)),
        'removed_energy_share_wake': float(rwk / max(rtot, 1e-300)),
    }


def row_spectrum(pi, rows, dx):
    """Mean power spectrum along x over the given rows."""
    f = np.fft.rfft(pi[rows].astype(np.float64), axis=1)
    return (np.abs(f) ** 2).mean(axis=0), np.fft.rfftfreq(pi.shape[1], d=dx) * 2 * np.pi


def process_geometry(geom, conf_path, n_frames):
    conf = copy.deepcopy(load_conf(str(conf_path)))
    conf['model']['use_grad_feature'] = False    # skip the gradmag build (not needed here)
    run_dir = conf['data']['runs'][0]            # the const member, as in the isolation script
    r = RunData(run_dir, conf)
    print(f'[{geom}] member {r.name}: grid {r.Ny}x{r.Nx}, D={r.D}, '
          f'x_c={r.x_c:.4f}, y_c={r.y_c:.4f}')

    frames = pick_val_frames(r, n_frames)
    times = r.times[frames]
    pis = [r.pi[k].astype(np.float64) for k in frames]
    x = (np.arange(r.Nx) + 0.5) * r.dx
    y = (np.arange(r.Ny) + 0.5) * r.dy

    # localization mask: far in y from the body CENTERLINE (excludes the whole
    # wake band at every x -- the isolation script's construction), sponge and
    # upstream-strip pixels excluded via r.valid
    far_y = np.abs(y - r.y_c)[:, None] > QUIESC_Y_D * r.D
    quiesc_loc = r.valid & far_y
    j0, j1, colrms, bg, peak = localize_band(pis, quiesc_loc)
    w = blend_weights(r.Nx, j0, j1)
    width_px = j1 - j0 + 1
    x_station = float(0.5 * (x[j0] + x[j1]))
    amp_ratio = peak / max(bg, 1e-300)
    print(f'[{geom}] band cols {j0}..{j1} ({width_px} px = {width_px * r.dx / r.D:.3f} D), '
          f'x={x_station:.3f} (x_c {r.x_c:.3f}), amplitude {amp_ratio:.1f}x background')

    inp = [filter_inpaint(p, j0, j1, w) for p in pis]
    nch = [filter_notch(p, j0, j1, w) for p in pis]
    ynt = [filter_ynotch(p, w) for p in pis]

    wake = r.sdf < WAKE_SDF_D * r.D
    quiesc = ~wake
    budget = {'inpaint': energy_budget(pis, inp, wake, quiesc),
              'notch': energy_budget(pis, nch, wake, quiesc),
              'ynotch': energy_budget(pis, ynt, wake, quiesc)}

    # artefact signature: adjacent-row correlation of the raw target inside
    # the band, quiescent rows only (a 2*dy checkerboard gives ~ -1)
    qi = np.where(far_y.all(axis=1))[0]
    qb = np.concatenate([p[qi][:, j0:j1 + 1].ravel() for p in pis])
    qb1 = np.concatenate([p[(qi + 1) % r.Ny][:, j0:j1 + 1].ravel() for p in pis])
    ylag = float(np.corrcoef(qb, qb1)[0, 1])

    gd = OUTD / r.name
    gd.mkdir(parents=True, exist_ok=True)

    # ---- figure 1: fields (first frame), shared symlog eval-style scale ---- #
    p0, i0, n0, y0 = pis[0], inp[0], nch[0], ynt[0]
    lin = max(np.percentile(np.abs(p0), 99), 1e-12)
    vmax = np.abs(p0).max()
    norm = SymLogNorm(linthresh=lin, vmin=-vmax, vmax=vmax, base=10)
    fig, axs = plt.subplots(2, 4, figsize=(22, 10))
    panels = [(axs[0, 0], p0, 'raw Pi_J target (symlog -- heavy-tailed)'),
              (axs[0, 1], i0, 'inpaint: band replaced by cubic through flanks'),
              (axs[0, 2], n0, 'x-notch: band low-passed along x'),
              (axs[0, 3], y0, 'y-notch: y-grid-Nyquist removed in band'),
              (axs[1, 1], p0 - i0, 'removed by inpaint (raw - filtered)'),
              (axs[1, 2], p0 - n0, 'removed by x-notch'),
              (axs[1, 3], p0 - y0, 'removed by y-notch (pure checkerboard)')]
    for ax, f2d, ttl in panels:
        im = ax.imshow(f2d, origin='lower', extent=[0, r.Lx, 0, r.Ly],
                       cmap='seismic', norm=norm, aspect='equal')
        ax.set_title(ttl, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)
    axl = axs[1, 0]
    axl.semilogy(x, colrms, color='#0072B2', lw=1.4)
    axl.axvspan(x[j0], x[j1], color='0.85', zorder=0)
    axl.axvline(r.x_c, color='k', lw=0.8, ls='--')
    axl.axhline(bg, color='0.5', lw=0.8, ls=':')
    axl.set_xlabel('x'); axl.set_ylabel('quiescent-band column RMS of Pi')
    axl.set_title(f'band localization: {amp_ratio:.0f}x background at the '
                  f'obstacle x-station (dashed)', fontsize=9)
    fig.suptitle(f'{r.name}: the ringing column filtered out of the target, '
                 f'domain kept everywhere (t={times[0]:.2f}; band {width_px} px '
                 f'= {width_px * r.dx / r.D:.2f} D at x={x_station:.2f}; in-band '
                 f'adjacent-row corr {ylag:+.2f} '
                 f'{"= a pure y-checkerboard" if ylag < -0.9 else "= NOT a pure y-checkerboard"})',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    ffields = gd / f'fields_t{times[0]:.2f}.png'
    fig.savefig(ffields, dpi=150); plt.close(fig)

    # ---- figure 2: x-cross-sections at 3 y-stations ------------------------ #
    stations = [(r.y_c, 'through the wake (y = y_c)'),
                (0.5 * (r.y_c + 0.92 * r.Ly), 'mid-height (between wake and edge)'),
                (0.92 * r.Ly, 'near the domain edge')]
    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for ax, (yy, lab) in zip(axs, stations):
        iy = int(round(yy / r.dy)) % r.Ny
        ax.plot(x, p0[iy], color='0.35', lw=1.6, label='raw')
        ax.plot(x, i0[iy], color='#0072B2', lw=1.0, label='inpaint')
        ax.plot(x, n0[iy], color='#D55E00', lw=1.0, ls='--', label='x-notch')
        ax.plot(x, y0[iy], color='#009E73', lw=1.0, ls='-.', label='y-notch')
        ax.axvspan(x[j0], x[j1], color='0.85', zorder=0)
        ax.set_yscale('symlog', linthresh=lin)
        ax.set_ylabel('Pi')
        ax.set_title(f'{lab}  (y = {y[iy]:.2f})', fontsize=10)
        ax.legend(fontsize=8, frameon=False, ncol=4)
    axs[-1].set_xlabel('x')
    fig.suptitle(f'{r.name}: Pi along x, raw vs filtered (grey band = detected '
                 f'artefact column; symlog y)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fxsec = gd / 'cross_sections.png'
    fig.savefig(fxsec, dpi=150); plt.close(fig)

    # ---- figure 3: x-wavenumber spectrum of quiescent rows ----------------- #
    qrows = np.where(quiesc_loc.mean(axis=1) > 0.8)[0]   # rows mostly quiescent
    if len(qrows) < 4:
        qrows = np.where(far_y.all(axis=1))[0]
    fig, ax = plt.subplots(figsize=(9, 6))
    for fields, lab, c in [(pis, 'raw', '0.35'), (inp, 'inpaint', '#0072B2'),
                           (nch, 'x-notch', '#D55E00'),
                           (ynt, 'y-notch', '#009E73')]:
        spec = None
        for f2d in fields:
            s, kx = row_spectrum(f2d, qrows, r.dx)
            spec = s if spec is None else spec + s
        ax.loglog(kx[1:], spec[1:] / len(fields), color=c, lw=1.4, label=lab)
    ax.set_xlabel('k_x'); ax.set_ylabel('mean |Pi_hat(k_x)|^2 over quiescent rows')
    ax.set_title(f'{r.name}: x-spectrum of the quiescent band (|y - y_c| > '
                 f'{QUIESC_Y_D:.0f}D, {len(qrows)} rows, {len(pis)} frames) -- '
                 f'the high-k artefact plateau before/after filtering', fontsize=10)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fspec = gd / 'spectrum.png'
    fig.savefig(fspec, dpi=150); plt.close(fig)

    (gd / 'README.txt').write_text(GEOM_README.format(
        member=r.name, geom=geom, t=times[0], qd=QUIESC_Y_D, j0=j0, j1=j1,
        w_px=width_px, w_D=width_px * r.dx / r.D, x_st=x_station, x_c=r.x_c,
        amp=amp_ratio, ylag=ylag))
    print(f'[{geom}] wrote {ffields}\n[{geom}] wrote {fxsec}\n[{geom}] wrote {fspec}')

    return {
        'member': r.name,
        'frames_t': [float(t) for t in times],
        'band_x_station': x_station,
        'band_x_offset_from_obstacle_D': float((x_station - r.x_c) / r.D),
        'band_cols': [int(j0), int(j1)],
        'band_width_px': int(width_px),
        'band_width_D': float(width_px * r.dx / r.D),
        'artefact_over_background_amplitude': float(amp_ratio),
        'background_colrms': bg,
        'peak_colrms': peak,
        'artefact_y_lag1_autocorr_in_band': ylag,
        'filters': budget,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--n-frames', type=int, default=4)
    ap.add_argument('--device', default='cpu', help='accepted for job-script '
                    'compatibility; this tool is numpy/CPU-only')
    args = ap.parse_args()

    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / 'ring_filter_prototype.txt').write_text(EXPLAINER)

    summary = {'what': 'prototype: filter the ringing column out of the Pi_FF '
                       'target; whole domain kept (no masking)',
               'n_val_frames_per_geometry': args.n_frames,
               'wake_region_def': f'sdf < {WAKE_SDF_D}D',
               'quiescent_band_def': f'|y - y_c| > {QUIESC_Y_D}D from the body '
                                     f'centerline (localization/spectra); '
                                     f'energy budget quiescent = NOT wake',
               'geometries': {}}
    for geom, conf_path in GEOMETRIES:
        summary['geometries'][geom] = process_geometry(geom, conf_path, args.n_frames)

    import yaml
    (OUTD / 'summary.yaml').write_text(yaml.safe_dump(summary, sort_keys=False))
    print(f'[done] wrote {OUTD / "summary.yaml"}')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
