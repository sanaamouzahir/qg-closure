#!/usr/bin/env python3
"""diag_cape_band_structure.py -- characterize the CAPE artefact band along y
(and x) before choosing its filter.

Background (diag_ring_filter_prototype.py, approved chain): the cylinder
artefact is a pure y-grid-Nyquist checkerboard (adjacent-row corr -0.99997)
and the y-notch [1,2,1]/4 removes it surgically -- SETTLED. The cape band
(cols ~102-127, x ~5.65, 45x background) has adjacent-row corr only +0.33:
NOT a pure checkerboard, the y-notch only captures part of it. This tool
answers: what IS the cape artefact along y, and which in-band filter removes
it without touching the wake?

On 4 val frames each from FPCape-const and FPCape-ramp:
  1. isolate the in-band signal: band re-localized with the prototype's
     detector; quiescent rows = y - y_c > 3D ABOVE the cape tip (the cape is
     bottom-attached -- there is no quiescent side below) and fully valid
     across the band (sponge/body excluded via RunData.valid);
  2. characterize along y: full 1-D k_y spectrum of the in-band columns
     (variance fractions at the exact Nyquist mode, near-Nyquist k >= 0.9 k_N,
     the top-25% k >= 0.75 k_N, and below), autocorrelation vs row lag 0-16;
     the same along x inside the band (x-ACF + x-spectrum: is it x-structured
     too?); secondary x-stations from the full high-pass column-RMS profile
     (the cylinder had an outlet-sponge-edge packet -- check the cape);
  3. test THREE candidate in-band filters (prototype metrics + in-band ones):
       ynotch  -- y-grid-Nyquist notch [1,2,1]/4 (the approved cylinder one);
       ylp75   -- stronger y low-pass: kill the top 25% of k_y in-band
                  (spectral, cosine taper 0.70->0.80 k_N);
       notch2d -- ylp75 + mild x low-pass (Gaussian sigma 1.5 px) in-band;
     report per filter: artefact energy removed in the quiescent band,
     residual band amplitude, wake Pi^2 touched (collateral);
  4. figures (CPU, seismic, symlog raw, aspect preserved) in
     pngs/cape_band_structure/ + README.txt; for the best filter a field
     panel [raw | filtered | removed] per run;
  5. summary.yaml with all numbers.

Cylinder follow-up (cheap): the y-notch FIELD panel [raw | filtered |
removed] for FPC-const, one frame, into
pngs/ring_filter_prototype/FPC-const/fields_ynotch_t*.png -- the prototype's
field panels only showed the two disqualified filters.

CPU/numpy only. No existing file is edited; reuses the prototype's functions
by import. Run from ml_closure/ (flat sibling imports).
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
import yaml

from dataset_piff import RunData, load_conf
from diag_ring_filter_prototype import (
    localize_band, blend_weights, filter_ynotch, energy_budget,
    pick_val_frames, HP_SIGMA_PX)

HERE = Path(__file__).resolve().parent
OUTD = HERE / 'pngs' / 'cape_band_structure'
PROTO_OUTD = HERE / 'pngs' / 'ring_filter_prototype'

CAPE_CONF = HERE / 'conf_piff_cape_gjs.yaml'
FPC_CONF = HERE / 'conf_piff_fpc_gjs.yaml'
CAPE_MEMBERS = ('FPCape-const', 'FPCape-ramp')

QUIESC_Y_D = 3.0        # quiescent rows: y - y_c > 3D above the tip (task spec;
                        # the cape is bottom-attached, no quiescent side below)
CYL_QUIESC_Y_D = 4.0    # cylinder follow-up: keep the prototype's 4D two-sided
WAKE_SDF_D = 3.0        # wake for the collateral budget = sdf < 3D (prototype)
MAX_LAG_Y = 16
MAX_LAG_X = 12          # band is ~26 px wide; 12 lags keep >= half the pairs
YLP_CUT = 0.75          # kill k_y >= 0.75 k_Nyquist ("top 25%")
YLP_TAPER = 0.05        # cosine taper 0.70 -> 0.80 k_N (avoid Gibbs ringing)
XLP_SIGMA_PX = 1.5      # mild x low-pass for the 2-D notch
WAKE_COLLATERAL_TOL = 0.10   # a candidate is admissible if it deletes < 10%
                             # of the wake's Pi^2 (ynotch cape baseline: 3.6%)
SECONDARY_SEP_PX = 10
SECONDARY_MIN_RATIO = 3.0    # HP column-RMS >= 3x its median = a real packet
X_SEARCH_D = (-2.0, 4.0)     # band search window around x_c: the artefact
                             # lives at the obstacle x-station; a global argmax
                             # can latch onto far-wake leakage above y_c + 3D
                             # instead (observed on FPCape-ramp at x ~ 16)

README = """cape_band_structure -- what these figures show (plain English)

WHAT: the Pi_J training target for the CAPE runs carries a numerical artefact
band at the obstacle x-station (columns ~102-127, ~1.28 D wide, ~45x the
quiescent background). For the CYLINDER the same kind of band turned out to
be a pure row-to-row checkerboard in y (adjacent rows anti-correlated at
-0.99997), so a tiny y-Nyquist notch removes it exactly -- that filter is
approved and settled. For the cape the adjacent-row correlation is only
+0.33, so the cape artefact is NOT the same animal, and before choosing its
filter we characterize what it actually is along y and x.

HOW: we look only at "quiescent" rows -- more than 3 obstacle-heights ABOVE
the cape tip, far from any real flow -- inside the detected band, on 4
validation frames each of FPCape-const and FPCape-ramp. Whatever signal
lives there is pure artefact. (Above the tip ONLY: the cape is
bottom-attached, so the rows below y_c - 3D are near-wall wake, not
quiescent -- using them misleads both the band detector and the earlier
+0.33 row-correlation. We also split the in-band signal into a smooth
per-column "pedestal" (the column mean in y) and the fluctuation around it:
the pedestal is what pushed the prototype's raw row-correlation up to +0.33
even though the fluctuation itself is strongly checkerboard.)

FILES (one set per run, in <run>/):

yspec_autocorr.png -- six panels:
  (a) y-wavenumber spectrum of the in-band quiescent columns, raw and after
      each candidate filter. The x-axis is k_y as a fraction of the grid
      Nyquist (1.0 = the 2-pixel wave). Where the raw curve carries its
      energy IS the artefact's y-structure.
  (b) autocorrelation of the in-band signal vs row lag (0-16). A pure
      checkerboard would alternate -1,+1,-1,...; a smooth blob would decay
      slowly and stay positive.
  (c) the same autocorrelation along x INSIDE the band (is the artefact
      x-structured too?).
  (d) the column-RMS profile that localizes the band (grey = band, dashed =
      obstacle x-station, dotted vertical = any secondary packet found,
      e.g. at the outlet-sponge edge). The band is searched only within a
      few D of the obstacle -- the artefact lives at the body x-station;
      the strongest packet elsewhere is characterized separately and its
      y-structure says whether it is artefact (checkerboard) or just the
      far wake leaking above y_c + 3D (smooth).
  (e) x-wavenumber spectrum inside the band (quiescent rows).
  (f) bar chart: fraction of the in-band quiescent variance sitting at the
      exact y-Nyquist mode / near-Nyquist (top 10% of k_y) / 0.75-0.9 k_N /
      below, raw vs after each filter -- the "what fraction is checkerboard"
      answer in one picture.

fields_<filter>_t*.png -- the winning filter shown on the actual field:
  [raw target | filtered | removed (raw - filtered)], shared symlog color
  scale (the target is heavy-tailed), seismic, aspect preserved. The
  "removed" panel must show ONLY the artefact band; wake structure appearing
  there = collateral damage.

CANDIDATE FILTERS (all applied only inside the band, cosine-blended edges,
every pixel of the domain kept):
  ynotch  -- remove the exact y-Nyquist (2-pixel checkerboard) component,
             the [1,2,1]/4 smoother; the approved cylinder filter.
  ylp75   -- stronger: remove ALL y-wavelengths shorter than ~2.7 pixels
             (top 25% of k_y), spectral with a smooth taper.
  notch2d -- ylp75 plus a mild x-smoothing (Gaussian, 1.5 px) in-band.

NUMBERS: summary.yaml next to this file. The judgement numbers per filter:
"frac_band_quiescent_pi2_removed" (how much of the pure-artefact energy it
kills -- want high), "residual_band_amp_ratio" (is the band still visible
above background afterwards -- want ~1), and "frac_wake_pi2_removed" (how
much REAL wake signal it deletes -- want ~0).
"""


# --------------------------------------------------------------------------- #
# characterization helpers
# --------------------------------------------------------------------------- #

def largest_contiguous(rows):
    """Largest contiguous run of row indices (rows sorted 1-D int array)."""
    if len(rows) == 0:
        return rows
    breaks = np.where(np.diff(rows) > 1)[0]
    segs = np.split(rows, breaks + 1)
    return max(segs, key=len)


def yspec_pooled(segs, dy):
    """Pooled per-bin y-energy of demeaned (M, C) segments (equal M). Returns
    (kn, e) with kn = k_y / k_Nyquist in (0, 1], e = energy per bin summed
    over columns/frames, k=0 excluded. Parseval weights: interior bins x2."""
    M = segs[0].shape[0]
    e = None
    for s in segs:
        sd = s - s.mean(axis=0, keepdims=True)
        P = np.abs(np.fft.rfft(sd, axis=0)) ** 2       # (M//2+1, C)
        wb = np.full(P.shape[0], 2.0)
        wb[0] = 1.0
        if M % 2 == 0:
            wb[-1] = 1.0
        eb = (wb[:, None] * P).sum(axis=1) / M
        e = eb if e is None else e + eb
    freq = np.fft.rfftfreq(M, d=1.0)                   # cycles / sample
    kn = freq / 0.5                                    # fraction of Nyquist
    return kn[1:], e[1:]


def spec_fractions(kn, e):
    tot = float(e.sum())
    if tot <= 0:
        return {k: 0.0 for k in ('frac_var_at_nyquist', 'frac_var_top10_ky',
                                 'frac_var_top25_ky', 'frac_var_below_top25')}
    at_nyq = float(e[kn >= 1.0 - 1e-9].sum() / tot)
    top10 = float(e[kn >= 0.9].sum() / tot)
    top25 = float(e[kn >= 0.75].sum() / tot)
    return {'frac_var_at_nyquist': at_nyq,
            'frac_var_top10_ky': top10,
            'frac_var_top25_ky': top25,
            'frac_var_below_top25': 1.0 - top25}


def acf_pooled(segs, max_lag, axis=0):
    """Energy-weighted autocorrelation vs lag over a list of 2-D segments,
    along `axis` (0 = y/rows, 1 = x/cols). Demeaned along that axis."""
    num = np.zeros(max_lag + 1)
    den = 0.0
    for s in segs:
        sd = np.swapaxes(s, 0, axis)                    # lag axis first
        sd = sd - sd.mean(axis=0, keepdims=True)
        M = sd.shape[0]
        den += float((sd * sd).sum()) / M
        for l in range(min(max_lag, M - 1) + 1):
            num[l] += float((sd[:M - l] * sd[l:]).sum()) / M
    return num / max(den, 1e-300)


def colrms_profiles(frames_pi, quiesc):
    """Raw and high-pass quiescent column-RMS profiles (localization stats,
    same construction as the prototype's localize_band, both returned)."""
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
    return rms_raw, rms_hp


def localize_band_windowed(frames_pi, quiesc, x, x_c, D):
    """Prototype band detector restricted to the obstacle x-window: peak of
    the high-pass quiescent column-RMS inside x_c + X_SEARCH_D, width walk on
    the raw profile with the prototype's geometric-mean threshold + margin."""
    from diag_ring_filter_prototype import BAND_MARGIN
    rms_raw, rms_hp = colrms_profiles(frames_pi, quiesc)
    nx = len(x)
    win = (x >= x_c + X_SEARCH_D[0] * D) & (x <= x_c + X_SEARCH_D[1] * D)
    hp_w = np.where(win, rms_hp, np.nan)
    jpk = int(np.nanargmax(hp_w))
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
    return j0, j1, rms_raw, rms_hp, bg, peak


def secondary_stations(rms_hp, j0, j1, x, valid):
    """Packets of the same kind OUTSIDE the primary band: greedy peaks of the
    HP column-RMS, separated >= SECONDARY_SEP_PX, amplitude >= 3x median."""
    prof = rms_hp.copy()
    med = float(np.nanmedian(prof))
    prof[max(j0 - SECONDARY_SEP_PX, 0):j1 + SECONDARY_SEP_PX + 1] = np.nan
    vcols = np.where(valid.any(axis=0))[0]
    sponge_edge_col = int(vcols.max())
    out = []
    for _ in range(3):
        if np.all(np.isnan(prof)):
            break
        jp = int(np.nanargmax(prof))
        ratio = float(prof[jp] / max(med, 1e-300))
        if ratio < SECONDARY_MIN_RATIO:
            break
        out.append({'col': jp, 'x': float(x[jp]), 'hp_rms_over_median': ratio,
                    'cols_from_sponge_edge': int(sponge_edge_col - jp)})
        prof[max(jp - SECONDARY_SEP_PX, 0):jp + SECONDARY_SEP_PX + 1] = np.nan
    return out, sponge_edge_col, med


# --------------------------------------------------------------------------- #
# candidate filters (in-band, blended -- domain kept everywhere)
# --------------------------------------------------------------------------- #

def ylowpass_field(pi, cut=YLP_CUT, taper=YLP_TAPER):
    """Spectral low-pass along y (periodic): H = 1 below (cut-taper) k_N,
    cosine to 0 at (cut+taper) k_N, 0 above."""
    ny = pi.shape[0]
    f = np.fft.rfft(pi, axis=0)
    kn = np.fft.rfftfreq(ny, d=1.0) / 0.5              # fraction of Nyquist
    lo, hi = cut - taper, cut + taper
    H = np.where(kn <= lo, 1.0,
                 np.where(kn >= hi, 0.0,
                          0.5 * (1.0 + np.cos(np.pi * (kn - lo) / (hi - lo)))))
    return np.fft.irfft(f * H[:, None], n=ny, axis=0)


def filter_ylp75(pi, w):
    lp = ylowpass_field(pi)
    return pi + w[None, :] * (lp - pi)


def filter_notch2d(pi, w):
    lp = ylowpass_field(pi)
    lp = gaussian_filter1d(lp, XLP_SIGMA_PX, axis=1, mode='wrap')
    return pi + w[None, :] * (lp - pi)


FILTERS = [('ynotch', lambda p, w: filter_ynotch(p, w)),
           ('ylp75', filter_ylp75),
           ('notch2d', filter_notch2d)]


# --------------------------------------------------------------------------- #
# per-run processing
# --------------------------------------------------------------------------- #

def band_energy_removed(frames_raw, frames_filt, region):
    """Fraction of Pi^2 inside `region` (bool mask) removed by the filter."""
    tot = rem = 0.0
    for raw, filt in zip(frames_raw, frames_filt):
        tot += (raw.astype(np.float64) ** 2)[region].sum()
        rem += ((raw - filt).astype(np.float64) ** 2)[region].sum()
    return float(rem / max(tot, 1e-300))


def residual_amp_ratio(frames_filt, quiesc, j0, j1):
    """Peak(in band)/median column-RMS of the FILTERED field -- is the band
    still visible above background?"""
    rms_raw, _ = colrms_profiles(frames_filt, quiesc)
    pk = float(np.nanmax(rms_raw[j0:j1 + 1]))
    bg = float(np.nanmedian(rms_raw))
    return pk / max(bg, 1e-300)


def field_panel(path, raw, filt, name, run_name, t, Lx, Ly):
    lin = max(np.percentile(np.abs(raw), 99), 1e-12)
    vmax = float(np.abs(raw).max())
    norm = SymLogNorm(linthresh=lin, vmin=-vmax, vmax=vmax, base=10)
    fig, axs = plt.subplots(1, 3, figsize=(19, 6))
    for ax, f2d, ttl in [(axs[0], raw, 'raw Pi_J target'),
                         (axs[1], filt, f'{name}-filtered'),
                         (axs[2], raw - filt, f'removed by {name} (raw - filtered)')]:
        im = ax.imshow(f2d, origin='lower', extent=[0, Lx, 0, Ly],
                       cmap='seismic', norm=norm, aspect='equal')
        ax.set_title(ttl, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f'{run_name}: {name} filter on the Pi_J target, t={t:.2f} '
                 f'(symlog, shared scale). The removed panel should show ONLY '
                 f'the artefact band.', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def process_cape_run(run_dir, conf, n_frames):
    r = RunData(run_dir, conf)
    print(f'[cape] member {r.name}: grid {r.Ny}x{r.Nx}, D={r.D}, '
          f'x_c={r.x_c:.4f}, y_c={r.y_c:.4f} (tip), Lx={r.Lx:.3f}')
    frames = pick_val_frames(r, n_frames)
    times = r.times[frames]
    pis = [r.pi[k].astype(np.float64) for k in frames]
    x = (np.arange(r.Nx) + 0.5) * r.dx
    y = (np.arange(r.Ny) + 0.5) * r.dy

    # ---- band localization: ONE-SIDED quiescent mask ------------------------ #
    # The cape is bottom-attached: |y - y_c| > 3D would include rows y < y_c -
    # 3D near the bottom, which are near-wall wake/recirculation, NOT
    # quiescent -- they drag the detector to the near-bottom wake (observed:
    # band jumped to x~7, "410x"). Above the tip only.
    far_y_loc = (y - r.y_c)[:, None] > QUIESC_Y_D * r.D
    quiesc_loc = r.valid & far_y_loc
    j0, j1, rms_raw_prof, rms_hp_prof, bg, peak = localize_band_windowed(
        pis, quiesc_loc, x, r.x_c, r.D)
    w = blend_weights(r.Nx, j0, j1)
    width_px = j1 - j0 + 1
    x_station = float(0.5 * (x[j0] + x[j1]))
    amp_ratio = peak / max(bg, 1e-300)
    print(f'[{r.name}] band cols {j0}..{j1} ({width_px} px = '
          f'{width_px * r.dx / r.D:.3f} D), x={x_station:.3f} '
          f'(x_c {r.x_c:.3f}), amplitude {amp_ratio:.1f}x background')

    # ---- pure-artefact rows: ABOVE the tip only (cape is bottom-attached), --- #
    # fully valid across the band, largest contiguous block, even length
    above = (y - r.y_c) > QUIESC_Y_D * r.D
    band_ok = r.valid[:, j0:j1 + 1].all(axis=1)
    rows = largest_contiguous(np.where(above & band_ok)[0])
    if len(rows) % 2 == 1:
        rows = rows[:-1]
    if len(rows) < 64:
        raise ValueError(f'{r.name}: only {len(rows)} quiescent rows -- mask wrong?')
    print(f'[{r.name}] quiescent rows y in [{y[rows[0]]:.2f}, {y[rows[-1]]:.2f}] '
          f'({len(rows)} rows, y_c + {QUIESC_Y_D}D = {r.y_c + QUIESC_Y_D * r.D:.2f})')

    segs_raw = [p[rows][:, j0:j1 + 1] for p in pis]

    # ---- pedestal vs fluctuation split -------------------------------------- #
    # The prototype's in-band adjacent-row corr (+0.33) pooled RAW values: a
    # smooth-in-y per-column pedestal (column mean) adds positive correlation
    # that masks the checkerboard. Split the in-band sum of squares into the
    # column-mean pedestal and the demeaned fluctuation, and report BOTH the
    # raw (prototype-comparable) and demeaned lag-1 correlations.
    ss_ped = ss_tot = 0.0
    for s in segs_raw:
        mu = s.mean(axis=0, keepdims=True)
        ss_ped += float((mu * mu).sum()) * s.shape[0]
        ss_tot += float((s * s).sum())
    pedestal_share = ss_ped / max(ss_tot, 1e-300)
    qb = np.concatenate([s[:-1].ravel() for s in segs_raw])
    qb1 = np.concatenate([s[1:].ravel() for s in segs_raw])
    ylag_raw = float(np.corrcoef(qb, qb1)[0, 1])

    # ---- characterization along y and x ------------------------------------- #
    kn, e_raw = yspec_pooled(segs_raw, r.dy)
    fr_raw = spec_fractions(kn, e_raw)
    acf_y = acf_pooled(segs_raw, MAX_LAG_Y, axis=0)
    acf_x = acf_pooled(segs_raw, MAX_LAG_X, axis=1)
    knx, e_x = yspec_pooled([s.T for s in segs_raw], r.dx)   # x-spectrum: transpose
    fr_x = spec_fractions(knx, e_x)
    print(f'[{r.name}] in-band quiescent variance (demeaned): '
          f'{fr_raw["frac_var_at_nyquist"]:.3f} at y-Nyquist, '
          f'{fr_raw["frac_var_top10_ky"]:.3f} top-10% k_y, '
          f'{fr_raw["frac_var_top25_ky"]:.3f} top-25%; row-lag1 corr {acf_y[1]:+.3f} '
          f'(raw undemeaned {ylag_raw:+.3f}, pedestal share {pedestal_share:.3f}); '
          f'x: lag1 {acf_x[1]:+.3f}, {fr_x["frac_var_at_nyquist"]:.3f} at x-Nyquist')

    secondaries, sponge_edge_col, hp_med = secondary_stations(
        rms_hp_prof, j0, j1, x, r.valid)
    for s in secondaries:
        print(f'[{r.name}] secondary packet: col {s["col"]} (x={s["x"]:.2f}), '
              f'{s["hp_rms_over_median"]:.1f}x median HP-RMS, '
              f'{s["cols_from_sponge_edge"]} cols from the sponge edge')
    # is the strongest secondary station artefact (checkerboard) or real
    # far-wake leakage into the y > y_c + 3D rows? Its y-structure decides.
    if secondaries:
        sc = secondaries[0]['col']
        s0, s1 = max(sc - 5, 0), min(sc + 5, r.Nx - 1)
        segs_s = [p[rows][:, s0:s1 + 1] for p in pis]
        kns, e_s = yspec_pooled(segs_s, r.dy)
        fr_s = spec_fractions(kns, e_s)
        acf_s = acf_pooled(segs_s, 2, axis=0)
        secondaries[0]['yspec_fractions'] = fr_s
        secondaries[0]['row_lag1_corr_demeaned'] = float(acf_s[1])
        secondaries[0]['verdict'] = (
            'checkerboard artefact' if acf_s[1] < -0.5 else
            'smooth in y -- real flow (far-wake/outlet leakage), not grid artefact'
            if acf_s[1] > 0.3 else 'mixed')
        print(f'[{r.name}] strongest secondary (col {sc}): lag1 '
              f'{acf_s[1]:+.3f}, Nyquist frac {fr_s["frac_var_at_nyquist"]:.3f} '
              f'-> {secondaries[0]["verdict"]}')

    # ---- candidate filters --------------------------------------------------- #
    wake = r.sdf < WAKE_SDF_D * r.D
    quiesc = ~wake
    band_q = np.zeros((r.Ny, r.Nx), dtype=bool)
    band_q[np.ix_(rows, np.arange(j0, j1 + 1))] = True

    filt_fields = {}
    budgets = {}
    for name, fn in FILTERS:
        ff = [fn(p, w) for p in pis]
        filt_fields[name] = ff
        b = energy_budget(pis, ff, wake, quiesc)
        b['frac_band_quiescent_pi2_removed'] = band_energy_removed(pis, ff, band_q)
        b['residual_band_amp_ratio'] = residual_amp_ratio(ff, quiesc_loc, j0, j1)
        segs_f = [p[rows][:, j0:j1 + 1] for p in ff]
        knf, e_f = yspec_pooled(segs_f, r.dy)
        b['residual_yspec_fractions'] = spec_fractions(knf, e_f)
        budgets[name] = b
        print(f'[{r.name}] {name}: band-quiescent removed '
              f'{b["frac_band_quiescent_pi2_removed"]:.3f}, residual amp '
              f'{b["residual_band_amp_ratio"]:.1f}x, wake touched '
              f'{b["frac_wake_pi2_removed"]:.4f}')

    # best = among wake-safe candidates, max artefact removal; removals within
    # 1% of each other are a tie -> take the LOWEST wake collateral (the
    # [1,2,1]/4 stencil is a broad cos^2 low-pass, not a pure Nyquist notch,
    # so it always costs a few % of in-band wake energy)
    admissible = [n for n, _ in FILTERS
                  if budgets[n]['frac_wake_pi2_removed'] < WAKE_COLLATERAL_TOL]
    pool = admissible if admissible else [n for n, _ in FILTERS]
    top = max(budgets[n]['frac_band_quiescent_pi2_removed'] for n in pool)
    tied = [n for n in pool
            if budgets[n]['frac_band_quiescent_pi2_removed'] > top - 0.01]
    best = min(tied, key=lambda n: budgets[n]['frac_wake_pi2_removed'])
    print(f'[{r.name}] best filter: {best} (admissible: {admissible}, '
          f'tied on removal: {tied})')

    # ---- figure: spectra + autocorrelations ---------------------------------- #
    gd = OUTD / r.name
    gd.mkdir(parents=True, exist_ok=True)
    colors = {'raw': '0.25', 'ynotch': '#009E73', 'ylp75': '#0072B2',
              'notch2d': '#D55E00'}
    fig, axs = plt.subplots(2, 3, figsize=(20, 11))

    ax = axs[0, 0]
    ax.semilogy(kn, e_raw / e_raw.sum(), color=colors['raw'], lw=1.8, label='raw')
    for name, _ in FILTERS:
        segs_f = [p[rows][:, j0:j1 + 1] for p in filt_fields[name]]
        knf, e_f = yspec_pooled(segs_f, r.dy)
        ax.semilogy(knf, e_f / e_raw.sum(), color=colors[name], lw=1.2, label=name)
    for kc, ls in [(0.75, ':'), (0.9, '--')]:
        ax.axvline(kc, color='0.6', lw=0.8, ls=ls)
    ax.set_xlabel('k_y / k_Nyquist')
    ax.set_ylabel('variance per bin / raw total')
    ax.set_title('(a) y-spectrum of in-band quiescent columns\n'
                 '(dotted 0.75 k_N, dashed 0.9 k_N)', fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax = axs[0, 1]
    lags = np.arange(MAX_LAG_Y + 1)
    ax.axhline(0, color='0.7', lw=0.8)
    ax.plot(lags, acf_y, 'o-', color=colors['raw'], lw=1.6, label='raw')
    for name, _ in FILTERS:
        segs_f = [p[rows][:, j0:j1 + 1] for p in filt_fields[name]]
        ax.plot(lags, acf_pooled(segs_f, MAX_LAG_Y, axis=0), 'o-', ms=3,
                color=colors[name], lw=1.0, label=name)
    ax.set_xlabel('row lag (pixels in y)')
    ax.set_ylabel('autocorrelation')
    ax.set_title(f'(b) in-band ACF vs row lag (lag-1 raw: {acf_y[1]:+.3f};\n'
                 'a pure checkerboard would be -1,+1,-1,...)', fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax = axs[0, 2]
    lagsx = np.arange(MAX_LAG_X + 1)
    ax.axhline(0, color='0.7', lw=0.8)
    ax.plot(lagsx, acf_x, 's-', color=colors['raw'], lw=1.6, label='raw, along x')
    ax.plot(lags, acf_y, 'o--', color='0.6', lw=1.0, label='along y (for comparison)')
    ax.set_xlabel('lag (pixels)')
    ax.set_ylabel('autocorrelation')
    ax.set_title(f'(c) in-band ACF along x (lag-1: {acf_x[1]:+.3f}) --\n'
                 'is the artefact x-structured too?', fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax = axs[1, 0]
    ax.semilogy(x, rms_raw_prof, color='#0072B2', lw=1.3, label='raw column RMS')
    ax.semilogy(x, rms_hp_prof, color='#D55E00', lw=1.0, label='high-pass column RMS')
    ax.axvspan(x[j0], x[j1], color='0.85', zorder=0)
    ax.axvline(r.x_c, color='k', lw=0.8, ls='--')
    for s in secondaries:
        ax.axvline(s['x'], color='0.4', lw=0.8, ls=':')
    ax.axvline(x[sponge_edge_col], color='0.4', lw=0.8, ls='-.')
    ax.set_xlabel('x')
    ax.set_ylabel('quiescent column RMS of Pi')
    ax.set_title(f'(d) band localization ({amp_ratio:.0f}x background; dotted = '
                 f'secondary packets,\ndash-dot = sponge edge)', fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax = axs[1, 1]
    ax.semilogy(knx, e_x / e_x.sum(), color=colors['raw'], lw=1.6)
    ax.set_xlabel('k_x / k_Nyquist')
    ax.set_ylabel('variance per bin / total')
    ax.set_title(f'(e) x-spectrum inside the band (quiescent rows);\n'
                 f'{fr_x["frac_var_at_nyquist"]:.1%} of variance at x-Nyquist', fontsize=10)

    ax = axs[1, 2]
    cats = ['at Nyquist', 'top 10% k_y', 'top 25% k_y', 'below 0.75 k_N']
    keys = ['frac_var_at_nyquist', 'frac_var_top10_ky', 'frac_var_top25_ky',
            'frac_var_below_top25']
    xs_bar = np.arange(len(cats))
    wbar = 0.2
    ax.bar(xs_bar - 1.5 * wbar, [fr_raw[k] for k in keys], wbar,
           color=colors['raw'], label='raw')
    for i, (name, _) in enumerate(FILTERS):
        rf = budgets[name]['residual_yspec_fractions']
        # residual fractions are of the RESIDUAL total; rescale to raw total
        scale = 1.0 - budgets[name]['frac_band_quiescent_pi2_removed']
        ax.bar(xs_bar + (i - 0.5) * wbar, [rf[k] * scale for k in keys], wbar,
               color=colors[name], label=f'{name} (of raw total)')
    ax.set_xticks(xs_bar)
    ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylabel('fraction of raw in-band quiescent variance')
    ax.set_title('(f) where the in-band variance lives, raw vs residual\n'
                 'after each filter', fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    fig.suptitle(f'{r.name}: cape artefact band structure -- in-band quiescent '
                 f'signal (rows y > y_c + {QUIESC_Y_D:.0f}D, cols {j0}..{j1}), '
                 f'{len(pis)} frames', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fspec = gd / 'yspec_autocorr.png'
    fig.savefig(fspec, dpi=150)
    plt.close(fig)
    print(f'[{r.name}] wrote {fspec}')

    # ---- field panel for the best filter (first frame) ----------------------- #
    ffield = gd / f'fields_{best}_t{times[0]:.2f}.png'
    field_panel(ffield, pis[0], filt_fields[best][0], best, r.name, times[0],
                r.Lx, r.Ly)
    print(f'[{r.name}] wrote {ffield}')

    for name in list(budgets):
        budgets[name] = {k: (v if isinstance(v, dict) else float(v))
                         for k, v in budgets[name].items()}

    return {
        'member': r.name,
        'frames_t': [float(t) for t in times],
        'band_cols': [int(j0), int(j1)],
        'band_width_px': int(width_px),
        'band_width_D': float(width_px * r.dx / r.D),
        'band_x_station': x_station,
        'artefact_over_background_amplitude': float(amp_ratio),
        'quiescent_rows_def': f'y - y_c > {QUIESC_Y_D}D above the tip, '
                              f'valid across the band, largest contiguous block',
        'quiescent_rows_y_range': [float(y[rows[0]]), float(y[rows[-1]])],
        'n_quiescent_rows': int(len(rows)),
        'pedestal_share_of_in_band_SS': float(pedestal_share),
        'row_lag1_corr_raw_undemeaned': ylag_raw,
        'row_lag1_corr_demeaned': float(acf_y[1]),
        'yspec_fractions_raw': fr_raw,
        'acf_y_lags_0_16': [float(v) for v in acf_y],
        'acf_x_lags_0_12': [float(v) for v in acf_x],
        'xspec_fractions_in_band': fr_x,
        'secondary_x_stations': secondaries,
        'sponge_edge_col': int(sponge_edge_col),
        'filters': budgets,
        'admissible_filters_wake_tol': admissible,
        'best_filter': best,
    }, best


# --------------------------------------------------------------------------- #
# cylinder follow-up: the APPROVED y-notch, shown on the field
# --------------------------------------------------------------------------- #

def cylinder_ynotch_panel(n_frames):
    conf = copy.deepcopy(load_conf(str(FPC_CONF)))
    conf['model']['use_grad_feature'] = False
    r = RunData(conf['data']['runs'][0], conf)
    print(f'[cyl] member {r.name}: grid {r.Ny}x{r.Nx}, D={r.D}, x_c={r.x_c:.4f}')
    frames = pick_val_frames(r, n_frames)
    k = frames[0]
    t = float(r.times[k])
    pi = r.pi[k].astype(np.float64)
    y = (np.arange(r.Ny) + 0.5) * r.dy
    far_y = np.abs(y - r.y_c)[:, None] > CYL_QUIESC_Y_D * r.D
    j0, j1, _, _, _ = localize_band([pi], r.valid & far_y)
    w = blend_weights(r.Nx, j0, j1)
    filt = filter_ynotch(pi, w)
    gd = PROTO_OUTD / r.name
    gd.mkdir(parents=True, exist_ok=True)
    fpath = gd / f'fields_ynotch_t{t:.2f}.png'
    field_panel(fpath, pi, filt, 'ynotch', r.name, t, r.Lx, r.Ly)
    print(f'[cyl] wrote {fpath}')
    return {'member': r.name, 'frame_t': t, 'band_cols': [int(j0), int(j1)],
            'panel': str(fpath)}


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--n-frames', type=int, default=4)
    ap.add_argument('--device', default='cpu', help='accepted for job-script '
                    'compatibility; this tool is numpy/CPU-only')
    args = ap.parse_args()

    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / 'README.txt').write_text(README)

    conf = copy.deepcopy(load_conf(str(CAPE_CONF)))
    conf['model']['use_grad_feature'] = False
    run_map = {Path(p).name: p for p in conf['data']['runs']}

    summary = {
        'what': 'cape artefact band: y/x structure characterization + in-band '
                'filter comparison (ynotch vs ylp75 vs notch2d)',
        'n_val_frames_per_run': args.n_frames,
        'quiescent_band_def': f'y - y_c > {QUIESC_Y_D}D above the cape tip '
                              f'(bottom-attached body: one-sided)',
        'wake_region_def': f'sdf < {WAKE_SDF_D}D',
        'filter_defs': {
            'ynotch': 'y-grid-Nyquist notch [1,2,1]/4 in-band (approved cylinder filter)',
            'ylp75': f'spectral y low-pass in-band, kill k_y >= {YLP_CUT} k_Nyquist '
                     f'(cosine taper {YLP_CUT - YLP_TAPER:.2f}-{YLP_CUT + YLP_TAPER:.2f})',
            'notch2d': f'ylp75 + mild x low-pass (Gaussian sigma {XLP_SIGMA_PX} px) in-band',
        },
        'wake_collateral_tolerance': WAKE_COLLATERAL_TOL,
        'runs': {},
    }
    votes = []
    for member in CAPE_MEMBERS:
        res, best = process_cape_run(run_map[member], conf, args.n_frames)
        summary['runs'][member] = res
        votes.append(best)
    summary['recommended_cape_filter'] = (
        votes[0] if len(set(votes)) == 1 else
        f'DISAGREEMENT between runs: {dict(zip(CAPE_MEMBERS, votes))}')

    summary['cylinder_followup'] = cylinder_ynotch_panel(args.n_frames)

    (OUTD / 'summary.yaml').write_text(yaml.safe_dump(summary, sort_keys=False))
    print(f'[done] wrote {OUTD / "summary.yaml"}')
    print(json.dumps(summary, indent=2, default=float))


if __name__ == '__main__':
    main()
