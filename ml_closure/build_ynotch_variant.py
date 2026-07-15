#!/usr/bin/env python3
"""build_ynotch_variant.py -- PRODUCTION: build in-band-filtered target
payloads DNS_LES_s4_gaussian_jonly_<filter>.npz (Sanaa full approval
2026-07-14; ylp75 extension per coordinator, same night).

Background: the Pi_J target carries a numerical artefact -- a ~0.6-1.6D-wide
band of columns at the obstacle x-footprint whose signal is a y-grid-Nyquist
checkerboard (cylinder adjacent-row corr ~ -0.99997; cape -0.99 after the
corrected one-sided measurement -- the earlier +0.33 was contaminated).
Filters (both applied ONLY inside the band, cosine-blended edges, prototype
code reused VERBATIM by import):

  ynotch -- [1,2,1]/4 y-Nyquist smoother (diag_ring_filter_prototype.py;
            approved cylinder filter, kept as the comparison arm);
  ylp75  -- spectral cut of k_y >= 0.75 k_Nyquist, cosine taper 0.70-0.80
            (diag_cape_band_structure.py; validated on BOTH geometries:
            same artefact removal, ~100x less wake collateral).

Band localization is geometry-aware:
  cylinder (FPC-*)  quiescent = |y - y_c| > 4D (two-sided, prototype);
  cape (FPCape-*)   quiescent = y - y_c > 3D ABOVE the tip (one-sided --
                    the corrected diag_cape_band_structure method; the body
                    is bottom-attached, below-tip rows are wake).
Detector: --localizer windowed (default) = the corrected x-windowed peak
search (x_c - 2D .. x_c + 4D) from diag_cape_band_structure -- REQUIRED for
capes and demonstrably necessary for FPC-sine, where the unwindowed prototype
detector latched onto the outlet-sponge packet (cols 438..499, x ~ 8.3D
downstream; found during the ynotch build). --localizer prototype reproduces
the original unwindowed cylinder search (the completed ynotch build used it).

Per run dir:
  1. load DNS_LES_s4_gaussian_jonly.npz; localize the artefact x-band;
  2. apply the blended in-band filter to pi_ff ONLY, every frame (float64
     compute, cast back to the source float32); ALL OTHER KEYS byte-identical;
  3. write DNS_LES_s4_gaussian_jonly_<filter>.npz (savez_compressed like the
     source; atomic tmp+rename; REFUSES to overwrite an existing target);
  4. verification, printed per run: (a) sha256 byte-identity of every non-pi
     key; (b) band location vs the characterized references (FPC-const
     245..286, FPCape-const 103..126); (c) in-band adjacent-row autocorr
     before/after; (d) fraction of wake Pi^2 (sdf < 3D) changed (~5% ynotch,
     ~0.03-0.1% ylp75); (e) out-of-band pixels asserted bitwise-identical;
  5. a [source | filtered | removed] figure per run (seismic, SymLog) into
     pngs/ynotch_variant_build/<run>/.

The variant mechanism (dataset_piff.py:91-100) then makes the models consume
this with data.variant: gaussian_jonly_<filter> -- zero loader changes.

CPU only, no training, no existing file edited. Run from ml_closure/ on a
compute node (piff_tool_job.sh).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

from dataset_piff import RunData, load_conf
# prototype/characterization pieces reused VERBATIM (import, not copy):
from diag_ring_filter_prototype import (
    localize_band, blend_weights, filter_ynotch, pick_val_frames,
    QUIESC_Y_D as CYL_QUIESC_Y_D, WAKE_SDF_D)
from diag_cape_band_structure import (
    filter_ylp75, localize_band_windowed, largest_contiguous,
    QUIESC_Y_D as CAPE_QUIESC_Y_D)

HERE = Path(__file__).resolve().parent
OUTD = HERE / 'pngs' / 'ynotch_variant_build'

VARIANT_SRC = 'gaussian_jonly'
FILTER_FNS = {'ynotch': filter_ynotch, 'ylp75': filter_ylp75}

# characterized reference bands (ring_filter_prototype + cape_band_structure
# summaries): the check that per-run localization lands where the measured
# artefact lives
PROTO_REF = {'FPC-const': (245, 286), 'FPCape-const': (103, 126)}
BAND_AGREE_PX = 5          # "within a few px" tolerance vs the reference


def sha256_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def quiescent_rows(r, y, far1d, j0, j1, geometry):
    """Rows used for the artefact autocorr signature. Cylinder: the prototype's
    construction (all rows beyond 4D either side). Cape: the corrected
    diag_cape_band_structure construction (above the tip, fully valid across
    the band, largest contiguous block)."""
    if geometry == 'cape':
        band_ok = r.valid[:, j0:j1 + 1].all(axis=1)
        return largest_contiguous(np.where(far1d & band_ok)[0]), 'block'
    return np.where(far1d)[0], 'wrap'


def inband_row_autocorr(frames_pi, rows, mode, j0, j1, Ny):
    """Adjacent-row (lag-1 in y) correlation of the in-band values over the
    quiescent rows. mode 'wrap' = prototype pairing (row, row+1 mod Ny);
    mode 'block' = cape-script pairing within the contiguous block."""
    if mode == 'block':
        qb = np.concatenate([np.asarray(p, np.float64)[rows][:-1, j0:j1 + 1].ravel()
                             for p in frames_pi])
        qb1 = np.concatenate([np.asarray(p, np.float64)[rows][1:, j0:j1 + 1].ravel()
                              for p in frames_pi])
    else:
        qb = np.concatenate([np.asarray(p, np.float64)[rows][:, j0:j1 + 1].ravel()
                             for p in frames_pi])
        qb1 = np.concatenate([np.asarray(p, np.float64)[(rows + 1) % Ny][:, j0:j1 + 1].ravel()
                              for p in frames_pi])
    return float(np.corrcoef(qb, qb1)[0, 1])


def process_run(run_dir, conf, n_frames, filt_name, localizer):
    run_dir = Path(run_dir)
    name = run_dir.name
    geometry = 'cape' if name.startswith('FPCape') else 'cylinder'
    filt_fn = FILTER_FNS[filt_name]
    src = run_dir / f'DNS_LES_s4_{VARIANT_SRC}.npz'
    dst = run_dir / f'DNS_LES_s4_{VARIANT_SRC}_{filt_name}.npz'
    if not src.exists():
        raise FileNotFoundError(f'{name}: source payload missing: {src}')
    if dst.exists():
        raise FileExistsError(f'{name}: REFUSING to overwrite existing {dst}')
    if geometry == 'cape' and localizer != 'windowed':
        raise ValueError(f'{name}: cape runs REQUIRE the windowed localizer '
                         f'(two-sided prototype method mislocalizes on the '
                         f'bottom-attached body)')

    # geometry/masks via the untouched loader (same payload: conf variant is
    # gaussian_jonly); grad feature off -- not needed here
    r = RunData(str(run_dir), conf)
    print(f'[{name}] {geometry}, grid {r.Ny}x{r.Nx}, D={r.D}, x_c={r.x_c:.4f}, '
          f'y_c={r.y_c:.4f}, T={r.T}, filter={filt_name}, localizer={localizer}')

    # ---- 1. localize the band (characterized methods, verbatim) ------------ #
    frames = pick_val_frames(r, n_frames)
    times = r.times[frames]
    pis_loc = [r.pi[k].astype(np.float64) for k in frames]
    x = (np.arange(r.Nx) + 0.5) * r.dx
    y = (np.arange(r.Ny) + 0.5) * r.dy
    if geometry == 'cape':
        far1d = (y - r.y_c) > CAPE_QUIESC_Y_D * r.D          # one-sided, above tip
    else:
        far1d = np.abs(y - r.y_c) > CYL_QUIESC_Y_D * r.D     # two-sided
    quiesc_loc = r.valid & far1d[:, None]
    if localizer == 'windowed':
        j0, j1, colrms, _rms_hp, bg, peak = localize_band_windowed(
            pis_loc, quiesc_loc, x, r.x_c, r.D)
    else:
        j0, j1, colrms, bg, peak = localize_band(pis_loc, quiesc_loc)
    w = blend_weights(r.Nx, j0, j1)
    width_px = j1 - j0 + 1
    x_station = float(0.5 * (x[j0] + x[j1]))
    amp_ratio = peak / max(bg, 1e-300)
    print(f'[{name}] band cols {j0}..{j1} ({width_px} px = '
          f'{width_px * r.dx / r.D:.3f} D), x={x_station:.3f} '
          f'(x_c {r.x_c:.3f}), amplitude {amp_ratio:.1f}x background')

    # localization sanity vs the characterized references / obstacle footprint
    notes = []
    if name in PROTO_REF:
        p0, p1 = PROTO_REF[name]
        if abs(j0 - p0) > BAND_AGREE_PX or abs(j1 - p1) > BAND_AGREE_PX:
            notes.append(f'band {j0}..{j1} DISAGREES with reference {p0}..{p1} '
                         f'(> {BAND_AGREE_PX} px)')
    off_D = abs(x_station - r.x_c) / r.D
    if off_D > 1.5:
        notes.append(f'band x-station {off_D:.2f} D from obstacle (> 1.5 D)')
    if not (0.3 <= width_px * r.dx / r.D <= 3.0):
        notes.append(f'band width {width_px * r.dx / r.D:.2f} D outside [0.3, 3] D')
    for n in notes:
        print(f'[{name}] WARNING: {n}')

    # ---- 2. filter pi_ff, every frame, float64 -> cast back ---------------- #
    z = np.load(src)
    pi_src = z['pi_ff']                       # (1, T, Ny, Nx) float32
    src_dtype = pi_src.dtype
    if pi_src.shape[1] != r.T:
        raise ValueError(f'{name}: payload T {pi_src.shape[1]} != loader T {r.T}')
    touch = w > 0.0                           # columns the filter may modify
    keep = ~touch
    pi_new = pi_src.copy()
    wake = r.sdf < WAKE_SDF_D * r.D
    quiesc = ~wake
    tot = wk = qu = rtot = rwk = rqu = 0.0    # Pi^2 budget, float64
    for t in range(pi_src.shape[1]):
        p64 = pi_src[0, t].astype(np.float64)
        f64 = filt_fn(p64, w)                 # prototype/cape filter, verbatim
        pi_new[0, t][:, touch] = f64[:, touch].astype(src_dtype)
        d64 = p64 - pi_new[0, t].astype(np.float64)   # as-stored removal
        p2, d2 = p64 * p64, d64 * d64
        tot += p2.sum(); wk += p2[wake].sum(); qu += p2[quiesc].sum()
        rtot += d2.sum(); rwk += d2[wake].sum(); rqu += d2[quiesc].sum()
    budget = {
        'frac_global_pi2_removed': float(rtot / tot),
        'frac_wake_pi2_removed': float(rwk / max(wk, 1e-300)),
        'frac_quiescent_pi2_removed': float(rqu / max(qu, 1e-300)),
        'removed_energy_share_quiescent': float(rqu / max(rtot, 1e-300)),
    }

    # ---- 3. write: all other keys byte-identical, atomic, no overwrite ----- #
    payload = {}
    for k in z.files:
        payload[k] = pi_new if k == 'pi_ff' else z[k]
    tmp = run_dir / f'.{dst.name}.tmp.npz'
    if tmp.exists():
        tmp.unlink()
    np.savez_compressed(tmp, **payload)
    if dst.exists():                          # re-check right before rename
        tmp.unlink()
        raise FileExistsError(f'{name}: {dst} appeared during build -- refusing')
    os.rename(tmp, dst)
    print(f'[{name}] wrote {dst} ({dst.stat().st_size / 1e9:.2f} GB)')

    # ---- 4. verification on the WRITTEN file -------------------------------- #
    z2 = np.load(dst)
    if set(z2.files) != set(z.files):
        raise ValueError(f'{name}: key set mismatch {sorted(z2.files)}')
    nonpi_ok = True
    for k in z.files:
        if k == 'pi_ff':
            continue
        same = (z2[k].dtype == z[k].dtype and z2[k].shape == z[k].shape
                and sha256_arr(z2[k]) == sha256_arr(z[k]))
        nonpi_ok &= same
        if not same:
            print(f'[{name}] FAIL: key {k} differs from source')
    if z2['pi_ff'].dtype != src_dtype:
        raise ValueError(f'{name}: pi_ff dtype changed to {z2["pi_ff"].dtype}')
    pi_out = z2['pi_ff']

    # (e) out-of-band bitwise identity
    ob_same = (np.ascontiguousarray(pi_out[:, :, :, keep]).tobytes()
               == np.ascontiguousarray(pi_src[:, :, :, keep]).tobytes())
    assert ob_same, f'{name}: out-of-band pixels NOT bitwise-identical'

    # (c) in-band adjacent-row autocorr before/after, quiescent rows
    rows, mode = quiescent_rows(r, y, far1d, j0, j1, geometry)
    corr_before = inband_row_autocorr([pi_src[0, k] for k in frames],
                                      rows, mode, j0, j1, r.Ny)
    corr_after = inband_row_autocorr([pi_out[0, k] for k in frames],
                                     rows, mode, j0, j1, r.Ny)

    print(f'[{name}] VERIFY: non-pi keys identical: {"yes" if nonpi_ok else "NO"}; '
          f'out-of-band bitwise-identical: yes; '
          f'in-band row-corr {corr_before:+.5f} -> {corr_after:+.5f}; '
          f'wake Pi^2 changed {100 * budget["frac_wake_pi2_removed"]:.4f}%; '
          f'quiescent Pi^2 changed {100 * budget["frac_quiescent_pi2_removed"]:.4f}%')
    if not nonpi_ok:
        raise ValueError(f'{name}: non-pi key mismatch -- payload rejected')
    if corr_after <= -0.3:
        print(f'[{name}] WARNING: post-filter in-band row-corr {corr_after:+.3f} '
              f'<= -0.3 (checkerboard not fully removed)')
        notes.append(f'post-filter corr {corr_after:+.3f} <= -0.3')

    # ---- 5. figure: [source | filtered | removed], one frame ---------------- #
    gd = OUTD / name
    gd.mkdir(parents=True, exist_ok=True)
    k0 = frames[0]
    p0 = pi_src[0, k0].astype(np.float64)
    f0 = pi_out[0, k0].astype(np.float64)
    lin = max(np.percentile(np.abs(p0), 99), 1e-12)
    vmax = np.abs(p0).max()
    norm = SymLogNorm(linthresh=lin, vmin=-vmax, vmax=vmax, base=10)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, f2d, ttl in [(axs[0], p0, 'source pi_ff (gaussian_jonly)'),
                         (axs[1], f0, f'filtered pi_ff ({filt_name} variant)'),
                         (axs[2], p0 - f0, f'removed by {filt_name} (in-band)')]:
        im = ax.imshow(f2d, origin='lower', extent=[0, r.Lx, 0, r.Ly],
                       cmap='seismic', norm=norm, aspect='equal')
        ax.set_title(ttl, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f'{name}: {filt_name} variant build (t={times[0]:.2f}; band cols '
                 f'{j0}..{j1} = {width_px * r.dx / r.D:.2f} D; in-band row-corr '
                 f'{corr_before:+.3f} -> {corr_after:+.3f}; wake Pi^2 changed '
                 f'{100 * budget["frac_wake_pi2_removed"]:.4f}%)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fpng = gd / f'fields_{filt_name}_t{times[0]:.2f}.png'
    fig.savefig(fpng, dpi=150)
    plt.close(fig)
    print(f'[{name}] wrote {fpng}')

    return {
        'run': name,
        'geometry': geometry,
        'filter': filt_name,
        'localizer': localizer,
        'output': str(dst),
        'figure': str(fpng),
        'band_cols': [int(j0), int(j1)],
        'band_width_px': int(width_px),
        'band_width_D': float(width_px * r.dx / r.D),
        'band_x_station': x_station,
        'band_x_offset_from_obstacle_D': float((x_station - r.x_c) / r.D),
        'artefact_over_background_amplitude': float(amp_ratio),
        'inband_row_corr_before': corr_before,
        'inband_row_corr_after': corr_after,
        'budget': budget,
        'nonpi_keys_identical': bool(nonpi_ok),
        'out_of_band_bitwise_identical': True,
        'localization_notes': notes,
        'loc_frames_t': [float(t) for t in times],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', nargs='+',
                    default=[str(HERE / 'conf_piff_fpc_gjs.yaml')],
                    help='conf(s) whose data.runs list the members (variant '
                         'must be gaussian_jonly, scale 4); pass the fpc AND '
                         'cape gjs confs for the 10-run ylp75 build')
    ap.add_argument('--filter', choices=sorted(FILTER_FNS), default='ynotch',
                    help='in-band filter -> output variant suffix')
    ap.add_argument('--localizer', choices=['windowed', 'prototype'],
                    default='windowed',
                    help='band detector: windowed = corrected x-windowed '
                         'search (diag_cape_band_structure; required for '
                         'capes, fixes FPC-sine); prototype = original '
                         'unwindowed cylinder search (ynotch build used it)')
    ap.add_argument('--runs', nargs='*', default=None,
                    help='optional run-name subset (basenames)')
    ap.add_argument('--n-frames', type=int, default=4,
                    help='val frames for band localization (prototype default)')
    ap.add_argument('--device', default='cpu',
                    help='accepted for job-script compatibility; numpy/CPU-only')
    args = ap.parse_args()

    jobs = []          # (run_dir, per-run conf)
    for cpath in args.config:
        conf = copy.deepcopy(load_conf(cpath))
        if str(conf['data'].get('variant') or '') != VARIANT_SRC:
            raise ValueError(f'{cpath}: variant {conf["data"].get("variant")!r} '
                             f'!= {VARIANT_SRC!r}')
        conf['model']['use_grad_feature'] = False   # skip gradmag build
        for rd in conf['data']['runs']:
            c = copy.deepcopy(conf)
            c['data']['runs'] = [rd]
            jobs.append((rd, c))
    if args.runs:
        want = set(args.runs)
        jobs = [(rd, c) for rd, c in jobs if Path(rd).name in want]
        missing = want - {Path(rd).name for rd, _ in jobs}
        if missing:
            raise ValueError(f'runs not in config(s): {sorted(missing)}')

    OUTD.mkdir(parents=True, exist_ok=True)
    results = [process_run(rd, c, args.n_frames, args.filter, args.localizer)
               for rd, c in jobs]

    # ---- final table --------------------------------------------------------- #
    print(f'\n===== {args.filter} variant build: verification table =====')
    hdr = (f'{"run":<14} {"band cols":<10} {"width D":>8} {"corr before":>12} '
           f'{"corr after":>11} {"wake %":>9} {"non-pi ident":>13} {"notes"}')
    print(hdr)
    print('-' * len(hdr))
    for s in results:
        print(f'{s["run"]:<14} {s["band_cols"][0]}..{s["band_cols"][1]:<7} '
              f'{s["band_width_D"]:>8.3f} {s["inband_row_corr_before"]:>+12.5f} '
              f'{s["inband_row_corr_after"]:>+11.5f} '
              f'{100 * s["budget"]["frac_wake_pi2_removed"]:>9.4f} '
              f'{"yes" if s["nonpi_keys_identical"] else "NO":>13} '
              f'{"; ".join(s["localization_notes"]) or "-"}')

    import yaml
    summary = {'what': f'{VARIANT_SRC}_{args.filter} payload build (in-band '
                       f'blended {args.filter} on pi_ff only; all other keys '
                       f'byte-identical)',
               'source_variant': VARIANT_SRC,
               'filter': args.filter,
               'localizer': args.localizer,
               'prototypes': ['diag_ring_filter_prototype.py',
                              'diag_cape_band_structure.py'],
               'runs': results}
    spath = OUTD / f'summary_{args.filter}.yaml'
    spath.write_text(yaml.safe_dump(summary, sort_keys=False))
    print(f'\n[done] wrote {spath}')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
