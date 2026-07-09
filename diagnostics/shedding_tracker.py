r"""shedding_tracker.py -- shedding-frequency tracking from scalars.npz (SGS-closure).

Implements AMENDMENT_01 SD and AMENDMENT_02 S4, against the binding theory doc
docs/briefs/Supervisor_simulation.md (S1, S3.1). Post-processing ONLY: reads the
in-run scalar recorder output (qg/_output/scalars.py contract: keys t, step,
U_inlet, Re_inlet, Fx, Fy, Cd_inst, Cl_inst, Cd_mid, Cl_mid, U_cyl, Re_cyl, E,
Z, probe_u/probe_v (n,B,5), meta json).

Pipeline per run (usable window t >= t_min, default T_wait = 30):
  1. Welch PSD of C_L (primary: Cl_mid; also Cl_inst) -- Hann, segment length
     5*T_sh_expected (~15 t.u. at Re_mid, Amendment 01 SD.1), 75% overlap,
     zero-padded FFT + parabolic peak refinement -> global f_sh. Drag line:
     Cd_mid PSD peak near 2 f_sh. Third harmonic: Cl PSD prominence at 3 f_sh.
     Compared against theory S1: f_sh in [0.189, 0.480]; and the S3.1
     no-aliasing statement (snapshot f_Nyq = 1/(2*0.25) = 2.0 vs measured
     lift/drag/3rd-harmonic lines; St is NEVER measured from snapshots).
  2. Instantaneous frequency: band-pass C_L in the expected shedding band
     (default [0.15, 0.55] per Amendment 01 SD.2; auto-scaled quasi-steady band
     St_ref*U/D*[0.6,1.6] in --gate-d1 mode where Re=200 puts f_sh ~ 0.016),
     Hilbert transform -> phase phi(t), f_sh(t) = dphi/dt/(2pi) lightly
     smoothed over ~1 T_sh, T_sh(t) = 1/f_sh(t). Cross-checks: zero-crossing
     intervals of band-passed C_L, and the same Hilbert analysis on the wake
     probe v(t) at x_c + 1D (probe index 0).
     Medians compared against T_sh(Re) = D^2/(St_ref nu Re): theory S1 table
     {5.304 @2200, 2.992 @3900, 2.084 @5600}; (theory, measured, ratio) at the
     run's median Re_inlet.
  3. St(t) = f_sh(t)*D/U(t) under BOTH normalizations (U_inlet and U_cyl) vs
     St_ref ~ 0.21. Gate D-1 mode (--gate-d1): the ONE place literature
     comparison is allowed (Amendment 01 SF) -- St vs the canonical Re=200
     value ~0.195-0.20, with citation string in the summary.
  4. Exports (machine-readable, alongside figures):
       shedding_summary.yaml   -- every scalar + (theory, measured, ratio)
       shedding_summary.npz    -- FULL series phi(t), f_sh(t), T_sh(t), St(t),
                                  Re(t), PSDs; consumed by audit_decorrelation
                                  and diagnostics_wake.
       fig_shedding_psd.png, fig_shedding_instfreq.png (Re trace panel + f_sh
       measured vs quasi-steady f_qs(t) = St_ref*U_inlet(t)/D + St(t): the
       first-class non-stationarity figure, Amendment 01 SD.3).

Precision: all computations float64 (inputs upcast on load; storage of the run
outputs is float32 per theory doc S9 -- summaries here stay float64, they are
tiny). No literature numbers are compared outside --gate-d1 (charter framing
constraint).

Usage:
    python shedding_tracker.py <scalars.npz> [--outdir DIR] [--t-min 30.0]
        [--batch 0] [--st-ref 0.21] [--band FLO FHI] [--gate-d1] [--tag NAME]
    python shedding_tracker.py --selftest

Selftest (no data dependencies, analytically known answers): stationary
sinusoid f0 = 0.33 + 2nd/3rd harmonics + noise (recover f0 to <1% by Welch AND
by Hilbert median; phase advances 2pi per period to <1%; drag line at 2 f0;
third harmonic detected) and a linear chirp (f_sh(t) tracks f_true(t) to
<1.5% median over the interior). Exits nonzero on any failure; prints a PASS
table. Runs via qsub only (Amendment 02 S3).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.signal import welch, hilbert

# ---------------------------------------------------------------------------- #
# Theory constants (Supervisor_simulation.md S1) -- comparison targets only.
THEORY = dict(
    st_ref=0.21,
    f_sh_range=(0.189, 0.480),
    T_sh_table={2200: 5.304, 3900: 2.992, 5600: 2.084},
    f_nyq_snapshots=2.0,          # 1/(2*dt_save) at dt_save = 0.25 (S3.1)
    dt_save=0.25,
)
GATE_D1_LIT = dict(
    st_range=(0.195, 0.20),
    cd_mean_range=(1.3, 1.4),
    cl_rms_range=(0.4, 0.7),
    citation=("2D laminar-shedding benchmarks (allowed at Re=200 ONLY, "
              "Amendment 01 SF): Henderson (1995) Phys. Fluids 7:2102 "
              "(St(200) ~ 0.197, mean Cd ~ 1.34); Williamson (1996) Annu. "
              "Rev. Fluid Mech. 28:477 (St(200) ~ 0.196-0.198, parallel "
              "shedding); rms Cl ~ 0.4-0.7 (2D DNS, e.g. Norberg 2003 "
              "J. Fluids Struct. 17:57)."),
)


# ---------------------------------------------------------------------------- #
# Core signal tools (float64 throughout)

def _next_pow2(n):
    return 1 << int(np.ceil(np.log2(max(int(n), 2))))


def welch_psd(x, fs, nperseg, nfft_pad=8):
    """Welch PSD: Hann window, 75% overlap (Amendment 01 SD.1), zero-padded
    FFT (nfft = pow2 >= nfft_pad*nperseg) so peak positions are not quantized
    to the 1/(5 T_sh) segment resolution."""
    x = np.asarray(x, dtype=np.float64)
    nperseg = int(min(len(x), nperseg))
    nfft = _next_pow2(nfft_pad * nperseg)
    f, P = welch(x, fs=fs, window='hann', nperseg=nperseg,
                 noverlap=int(0.75 * nperseg), nfft=nfft, detrend='constant')
    return f, P


def peak_in_band(f, P, band):
    """Discrete peak inside band + parabolic refinement on log10 P.
    Returns (f_peak, P_peak). NaNs if the band holds no sample."""
    m = (f >= band[0]) & (f <= band[1])
    if not np.any(m):
        return np.nan, np.nan
    idx = np.flatnonzero(m)
    k = idx[np.argmax(P[idx])]
    fpk = f[k]
    if 0 < k < len(f) - 1 and P[k] > 0 and P[k - 1] > 0 and P[k + 1] > 0:
        lp = np.log10([P[k - 1], P[k], P[k + 1]])
        denom = lp[0] - 2.0 * lp[1] + lp[2]
        if denom < 0:
            delta = 0.5 * (lp[0] - lp[2]) / denom
            if abs(delta) <= 1.0:
                fpk = f[k] + delta * (f[1] - f[0])
    return float(fpk), float(P[k])


def bandpass(x, dt, f_lo, f_hi, taper_frac=0.15):
    """Zero-phase FFT band-pass with raised-cosine edges (taper_frac of each
    band edge) to limit ringing. Mean removed first."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, d=dt)
    H = np.zeros_like(f)
    lo0, lo1 = f_lo * (1.0 - taper_frac), f_lo
    hi0, hi1 = f_hi, f_hi * (1.0 + taper_frac)
    H[(f >= lo1) & (f <= hi0)] = 1.0
    ramp_lo = (f > lo0) & (f < lo1)
    H[ramp_lo] = 0.5 * (1.0 - np.cos(np.pi * (f[ramp_lo] - lo0) / (lo1 - lo0)))
    ramp_hi = (f > hi0) & (f < hi1)
    H[ramp_hi] = 0.5 * (1.0 + np.cos(np.pi * (f[ramp_hi] - hi0) / (hi1 - hi0)))
    return np.fft.irfft(X * H, n=n)


def hilbert_instfreq(x_bp, t, smooth_T):
    """Analytic signal -> unwrapped phase phi(t), instantaneous frequency
    f(t) = dphi/dt / 2pi, lightly smoothed (moving average over ~smooth_T).
    Returns (phi, f_raw, f_smooth, n_edge) with n_edge the number of samples
    on each end contaminated by the smoother/transform (exclude from medians)."""
    dt = float(np.median(np.diff(t)))
    an = hilbert(np.asarray(x_bp, dtype=np.float64))
    phi = np.unwrap(np.angle(an))
    f_raw = np.gradient(phi, t) / (2.0 * np.pi)
    n_sm = max(3, int(round(smooth_T / dt)) | 1)          # odd
    kern = np.ones(n_sm) / n_sm
    f_smooth = np.convolve(f_raw, kern, mode='same')
    return phi, f_raw, f_smooth, n_sm


def zero_crossing_freq(x_bp, t):
    """Mean frequency from upward zero crossings of the band-passed signal
    (Amendment 01 SD.2 cross-check). Returns (f_zc, periods)."""
    s = np.sign(x_bp)
    up = np.flatnonzero((s[:-1] <= 0) & (s[1:] > 0))
    if len(up) < 3:
        return np.nan, np.array([])
    # sub-sample crossing times by linear interpolation
    tc = t[up] - x_bp[up] * (t[up + 1] - t[up]) / (x_bp[up + 1] - x_bp[up])
    periods = np.diff(tc)
    return float(1.0 / np.median(periods)), periods


# ---------------------------------------------------------------------------- #
# Analysis on a scalars dict

def _load_scalars(path):
    z = np.load(path, allow_pickle=False)
    data = {k: np.asarray(z[k]) for k in z.files if k != 'meta'}
    for k, v in data.items():
        if np.issubdtype(v.dtype, np.floating):
            data[k] = v.astype(np.float64)      # F: upcast float32 storage
    data['meta'] = json.loads(str(z['meta']))
    return data


def analyze(data, outdir, batch=0, t_min=30.0, st_ref=0.21, band=None,
            gate_d1=False, tag=''):
    """Full shedding analysis on a scalars dict. Returns the summary dict
    (also written to shedding_summary.yaml/.npz + figures in outdir)."""
    os.makedirs(outdir, exist_ok=True)
    meta = data['meta']
    D = float(meta['length'])
    nu = float(meta['nu'])
    u_mid = float(meta.get('u_mid', 2.0))

    t_all = data['t']
    b = int(batch)

    def col(name):
        a = data[name]
        return a[:, b] if a.ndim == 2 else a

    cl_mid_all = col('Cl_mid')
    win = (t_all >= t_min) & np.isfinite(cl_mid_all)
    t = t_all[win]
    n = len(t)
    if n < 100:
        raise ValueError(f'usable window too short: {n} samples at t >= {t_min}')
    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt

    cl_mid = cl_mid_all[win]
    cl_inst = col('Cl_inst')[win]
    cd_mid = col('Cd_mid')[win]
    cd_inst = col('Cd_inst')[win]
    U_inlet = col('U_inlet')[win]
    U_cyl = col('U_cyl')[win]
    Re_inlet = col('Re_inlet')[win]
    Re_cyl = col('Re_cyl')[win]
    pv = data['probe_v']
    probe_v0 = pv[win, b, 0] if pv.ndim == 3 else pv[win, 0]

    U_med = float(np.nanmedian(U_inlet))
    Re_med = float(np.nanmedian(Re_inlet))
    T_sh_exp = D / (st_ref * U_med)               # quasi-steady expectation

    # ---- 1. Welch PSDs + spectral lines ---------------------------------- #
    nperseg = max(256, int(round(5.0 * T_sh_exp / dt)))   # SD.1: 5*T_sh ~ 15 t.u.
    if band is None:
        if gate_d1:
            band = (0.6 * st_ref * U_med / D, 1.6 * st_ref * U_med / D)
        else:
            band = (0.15, 0.55)                   # Amendment 01 SD.2 default
    band = (float(band[0]), float(band[1]))

    f_w, P_cl_mid = welch_psd(cl_mid, fs, nperseg)
    _, P_cl_inst = welch_psd(cl_inst, fs, nperseg)
    _, P_cd_mid = welch_psd(cd_mid, fs, nperseg)
    _, P_pv0 = welch_psd(probe_v0, fs, nperseg)

    f_sh, _ = peak_in_band(f_w, P_cl_mid, band)
    f_sh_clinst, _ = peak_in_band(f_w, P_cl_inst, band)
    f_sh_probe, _ = peak_in_band(f_w, P_pv0, band)

    # drag line at 2 f_sh (Cd PSD peak within +-20%)
    f_drag, _ = peak_in_band(f_w, P_cd_mid, (1.6 * f_sh, 2.4 * f_sh))
    # third harmonic prominence in the Cl PSD: peak power at 3 f_sh vs the
    # local background median in [2.6, 3.4] f_sh excluding +-6% of the line
    f3 = 3.0 * f_sh
    m_line = (f_w >= 0.94 * f3) & (f_w <= 1.06 * f3)
    m_bg = (f_w >= 2.6 * f_sh) & (f_w <= 3.4 * f_sh) & ~m_line
    if np.any(m_line) and np.any(m_bg):
        third_ratio = float(P_cl_mid[m_line].max() / np.median(P_cl_mid[m_bg]))
    else:
        third_ratio = np.nan
    third_present = bool(np.isfinite(third_ratio) and third_ratio > 5.0)

    # ---- 2. Hilbert instantaneous frequency ------------------------------ #
    cl_bp = bandpass(cl_mid, dt, band[0], band[1])
    phi, f_raw, f_t, n_edge = hilbert_instfreq(cl_bp, t, smooth_T=T_sh_exp)
    core = slice(n_edge, n - n_edge) if n > 4 * n_edge else slice(0, n)
    f_sh_median = float(np.median(f_t[core]))
    T_sh_t = 1.0 / np.maximum(f_t, 1e-12)
    T_sh_median = float(np.median(T_sh_t[core]))

    # cross-checks (SD.2): zero crossings of Cl; Hilbert on probe v at xc+1D
    f_zc, _ = zero_crossing_freq(cl_bp, t)
    pv_bp = bandpass(probe_v0, dt, band[0], band[1])
    _, _, f_t_pv, _ = hilbert_instfreq(pv_bp, t, smooth_T=T_sh_exp)
    f_sh_median_probe = float(np.median(f_t_pv[core]))

    # ---- 3. Strouhal, both normalizations -------------------------------- #
    St_inlet_t = f_t * D / np.where(np.isfinite(U_inlet) & (U_inlet != 0),
                                    U_inlet, np.nan)
    St_cyl_t = f_t * D / np.where(np.isfinite(U_cyl) & (U_cyl != 0),
                                  U_cyl, np.nan)
    St_inlet_median = float(np.nanmedian(St_inlet_t[core]))
    St_cyl_median = float(np.nanmedian(St_cyl_t[core]))
    f_qs_t = st_ref * U_inlet / D                 # quasi-steady prediction

    # ---- theory comparisons (Supervisor_simulation.md S1, S3.1) ----------- #
    T_sh_theory = D * D / (st_ref * nu * Re_med)  # T_sh(Re) at the run's Re
    lo, hi = THEORY['f_sh_range']
    fN_snap = THEORY['f_nyq_snapshots']
    summary = {
        'tag': tag or os.path.basename(os.path.dirname(os.path.abspath(outdir))),
        'inputs': dict(batch=b, t_min=float(t_min), n_samples=int(n),
                       dt_scalars=dt, window=[float(t[0]), float(t[-1])],
                       band=list(band), st_ref=float(st_ref),
                       D=D, nu=nu, u_mid=u_mid, gate_d1=bool(gate_d1)),
        'welch': dict(
            nperseg=int(min(n, nperseg)), overlap=0.75, window='hann',
            f_sh_peak_Cl_mid=f_sh, f_sh_peak_Cl_inst=f_sh_clinst,
            f_sh_peak_probe_v0=f_sh_probe,
            f_drag_peak=f_drag,
            f_drag_over_2fsh=(float(f_drag / (2.0 * f_sh))
                              if np.isfinite(f_drag) and f_sh else np.nan),
            third_harmonic_ratio=third_ratio,
            third_harmonic_present=third_present),
        'hilbert': dict(
            f_sh_median=f_sh_median, T_sh_median=T_sh_median,
            f_sh_median_probe_v0=f_sh_median_probe,
            f_zero_crossing=f_zc,
            phase_advance_cycles=float((phi[-1] - phi[0]) / (2.0 * np.pi)),
            smooth_T=float(T_sh_exp)),
        'strouhal': dict(
            St_inlet_median=St_inlet_median, St_cyl_median=St_cyl_median,
            St_ref=float(st_ref),
            Re_inlet_median=Re_med, Re_cyl_median=float(np.nanmedian(Re_cyl)),
            U_inlet_median=U_med, U_cyl_median=float(np.nanmedian(U_cyl))),
        'theory_comparison': {
            'f_sh_range_S1': dict(theory=[lo, hi], measured=f_sh,
                                  in_range=bool(lo <= f_sh <= hi)),
            'T_sh_at_median_Re': dict(theory=float(T_sh_theory),
                                      measured=T_sh_median,
                                      ratio=float(T_sh_median / T_sh_theory)),
            'T_sh_reference_table_S1': {int(k): float(v) for k, v in
                                        THEORY['T_sh_table'].items()},
            'St_vs_ref': dict(theory=float(st_ref), measured=St_inlet_median,
                              ratio=float(St_inlet_median / st_ref)),
            'no_aliasing_S3_1': dict(
                f_nyq_snapshots=fN_snap,
                f_nyq_scalars=float(0.5 * fs),
                lift_line=f_sh, drag_line=(2.0 * f_sh),
                third_harmonic=(3.0 * f_sh),
                snapshots_alias_free=bool(3.0 * f_sh < fN_snap),
                note=('St measured from the dense scalar series only; '
                      'snapshots are NOT used for St (theory doc S3.1)')),
        },
    }
    if gate_d1:
        g = GATE_D1_LIT
        summary['gate_d1'] = dict(
            st_measured_inlet=St_inlet_median,
            st_canonical_Re200=list(g['st_range']),
            st_pass=bool(g['st_range'][0] * 0.95 <= St_inlet_median
                         <= g['st_range'][1] * 1.05),
            cd_mean_inst=float(np.nanmean(cd_inst)),
            cd_mean_canonical=list(g['cd_mean_range']),
            cl_rms_inst=float(np.nanstd(cl_inst)),
            cl_rms_canonical=list(g['cl_rms_range']),
            citation=g['citation'])

    # ---- exports ---------------------------------------------------------- #
    np.savez(
        os.path.join(outdir, 'shedding_summary.npz'),
        t=t, phi=phi, f_sh_t=f_t, f_sh_t_raw=f_raw, T_sh_t=T_sh_t,
        St_inlet_t=St_inlet_t, St_cyl_t=St_cyl_t, f_qs_t=f_qs_t,
        Re_inlet_t=Re_inlet, Re_cyl_t=Re_cyl,
        U_inlet_t=U_inlet, U_cyl_t=U_cyl,
        f_welch=f_w, P_cl_mid=P_cl_mid, P_cl_inst=P_cl_inst,
        P_cd_mid=P_cd_mid, P_probe_v0=P_pv0,
        f_sh_peak=np.float64(f_sh), f_sh_median=np.float64(f_sh_median),
        T_sh_median=np.float64(T_sh_median),
        St_inlet_median=np.float64(St_inlet_median),
        St_cyl_median=np.float64(St_cyl_median),
        n_edge=np.int64(n_edge),
        meta=json.dumps(summary, default=float))
    with open(os.path.join(outdir, 'shedding_summary.yaml'), 'w') as fh:
        yaml.safe_dump(json.loads(json.dumps(summary, default=float)), fh,
                       sort_keys=False)

    _figures(outdir, t, cl_mid, f_w, P_cl_mid, P_cl_inst, P_cd_mid, f_sh,
             Re_inlet, Re_cyl, f_t, f_qs_t, St_inlet_t, St_cyl_t, st_ref,
             band, core)
    return summary


def _figures(outdir, t, cl_mid, f_w, P_cl_mid, P_cl_inst, P_cd_mid, f_sh,
             Re_inlet, Re_cyl, f_t, f_qs_t, St_inlet_t, St_cyl_t, st_ref,
             band, core):
    # PSD figure
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.semilogy(f_w, P_cl_mid, lw=1.2, label=r'$C_L$ (mid)')
    ax.semilogy(f_w, P_cl_inst, lw=0.8, alpha=0.7, label=r'$C_L$ (inst)')
    ax.semilogy(f_w, P_cd_mid, lw=0.8, alpha=0.7, label=r'$C_D$ (mid)')
    for m, ls, lb in [(1, '-', r'$f_{sh}$'), (2, '--', r'$2f_{sh}$'),
                      (3, ':', r'$3f_{sh}$')]:
        if np.isfinite(f_sh):
            ax.axvline(m * f_sh, color='k', ls=ls, lw=0.8, alpha=0.6, label=lb)
    ax.axvspan(band[0], band[1], color='0.9', zorder=0)
    if np.isfinite(f_sh):
        ax.set_xlim(0, min(5.0 * f_sh, f_w[-1]))
    ax.set_xlabel('f  [1/t.u.]')
    ax.set_ylabel('PSD (Welch, Hann, 75% overlap)')
    ax.legend(fontsize=8, ncol=2)
    ax.set_title('Force-coefficient spectra')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig_shedding_psd.png'), dpi=150)
    plt.close(fig)

    # instantaneous-frequency figure (Amendment 01 SD.3 -- first-class result)
    fig, axs = plt.subplots(3, 1, figsize=(8.0, 7.5), sharex=True)
    axs[0].plot(t, Re_inlet, lw=1.0, label=r'$Re_{inlet}(t)$')
    axs[0].plot(t, Re_cyl, lw=1.0, alpha=0.8, label=r'$Re_{cyl}(t)$')
    axs[0].set_ylabel('Re')
    axs[0].legend(fontsize=8)
    axs[1].plot(t, f_t, lw=1.0, label=r'$f_{sh}(t)$ (Hilbert)')
    axs[1].plot(t, f_qs_t, lw=1.0, ls='--',
                label=r'$f_{qs}(t) = St_{ref} U_{inlet}(t)/D$')
    axs[1].set_ylabel('f  [1/t.u.]')
    axs[1].legend(fontsize=8)
    axs[2].plot(t, St_inlet_t, lw=1.0, label=r'$St(t)$, $U_{inlet}$')
    axs[2].plot(t, St_cyl_t, lw=1.0, alpha=0.8, label=r'$St(t)$, $U_{cyl}$')
    axs[2].axhline(st_ref, color='k', ls=':', lw=0.8,
                   label=rf'$St_{{ref}}={st_ref}$')
    axs[2].set_ylabel('St')
    axs[2].set_xlabel('t  [t.u.]')
    axs[2].legend(fontsize=8)
    tc = t[core]
    for ax in axs:
        ax.axvspan(t[0], tc[0], color='0.92', zorder=0)
        ax.axvspan(tc[-1], t[-1], color='0.92', zorder=0)
    fig.suptitle('Shedding under modulation: measured vs quasi-steady')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig_shedding_instfreq.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------- #
# Selftest: synthetic signals with analytically known answers.

def _synthetic_scalars(kind, f0=0.33, dt=2.5e-3, T=120.0, D=1.256637,
                       nu=6.4443e-4, U=2.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, T, dt)
    n = len(t)
    if kind == 'stationary':
        phase = 2.0 * np.pi * f0 * t
        f_true = np.full(n, f0)
    elif kind == 'chirp':
        f_lo_c, f_hi_c = 0.25, 0.45
        f_true = f_lo_c + (f_hi_c - f_lo_c) * t / T
        phase = 2.0 * np.pi * np.cumsum(f_true) * dt
    else:
        raise ValueError(kind)
    cl = (np.sin(phase) + 0.15 * np.sin(3.0 * phase + 0.5)
          + 0.05 * rng.standard_normal(n))
    cd = 1.2 + 0.3 * np.sin(2.0 * phase + 1.0) + 0.05 * rng.standard_normal(n)
    ones = np.ones((n, 1))
    data = dict(
        t=t, step=np.arange(n),
        Cl_mid=cl[:, None], Cl_inst=cl[:, None] * (U / 2.0) ** -2,
        Cd_mid=cd[:, None], Cd_inst=cd[:, None] * (U / 2.0) ** -2,
        U_inlet=U * ones, U_cyl=0.95 * U * ones,
        Re_inlet=(U * D / nu) * ones, Re_cyl=(0.95 * U * D / nu) * ones,
        E=ones, Z=ones,
        probe_v=np.repeat(cl[:, None, None], 5, axis=2),
        meta=dict(length=D, nu=nu, u_mid=2.0, dt=2.5e-4, scalar_rate=10),
    )
    return data, f_true


def selftest():
    outroot = tempfile.mkdtemp(prefix='shed_selftest_',
                               dir=os.environ.get('TMPDIR', '/tmp'))
    checks = []       # (name, target, got, tol_rel, ok)

    def add(name, target, got, tol):
        ok = np.isfinite(got) and abs(got - target) <= tol * max(abs(target), 1e-30)
        checks.append((name, target, got, tol, bool(ok)))

    # --- stationary tone: f0 = 0.33, drag at 2 f0, 3rd harmonic ------------ #
    f0 = 0.33
    data, _ = _synthetic_scalars('stationary', f0=f0)
    s = analyze(data, os.path.join(outroot, 'stationary'), t_min=10.0,
                tag='selftest-stationary')
    add('welch f_sh (Cl_mid)', f0, s['welch']['f_sh_peak_Cl_mid'], 0.01)
    add('welch f_sh (Cl_inst)', f0, s['welch']['f_sh_peak_Cl_inst'], 0.01)
    add('hilbert median f_sh', f0, s['hilbert']['f_sh_median'], 0.01)
    add('zero-crossing f_sh', f0, s['hilbert']['f_zero_crossing'], 0.01)
    add('drag line at 2 f_sh', 2.0 * f0, s['welch']['f_drag_peak'], 0.02)
    add('T_sh median', 1.0 / f0, s['hilbert']['T_sh_median'], 0.01)
    # phase advances 2pi per period: total cycles == f0 * window length
    win = s['inputs']['window']
    add('phase advance [cycles]', f0 * (win[1] - win[0]),
        s['hilbert']['phase_advance_cycles'], 0.01)
    add('St median (U_inlet)', f0 * 1.256637 / 2.0,
        s['strouhal']['St_inlet_median'], 0.01)
    ok3 = s['welch']['third_harmonic_present']
    checks.append(('third harmonic present', True, ok3, 0.0, bool(ok3)))

    # --- chirp: f(t) tracked to <1.5% median over the interior ------------- #
    data, f_true = _synthetic_scalars('chirp')
    outdir = os.path.join(outroot, 'chirp')
    s = analyze(data, outdir, t_min=10.0, tag='selftest-chirp')
    z = np.load(os.path.join(outdir, 'shedding_summary.npz'))
    t, f_t, ne = z['t'], z['f_sh_t'], int(z['n_edge'])
    f_ref = np.interp(t, np.arange(len(f_true)) * 2.5e-3, f_true)
    core = slice(4 * ne, len(t) - 4 * ne)
    med_err = float(np.median(np.abs(f_t[core] - f_ref[core]) / f_ref[core]))
    add('chirp median |f_t - f_true|/f_true', 0.0, med_err, np.inf)
    checks[-1] = ('chirp median rel err (< 1.5%)', 0.015, med_err, 0.015,
                  bool(med_err < 0.015))

    # --- PASS table --------------------------------------------------------- #
    print('\n===== shedding_tracker selftest =====')
    print(f"{'check':<38}{'target':>14}{'got':>14}{'tol':>8}  verdict")
    all_ok = True
    for name, tgt, got, tol, ok in checks:
        all_ok &= ok
        print(f'{name:<38}{tgt!s:>14}{got!s:>14.14}{tol!s:>8}  '
              f"{'PASS' if ok else 'FAIL'}")
    print(f"===== overall: {'PASS' if all_ok else 'FAIL'} "
          f'({sum(c[-1] for c in checks)}/{len(checks)}) =====')
    print(f'artifacts: {outroot}')
    return 0 if all_ok else 1


# ---------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('scalars', nargs='?', help='path to scalars.npz')
    p.add_argument('--outdir', default=None,
                   help='output dir (default: alongside scalars.npz)')
    p.add_argument('--batch', type=int, default=0)
    p.add_argument('--t-min', type=float, default=30.0,
                   help='usable-window start (T_wait, theory doc S1)')
    p.add_argument('--st-ref', type=float, default=0.21)
    p.add_argument('--band', type=float, nargs=2, default=None,
                   metavar=('FLO', 'FHI'),
                   help='shedding band-pass (default 0.15 0.55; '
                        'auto quasi-steady in --gate-d1)')
    p.add_argument('--gate-d1', action='store_true',
                   help='Gate D-1 mode: Re=200 literature comparison '
                        '(ONLY place it is allowed)')
    p.add_argument('--tag', default='')
    p.add_argument('--selftest', action='store_true')
    args = p.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if not args.scalars:
        p.error('scalars.npz path required (or --selftest)')
    outdir = args.outdir or os.path.dirname(os.path.abspath(args.scalars))
    data = _load_scalars(args.scalars)
    s = analyze(data, outdir, batch=args.batch, t_min=args.t_min,
                st_ref=args.st_ref, band=args.band, gate_d1=args.gate_d1,
                tag=args.tag)
    print(yaml.safe_dump(json.loads(json.dumps(s, default=float)),
                         sort_keys=False))


if __name__ == '__main__':
    main()
