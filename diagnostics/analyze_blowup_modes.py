"""analyze_blowup_modes.py -- WHICH MODES BLOW UP, HOW FAST, FROM WHEN.

Reads the npz written by `rollout_aposteriori.py --instrument-blowup` and
answers the question Sanaa posed 2026-07-19 (GO): two interventions aimed at
LINEAR/spectral mechanisms (p3 frozen-coefficient certificate |G_eff| 1.7 ->
0.96; the 2026-07-13 dissipative projection) each moved the a-posteriori
blowup by <=3 steps. So the mechanism is NOT the one we keep modelling. Stop
proposing training arms; instrument one blowing run.

Inputs (per arm, per integer mode-radius shell kappa, EVERY step):
    Z_k   state enstrophy                    (what grows)
    Ca_k  applied ANALYTIC closure term      (-c12 L^2 N [+ -coef f_anal])
    Cn_k  applied NN closure term            (-coef gamma f_NN)
    Ci_k  applied IMPLICIT L^3 term          (-c12 L^3 w_{n+1})
    C_k   the three summed as fields
    R_k   bare rhs increment DT*(0.5 L w + AB2 N)   (scale reference)

Outputs:
  (a) per-shell growth rate g_k(s) = d log Z_k / ds  (centered differences)
  (b) ONSET table: first step where g_k > --g-thresh and SUSTAINS it >=
      --sustain steps; sorted by onset; per shell reports mean g_k after
      onset and CONSTANT vs ACCELERATING (linear fit of g_k in s).
  (c) LEAD/LAG of the closure-dominance ratio C_k/R_k against Z_k, per shell,
      split Cn (NN) vs Ca (analytic) -- does the correction DRIVE or REACT.
  (d) 4-panel figure (--fig).
  (e) VERDICT block: linear-vs-nonlinear, which band starts first, NN vs
      analytic lead.

Usage (from anywhere):
  python diagnostics/analyze_blowup_modes.py \
      --npz <...>/blowup_instr_kf4_ic912.npz --arm closure \
      [--control-arm bare] [--fig <...>/blowup_modes.png] \
      [--g-thresh 0.05] [--sustain 3]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# 512^2 mode-index band convention (spectral_error_profile.BANDS)
BANDS = [('low_k<60', 0.0, 60.0),
         ('mid_60-170', 60.0, 170.67),
         ('annulus_171-241', 170.67, 241.4),
         ('beyond_241', 241.4, np.inf)]
EPS = 1e-300


def band_of(kappa):
    for name, lo, hi in BANDS:
        if lo <= kappa < hi:
            return name
    return BANDS[-1][0]


def load_arm(npz, arm):
    """Pull one arm's block out of the instrumentation npz."""
    need = ['Z_k', 'C_k', 'Ca_k', 'Cn_k', 'Ci_k', 'R_k', 'Z', 'E', 'cfl',
            'max_omega', 'step', 'kshell', 'kphys', 'blowup']
    miss = [k for k in need if f'{arm}_{k}' not in npz]
    if miss:
        raise SystemExit(f'[blowup] npz has no arm {arm!r} '
                         f'(missing {miss[:3]}...). Present keys start: '
                         f'{sorted({k.split("_")[0] for k in npz.files})}')
    d = {k: npz[f'{arm}_{k}'] for k in need}
    d['blowup'] = int(d['blowup'])
    d['arm'] = arm
    return d


def growth_rates(Z_k):
    """g_k(s) = d log Z_k / ds, centered in the interior, one-sided at the
    ends. Shells that are identically zero (or hit non-finite) give nan."""
    Z = np.asarray(Z_k, np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        lz = np.log(np.where(Z > 0.0, Z, np.nan))
    g = np.full_like(lz, np.nan)
    if lz.shape[0] >= 3:
        g[1:-1] = 0.5 * (lz[2:] - lz[:-2])
    if lz.shape[0] >= 2:
        g[0] = lz[1] - lz[0]
        g[-1] = lz[-1] - lz[-2]
    return g


def first_sustained(x, thresh, sustain):
    """First index i where x[i:i+sustain] are all finite and > thresh."""
    n = len(x)
    for i in range(n - sustain + 1):
        w = x[i:i + sustain]
        if np.all(np.isfinite(w)) and np.all(w > thresh):
            return i
    return None


def fit_slope(y, x):
    """Least-squares slope of y vs x + a crude significance heuristic
    (|slope| vs its standard error). Returns (slope, t_like)."""
    m = np.isfinite(y) & np.isfinite(x)
    if m.sum() < 3:
        return np.nan, np.nan
    xx, yy = x[m], y[m]
    A = np.vstack([xx, np.ones_like(xx)]).T
    coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
    resid = yy - A @ coef
    dof = max(len(yy) - 2, 1)
    sxx = float(((xx - xx.mean()) ** 2).sum())
    se = float(np.sqrt((resid ** 2).sum() / dof / max(sxx, EPS)))
    return float(coef[0]), (abs(float(coef[0])) / se if se > 0 else np.inf)


def onset_table(d, g, g_thresh, sustain):
    """Per-shell onset of sustained growth + CONSTANT/ACCELERATING verdict."""
    steps = np.asarray(d['step'], np.float64)
    rows = []
    for j in range(g.shape[1]):
        gk = g[:, j]
        i0 = first_sustained(gk, g_thresh, sustain)
        if i0 is None:
            continue
        post = gk[i0:]
        s_post = steps[i0:]
        slope, tstat = fit_slope(post, s_post)
        if not np.isfinite(slope) or tstat < 2.0:
            kind = 'CONSTANT'
        elif slope > 0:
            kind = 'ACCELERATING'
        else:
            kind = 'DECELERATING'
        rows.append(dict(shell=int(d['kshell'][j]), kphys=float(d['kphys'][j]),
                         band=band_of(float(d['kshell'][j])),
                         onset=int(steps[i0]),
                         g_mean=float(np.nanmean(post)),
                         g_max=float(np.nanmax(post)),
                         slope=slope, tstat=tstat, kind=kind,
                         Z_gain=float(d['Z_k'][-1, j]
                                      / max(d['Z_k'][i0, j], EPS))))
    rows.sort(key=lambda r: (r['onset'], -r['g_mean']))
    return rows


def ratio_onset(num, den, steps, sustain):
    """First step where the dominance ratio num/den rises a decade above its
    own pre-run baseline (median of the first third) and sustains it."""
    with np.errstate(divide='ignore', invalid='ignore'):
        r = num / np.where(den > 0.0, den, np.nan)
    n0 = max(len(r) // 3, 2)
    base = np.nanmedian(r[:n0])
    if not np.isfinite(base) or base <= 0:
        return None
    i0 = first_sustained(np.log10(np.where(r > 0, r, np.nan) / base),
                         1.0, sustain)
    return None if i0 is None else int(steps[i0])


def lead_lag(d, rows, sustain, n_shells=8):
    """(c) Does closure dominance rise BEFORE the shell's enstrophy does?
    Negative lag = correction LEADS (drives); positive = it REACTS."""
    steps = np.asarray(d['step'], np.float64)
    out = []
    for r in rows[:n_shells]:
        j = int(np.where(d['kshell'] == r['shell'])[0][0])
        den = d['R_k'][:, j]
        res = dict(shell=r['shell'], band=r['band'], z_onset=r['onset'])
        for lab, arr in (('C', d['C_k']), ('Cn', d['Cn_k']),
                         ('Ca', d['Ca_k'])):
            on = ratio_onset(arr[:, j], den, steps, sustain)
            res[f'{lab}_onset'] = on
            res[f'{lab}_lag'] = None if on is None else on - r['onset']
        out.append(res)
    return out


def fmt_onset_table(rows, limit=25):
    L = ['  shell   |k|_phys  band              onset  g_mean   g_max'
         '   slope/step   verdict',
         '  ' + '-' * 84]
    for r in rows[:limit]:
        L.append(f"  {r['shell']:5d}  {r['kphys']:8.2f}  {r['band']:<16s}"
                 f"  {r['onset']:5d}  {r['g_mean']:6.3f}  {r['g_max']:6.3f}"
                 f"  {r['slope']:+10.4f}   {r['kind']}")
    if len(rows) > limit:
        L.append(f'  ... {len(rows) - limit} further shells above threshold')
    if not rows:
        L.append('  (no shell sustained growth above the threshold)')
    return '\n'.join(L)


def fmt_leadlag(ll):
    L = ['  shell  band              Z_onset  C_onset(lag)'
         '  Cn_onset(lag)  Ca_onset(lag)',
         '  ' + '-' * 78]

    def cell(on, lag):
        return ('--'.rjust(14) if on is None
                else f'{on:5d} ({lag:+d})'.rjust(14))
    for r in ll:
        L.append(f"  {r['shell']:5d}  {r['band']:<16s}  {r['z_onset']:7d}"
                 f"  {cell(r['C_onset'], r['C_lag'])}"
                 f"{cell(r['Cn_onset'], r['Cn_lag'])}"
                 f"{cell(r['Ca_onset'], r['Ca_lag'])}")
    return '\n'.join(L)


def verdict(d, rows, ll, ctrl_rows):
    """(e) plain-English mechanism classification."""
    L = ['=' * 78, 'VERDICT -- blowup mechanism classification', '=' * 78]
    arm = d['arm']
    bl = d['blowup']
    L.append(f"arm={arm}   steps recorded={len(d['step'])}   "
             f"blowup step={'none' if bl < 0 else bl}   "
             f"cfl_max={np.nanmax(d['cfl']):.3f}   "
             f"max|omega| final={d['max_omega'][-1]:.3e}")
    if not rows:
        L.append('No shell sustained growth above the threshold: this run '
                 'does not blow up by the growth criterion. Nothing to '
                 'classify -- lower --g-thresh or check the arm.')
        L.append('=' * 78)
        return '\n'.join(L)

    # 1. linear vs nonlinear, from acceleration of g_k
    kinds = [r['kind'] for r in rows]
    n_acc = kinds.count('ACCELERATING')
    frac_acc = n_acc / len(kinds)
    if frac_acc >= 0.5:
        mech = ('NONLINEAR / SELF-AMPLIFYING. The per-shell growth rate '
                'itself increases with step in the majority of growing '
                'shells (%d/%d ACCELERATING). A frozen-coefficient linear '
                'instability grows at a CONSTANT rate; this does not. That '
                'is why two linear/spectral interventions (p3 certificate, '
                'dissipative projection) each bought only ~1-3 steps: they '
                'change the rate, not the feedback.'
                % (n_acc, len(kinds)))
    else:
        mech = ('LINEAR / CONSTANT-RATE. Most growing shells hold a steady '
                'g_k (%d/%d CONSTANT), consistent with a fixed-coefficient '
                'amplification factor |G|>1. A correctly targeted spectral '
                'intervention SHOULD then move the blowup step materially -- '
                'if it did not, the certificate is being evaluated on the '
                'wrong state or the wrong operator.'
                % (len(kinds) - n_acc, len(kinds)))
    L += ['', '1) LINEAR vs NONLINEAR: ' + mech]

    # 2. which band goes first
    first = rows[0]
    by_band = {}
    for r in rows:
        by_band.setdefault(r['band'], r['onset'])
    order = sorted(by_band.items(), key=lambda kv: kv[1])
    L += ['', '2) BAND ORDERING (earliest onset first): '
          + ', '.join(f'{b} @ step {s}' for b, s in order),
          f"   First shell to go: kappa={first['shell']} "
          f"(|k|={first['kphys']:.1f}, {first['band']}) at step "
          f"{first['onset']}, mean g={first['g_mean']:.3f}/step."]
    if first['band'].startswith('annulus'):
        L.append('   => it STARTS in the alias-contaminated annulus '
                 '(170.7-241.4): aliasing of the correction is implicated, '
                 'and a projection to the alias-safe radius is on-target.')
    elif first['band'].startswith('low'):
        L.append('   => it STARTS at LARGE SCALES (k<60), NOT in the '
                 'aliasing annulus. High-k remediations (kcut, alias-safe '
                 'projection, dissipative shell projection) cannot fix this '
                 '-- they act where the growth is not.')
    else:
        L.append('   => it STARTS at MID scales; neither a pure large-scale '
                 'nor a pure aliasing story.')

    # 3. NN vs analytic lead
    def med(key):
        v = [r[key] for r in ll if r.get(key) is not None]
        return float(np.median(v)) if v else None
    lag_n, lag_a = med('Cn_lag'), med('Ca_lag')
    L += ['', '3) NN vs ANALYTIC lead:']
    if lag_n is None and lag_a is None:
        L.append('   Neither correction term shows a dominance rise over the '
                 'bare rhs in the earliest shells -- the growth is not '
                 'closure-dominance driven in those shells (check R_k: the '
                 'bare tendency itself may be the one growing).')
    else:
        for lab, lag in (('NN (Cn)', lag_n), ('analytic (Ca)', lag_a)):
            if lag is None:
                L.append(f'   {lab}: no sustained dominance rise.')
            elif lag < 0:
                L.append(f'   {lab}: LEADS the enstrophy growth by '
                         f'{-lag:.0f} step(s) -- it DRIVES.')
            elif lag == 0:
                L.append(f'   {lab}: rises SIMULTANEOUSLY (lag 0) -- '
                         f'co-moving; lead/lag cannot separate cause here.')
            else:
                L.append(f'   {lab}: LAGS by {lag:.0f} step(s) -- it REACTS '
                         f'to a state that is already growing.')
        if lag_n is not None and lag_a is not None:
            who = ('the NN term' if lag_n < lag_a
                   else 'the analytic term' if lag_a < lag_n
                   else 'neither (tied)')
            L.append(f'   => {who} moves first.')
        elif lag_n is not None:
            L.append('   => only the NN term shows a dominance rise: the '
                     'NN piece is the mover.')
        else:
            L.append('   => only the analytic term shows a dominance rise: '
                     'this is NOT an NN-injected instability.')

    # 4. control
    L += ['', '4) CONTROL (bare arm):']
    if ctrl_rows is None:
        L.append('   not supplied (--control-arm) -- no growth-signature '
                 'comparison.')
    elif not ctrl_rows:
        L.append('   CLEAN: no shell in the bare arm sustains growth above '
                 'the same threshold. The growth signature above belongs to '
                 'the closure correction, not to the flow or the scheme.')
    else:
        L.append(f'   {len(ctrl_rows)} shell(s) also grow in the bare arm '
                 f'(earliest step {ctrl_rows[0]["onset"]}, band '
                 f'{ctrl_rows[0]["band"]}). Subtract this baseline before '
                 f'attributing the closure arm signature.')
    L.append('=' * 78)
    return '\n'.join(L)


def make_figure(d, g, rows, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    steps = np.asarray(d['step'], np.float64)
    Z = np.asarray(d['Z_k'], np.float64)
    top = rows[:6]
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 9.5))

    # (1) Z_k heatmap, log color. Unsigned magnitude -> viridis; 'seismic' is
    # the branch rule for SIGNED fields (panel 3 below).
    a = ax[0, 0]
    Zp = np.where(np.isfinite(Z) & (Z > 0), Z, np.nan)
    vmin = np.nanpercentile(Zp, 5) if np.isfinite(Zp).any() else 1e-30
    im = a.pcolormesh(steps, d['kshell'], Zp.T,
                      norm=LogNorm(vmin=max(vmin, 1e-30),
                                   vmax=np.nanmax(Zp)),
                      cmap='viridis', shading='nearest')
    fig.colorbar(im, ax=a, label=r'$Z_\kappa$')
    a.set_xlabel('step'); a.set_ylabel(r'shell $\kappa$')
    a.set_title(f"({d['arm']}) shell enstrophy $Z_\\kappa(s)$")
    if d['blowup'] > 0:
        a.axvline(d['blowup'], color='w', ls='--', lw=1.2)

    # (2) g_k for the 6 earliest-onset shells
    a = ax[0, 1]
    for r in top:
        j = int(np.where(d['kshell'] == r['shell'])[0][0])
        a.plot(steps, g[:, j], lw=1.3,
               label=f"$\\kappa$={r['shell']} (onset {r['onset']})")
    a.axhline(0.0, color='k', lw=0.8)
    a.set_xlabel('step')
    a.set_ylabel(r'$g_\kappa = \frac{d\,\log Z_\kappa}{ds}$')
    a.set_title('growth rate, earliest-onset shells')
    a.legend(fontsize=7)

    # (3) closure dominance C_k/R_k (signed exponent -> seismic, per rule 10)
    a = ax[1, 0]
    for r in top:
        j = int(np.where(d['kshell'] == r['shell'])[0][0])
        with np.errstate(divide='ignore', invalid='ignore'):
            rat = d['C_k'][:, j] / np.where(d['R_k'][:, j] > 0,
                                            d['R_k'][:, j], np.nan)
        a.semilogy(steps, rat, lw=1.3, label=f"$\\kappa$={r['shell']}")
    a.axhline(1.0, color='k', ls=':', lw=0.9)
    a.set_xlabel('step'); a.set_ylabel(r'$C_\kappa / R_\kappa$')
    a.set_title('closure dominance over the bare rhs')
    a.legend(fontsize=7)

    # (4) spectra at step 0 / onset / last
    a = ax[1, 1]
    i_on = (int(np.argmin(np.abs(steps - top[0]['onset']))) if top else 0)
    for lab, i, st in (('step 0', 0, '-'),
                       (f'onset (s={int(steps[i_on])})', i_on, '--'),
                       (f'last (s={int(steps[-1])})', len(steps) - 1, '-')):
        a.loglog(np.maximum(d['kshell'], 1.0), np.maximum(Z[i], 1e-300),
                 st, lw=1.4, label=lab)
    for _, lo, hi in BANDS[:-1]:
        if lo > 0:
            a.axvline(lo, color='grey', lw=0.6, ls=':')
    a.set_xlabel(r'shell $\kappa$'); a.set_ylabel(r'$Z_\kappa$')
    a.set_title('enstrophy spectrum: start vs onset vs blowup')
    a.legend(fontsize=8)

    for row in ax:
        for aa in row:
            aa.set_box_aspect(0.75)     # aspect-preserving, no stretch
    bl = ('' if d['blowup'] < 0 else f", blowup @ step {d['blowup']}")
    fig.suptitle(f"blowup modes -- arm {d['arm']}" + bl)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return out_png


def main():
    fmt = argparse.RawDescriptionHelpFormatter
    p = argparse.ArgumentParser(description=__doc__, formatter_class=fmt)
    p.add_argument('--npz', type=Path, required=True,
                   help='output of rollout_aposteriori.py --instrument-blowup')
    p.add_argument('--arm', default='closure', help='arm to analyze')
    p.add_argument('--control-arm', default=None,
                   help='non-blowing control arm (e.g. bare) for the '
                        'growth-signature comparison')
    p.add_argument('--g-thresh', type=float, default=0.05,
                   help='growth-rate threshold, per step (default 0.05)')
    p.add_argument('--sustain', type=int, default=3,
                   help='steps the threshold must be held (default 3)')
    p.add_argument('--fig', type=Path, default=None, help='4-panel png')
    p.add_argument('--table-limit', type=int, default=25)
    args = p.parse_args()

    npz = np.load(args.npz, allow_pickle=False)
    d = load_arm(npz, args.arm)
    g = growth_rates(d['Z_k'])
    rows = onset_table(d, g, args.g_thresh, args.sustain)
    ll = lead_lag(d, rows, args.sustain)

    ctrl_rows = None
    if args.control_arm:
        c = load_arm(npz, args.control_arm)
        ctrl_rows = onset_table(c, growth_rates(c['Z_k']),
                                args.g_thresh, args.sustain)

    print(f'[blowup] {args.npz}  arm={args.arm}  '
          f'threshold g>{args.g_thresh}/step sustained {args.sustain} steps')
    print()
    print('ONSET TABLE (sorted by onset step)')
    print(fmt_onset_table(rows, args.table_limit))
    print()
    print('LEAD/LAG -- correction dominance vs enstrophy growth '
          '(negative = correction leads/drives)')
    print(fmt_leadlag(ll))
    if ctrl_rows is not None:
        print()
        print(f'CONTROL ARM {args.control_arm}: '
              f'{len(ctrl_rows)} shell(s) above the same threshold')
        if ctrl_rows:
            print(fmt_onset_table(ctrl_rows, 10))
    print()
    print(verdict(d, rows, ll, ctrl_rows))
    if args.fig:
        if not rows:
            print(f'[blowup] no growing shells -- figure skipped ({args.fig})')
        else:
            print(f'[blowup] figure -> {make_figure(d, g, rows, args.fig)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
