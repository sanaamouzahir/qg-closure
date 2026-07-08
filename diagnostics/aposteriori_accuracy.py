"""
aposteriori_accuracy.py -- 3a: accuracy diagnostics from a rollout_apost npz.

Consumes rollout_apost_<tag>.npz/.json written by training/rollout_aposteriori.py
(run with the truth arm on) and produces, under --out-dir:

  apost_accuracy_<tag>.png : (1) rel-L2(t) of omega vs truth for every arm
                             (closure / bare / r3only); (2) energy spectra
                             E(k) at 3 horizon times (early/mid/final), truth
                             overlaid; (3) pattern correlation(t).
  apost_accuracy_<tag>.csv : t, per-arm rel-L2, per-arm pattern correlation.

Pattern correlation: corr(t) = <w_arm w_truth> / (|w_arm| |w_truth|) over the
domain (the standard anomaly-correlation with zero reference mean; both fields
are zero-mean here by periodicity of the vorticity).

Spectra use the radial-SUM convention of rollout_multistep_comparison.py
(sum within each |k| annulus / dk, physical wavenumbers) -- NOT the radial
mean, which flattens the k^-3 enstrophy-cascade slope.

Run on qlogin/qrsh (analysis-only, CPU is fine):
    python aposteriori_accuracy.py \
        --npz  ../diagnostics/Results/apost_b2_5em3/rollout_apost_<tag>.npz \
        --json ../diagnostics/Results/apost_b2_5em3/rollout_apost_<tag>.json \
        --out-dir ../diagnostics/Results/apost_b2_5em3
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                     # noqa: E402


def radial_sum_spectrum(field2d, Lx, Ly):
    Ny, Nx = field2d.shape
    kx = np.fft.fftfreq(Nx, d=Lx / Nx) * 2.0 * np.pi
    ky = np.fft.fftfreq(Ny, d=Ly / Ny) * 2.0 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    Kmag = np.sqrt(KX ** 2 + KY ** 2)
    fhat = np.fft.fft2(field2d) / (Nx * Ny)
    psd = np.abs(fhat) ** 2
    dk = max(2.0 * np.pi / Lx, 2.0 * np.pi / Ly)
    bins = np.arange(0.0, float(Kmag.max()) + dk, dk)
    sp, _ = np.histogram(Kmag.ravel(), bins=bins, weights=psd.ravel())
    kc = 0.5 * (bins[:-1] + bins[1:])
    return kc, sp / dk


def energy_spectrum(omega, Lx, Ly):
    kc, sp = radial_sum_spectrum(omega, Lx, Ly)
    k_safe = np.where(kc <= 0, np.inf, kc)
    return kc, 0.5 * sp / (k_safe ** 2)


def rel_l2(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)) / max(np.sqrt(np.mean(b ** 2)), 1e-30))


def pattern_corr(a, b):
    num = float(np.mean(a * b))
    den = float(np.sqrt(np.mean(a ** 2) * np.mean(b ** 2)))
    return num / max(den, 1e-30)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--npz', type=Path, required=True)
    p.add_argument('--json', type=Path, default=None,
                   help='matching .json (default: npz with suffix swapped)')
    p.add_argument('--Lx', type=float, default=None,
                   help='domain size (default: from the json config manifest '
                        'entry if present, else 4*pi)')
    p.add_argument('--out-dir', type=Path, default=None)
    p.add_argument('--tag', type=str, default=None)
    args = p.parse_args()

    jpath = args.json or args.npz.with_suffix('.json')
    meta = json.loads(jpath.read_text()) if jpath.exists() else {}
    tag = args.tag or args.npz.stem.replace('rollout_apost_', '')
    out_dir = args.out_dir or args.npz.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(args.npz)
    if 'truth_stack' not in d.files:
        raise SystemExit('npz has no truth_stack -- run the driver WITH the '
                         'truth arm for accuracy diagnostics (3a).')
    cp_steps = d['cp_steps']
    cp_times = d['cp_times']
    truth = d['truth_stack']
    arms = [a for a in ('closure', 'closure2', 'bare', 'r3only', 'r3anal')
            if f'{a}_stack' in d.files]
    Delta_T = float(meta.get('Delta_T', cp_times[1] / max(cp_steps[1], 1)
                             if len(cp_times) > 1 else 1.0))
    Lx = args.Lx or 4 * np.pi
    print(f'[3a] tag={tag}  arms={arms}  checkpoints={len(cp_steps)}  '
          f'Delta_T={Delta_T}')

    # align stacks: each arm may have stopped early (blowup); use avail lists
    rows = []
    corr = {a: [] for a in arms}
    rel = {a: [] for a in arms}
    times = {a: [] for a in arms}
    for a in arms:
        avail = d[f'{a}_cp_avail'] if f'{a}_cp_avail' in d.files else cp_steps
        stack = d[f'{a}_stack']
        idx_of = {int(s): i for i, s in enumerate(cp_steps)}
        for i, s in enumerate(np.asarray(avail)):
            j = idx_of.get(int(s))
            if j is None or j >= len(truth):
                continue
            times[a].append(float(cp_times[j]))
            rel[a].append(rel_l2(stack[i], truth[j]))
            corr[a].append(pattern_corr(stack[i], truth[j]))

    # csv (aligned on the union of checkpoint times)
    all_t = sorted({t for a in arms for t in times[a]})
    with open(out_dir / f'apost_accuracy_{tag}.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t'] + [f'relL2_{a}' for a in arms]
                   + [f'corr_{a}' for a in arms])
        for t in all_t:
            row = [t]
            for a in arms:
                row.append(rel[a][times[a].index(t)] if t in times[a] else '')
            for a in arms:
                row.append(corr[a][times[a].index(t)] if t in times[a] else '')
            w.writerow(row)

    # figure
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6))
    colors = {'bare': 'C0', 'r3only': 'C1', 'r3anal': 'C2', 'closure': 'C3',
              'closure2': 'C4'}
    for a in arms:
        ax[0].semilogy(times[a], rel[a], 'o-', ms=3, color=colors[a], label=a)
        ax[2].plot(times[a], corr[a], 'o-', ms=3, color=colors[a], label=a)
    ax[0].set_xlabel('t'); ax[0].set_ylabel('rel-L2 vs truth')
    ax[0].set_title('error growth'); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[2].set_xlabel('t'); ax[2].set_ylabel('pattern correlation')
    ax[2].set_ylim(None, 1.001); ax[2].set_title('pattern correlation(t)')
    ax[2].legend(); ax[2].grid(alpha=0.3)

    # spectra at 3 horizon times: ~25%, ~60%, final (common to all arms)
    n_cp = len(cp_steps)
    picks = sorted({max(1, int(0.25 * (n_cp - 1))),
                    max(1, int(0.6 * (n_cp - 1))), n_cp - 1})
    styles = ['--', '-.', '-']
    for st, j in zip(styles, picks):
        kc, Et = energy_spectrum(truth[j], Lx, Lx)
        ax[1].loglog(kc[1:], Et[1:], 'k' + st, lw=1.2, alpha=0.7,
                     label=f'truth t={cp_times[j]:.2f}')
        for a in arms:
            avail = list(np.asarray(d[f'{a}_cp_avail']
                                    if f'{a}_cp_avail' in d.files else cp_steps))
            if int(cp_steps[j]) in [int(x) for x in avail]:
                i = [int(x) for x in avail].index(int(cp_steps[j]))
                kc2, Ea = energy_spectrum(d[f'{a}_stack'][i], Lx, Lx)
                ax[1].loglog(kc2[1:], Ea[1:], st, color=colors[a], lw=1.0,
                             alpha=0.8)
    ax[1].set_xlabel('|k|'); ax[1].set_ylabel('E(k)')
    ax[1].set_title(f'energy spectra (colors = arms, styles = 3 times)')
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which='both')

    fig.suptitle(f'a-posteriori accuracy -- {tag}  '
                 f'(model {meta.get("model", "?")})')
    fig.tight_layout()
    fig.savefig(out_dir / f'apost_accuracy_{tag}.png', dpi=140)

    finals = {a: rel[a][-1] for a in arms if rel[a]}
    verdict = (f"[3a verdict] {tag}: final rel-L2 " +
               ", ".join(f"{a}={v:.3e}" for a, v in finals.items()))
    if 'bare' in finals and 'closure' in finals:
        verdict += (f"; closure improves over bare by "
                    f"{finals['bare']/max(finals['closure'],1e-30):.1f}x at "
                    f"t={all_t[-1]:.2f}")
    print(verdict)
    (out_dir / f'apost_accuracy_{tag}_verdict.txt').write_text(verdict + '\n')
    print(f'[3a] wrote apost_accuracy_{tag}.png/.csv in {out_dir}')


if __name__ == '__main__':
    main()
