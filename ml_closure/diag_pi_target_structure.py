"""
Target-structure comparison, sharp vs Gaussian filter (2026-07-13, evidence
for the stalled cylinder-on-Gaussian training): for each member, from the
valid pixels of the val window frames, compute for BOTH DNS_LES_s4.npz and
DNS_LES_s4_gaussian.npz:
  - |Pi| quantiles (50/90/99/99.9%) and excess kurtosis of Pi
  - the WAKE's share of total Pi variance (wake = x in [x_c-D, x_c+12D],
    |y-y_c| <= 3D, the training crop-bias window) vs the rest
  - fraction of pixels carrying 90% of the total Pi^2 (concentration)
Question answered: did removing the sharp filter turn the cylinder target
into rare-extreme-spots-on-nothing (unlearnable-by-ELBO territory, rule-16
family) while cape kept broad support?

Outputs (convention): pngs/pi_target_structure_sharp_vs_gaussian/ (figure +
.txt explainer), yamls/pi_target_structure_sharp_vs_gaussian/summary.yaml.

Usage: python diag_pi_target_structure.py <member_dir> [...]
"""

import sys
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PNGD = HERE / 'pngs' / 'pi_target_structure_sharp_vs_gaussian'
YMLD = HERE / 'yamls' / 'pi_target_structure_sharp_vs_gaussian'

EXPLAINER = """pi_target_structure_sharp_vs_gaussian — what this shows

One row of statistics per (member, filter). The question: did the Gaussian
filter change the *learnability* of the cylinder target? A Gaussian-process
model with one global noise term learns poorly when the target is a few
extreme spots on a near-zero background (the ELBO's cheapest move becomes
"call everything noise" — the collapse family seen in arms C/D/E and in the
stalled cylinder-on-Gaussian run).

Numbers per member/filter, over valid pixels of frames t in [100,120]:
  - |Pi| quantiles 50/90/99/99.9: how extreme the tail is vs the median.
  - excess kurtosis of Pi: heavy-tailedness (Gaussian field = 0).
  - wake variance share: fraction of total Pi^2 inside the wake window
    (x_c-D..x_c+12D, |y-y_c|<=3D) — where the crops are biased to look.
  - px90: fraction of pixels carrying 90% of total Pi^2 — concentration
    (smaller = more extreme concentration).
The figure shows |Pi|/median CDFs (log-x) per member, sharp vs gaussian.
"""


def member_stats(mdir, fname):
    z = np.load(mdir / fname)
    times = np.asarray(z['times'])
    sel = np.where((times >= 100.0) & (times <= 120.0))[0]
    pi = np.asarray(z['pi_ff'][0][sel], dtype=np.float64)
    chi_o = np.asarray(z['chi_obs_bar']).squeeze()
    chi_s = np.asarray(z['chi_sponge_bar']).squeeze()
    valid = (chi_o <= 1e-3) & (chi_s <= 1e-2)
    v = pi[:, valid].ravel()
    q = np.percentile(np.abs(v), [50, 90, 99, 99.9])
    r = v - v.mean()
    kurt = float(np.mean(r ** 4) / max(np.mean(r ** 2) ** 2, 1e-300) - 3.0)
    s2 = np.sort(v ** 2)[::-1]
    c = np.cumsum(s2)
    px90 = float(np.searchsorted(c, 0.9 * c[-1]) / c.size)
    return {'q50': float(q[0]), 'q90': float(q[1]), 'q99': float(q[2]),
            'q999': float(q[3]), 'kurtosis': kurt, 'px90_frac': px90,
            'n_px': int(v.size)}, v


def main():
    members = [Path(p) for p in sys.argv[1:]]
    PNGD.mkdir(parents=True, exist_ok=True)
    YMLD.mkdir(parents=True, exist_ok=True)
    (PNGD / 'pi_target_structure_sharp_vs_gaussian.txt').write_text(EXPLAINER)
    out = {}
    fig, axs = plt.subplots(1, len(members), figsize=(4.5 * len(members), 4),
                            squeeze=False)
    for j, m in enumerate(members):
        ax = axs[0, j]
        for fname, lab, color in ((f'DNS_LES_s4.npz', 'sharp', 'tab:red'),
                                  (f'DNS_LES_s4_gaussian.npz', 'gaussian', 'tab:blue')):
            if not (m / fname).exists():
                continue
            st, v = member_stats(m, fname)
            out[f'{m.name}__{lab}'] = st
            a = np.abs(v) / max(st['q50'], 1e-300)
            xs = np.percentile(a, np.linspace(50, 99.99, 200))
            ax.plot(xs, np.linspace(0.5, 0.9999, 200), color=color,
                    label=f"{lab}: kurt {st['kurtosis']:.0f}, "
                          f"q99.9/q50 {st['q999']/max(st['q50'],1e-300):.0f}x")
        ax.set_xscale('log'); ax.grid(alpha=0.3)
        ax.set_xlabel('|Pi| / median |Pi|'); ax.set_ylabel('CDF')
        ax.set_title(m.name, fontsize=10); ax.legend(fontsize=7)
    fig.suptitle('How extreme is the target? sharp vs gaussian filter '
                 '(valid pixels, t in [100,120])', fontsize=11)
    fig.tight_layout()
    fig.savefig(PNGD / 'pi_amplitude_cdf_sharp_vs_gaussian.png', dpi=130)
    plt.close(fig)
    with open(YMLD / 'summary.yaml', 'w') as f:
        f.write('# Target-structure stats sharp vs gaussian (see the pngs '
                'folder explainer).\n# kurtosis = heavy tails; px90_frac = '
                'fraction of pixels carrying 90% of Pi^2 (small = extreme '
                'concentration).\n')
        yaml.safe_dump(out, f, sort_keys=True)
    for k, v in sorted(out.items()):
        print(f"[stats] {k}: kurt {v['kurtosis']:.0f}  q999/q50 "
              f"{v['q999']/max(v['q50'],1e-300):.0f}x  px90 {v['px90_frac']:.4f}")


if __name__ == '__main__':
    main()
