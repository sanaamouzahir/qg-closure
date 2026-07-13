"""
Field portraits of the GAUSSIAN-FILTERED training data (Sanaa request,
2026-07-13): for each member, one figure showing the model's INPUTS as the
model sees them — filtered vorticity omega_bar, filtered velocities ubar/vbar,
the obstacle+sponge exclusion picture — and the OUTPUT it must learn, the
closure field Pi, on a symlog color scale (linthresh = 99th pct |Pi|).
All from DNS_LES_s4_gaussian.npz (the all-Gaussian filter rebuild).

Output (diagnostics convention 2026-07-13):
  pngs/gaussian_filtered_inputs_and_closure_field/<member>_gaussian_inputs_and_closure_t<T>.png
  + the folder's plain-English explainer .txt (written once).

CPU-only, numpy+matplotlib. Usage:
  python plot_gaussian_fields.py <member_dir> [...] [--t-target 81.0]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / 'pngs' / 'gaussian_filtered_inputs_and_closure_field'

EXPLAINER = """gaussian_filtered_inputs_and_closure_field — what these figures show

One figure per simulation member, all fields from DNS_LES_s4_gaussian.npz
(the coarse data produced with the ALL-GAUSSIAN filter that replaced the
sharp spectral cutoff on 2026-07-13 — the filter change that removed the
vertical ringing streaks).

Panels, left to right:
  1. filtered vorticity omega_bar — the model's main input: the coarse
     picture of the flow after the small eddies are smoothed away.
  2. filtered x-velocity ubar and 3. filtered y-velocity vbar — the other
     two flow inputs, derived from omega_bar exactly as the coarse solver
     would (psi = inverse Laplacian, u = -dpsi/dy, v = +dpsi/dx, inlet
     speed U(t) as the mean mode).
  4. exclusion picture — obstacle mask (dark) and sponge zone (shaded):
     the pixels the model never trains on and is never scored on.
  5. the closure field Pi — the OUTPUT the model must learn: the push of
     the discarded small eddies on the coarse flow,
         Pi = filter(J + obstacle penalty + sponge)_fine
            - (same terms)(filtered fields),
     shown on a SYMLOG color scale (linear inside the 99th percentile of
     |Pi|, logarithmic beyond) so both the wake structure and the extremes
     are visible at once — a linear-to-max scale hides the wake entirely.

Inputs: the member's DNS_LES_s4_gaussian.npz. Outputs: these pngs.
Purpose: visual confirmation that the Gaussian-filter data is clean (no
ringing streaks) and a reference gallery of what the model actually sees.
Frame chosen at t ~ 81 (developed flow), title states exact t, U(t), Re(t).
"""


def plot_member(mdir, t_target):
    z = np.load(mdir / 'DNS_LES_s4_gaussian.npz')
    times = np.asarray(z['times'])
    fi = int(np.argmin(np.abs(times - t_target)))
    om, u, v, pi = (np.asarray(z[k][0][fi], dtype=np.float64)
                    for k in ('omega_bar', 'ubar', 'vbar', 'pi_ff'))
    chi_o = np.asarray(z['chi_obs_bar']).squeeze()
    chi_s = np.asarray(z['chi_sponge_bar']).squeeze()
    U, Re, t = float(z['U_snap'][fi]), float(z['Re_snap'][fi]), float(times[fi])
    meta = json.loads(str(z['meta'])) if 'meta' in z.files else {}

    fig, axs = plt.subplots(1, 5, figsize=(22, 4.4))
    for ax, f2d, ttl in zip(axs[:3], (om, u, v),
                            (f'filtered vorticity  omega_bar',
                             f'filtered x-velocity  ubar',
                             f'filtered y-velocity  vbar')):
        vmax = np.percentile(np.abs(f2d), 99.5)
        im = ax.imshow(f2d, cmap='seismic', vmin=-vmax, vmax=vmax,
                       origin='lower', aspect='equal')
        ax.set_title(ttl, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)

    excl = (chi_o > 1e-3).astype(float) * 2.0 + (chi_s > 1e-2).astype(float)
    im = axs[3].imshow(excl, cmap='Greys', origin='lower', aspect='equal',
                       vmin=0, vmax=2)
    axs[3].set_title('excluded pixels: obstacle (dark), sponge (grey)', fontsize=10)
    fig.colorbar(im, ax=axs[3], fraction=0.046)

    lin = max(np.percentile(np.abs(pi), 99.0), 1e-12)
    im = axs[4].imshow(pi, cmap='seismic', origin='lower', aspect='equal',
                       norm=SymLogNorm(linthresh=lin, vmin=-np.abs(pi).max(),
                                       vmax=np.abs(pi).max(), base=10))
    axs[4].set_title('closure field Pi (symlog)', fontsize=10)
    fig.colorbar(im, ax=axs[4], fraction=0.046)

    fig.suptitle(f"{mdir.name} — GAUSSIAN-filtered data, t={t:.2f}, "
                 f"U={U:.3f}, Re={Re:.0f}"
                 + (f"  [{meta.get('filter','')[:60]}...]" if meta else ''),
                 fontsize=11)
    fig.tight_layout()
    out = OUTDIR / f"{mdir.name}_gaussian_inputs_and_closure_t{t:.0f}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[plot] {out.name}  (|Pi| p99 {lin:.3e}, max {np.abs(pi).max():.3e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('member_dirs', nargs='+')
    ap.add_argument('--t-target', type=float, default=81.0)
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / 'gaussian_filtered_inputs_and_closure_field.txt').write_text(EXPLAINER)
    for m in args.member_dirs:
        plot_member(Path(m), args.t_target)
    print(f"[plot] done — {len(args.member_dirs)} members in {OUTDIR}")


if __name__ == '__main__':
    main()
