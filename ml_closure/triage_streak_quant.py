"""Quantify the vertical streak through the obstacle column from the saved
A/B fields (P2 follow-up): median |Pi| in the obstacle-column band, restricted
to rows far from the wake/obstacle, sharp vs Gaussian filter; same for
omega_bar (relative to its own field scale). CPU, reads the ab_filter npz."""
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / 'triage_plot_20260713' / 'ab_filter'
# obstacle x-centers (domain fraction): cylinder at x_c ~ Lx/2, cape at ~x=5/25
COL_FRAC = {'FPC-const': 0.5, 'FPCape-const': 0.2}
# rows far from the obstacle/wake (domain fraction bands)
ROW_FRAC = {'FPC-const': (0.70, 0.92), 'FPCape-const': (0.55, 0.90)}

for name, cf in COL_FRAC.items():
    z = np.load(OUT / f'ab_filter_{name}.npz', allow_pickle=True)
    ps, pg, ob = z['pi_sharp'], z['pi_gauss'], z['omega_bar']
    obg = z['omega_bar_gauss']
    Ny, Nx = ps.shape
    c0, c1 = int((cf - 0.04) * Nx), int((cf + 0.04) * Nx)
    r0, r1 = int(ROW_FRAC[name][0] * Ny), int(ROW_FRAC[name][1] * Ny)
    # background: same rows, columns well away from the obstacle column
    b0, b1 = int((cf + 0.20) * Nx), int((cf + 0.35) * Nx)
    for tag, f in [('Pi sharp', ps), ('Pi gauss', pg)]:
        col = np.median(np.abs(f[r0:r1, c0:c1]))
        bg = np.median(np.abs(f[r0:r1, b0:b1]))
        print(f'[{name}] {tag}: median |Pi| on obstacle column = {col:.4e}, '
              f'background = {bg:.4e}, ratio = {col / max(bg, 1e-300):.1f}x')
    for tag, f in [('omega sharp', ob), ('omega gauss', obg)]:
        col = np.median(np.abs(f[r0:r1, c0:c1]))
        bg = np.median(np.abs(f[r0:r1, b0:b1]))
        sc = np.percentile(np.abs(f), 99.5)
        print(f'[{name}] {tag}: column/background = {col / max(bg, 1e-300):.2f}x, '
              f'column/fieldscale = {col / sc:.2e}')
print('[streak-quant] done')
sys.exit(0)
