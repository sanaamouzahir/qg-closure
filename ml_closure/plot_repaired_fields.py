"""Upstream input-repair verification (Sanaa 2026-07-13 night): side-by-side
ORIGINAL vs REPAIRED model inputs (omega*, u*, v*) + difference, full frame,
with the mask/repair line (x_c - 1.5D) and blend zone drawn. Purpose: eyeball
that the repair replaces the sponge-reflection artifact with clean freestream
and introduces NO new seam/error. A few members x 2 frames. CPU only.
Outputs: pngs/upstream_repair_verification/ + explainer txt."""
import sys, copy
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import dataset_piff as dp

HERE = Path(__file__).resolve().parent
OUT = HERE / 'pngs' / 'upstream_repair_verification'
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'upstream_repair_verification.txt').write_text(
    "upstream_repair_verification — original vs repaired model inputs.\n\n"
    "Rows: filtered vorticity omega*, u*, v*. Columns: ORIGINAL (what the\n"
    "model used to see: includes the sponge-reflection artifact upstream),\n"
    "REPAIRED (upstream replaced by analytic freestream omega*=0, u*=1, v*=0\n"
    "via an 8-px blend), DIFFERENCE (original - repaired: should be nonzero\n"
    "ONLY left of the dashed line = exactly the artifact we removed).\n"
    "Dashed line: x_c - 1.5 D (mask + repair edge); dotted: blend start.\n"
    "Inputs: DNS_LES_s4_gaussian_jonly.npz per member. Purpose: confirm the\n"
    "repair is clean before back-porting it to all stored runs.\n")

conf_gjs = dp.load_conf(HERE / ('conf_piff_fpc_gjs.yaml'))
for mdir in map(Path, sys.argv[1:]):
    is_cape = 'Cape' in mdir.name
    base = dp.load_conf(HERE / ('conf_piff_cape_gjs.yaml' if is_cape else 'conf_piff_fpc_gjs.yaml'))
    c_rep = copy.deepcopy(base); c_rep['data']['runs'] = [str(mdir)]
    c_orig = copy.deepcopy(c_rep); c_orig['data'].pop('upstream_mask_x_lo_D')
    r_rep = dp.RunData(str(mdir), c_rep)
    r_org = dp.RunData(str(mdir), c_orig)
    x_edge = r_rep.x_c - 1.5 * r_rep.D
    for tt in (105.0, 116.0):
        fi = int(np.argmin(np.abs(r_rep.times - tt)))
        xo = r_org.full_frame(fi)[0].numpy()
        xr = r_rep.full_frame(fi)[0].numpy()
        fig, axs = plt.subplots(3, 3, figsize=(15, 11))
        names = ['omega*', 'u*', 'v*']
        for row in range(3):
            fo, fr = xo[row], xr[row]
            vmax = np.percentile(np.abs(fo), 99.5)
            for col, (f2d, ttl) in enumerate((
                    (fo, f'{names[row]} ORIGINAL'),
                    (fr, f'{names[row]} REPAIRED'),
                    (fo - fr, f'{names[row]} difference'))):
                ax = axs[row, col]
                vm = vmax if col < 2 else max(np.abs(fo - fr).max(), 1e-12)
                im = ax.imshow(f2d, cmap='seismic', vmin=-vm, vmax=vm,
                               origin='lower', aspect='equal',
                               extent=[0, r_rep.Lx, 0, r_rep.Ly])
                ax.axvline(x_edge, color='k', ls='--', lw=0.8)
                ax.axvline(x_edge - 8 * r_rep.dx, color='k', ls=':', lw=0.6)
                ax.set_title(ttl, fontsize=9)
                fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"{mdir.name}  t={r_rep.times[fi]:.2f}  (u* shown around "
                     f"its freestream value 1)", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / f'{mdir.name}_repair_check_t{r_rep.times[fi]:.0f}.png',
                    dpi=120)
        plt.close(fig)
        print(f"[repair-check] {mdir.name} t={r_rep.times[fi]:.2f} done")
print("[repair-check] all done ->", OUT)
