"""
Pi decomposition under the GAUSSIAN filter (Sanaa order 2026-07-13): split the
full-RHS commutator into its JACOBIAN part and its BODY part (obstacle penalty
+ sponge), and ask which one owns the extreme concentration that stalled the
cylinder training (full target: 90% of energy in 0.08% of pixels).

    Pi_full = filter(J + Brinkman + Sponge)_fine - (J + Brinkman + Sponge)(filtered)
    Pi_J    = filter(J)_fine                    - J(filtered fields)
    Pi_body = Pi_full - Pi_J                     (penalty + sponge commutator)

Same production code path as the rebuild (qg.compute_pi_ff internals), same
all-Gaussian filter (P2-validated), float64, CPU. Frames: every other frame of
t in [100,120] (the val window). Stats per part: excess kurtosis, |.| quantiles,
px90 (fraction of pixels carrying 90% of the part's energy), wake share.

DECISION RULE (pre-registered): if px90(Pi_J) >= 10 x px90(Pi_full) on the
cylinder members (i.e. the Jacobian part has an order of magnitude broader
support), the cylinder relaunch trains on JACOBIAN-ONLY gaussian targets.

Outputs: pngs/pi_jacobian_vs_body_split_gaussian/ (field panels + CDF + .txt),
yamls/pi_jacobian_vs_body_split_gaussian/summary.yaml, and
<member>/DNS_LES_s4_gaussian_jonly_valwin.npz (the Pi_J frames computed here,
val window only — reusable for a quick-look, NOT a training file).

Usage: python diag_pi_jacobian_split.py <member_dir> [...]
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

from compute_pi_ff_gaussian_rebuild import load_cpf, make_all_gaussian_filter

HERE = Path(__file__).resolve().parent
PNGD = HERE / 'pngs' / 'pi_jacobian_vs_body_split_gaussian'
YMLD = HERE / 'yamls' / 'pi_jacobian_vs_body_split_gaussian'

EXPLAINER = """pi_jacobian_vs_body_split_gaussian — what these figures show

The closure target Pi (gaussian filter) is the commutator of the FULL
right-hand side: turbulence (Jacobian) + obstacle penalty + sponge. The
stalled cylinder training suggested the target is unlearnably concentrated
(90% of energy in 0.08% of pixels). These figures split Pi into:
  Pi_J    — the turbulence-only part: filter(J(psi,omega))_fine - J(filtered)
  Pi_body — the rest (obstacle penalty + sponge commutator) = Pi_full - Pi_J
Panels per member: Pi_full | Pi_J | Pi_body, same symlog scale, one developed
frame; plus the |Pi|/median CDFs and the concentration stats per part.
Inputs: DNS_FR.npz (fine fields) + the all-Gaussian filter. Outputs: these
pngs + yamls/.../summary.yaml + a val-window Pi_J quick-look npz per member.
Purpose: decide (pre-registered rule in the script header) whether the
cylinder relaunch should train on the Jacobian-only target.
"""


def part_stats(frames, valid):
    v = np.concatenate([f[valid].ravel() for f in frames]).astype(np.float64)
    q = np.percentile(np.abs(v), [50, 99, 99.9])
    r = v - v.mean()
    kurt = float(np.mean(r ** 4) / max(np.mean(r ** 2) ** 2, 1e-300) - 3.0)
    s2 = np.sort(v ** 2)[::-1]
    c = np.cumsum(s2)
    px90 = float(np.searchsorted(c, 0.9 * c[-1]) / c.size)
    return {'q50': float(q[0]), 'q99': float(q[1]), 'q999': float(q[2]),
            'kurtosis': kurt, 'px90_frac': px90,
            'var': float(v.var()), 'n_px': int(v.size)}


def main():
    members = [Path(p) for p in sys.argv[1:]]
    PNGD.mkdir(parents=True, exist_ok=True)
    YMLD.mkdir(parents=True, exist_ok=True)
    (PNGD / 'pi_jacobian_vs_body_split_gaussian.txt').write_text(EXPLAINER)
    cpf = load_cpf()
    from qg._output.filter import LESFilter
    from qg.solver.opt.basis import to_physical
    out = {}
    for mdir in members:
        z = np.load(mdir / 'DNS_FR.npz')
        with open(mdir / 'DNS_FR_params.yaml') as f:
            params = yaml.safe_load(f)
        gz = np.load(mdir / 'DNS_LES_s4_gaussian.npz')
        times = np.asarray(gz['times'])
        sel = np.where((times >= 100.0) & (times <= 120.0))[0][::2]
        omega_FR_np = z['omega_FR']
        chi_obs_np = z.get('chi_obs') if 'chi_obs' in z.files else None
        chi_sponge_np = z.get('chi_sponge_ramp') if 'chi_sponge_ramp' in z.files else None
        dt = float(params['time']['dt'])
        penalty = float(params['pde']['penalty'])
        sponge_eta = float(params['bc'].get('sponge', 0))
        gp = params['grid']
        _, _, Ny, Nx = omega_FR_np.shape
        grid_FR, deriv_FR = cpf._build_grid_and_derivative(gp, device='cpu')
        cp = dict(gp); cp['Nx'] = Nx // 4; cp['Ny'] = Ny // 4
        _, deriv_LES = cpf._build_grid_and_derivative(cp, device='cpu')
        filt = make_all_gaussian_filter(LESFilter, grid_FR, deriv_FR, 4, 1.5, 2.0)
        chi_obs_FR = torch.tensor(chi_obs_np, dtype=grid_FR.ftype).squeeze() \
            if chi_obs_np is not None and penalty > 0 else None
        chi_sponge_FR = torch.tensor(chi_sponge_np, dtype=grid_FR.ftype).squeeze() \
            if chi_sponge_np is not None and sponge_eta > 0 else None
        chi_obs_bar = filt.from_physical(chi_obs_FR) if chi_obs_FR is not None else None
        chi_sponge_bar = filt.from_physical(chi_sponge_FR) if chi_sponge_FR is not None else None

        full_f, j_f = [], []
        with torch.no_grad():
            for fi in sel:
                om = torch.tensor(omega_FR_np[0, fi], dtype=grid_FR.ftype)
                om_bar = filt.from_physical(om)
                sf_full = cpf._sources_on_grid(om, deriv_FR, dt, penalty,
                                               chi_obs_FR, chi_sponge_FR, sponge_eta)
                sb_full = cpf._sources_on_grid(om_bar, deriv_LES, dt, penalty,
                                               chi_obs_bar, chi_sponge_bar, sponge_eta)
                pi_full = (filt.from_spectral(sf_full, output='physical')
                           - to_physical(sb_full)).numpy()
                sf_j = cpf._sources_on_grid(om, deriv_FR, dt, 0.0, None, None, 0.0)
                sb_j = cpf._sources_on_grid(om_bar, deriv_LES, dt, 0.0, None, None, 0.0)
                pi_j = (filt.from_spectral(sf_j, output='physical')
                        - to_physical(sb_j)).numpy()
                # filter/to_physical may carry a leading batch axis — always 2-D here
                full_f.append(np.asarray(pi_full).squeeze())
                j_f.append(np.asarray(pi_j).squeeze())

        chi_o_c = chi_obs_bar.numpy().squeeze() if chi_obs_bar is not None else np.zeros_like(full_f[0])
        chi_s_c = chi_sponge_bar.numpy().squeeze() if chi_sponge_bar is not None else np.zeros_like(full_f[0])
        valid = (chi_o_c <= 1e-3) & (chi_s_c <= 1e-2)
        body_f = [a - b for a, b in zip(full_f, j_f)]
        st = {'full': part_stats(full_f, valid), 'jacobian': part_stats(j_f, valid),
              'body': part_stats(body_f, valid)}
        st['jacobian_var_share'] = st['jacobian']['var'] / max(st['full']['var'], 1e-300)
        out[mdir.name] = st
        np.savez_compressed(mdir / 'DNS_LES_s4_gaussian_jonly_valwin.npz',
                            pi_j=np.stack(j_f).astype(np.float32)[None],
                            frames=sel, times=times[sel],
                            meta=json.dumps({'note': 'val-window quick-look, NOT training data'}))

        k = len(sel) // 2
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
        vmax = np.abs(full_f[k]).max()
        lin = max(np.percentile(np.abs(full_f[k]), 99.0), 1e-12)
        for ax, f2d, ttl in zip(axs, (full_f[k], j_f[k], body_f[k]),
                                ('Pi FULL (J + body forces)', 'Pi JACOBIAN only',
                                 'Pi BODY (penalty + sponge)')):
            im = ax.imshow(f2d, cmap='seismic', origin='lower', aspect='equal',
                           norm=SymLogNorm(linthresh=lin, vmin=-vmax, vmax=vmax, base=10))
            s = out[mdir.name]['full' if 'FULL' in ttl else ('jacobian' if 'JACOBIAN' in ttl else 'body')]
            ax.set_title(f"{ttl}\nkurt {s['kurtosis']:.0f}, px90 {s['px90_frac']*100:.3f}%",
                         fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"{mdir.name} — gaussian filter, t={times[sel[k]]:.1f} "
                     f"(same symlog scale all panels)", fontsize=11)
        fig.tight_layout()
        fig.savefig(PNGD / f'{mdir.name}_pi_full_vs_jacobian_vs_body_t{times[sel[k]]:.0f}.png',
                    dpi=130)
        plt.close(fig)
        print(f"[split] {mdir.name}: full px90 {st['full']['px90_frac']*100:.4f}% kurt "
              f"{st['full']['kurtosis']:.0f} | J px90 {st['jacobian']['px90_frac']*100:.4f}% "
              f"kurt {st['jacobian']['kurtosis']:.0f} | body px90 "
              f"{st['body']['px90_frac']*100:.4f}% | J var share {st['jacobian_var_share']:.3f}")

    with open(YMLD / 'summary.yaml', 'w') as f:
        f.write('# Pi decomposition (gaussian filter): full vs jacobian-only vs '
                'body (penalty+sponge).\n# px90_frac = fraction of pixels carrying '
                '90% of that part\'s energy (small = concentrated).\n# Decision rule '
                'in diag_pi_jacobian_split.py header.\n')
        yaml.safe_dump(out, f, sort_keys=True)
    print('[split] done')


if __name__ == '__main__':
    main()
