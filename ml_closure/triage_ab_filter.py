"""
triage_ab_filter.py — P2 of the 2026-07-13 field-plot triage (standalone).

Question: the Pi_FF panels show suspicious VERTICAL STREAKS. Are they sinc
ringing of the LES filter's sharp axis-wise spectral cutoff (benign, a
filter-definition footnote), or a mask/dealias bug in compute_pi_ff
(training-data poison)?

Test: recompute Pi_FF for ONE developed-flow frame per member with
  A) the production filter  : Gaussian(width*dx_FR) o sharp cutoff |k|<pi/dx_LES o avg-pool
  B) an all-Gaussian filter : Gaussian(width*dx_FR) o Gaussian(Delta=2*dx_LES)  o avg-pool
(same pipeline otherwise, float64 end to end). If the streaks collapse under
(B), they are ringing of the sharp cutoff. If they persist, the bug is
elsewhere in compute_pi_ff => FLAG.

Also answers: is the streak present in omega_bar itself? (It should not be.)

Reuses the PRODUCTION code paths (qg.compute_pi_ff helpers + LESFilter) —
nothing in production is edited; the Gaussian variant only swaps the two
cutoff masks on a second LESFilter instance.

Usage (GPU, via piff_tool_job.sh):
  python triage_ab_filter.py --member-dir <...>/FPC-const --t-target 81.0 \
      --scale 4 --outdir triage_plot_20260713/ab_filter [--device cuda]
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

torch.set_default_dtype(torch.float64)   # hard rule: f64 for the recompute


def load_cpf():
    """Import the PRODUCTION compute_pi_ff module (read-only reuse)."""
    try:
        import qg.compute_pi_ff as cpf
        return cpf
    except ImportError:
        import qg
        p = Path(qg.__file__).parent / 'compute_pi_ff.py'
        spec = importlib.util.spec_from_file_location('cpf_prod', p)
        cpf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cpf)
        return cpf


def streak_amplitude(field2d, rows, cols, smooth=16):
    """Std of the detrended column-mean profile in a freestream band,
    normalized by the field std in the band. Isolates x-periodic vertical
    streaks from large-scale structure."""
    band = np.asarray(field2d, dtype=np.float64)[np.ix_(rows, cols)]
    prof = band.mean(axis=0)
    k = np.ones(smooth) / smooth
    trend = np.convolve(prof, k, mode='same')
    osc = prof - trend
    denom = max(band.std(), 1e-300)
    return float(osc.std() / denom), prof


def pick_freestream_rows(chi_obs_c, chi_sponge_c, frac=0.15):
    """Contiguous row band (height frac*Ny) with the least obstacle+sponge
    coverage — works for both cylinder and cape geometries."""
    cov = (chi_obs_c > 1e-3).mean(axis=1) + (chi_sponge_c > 1e-2).mean(axis=1)
    Ny = cov.size
    h = max(4, int(round(frac * Ny)))
    csum = np.insert(np.cumsum(cov), 0, 0.0)
    scores = csum[h:] - csum[:-h]
    i0 = int(np.argmin(scores))
    return np.arange(i0, i0 + h)


def compute_pi_variant(cpf, filt, omega_FR, deriv_FR, deriv_LES, dt, penalty,
                       chi_obs_FR, chi_sponge_FR, sponge_eta):
    """Exact production Pi_FF assembly (compute_pi_ff.py lines 179-190) with a
    pluggable filter instance."""
    from qg.solver.opt.basis import to_physical
    chi_obs_bar = filt.from_physical(chi_obs_FR) if chi_obs_FR is not None else None
    chi_sponge_bar = filt.from_physical(chi_sponge_FR) if chi_sponge_FR is not None else None
    src_FR = cpf._sources_on_grid(omega_FR, deriv_FR, dt, penalty,
                                  chi_obs_FR, chi_sponge_FR, sponge_eta)
    src_FR_f = filt.from_spectral(src_FR, output='physical')
    omega_bar = filt.from_physical(omega_FR)
    src_bar = cpf._sources_on_grid(omega_bar, deriv_LES, dt, penalty,
                                   chi_obs_bar, chi_sponge_bar, sponge_eta)
    pi = src_FR_f - to_physical(src_bar)
    return omega_bar.detach().cpu().numpy(), pi.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--member-dir', required=True)
    ap.add_argument('--t-target', type=float, default=81.0)
    ap.add_argument('--scale', type=int, default=4)
    ap.add_argument('--alpha', type=float, default=1.5)
    ap.add_argument('--gauss-delta-les', type=float, default=2.0,
                    help='Gaussian replacement width in dx_LES units')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    cpf = load_cpf()
    from qg._output.filter import LESFilter

    mdir = Path(args.member_dir)
    name = mdir.name
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(mdir / 'DNS_FR_params.yaml') as f:
        params = yaml.safe_load(f)
    z = np.load(mdir / 'DNS_FR.npz')
    times_raw = np.asarray(z['times'], dtype=np.float64)
    # frame k of the omega array: k=0 is the IC at t=0, k>=1 <-> times_raw[k-1]
    # (the audit-A off-by-one, fixed downstream in make_dataset_manifest.py)
    omega_mm = np.load(mdir / 'DNS_FR_omega.npy', mmap_mode='r')
    n_frames = omega_mm.shape[1]
    times_full = np.concatenate([[0.0], times_raw])
    if len(times_full) != n_frames:
        raise ValueError(f'{name}: {n_frames} frames vs {len(times_full)} times(+IC)')
    idx = int(np.argmin(np.abs(times_full - args.t_target)))
    t_used = float(times_full[idx])
    print(f'[{name}] frame {idx} at t={t_used:.2f} (target {args.t_target})')

    dev = args.device
    grid_params = params['grid']
    grid_FR, deriv_FR = cpf._build_grid_and_derivative(grid_params, device=dev)
    coarse = dict(grid_params)
    coarse['Nx'] = grid_FR.Nx // args.scale
    coarse['Ny'] = grid_FR.Ny // args.scale
    grid_LES, deriv_LES = cpf._build_grid_and_derivative(coarse, device=dev)
    print(f'[{name}] FR {grid_FR.Nx}x{grid_FR.Ny} -> LES {grid_LES.Nx}x{grid_LES.Ny}, '
          f'ftype {grid_FR.ftype}')

    omega_FR = torch.tensor(np.asarray(omega_mm[0, idx], dtype=np.float64),
                            dtype=grid_FR.ftype, device=grid_FR.device)
    dt = float(params['time']['dt'])
    penalty = float(params['pde']['penalty'])
    sponge_eta = float(params['bc'].get('sponge', 0))
    chi_obs_FR = torch.tensor(np.asarray(z['chi_obs'], dtype=np.float64),
                              dtype=grid_FR.ftype, device=grid_FR.device) \
        if ('chi_obs' in z.files and penalty > 0) else None
    chi_sponge_FR = torch.tensor(np.asarray(z['chi_sponge_ramp'], dtype=np.float64),
                                 dtype=grid_FR.ftype, device=grid_FR.device) \
        if ('chi_sponge_ramp' in z.files and sponge_eta > 0) else None

    # ---- filter A: production (sharp cutoff) ------------------------------- #
    filt_sharp = LESFilter(grid_FR, deriv_FR, scale=args.scale, width=args.alpha)
    # ---- filter B: Gaussian replaces the sharp cutoff ---------------------- #
    filt_gauss = LESFilter(grid_FR, deriv_FR, scale=args.scale, width=args.alpha)
    dx_LES = args.scale * grid_FR.dx
    Delta = args.gauss_delta_les * dx_LES
    gauss_cut = torch.exp(-grid_FR.ksq * (Delta ** 2) / 6)
    filt_gauss._cutoff_y = gauss_cut          # isotropic Gaussian in one go
    filt_gauss._cutoff_x = torch.ones((), dtype=grid_FR.ftype, device=grid_FR.device)

    ob_sharp, pi_sharp = compute_pi_variant(
        cpf, filt_sharp, omega_FR, deriv_FR, deriv_LES, dt, penalty,
        chi_obs_FR, chi_sponge_FR, sponge_eta)
    ob_gauss, pi_gauss = compute_pi_variant(
        cpf, filt_gauss, omega_FR, deriv_FR, deriv_LES, dt, penalty,
        chi_obs_FR, chi_sponge_FR, sponge_eta)
    diff = pi_sharp - pi_gauss

    # consistency: recomputed sharp Pi vs the stored piff_s4 product ---------- #
    stored = np.load(mdir / f'piff_s{args.scale}' / 'DNS_LES.npz')
    pi_stored = np.asarray(stored['pi_ff'][0, idx], dtype=np.float64)
    rel = np.linalg.norm(pi_sharp - pi_stored) / max(np.linalg.norm(pi_stored), 1e-300)
    print(f'[{name}] sharp recompute vs stored piff_s{args.scale} frame {idx}: '
          f'rel L2 = {rel:.3e} (expect ~f32-storage level)')

    # coarse masks for the freestream band ----------------------------------- #
    chi_obs_c = filt_sharp.from_physical(chi_obs_FR).detach().cpu().numpy()[0] \
        if chi_obs_FR is not None else np.zeros_like(pi_sharp[0] if pi_sharp.ndim == 3 else pi_sharp)
    chi_sp_c = filt_sharp.from_physical(chi_sponge_FR).detach().cpu().numpy()[0] \
        if chi_sponge_FR is not None else np.zeros_like(chi_obs_c)

    # squeeze possible leading batch dim
    def sq(a):
        return a[0] if a.ndim == 3 else a
    pi_sharp, pi_gauss, diff = sq(pi_sharp), sq(pi_gauss), sq(diff)
    ob_sharp, ob_gauss = sq(ob_sharp), sq(ob_gauss)

    Ny, Nx = pi_sharp.shape
    rows = pick_freestream_rows(chi_obs_c, chi_sp_c)
    cols = np.arange(int(0.05 * Nx), int(0.85 * Nx))
    res = {}
    for tag, f2d in [('pi_sharp', pi_sharp), ('pi_gauss', pi_gauss),
                     ('omega_bar', ob_sharp)]:
        amp, prof = streak_amplitude(f2d, rows, cols)
        res[tag] = amp
        res[f'{tag}_profile'] = prof
    print(f'[{name}] streak amplitude (detrended column-mean std / band std):')
    print(f'          Pi sharp filter    : {res["pi_sharp"]:.4f}')
    print(f'          Pi Gaussian filter : {res["pi_gauss"]:.4f}')
    print(f'          omega_bar          : {res["omega_bar"]:.4f}')
    ratio = res['pi_gauss'] / max(res['pi_sharp'], 1e-300)
    verdict = ('RINGING (benign): streaks collapse under the Gaussian filter'
               if ratio < 0.5 else
               'PERSISTS: streaks survive the Gaussian filter -> suspect a bug (FLAG)')
    print(f'[{name}] gauss/sharp streak ratio = {ratio:.3f}  =>  {verdict}')

    # ---- figure: stacked panels, identical x-extent ------------------------- #
    Lx, Ly = float(grid_params['Lx']), float(grid_params['Ly'])
    lt = float(np.percentile(np.abs(pi_sharp), 99.0))
    vmax = float(np.abs(pi_sharp).max())
    norm = SymLogNorm(linthresh=max(lt, 1e-12), vmin=-vmax, vmax=vmax, base=10)
    ovmax = float(np.percentile(np.abs(ob_sharp), 99.5))
    fig, axs = plt.subplots(4, 1, figsize=(7.5, 26), constrained_layout=True)
    panels = [
        (ob_sharp, f'filtered vorticity omega_bar  ({name}, t={t_used:.1f}) — '
                   'linear scale, no streaks expected here', dict(vmin=-ovmax, vmax=ovmax)),
        (pi_sharp, 'Pi_FF, PRODUCTION sharp-cutoff filter — symlog', dict(norm=norm)),
        (pi_gauss, f'Pi_FF, all-Gaussian filter (Delta={args.gauss_delta_les:g} dx_LES) — same symlog',
         dict(norm=norm)),
        (diff, 'difference (sharp - Gaussian) — same symlog', dict(norm=norm)),
    ]
    yb0, yb1 = rows[0] / Ny * Ly, (rows[-1] + 1) / Ny * Ly
    for ax, (f2d, ttl, kw) in zip(axs, panels):
        im = ax.imshow(f2d, cmap='seismic', origin='lower',
                       extent=[0, Lx, 0, Ly], aspect='equal', **kw)
        ax.set_title(ttl, fontsize=10)
        ax.axhline(yb0, color='k', ls=':', lw=0.6)
        ax.axhline(yb1, color='k', ls=':', lw=0.6)
        fig.colorbar(im, ax=ax, fraction=0.035)
    axs[0].text(0.01, 0.02, 'dotted lines: freestream band used for the streak metric',
                transform=axs[0].transAxes, fontsize=7)
    fp = outdir / f'ab_filter_{name}_t{t_used:.1f}.png'
    fig.savefig(fp, dpi=130)
    plt.close(fig)
    print(f'[{name}] figure -> {fp}')

    np.savez(outdir / f'ab_filter_{name}.npz',
             pi_sharp=pi_sharp.astype(np.float64), pi_gauss=pi_gauss.astype(np.float64),
             omega_bar=ob_sharp.astype(np.float64), omega_bar_gauss=ob_gauss.astype(np.float64),
             t=t_used, frame=idx, rel_l2_vs_stored=rel,
             streak_pi_sharp=res['pi_sharp'], streak_pi_gauss=res['pi_gauss'],
             streak_omega=res['omega_bar'], gauss_delta_les=args.gauss_delta_les,
             band_rows=rows, band_cols=cols)
    print(f'[{name}] VERDICT: {verdict}')


if __name__ == '__main__':
    main()
