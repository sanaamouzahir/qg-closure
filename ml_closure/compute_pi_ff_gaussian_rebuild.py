"""
Gaussian-filter Pi_FF data rebuild (Sanaa order, 2026-07-13 chat).

For one member run dir, recompute the LES products with an ALL-GAUSSIAN filter
(the sharp axis-wise cutoff of the production composite replaced by a second
Gaussian of width gauss_delta_les * dx_LES — exactly the variant validated in
the same-day P2 streak triage, where the sharp-filter recompute matched the
stored training product to rel-L2 ~3e-8 and the streaks collapsed ~100x).

Target definition (triple-checked against qg/compute_pi_ff.py:99-190, same
here, only the filter changes):
    Pi = filter[ J(psi,omega) + Brinkman + Sponge ]_fine
       - [ J(psi_bar,omega_bar) + Brinkman(chi_bar,..) + Sponge(..) ]_coarse
i.e. the commutator of the FULL RHS source (Jacobian + obstacle penalty +
sponge), NOT the bare Jacobian commutator — flagged to Sanaa 2026-07-13.
Inputs are the true filtered fields: omega_bar = filter(omega_fine);
ubar/vbar from psi_bar = inv_lap(omega_bar) + U(t) mean mode, which equals
filter(u_fine) exactly for a spectral-multiplier filter.

Output per scale, in the SAME member folder:  DNS_LES_s<scale>_gaussian.npz
with the full canonical key set (omega_bar, pi_ff, ubar, vbar, times, U_snap,
Re_snap, chi_obs_bar, chi_sponge_bar, meta) — times/U_snap/Re_snap copied from
the existing canonical DNS_LES_s<scale>.npz (frame-count asserted).

CPU by design (Sanaa: "cpu job NOT a gpu job"). float64 compute (default
grid dtype), float32 storage — branch precision policy.

Usage (from ml_closure/, via scripts/sge/gaussian_rebuild_job.sh):
    python compute_pi_ff_gaussian_rebuild.py <member_dir> [--scales 4 2 8]
        [--alpha 1.5] [--gauss-delta-les 2.0] [--name DNS]
"""

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from make_dataset_manifest import compute_uv_from_omega

torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))


def load_cpf():
    """Import the PRODUCTION compute_pi_ff module (read-only reuse — same
    loader as triage_ab_filter.py)."""
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


def make_all_gaussian_filter(LESFilter, grid, deriv, scale, alpha, delta_les):
    """Production composite with the sharp cutoff masks disabled and a second
    Gaussian of width delta_les*dx_LES multiplied in (P2-validated variant)."""
    filt = LESFilter(grid, deriv, scale=scale, width=alpha)
    sigma_les = float(delta_les) * scale * grid.dx
    filt._gaussian = filt._gaussian * torch.exp(-grid.ksq * (sigma_les ** 2) / 6)
    filt._cutoff_x = torch.ones_like(filt._cutoff_x)
    filt._cutoff_y = torch.ones_like(filt._cutoff_y)
    return filt


def main():
    ap = argparse.ArgumentParser(description="All-Gaussian Pi_FF rebuild for one member")
    ap.add_argument('member_dir')
    ap.add_argument('--scales', type=int, nargs='+', default=[4, 2, 8],
                    help='training scale 4 FIRST so the A/B-critical file lands early')
    ap.add_argument('--alpha', type=float, default=1.5)
    ap.add_argument('--gauss-delta-les', type=float, default=2.0)
    ap.add_argument('--name', default='DNS')
    ap.add_argument('--jacobian-only', action='store_true',
                    help="Pi = gaussianfilter(J)_fine - J(gaussianfilter(fields)) ONLY "
                         "(Sanaa convention ruling 2026-07-13: no body-force commutator "
                         "in the target; physics quantities only). Output suffix "
                         "_gaussian_jonly. chi masks still SAVED (the dataset needs "
                         "them for the valid mask) — they just don't enter Pi.")
    args = ap.parse_args()

    cpf = load_cpf()
    from qg._output.filter import LESFilter
    from qg.solver.opt.basis import to_physical

    mdir = Path(args.member_dir)
    z = np.load(mdir / f'{args.name}_FR.npz')
    with open(mdir / f'{args.name}_FR_params.yaml') as f:
        params = yaml.safe_load(f)

    omega_FR_np = z['omega_FR']
    chi_obs_np = z.get('chi_obs') if 'chi_obs' in z.files else None
    chi_sponge_np = z.get('chi_sponge_ramp') if 'chi_sponge_ramp' in z.files else None
    B, T, Ny, Nx = omega_FR_np.shape
    assert B == 1, f"expected single-batch FR data, got B={B}"
    dt = float(params['time']['dt'])
    penalty = float(params['pde']['penalty'])
    sponge_eta = float(params['bc'].get('sponge', 0))
    gp = params['grid']
    Lx, Ly = float(gp['Lx']), float(gp['Ly'])
    print(f"[gauss-rebuild] {mdir.name}: FR ({Ny}x{Nx}) x {T} frames, "
          f"scales {args.scales}, alpha {args.alpha}, delta_les {args.gauss_delta_les}")

    grid_FR, deriv_FR = cpf._build_grid_and_derivative(gp, device='cpu')

    # Jacobian-only mode: body forces OUT of the target in BOTH legs
    src_penalty = 0.0 if args.jacobian_only else penalty
    src_sponge = 0.0 if args.jacobian_only else sponge_eta
    suffix = '_gaussian_jonly' if args.jacobian_only else '_gaussian'

    for scale in args.scales:
        t0 = time.time()
        canon_path = mdir / f'DNS_LES_s{scale}.npz'
        if not canon_path.exists():
            print(f"[gauss-rebuild] s{scale}: no canonical {canon_path.name} — SKIP "
                  f"(times/U_snap source missing; run step0 first)")
            continue
        out_path = mdir / f'DNS_LES_s{scale}{suffix}.npz'
        if out_path.exists():
            print(f"[gauss-rebuild] s{scale}: {out_path.name} exists — SKIP (no overwrite)")
            continue
        canon = np.load(canon_path)
        times_c = np.asarray(canon['times'])
        U_snap = np.asarray(canon['U_snap'], dtype=np.float64)
        Re_snap = np.asarray(canon['Re_snap'])
        if len(times_c) != T:
            raise ValueError(f"s{scale}: canonical has {len(times_c)} frames, FR has {T}")

        cp = dict(gp); cp['Nx'] = Nx // scale; cp['Ny'] = Ny // scale
        grid_LES, deriv_LES = cpf._build_grid_and_derivative(cp, device='cpu')
        filt = make_all_gaussian_filter(LESFilter, grid_FR, deriv_FR,
                                        scale, args.alpha, args.gauss_delta_les)

        chi_obs_FR = chi_obs_bar = None
        if chi_obs_np is not None and penalty > 0:
            chi_obs_FR = torch.tensor(chi_obs_np, dtype=grid_FR.ftype)
            chi_obs_bar = filt.from_physical(chi_obs_FR)
        chi_sponge_FR = chi_sponge_bar = None
        if chi_sponge_np is not None and sponge_eta > 0:
            chi_sponge_FR = torch.tensor(chi_sponge_np, dtype=grid_FR.ftype)
            chi_sponge_bar = filt.from_physical(chi_sponge_FR)

        ob = np.zeros((T, cp['Ny'], cp['Nx']), dtype=np.float32)
        pi = np.zeros_like(ob)
        with torch.no_grad():
            for t in range(T):
                om = torch.tensor(omega_FR_np[0, t], dtype=grid_FR.ftype)
                src_FR = cpf._sources_on_grid(
                    om, deriv_FR, dt, src_penalty,
                    None if args.jacobian_only else chi_obs_FR,
                    None if args.jacobian_only else chi_sponge_FR, src_sponge)
                src_f = filt.from_spectral(src_FR, output='physical')
                om_bar = filt.from_physical(om)
                src_bar = cpf._sources_on_grid(
                    om_bar, deriv_LES, dt, src_penalty,
                    None if args.jacobian_only else chi_obs_bar,
                    None if args.jacobian_only else chi_sponge_bar, src_sponge)
                ob[t] = om_bar.numpy()
                pi[t] = (src_f - to_physical(src_bar)).numpy()
                if t % 50 == 0:
                    print(f"[gauss-rebuild] s{scale} frame {t}/{T} ({time.time()-t0:.0f}s)")

        ubar, vbar = compute_uv_from_omega(ob, Lx, Ly, U_snap)
        meta = {
            'filter': (f"ALL-GAUSSIAN: exp(-k^2 (alpha*dx_FR)^2/6) * "
                       f"exp(-k^2 ({args.gauss_delta_les}*dx_LES)^2/6) o avg-pool "
                       f"(sharp cutoff REMOVED; P2-validated variant 2026-07-13)"),
            'pi_definition': ('gaussianfilter(J)_fine - J(gaussianfilter(fields)) '
                              'JACOBIAN-ONLY (Sanaa convention 2026-07-13)'
                              if args.jacobian_only else
                              'filter(J+Brinkman+Sponge)_fine - (J+Brinkman+Sponge)(filtered fields)'),
            'alpha': args.alpha, 'gauss_delta_les': args.gauss_delta_les,
            'scale': int(scale), 'source': f'{args.name}_FR.npz',
            'times_U_Re_from': canon_path.name,
        }
        np.savez_compressed(
            out_path,
            omega_bar=ob[None], pi_ff=pi[None],
            ubar=ubar[None], vbar=vbar[None],
            times=times_c, U_snap=canon['U_snap'], Re_snap=Re_snap,
            # squeeze then [None]: chi arrays in DNS_FR.npz may already carry a
            # batch axis — guarantee (1, Ny, Nx) like the canonical files
            # (2026-07-13 bug: double batch dim crashed signed_distance rank-2 check)
            chi_obs_bar=(np.asarray(chi_obs_bar.numpy(), dtype=np.float32).squeeze()[None]
                         if chi_obs_bar is not None else np.zeros((1, cp['Ny'], cp['Nx']), np.float32)),
            chi_sponge_bar=(np.asarray(chi_sponge_bar.numpy(), dtype=np.float32).squeeze()[None]
                            if chi_sponge_bar is not None else np.zeros((1, cp['Ny'], cp['Nx']), np.float32)),
            meta=json.dumps(meta))
        print(f"[gauss-rebuild] s{scale}: wrote {out_path.name} "
              f"(pi range [{pi.min():.3e}, {pi.max():.3e}]) in {time.time()-t0:.0f}s")

    print(f"[gauss-rebuild] {mdir.name} done")


if __name__ == '__main__':
    main()
