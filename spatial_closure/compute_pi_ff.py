"""
Stage 2: compute (omega_bar, Pi_FF) training pairs from a Stage-1 FR dataset.

Implements Eq. 8 of arXiv 2508.06678:

    Pi_FF =  filter[ J(psi_FR, omega_FR) + Brinkman(chi_FR, u_FR, v_FR) + Sponge(omega_FR, chi_sponge_FR) ]
           - [ J(psi_bar, omega_bar)   + Brinkman(chi_bar, u_bar, v_bar)  + Sponge(omega_bar, chi_sponge_bar) ]

where the LES filter (`filter`) is the Eq.5 composite (Gaussian + sharp cutoff +
average-pool) and the coarse grid is obtained by downsampling the FR grid by an
integer factor `scale`. The Gaussian bandwidth is set independently by `alpha`
(paper uses 1.5).

Conventions:
  * chi_bar = filter(chi_FR)  (Option A from project memory).
  * No 2/3 dealiasing of the kernel calls. The LES filter's sharp cutoff at
    |k| >= pi/dx_LES already kills modes above coarse-Nyquist.
  * Outputs live on the coarse (LES) grid as physical-space tensors.

Usage:
    python compute_pi_ff.py <save_path> [--scale N] [--alpha 1.5] [--name DNS]

By default reads `{save_path}/{name}_FR.npz`, writes `{save_path}/{name}_LES.npz`,
emits diagnostic videos, and re-encodes them in place to a laptop-compatible
format (H.264 + yuv420p). The original .mp4 files are replaced.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Local QG imports
from qg.solver.grid.cartesian import CartesianGrid
from qg.solver.opt.derivative import Derivative
from qg.solver.opt.basis import puv, to_spectral, to_physical
from qg.solver.opt.operator.jacobian import jacobian_pq
from qg.solver.opt.operator.obstacle import brinkman_no_slip_penalty
from qg._input.sources.bc import Sponge
from qg._output.filter import LESFilter

# Optional: video rendering
try:
    import jpcm.draw as draw
    _HAVE_DRAW = True
except Exception:
    _HAVE_DRAW = False


# --------------------------------------------------------------------------- #
# Lightweight stand-ins for State and Operator that the kernels expect.
class _State:
    def __init__(self, qh, ph, uh, vh, dt):
        self.qh, self.ph, self.uh, self.vh, self.dt = qh, ph, uh, vh, dt


class _Op:
    class _Params: pass
    def __init__(self, derivative, penalty, dt):
        self.derivative = derivative
        self.params = _Op._Params()
        self.params.penalty = penalty
        self.dt = dt


# --------------------------------------------------------------------------- #
def _build_grid_and_derivative(grid_params, device):
    """Build a CartesianGrid + Derivative from a params dict."""
    kw = {k: v for k, v in grid_params.items() if v is not None and k != 'device'}
    if device is not None:
        kw['device'] = device
    grid = CartesianGrid(**kw)
    deriv = Derivative(grid).to(grid.device)
    return grid, deriv


def _make_state_from_omega(omega_phys, derivative, dt):
    """Build a state-like object from physical-space vorticity."""
    qh = to_spectral(omega_phys)
    ph, uh, vh = puv(qh, derivative)
    return _State(qh=qh, ph=ph, uh=uh, vh=vh, dt=dt)


def _sources_on_grid(omega_phys, derivative, dt, penalty, chi_obs, chi_sponge_ramp, sponge_eta):
    """
    Evaluate the SGS-relevant source sum on whatever grid the inputs are on:
        J(psi, omega) + Brinkman(chi_obs, u, v) + Sponge(omega, chi_sponge)
    Returns spectral-space tensor on the input grid; no dealiasing.
    """
    state = _make_state_from_omega(omega_phys, derivative, dt)
    op = _Op(derivative=derivative, penalty=penalty, dt=dt)

    src = jacobian_pq(op, state)
    if chi_obs is not None and penalty > 0:
        chi_velocity = torch.zeros(2, device=state.qh.device, dtype=state.qh.real.dtype)
        src = src + brinkman_no_slip_penalty(op, state, chi_obs, chi_velocity)
    if chi_sponge_ramp is not None and sponge_eta > 0:
        src = src + Sponge.vorticity_sponge(state, derivative, sponge_eta, chi_sponge_ramp, None, None)
    return src


# --------------------------------------------------------------------------- #
def compute_pi_ff(save_path, name='DNS', scale=2, alpha=1.5, device='cpu'):
    save_path = Path(save_path)
    fr_npz_path = save_path / f'{name}_FR.npz'
    fr_yaml_path = save_path / f'{name}_FR_params.yaml'
    if not fr_npz_path.exists():
        raise FileNotFoundError(f"Stage-1 FR data not found: {fr_npz_path}")
    if not fr_yaml_path.exists():
        raise FileNotFoundError(f"Stage-1 params not found: {fr_yaml_path}")

    print(f"Loading FR data from {fr_npz_path}")
    z = np.load(fr_npz_path)
    with open(fr_yaml_path) as f:
        params = yaml.safe_load(f)

    omega_FR_np = z['omega_FR']
    times = z['times']
    chi_obs_np = z.get('chi_obs') if 'chi_obs' in z.files else None
    chi_sponge_np = z.get('chi_sponge_ramp') if 'chi_sponge_ramp' in z.files else None

    B, T_save, Ny, Nx = omega_FR_np.shape
    print(f"  shape (B,T,Ny,Nx) = {omega_FR_np.shape}")
    print(f"  chi_obs={'yes' if chi_obs_np is not None else 'no'}, "
          f"chi_sponge={'yes' if chi_sponge_np is not None else 'no'}")

    # Build FR + coarse grids
    grid_params = params['grid']
    grid_FR, deriv_FR = _build_grid_and_derivative(grid_params, device=device)

    coarse_params = dict(grid_params)
    coarse_params['Nx'] = Nx // scale
    coarse_params['Ny'] = Ny // scale
    grid_LES, deriv_LES = _build_grid_and_derivative(coarse_params, device=device)
    print(f"  FR grid: {grid_FR.Nx}x{grid_FR.Ny}, LES grid: {grid_LES.Nx}x{grid_LES.Ny}")
    print(f"  filter: scale={scale}, width(alpha)={alpha}")

    # LES filter: independent scale (downsampling) and width (Gaussian bandwidth)
    les_filter = LESFilter(grid_FR, deriv_FR, scale=scale, width=alpha)

    # Filter static masks
    chi_obs_FR = chi_obs_bar = None
    if chi_obs_np is not None and params['pde']['penalty'] > 0:
        chi_obs_FR = torch.tensor(chi_obs_np, dtype=grid_FR.ftype, device=grid_FR.device)
        chi_obs_bar = les_filter.from_physical(chi_obs_FR)

    chi_sponge_FR = chi_sponge_bar = None
    if chi_sponge_np is not None and params['bc'].get('sponge', 0) > 0:
        chi_sponge_FR = torch.tensor(chi_sponge_np, dtype=grid_FR.ftype, device=grid_FR.device)
        chi_sponge_bar = les_filter.from_physical(chi_sponge_FR)

    dt = float(params['time']['dt'])
    penalty = float(params['pde']['penalty'])
    sponge_eta = float(params['bc'].get('sponge', 0))

    omega_bar_all = np.zeros((B, T_save, grid_LES.Ny, grid_LES.Nx), dtype=np.float32)
    pi_ff_all = np.zeros_like(omega_bar_all)

    print(f"  Computing Pi_FF for {B*T_save} snapshots...")
    for b in range(B):
        for t in range(T_save):
            omega_FR = torch.tensor(omega_FR_np[b, t], dtype=grid_FR.ftype, device=grid_FR.device)

            # FR-grid source (spectral) -> filter to coarse physical
            src_FR = _sources_on_grid(omega_FR, deriv_FR, dt, penalty,
                                      chi_obs_FR, chi_sponge_FR, sponge_eta)
            src_FR_filtered_phys = les_filter.from_spectral(src_FR, output='physical')

            # Coarse omega and coarse-grid source
            omega_bar_phys = les_filter.from_physical(omega_FR)
            src_bar = _sources_on_grid(omega_bar_phys, deriv_LES, dt, penalty,
                                       chi_obs_bar, chi_sponge_bar, sponge_eta)
            src_bar_phys = to_physical(src_bar)

            pi_ff_phys = src_FR_filtered_phys - src_bar_phys

            omega_bar_all[b, t] = omega_bar_phys.detach().cpu().numpy()
            pi_ff_all[b, t] = pi_ff_phys.detach().cpu().numpy()

    print(f"  omega_bar range: [{omega_bar_all.min():.3e}, {omega_bar_all.max():.3e}]")
    print(f"  pi_ff range:     [{pi_ff_all.min():.3e}, {pi_ff_all.max():.3e}]")

    out_path = save_path / f'{name}_LES.npz'
    out_dict = {
        'omega_bar': omega_bar_all.astype(np.float32),
        'pi_ff': pi_ff_all.astype(np.float32),
        'times': times,
    }
    if chi_obs_bar is not None:
        out_dict['chi_obs_bar'] = chi_obs_bar.detach().cpu().numpy().astype(np.float32)
    if chi_sponge_bar is not None:
        out_dict['chi_sponge_bar'] = chi_sponge_bar.detach().cpu().numpy().astype(np.float32)
    out_dict['_scale'] = np.array([scale], dtype=np.int32)
    out_dict['_alpha'] = np.array([alpha], dtype=np.float32)

    np.savez_compressed(out_path, **out_dict)
    print(f"Wrote {out_path}")

    summary = {
        'source': str(fr_npz_path.name),
        'scale': int(scale),
        'alpha': float(alpha),
        'FR_shape': list(omega_FR_np.shape),
        'LES_shape': list(omega_bar_all.shape),
        'omega_bar_range': [float(omega_bar_all.min()), float(omega_bar_all.max())],
        'pi_ff_range': [float(pi_ff_all.min()), float(pi_ff_all.max())],
    }
    with open(save_path / f'{name}_LES_summary.yaml', 'w') as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    print(f"Wrote {save_path / (name + '_LES_summary.yaml')}")

    return out_dict


# --------------------------------------------------------------------------- #
# Video helpers
def _reencode_in_place(mp4_path):
    """
    Re-encode a video with H.264 + yuv420p (laptop-compatible) and replace
    the original file in place. Uses an adjacent .tmp file because ffmpeg
    cannot read and write the same path simultaneously. We pass -f mp4
    explicitly because the .tmp suffix would otherwise confuse format detection.
    """
    if not shutil.which('ffmpeg'):
        return False
    mp4_path = Path(mp4_path)
    tmp_path = mp4_path.with_suffix('.mp4.tmp')
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', str(mp4_path),
             '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
             '-f', 'mp4', str(tmp_path)],
            check=True, capture_output=True, text=True
        )
        os.replace(tmp_path, mp4_path)
        return True
    except subprocess.CalledProcessError as e:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        print(f"  [reencode] ffmpeg failed for {mp4_path.name}: {e.stderr[-300:] if e.stderr else e}")
        return False


def render_videos(save_path, name='DNS', fps=20, clamp=0.3, reencode=True):
    """
    Make MP4s of omega_bar and pi_ff for sanity checking. If `reencode` is
    true (default), each rendered file is then re-encoded in place with
    H.264 + yuv420p so it plays on any laptop without an extra step.
    """
    if not _HAVE_DRAW:
        print("[videos] jpcm.draw not available; skipping videos.")
        return

    save_path = Path(save_path)
    z = np.load(save_path / f'{name}_LES.npz')
    omega_bar = z['omega_bar']
    pi_ff = z['pi_ff']

    def _prep(x):
        # Flip y so row 0 of the array (physical y=0) renders at the BOTTOM
        # of the image. jpcm.draw.mp4 / imshow defaults to origin='upper' which
        # otherwise puts y=0 at the top.
        x = x[..., ::-1, :].copy()
        # jpcm.draw.mp4 expects (T, B_subset, C, H, W); we have (B, T, H, W).
        x4 = x[:, :, None, ...]
        x4 = x4[:4, ...]
        return np.transpose(x4, (1, 0, 2, 3, 4))

    raw_outputs = []
    targets = [
        ('omega_bar',          _prep(omega_bar), {}),
        ('omega_bar_clamped',  _prep(omega_bar), {'clamp': clamp}),
        ('omega_bar_seismic',  _prep(omega_bar), {'cmap': 'seismic', 'clamp': clamp}),
        ('pi_ff',              _prep(pi_ff),     {}),
        ('pi_ff_seismic',      _prep(pi_ff),     {'cmap': 'seismic', 'clamp': clamp}),
    ]
    for tag, frames, kwargs in targets:
        out_mp4 = save_path / f'{name}_LES_{tag}.mp4'
        try:
            draw.mp4(str(out_mp4), frames, fps=fps, triplet=True, **kwargs)
            raw_outputs.append(out_mp4)
        except Exception as e:
            print(f"  [videos] {tag} render failed: {type(e).__name__}: {e}")

    if not raw_outputs:
        print("[videos] no videos rendered; skipping re-encode.")
        return

    if reencode:
        print(f"[videos] re-encoding {len(raw_outputs)} files in place (H.264 + yuv420p)...")
        for mp4 in raw_outputs:
            ok = _reencode_in_place(mp4)
            print(f"  {'->' if ok else '[fail]'} {mp4.name}")
        print()
        print("To copy videos to your laptop, on your laptop run:")
        print(f"  scp 'sanaamz@<cluster>:{save_path.resolve()}/{name}_LES_*.mp4' ~/Downloads/")
    else:
        print("[videos] re-encoding disabled; .mp4 files may not play on all laptops as-is.")


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Compute (omega_bar, Pi_FF) from a Stage-1 FR dataset.")
    p.add_argument('save_path')
    p.add_argument('--name', default='DNS')
    p.add_argument('--scale', type=int, default=2,
                   help="Integer downsampling factor (default: 2)")
    p.add_argument('--alpha', type=float, default=1.5,
                   help="Gaussian-filter bandwidth multiplier (paper uses 1.5)")
    p.add_argument('--device', default='cpu')
    p.add_argument('--no-videos', action='store_true')
    p.add_argument('--no-reencode', action='store_true',
                   help="Skip the in-place re-encode step (default: re-encode for laptop compatibility)")
    args = p.parse_args()

    compute_pi_ff(args.save_path, name=args.name, scale=args.scale,
                  alpha=args.alpha, device=args.device)

    if not args.no_videos:
        render_videos(args.save_path, name=args.name, reencode=not args.no_reencode)


if __name__ == '__main__':
    main()
