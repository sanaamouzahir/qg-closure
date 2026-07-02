"""
Re-render FR and LES videos from existing data files with correct y-orientation
(physical y=0 at the bottom of the frame).

Use this when the simulation is already done and saved, but the videos came out
upside-down due to the imshow origin convention.

Usage:
    python rerender_videos.py <save_path> [--name DNS] [--clamp 0.3]

What it reads:
    - {save_path}/{name}.npy           (FR 4-channel state, optional)
    - {save_path}/{name}_LES.npz       (LES omega_bar + pi_ff, optional)

What it writes (replaces existing):
    - {save_path}/{name}.mp4
    - {save_path}/{name}_clamped.mp4
    - {save_path}/{name}_seismic.mp4
    - {save_path}/{name}_LES_omega_bar.mp4
    - {save_path}/{name}_LES_omega_bar_clamped.mp4
    - {save_path}/{name}_LES_omega_bar_seismic.mp4
    - {save_path}/{name}_LES_pi_ff.mp4
    - {save_path}/{name}_LES_pi_ff_seismic.mp4

Each .mp4 is re-encoded in place to H.264 + yuv420p so it plays on any laptop.
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

try:
    import jpcm.draw as draw
    _HAVE_DRAW = True
except Exception as _e:
    _HAVE_DRAW = False
    print(f"[error] jpcm.draw not available: {_e}")


def _reencode_in_place(mp4_path):
    """Re-encode a video with H.264 + yuv420p, replacing it in place."""
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
        print(f"  [reencode] ffmpeg failed for {mp4_path.name}: "
              f"{e.stderr[-300:] if e.stderr else e}")
        return False


def render_fr_videos(save_path, name, fps, clamp):
    """Re-render the 3 FR videos from {name}.npy with correct y-orientation."""
    save_path = Path(save_path)
    npy_path = save_path / f'{name}.npy'
    if not npy_path.exists():
        print(f"[fr] {npy_path} not found; skipping FR videos.")
        return []

    print(f"[fr] loading {npy_path}")
    sol = np.load(npy_path)  # (B, T, C, H, W)
    print(f"[fr] sol shape: {sol.shape}")

    # qg.py does: solution_b = np.transpose(solution[0:4,:,:1,...], (1,0,2,3,4))
    # i.e. (T, selected_B, C=1, H, W)
    solution_b = np.transpose(sol[0:4, :, :1, ...], (1, 0, 2, 3, 4))
    # FLIP y so row 0 (physical y=0) renders at the BOTTOM of the frame
    solution_b = solution_b[..., ::-1, :].copy()

    outputs = []
    targets = [
        ('',         {}),
        ('clamped',  {'clamp': clamp}),
        ('seismic',  {'cmap': 'seismic', 'clamp': clamp}),
    ]
    for tag, kwargs in targets:
        suffix = f'_{tag}' if tag else ''
        out_mp4 = save_path / f'{name}{suffix}.mp4'
        try:
            draw.mp4(str(out_mp4), solution_b, fps=fps, triplet=True, **kwargs)
            outputs.append(out_mp4)
            print(f"[fr] rendered {out_mp4.name}")
        except Exception as e:
            print(f"[fr] {out_mp4.name} render failed: {type(e).__name__}: {e}")
    return outputs


def render_les_videos(save_path, name, fps, clamp):
    """Re-render the 5 LES videos from {name}_LES.npz with correct y-orientation."""
    save_path = Path(save_path)
    les_path = save_path / f'{name}_LES.npz'
    if not les_path.exists():
        print(f"[les] {les_path} not found; skipping LES videos.")
        return []

    print(f"[les] loading {les_path}")
    z = np.load(les_path)
    omega_bar = z['omega_bar']    # (B, T, Ny, Nx)
    pi_ff = z['pi_ff']
    print(f"[les] omega_bar shape: {omega_bar.shape}")

    def _prep(x):
        # FLIP y first, then add C axis, take up to 4 batches, transpose to (T, B, C, H, W)
        x = x[..., ::-1, :].copy()
        x4 = x[:, :, None, ...]
        x4 = x4[:4, ...]
        return np.transpose(x4, (1, 0, 2, 3, 4))

    outputs = []
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
            outputs.append(out_mp4)
            print(f"[les] rendered {out_mp4.name}")
        except Exception as e:
            print(f"[les] {out_mp4.name} render failed: {type(e).__name__}: {e}")
    return outputs


def main():
    p = argparse.ArgumentParser(description="Re-render FR and LES videos with correct y-orientation.")
    p.add_argument('save_path')
    p.add_argument('--name', default='DNS')
    p.add_argument('--fps', type=int, default=20)
    p.add_argument('--clamp', type=float, default=0.3)
    p.add_argument('--no-reencode', action='store_true')
    p.add_argument('--fr-only', action='store_true')
    p.add_argument('--les-only', action='store_true')
    args = p.parse_args()

    if not _HAVE_DRAW:
        print("Cannot proceed without jpcm.draw. Make sure the venv is active.")
        return 1

    outputs = []
    if not args.les_only:
        outputs += render_fr_videos(args.save_path, args.name, args.fps, args.clamp)
    if not args.fr_only:
        outputs += render_les_videos(args.save_path, args.name, args.fps, args.clamp)

    if not outputs:
        print("No videos rendered.")
        return 1

    if not args.no_reencode:
        print(f"\nRe-encoding {len(outputs)} files in place (H.264 + yuv420p)...")
        for mp4 in outputs:
            ok = _reencode_in_place(mp4)
            print(f"  {'->' if ok else '[fail]'} {mp4.name}")

    save_path_abs = Path(args.save_path).resolve()
    print()
    print("To copy videos to your laptop, on your laptop run:")
    print(f"  scp 'sanaamz@<cluster>:{save_path_abs}/{args.name}*.mp4' ~/Downloads/")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
