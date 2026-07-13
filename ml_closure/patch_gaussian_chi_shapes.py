"""
One-shot repair of the 2026-07-13 Gaussian rebuild chi-shape bug: the
DNS_LES_s4_gaussian.npz files were written with chi_obs_bar/chi_sponge_bar of
shape (1,1,Ny,Nx) (double batch dim — the FR npz masks already carried one);
dataset_piff's signed_distance needs (Ny,Nx) after the [0] index. Squeeze both
mask arrays to (1,Ny,Nx) in place, then GATE: load each member end-to-end
through dataset_piff.RunData with variant='gaussian' and print the valid-pixel
count (the real G2 data gate — a member that fails here fails loudly).

Usage: python patch_gaussian_chi_shapes.py <member_dir> [<member_dir> ...]
"""

import json
import sys
from pathlib import Path

import numpy as np

import dataset_piff as dp


def patch(npz_path):
    z = dict(np.load(npz_path, allow_pickle=False))
    changed = False
    for k in ('chi_obs_bar', 'chi_sponge_bar'):
        a = z[k]
        if a.ndim > 3:
            z[k] = np.ascontiguousarray(a.squeeze()[None].astype(np.float32))
            changed = True
    if changed:
        tmp = npz_path.with_suffix('.npz.tmp')
        # write via OPEN HANDLE: np.savez appends '.npz' to non-.npz filenames
        # (the 2026-07-09 scalars.py trap — same fix)
        with open(tmp, 'wb') as f:
            np.savez_compressed(f, **z)
        tmp.replace(npz_path)
    return changed, {k: z[k].shape for k in ('chi_obs_bar', 'chi_sponge_bar')}


def main():
    members = [Path(p) for p in sys.argv[1:]]
    conf = dp.load_conf(Path(__file__).parent / 'conf_piff.yaml')
    conf['data']['variant'] = 'gaussian'
    ok = 0
    for m in members:
        p = m / 'DNS_LES_s4_gaussian.npz'
        changed, shapes = patch(p)
        print(f"[patch] {m.name}: {'FIXED' if changed else 'already-ok'} {shapes}")
        # stale variant SDF cache would have been built from the bad mask — drop
        sdf = m / 'SDF_s4_gaussian.npy'
        if sdf.exists():
            sdf.unlink()
            print(f"[patch] {m.name}: removed stale {sdf.name}")
        conf['data']['runs'] = [str(m)]
        r = dp.RunData(str(m), conf)          # the G2 gate: loud or nothing
        print(f"[gate] {m.name}: OK — {r.T} frames, {r.n_valid} valid px, "
              f"zeta [{r.zeta_snap.min():.3f},{r.zeta_snap.max():.3f}]")
        ok += 1
    print(f"[patch] {ok}/{len(members)} members patched+gated")


if __name__ == '__main__':
    main()
