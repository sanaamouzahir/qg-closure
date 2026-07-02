"""
deriv_dataset.py
================
Pooled training data for the *derivative-loss* temporal closure (the pre-6.1.2
model): the network predicts the local N-time-derivatives [Ndot, Nddot, N3dot]
directly, and the L^k weightings are applied analytically at assembly/inference.

A sweep_dT_<tag> dir (sliced by slice_delta_sweep.py, then augmented by
add_deriv_targets.py) packs:
    packed/inputs.npy          (N, 2S, Ny, Nx)  [omega_0..m{S-1}, psi_0..m{S-1}]
    packed/deriv_anal_f64.npy  (N, 3,  Ny, Nx)  [Ndot, Nddot, N3dot]  (the target)
    split.npz, manifest.json

Thin sibling of delta_dataset.py: same ConcatDataset + GridHomogeneousBatchSampler
machinery, but serves the 3-channel derivative target instead of delta. Always
returns the per-sample regime vector [Delta_T, beta, nu, mu, dx, dy] -- Delta_T is
a model input via the FD time-scaling, and dx,dy drive the MULTIGRID per-sample
spatial rescale in cheap_deriv.SpatialGrad.

Usage:
    from deriv_dataset import make_deriv_loaders
    tl, vl, te, *ds = make_deriv_loaders(
        sweep_roots, batch_size=4, n_snapshots=4, compute_dtype='float64')
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from concat_dataset import GridHomogeneousBatchSampler, snapshot_input_fields


class DerivMemberDataset(Dataset):
    """One sweep_dT_<tag> dir: inputs.npy + deriv_anal_f64.npy, one split."""

    def __init__(self, root, split: str = 'train', n_snapshots: int = 4,
                 compute_dtype: str = 'float64', packed_subdir: str = 'packed'):
        super().__init__()
        root = Path(root)
        self.root = root
        self.man = json.loads((root / 'manifest.json').read_text())
        self.S = int(self.man['n_snapshots_per_sample'])
        self.n = int(n_snapshots)
        if self.n > self.S:
            raise ValueError(f"n_snapshots={self.n} > sliced S={self.S} at {root.name}")
        pdir = root / packed_subdir
        self.inp = np.load(pdir / 'inputs.npy', mmap_mode='r')          # (N,2S,Ny,Nx)
        tfile = pdir / 'deriv_anal_f64.npy'
        if not tfile.exists():
            raise FileNotFoundError(
                f"{tfile.name} missing in {root.name}; run add_deriv_targets.py first.")
        self.tgt = np.load(tfile, mmap_mode='r')                        # (N,3,Ny,Nx)
        sp = np.load(root / 'split.npz')
        self.idx = sp[f'{split}_idx']
        self.Ny, self.Nx = int(self.man['Ny']), int(self.man['Nx'])
        self.cdtype = torch.float64 if compute_dtype == 'float64' else torch.float32
        self.input_fields = snapshot_input_fields(self.n)
        self.regime_vec = torch.tensor(
            [float(self.man['Delta_T']), float(self.man['beta']),
             float(self.man['nu']), float(self.man['mu']),
             float(self.man['Lx']) / int(self.man['Nx']),   # MULTIGRID: dx
             float(self.man['Ly']) / int(self.man['Ny'])],  # MULTIGRID: dy
            dtype=torch.float32)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        g = int(self.idx[i])
        om = self.inp[g, 0:self.n]                       # (n,Ny,Nx)
        ps = self.inp[g, self.S:self.S + self.n]         # (n,Ny,Nx)
        x = torch.from_numpy(np.concatenate([om, ps], 0).copy()).to(self.cdtype)
        y = torch.from_numpy(np.asarray(self.tgt[g]).copy()).to(self.cdtype)  # (3,Ny,Nx)
        return x, y, self.regime_vec


class ConcatDerivDataset(Dataset):
    """Concatenate DerivMemberDataset over many sweep dirs (Delta_T x regime)."""

    def __init__(self, roots: Sequence, split: str = 'train', n_snapshots: int = 4,
                 compute_dtype: str = 'float64'):
        super().__init__()
        if isinstance(roots, (str, Path)):
            roots = [roots]
        self.subsets, self.grid_shapes, self._cum = [], [], []
        total = 0
        for rd in roots:
            ds = DerivMemberDataset(rd, split=split, n_snapshots=n_snapshots,
                                    compute_dtype=compute_dtype)
            if len(ds) == 0:
                continue
            self.subsets.append(ds)
            self.grid_shapes.append((ds.Ny, ds.Nx))
            total += len(ds)
            self._cum.append(total)
        if not self.subsets:
            raise ValueError(f"no non-empty '{split}' across {[str(r) for r in roots]}")
        self.n_snapshots = n_snapshots

    def __len__(self):
        return self._cum[-1] if self._cum else 0

    def _locate(self, i):
        import bisect
        s = bisect.bisect_right(self._cum, i)
        lo = self._cum[s - 1] if s > 0 else 0
        return s, i - lo

    def sample_grid_shape(self, i):
        s, _ = self._locate(i)
        return self.grid_shapes[s]

    def __getitem__(self, i):
        s, j = self._locate(i)
        return self.subsets[s][j]


def make_deriv_loaders(roots, batch_size=4, num_workers=2, n_snapshots=4,
                       compute_dtype='float64', seed=0):
    """(train, val, test) loaders + datasets over a pooled derivative-target sweep."""
    from torch.utils.data import DataLoader

    def make(split, shuffle):
        ds = ConcatDerivDataset(roots, split=split, n_snapshots=n_snapshots,
                                compute_dtype=compute_dtype)
        multigrid = len(set(ds.grid_shapes)) > 1
        pw = num_workers > 0
        if multigrid:
            sampler = GridHomogeneousBatchSampler(ds, batch_size, shuffle=shuffle,
                                                  drop_last=False, seed=seed)
            loader = DataLoader(ds, batch_sampler=sampler, num_workers=num_workers,
                                pin_memory=True, persistent_workers=pw)
        else:
            loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                num_workers=num_workers, pin_memory=True,
                                drop_last=False, persistent_workers=pw)
        return loader, ds

    tl, train_ds = make('train', True)
    vl, val_ds = make('val', False)
    te, test_ds = make('test', False)
    dTs = sorted({float(s.man['Delta_T']) for s in train_ds.subsets})
    print(f"[deriv] members(train)={len(train_ds.subsets)}  "
          f"n_snapshots={n_snapshots} in_ch={2*n_snapshots}  dT_sweep={dTs}  "
          f"grids={sorted(set(train_ds.grid_shapes))}  "
          f"train/val/test={len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    return tl, vl, te, train_ds, val_ds, test_ds


if __name__ == '__main__':
    import sys
    roots = sys.argv[1:]
    if not roots:
        print("usage: python deriv_dataset.py SWEEP_ROOT [SWEEP_ROOT ...]")
        raise SystemExit(0)
    ds = ConcatDerivDataset(roots, split='train', n_snapshots=4)
    x, y, r = ds[0]
    print(f"len={len(ds)} x={tuple(x.shape)} y={tuple(y.shape)} "
          f"regime(dT,beta,nu,mu)={r.tolist()} grids={set(ds.grid_shapes)}")
