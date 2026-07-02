"""
delta_dataset.py

Pooled training data for the *empirical* temporal closure (the delta-R pivot).

A sweep_dT_<tag> dir (produced by slice_delta_sweep.py) packs:
    packed/inputs.npy          (N, 2S, Ny, Nx)  [omega_0..m{S-1}, psi_0..m{S-1}]
    packed/delta_exact_f64.npy (N, 1,  Ny, Nx)  ref = fine RK4 ~ exact : R3,R4,R5,...
    packed/delta_rk4_f64.npy   (N, 1,  Ny, Nx)  ref = coarse RK4       : R3,R4,R5-T5,...
    split.npz, manifest.json   (Delta_T, beta, nu, mu, n_snapshots_per_sample=S)

This loader pools any list of those dirs (the sweep across Delta_T AND the ensemble
across regime), pulls n_snapshots <= S lags, serves the chosen reference's delta as
the target, and ALWAYS returns the per-sample regime vector [Delta_T, beta, nu, mu]
-- Delta_T is now a model input (the closure is Delta_T-dependent), so it must travel.

Self-contained: mmaps the packed arrays directly (no dependence on dataset.py's
PackedClosureDataset), and reuses GridHomogeneousBatchSampler from concat_dataset
so mixed 256^2/512^2 pools still collate.

Usage:
    from delta_dataset import make_delta_loaders
    tl, vl, te, *ds = make_delta_loaders(
        sweep_roots, batch_size=4, n_snapshots=7,
        reference='exact',          # or 'rk4'
        compute_dtype='float64')    # clean FD; needs a float64 build for small dT
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from concat_dataset import GridHomogeneousBatchSampler, snapshot_input_fields


class DeltaMemberDataset(Dataset):
    """One sweep_dT_<tag> dir: inputs.npy + delta_{reference}_f64.npy, one split."""

    def __init__(self, root, split: str = 'train', n_snapshots: int = 7,
                 reference: str = 'exact', compute_dtype: str = 'float64',
                 packed_subdir: str = 'packed'):
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
        self.both = (reference == 'both')
        refs = ['exact', 'rk4'] if self.both else [reference]
        if not self.both and reference not in ('exact', 'rk4'):
            raise ValueError(f"reference must be exact|rk4|both, got {reference!r}")
        self.dys = []
        for ref in refs:
            dfile = pdir / f'delta_{ref}_f64.npy'
            if not dfile.exists():
                raise FileNotFoundError(
                    f"{dfile.name} missing in {root.name}; re-slice with the two-delta "
                    f"slicer.")
            self.dys.append(np.load(dfile, mmap_mode='r'))               # each (N,1,Ny,Nx)
        sp = np.load(root / 'split.npz')
        self.idx = sp[f'{split}_idx']
        self.Ny, self.Nx = int(self.man['Ny']), int(self.man['Nx'])
        self.Lx, self.Ly = float(self.man['Lx']), float(self.man['Ly'])
        self.dx, self.dy = self.Lx / self.Nx, self.Ly / self.Ny
        self.sig = (self.Ny, self.Nx, round(self.Lx, 4), round(self.Ly, 4))
        self.cdtype = torch.float64 if compute_dtype == 'float64' else torch.float32
        self.input_fields = snapshot_input_fields(self.n)
        self.reference = reference
        # regime carries physics [dT,beta,nu,mu] AND the grid signature
        # [Ny,Nx,Lx,Ly]. The trainer selects per-grid spectral ops + spacing per
        # batch from [4:8] (batches are signature-homogeneous); normalize_regime /
        # FiLM use only [:,0:4]. Putting the signature on the sample (not a per-
        # dataset index) keeps the key canonical across train/val/test splits.
        self.regime_vec = torch.tensor(
            [float(self.man['Delta_T']), float(self.man['beta']),
             float(self.man['nu']), float(self.man['mu']),
             float(self.Ny), float(self.Nx), self.Lx, self.Ly], dtype=torch.float32)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        g = int(self.idx[i])
        om = self.inp[g, 0:self.n]                       # (n,Ny,Nx)
        ps = self.inp[g, self.S:self.S + self.n]         # (n,Ny,Nx)
        x = torch.from_numpy(np.concatenate([om, ps], 0).copy()).to(self.cdtype)
        ys = [torch.from_numpy(np.asarray(dy[g]).copy()).to(self.cdtype)   # (1,Ny,Nx)
              for dy in self.dys]
        if self.both:
            return x, ys[0], ys[1], self.regime_vec      # (x, delta_exact, delta_rk4, reg)
        return x, ys[0], self.regime_vec


class ConcatDeltaDataset(Dataset):
    """Concatenate DeltaMemberDataset over many sweep dirs (Delta_T x regime)."""

    def __init__(self, roots: Sequence, split: str = 'train', n_snapshots: int = 7,
                 reference: str = 'exact', compute_dtype: str = 'float64'):
        super().__init__()
        if isinstance(roots, (str, Path)):
            roots = [roots]
        self.subsets, self.grid_shapes, self._cum = [], [], []
        total = 0
        for rd in roots:
            ds = DeltaMemberDataset(rd, split=split, n_snapshots=n_snapshots,
                                    reference=reference, compute_dtype=compute_dtype)
            if len(ds) == 0:
                continue
            self.subsets.append(ds)
            self.grid_shapes.append(ds.sig)        # full sig (Ny,Nx,Lx,Ly): the
                                                   # batch sampler buckets on this,
                                                   # so each batch is one grid+domain
            total += len(ds)
            self._cum.append(total)
        if not self.subsets:
            raise ValueError(f"no non-empty '{split}' across {[str(r) for r in roots]}")
        self.n_snapshots = n_snapshots
        self.reference = reference

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


def make_delta_loaders(roots, batch_size=4, num_workers=2, n_snapshots=7,
                       reference='exact', compute_dtype='float64', seed=0):
    """(train, val, test) loaders + datasets over a pooled delta sweep.

    Always returns the regime (x, y, regime); uses GridHomogeneousBatchSampler
    when the pool mixes grids.
    """
    from torch.utils.data import DataLoader

    def make(split, shuffle):
        ds = ConcatDeltaDataset(roots, split=split, n_snapshots=n_snapshots,
                                reference=reference, compute_dtype=compute_dtype)
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
    print(f"[delta] members(train)={len(train_ds.subsets)}  ref={reference}  "
          f"n_snapshots={n_snapshots} in_ch={2*n_snapshots}  dT_sweep={dTs}  "
          f"grids={sorted(set(train_ds.grid_shapes))}  "
          f"train/val/test={len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    return tl, vl, te, train_ds, val_ds, test_ds


if __name__ == '__main__':
    import sys
    roots = sys.argv[1:]
    if not roots:
        print("usage: python delta_dataset.py SWEEP_ROOT [SWEEP_ROOT ...]")
        raise SystemExit(0)
    for ref in ('exact', 'rk4'):
        try:
            ds = ConcatDeltaDataset(roots, split='train', n_snapshots=7, reference=ref)
            x, y, r = ds[0]
            print(f"ref={ref}: len={len(ds)} x={tuple(x.shape)} y={tuple(y.shape)} "
                  f"regime(dT,beta,nu,mu)={r.tolist()} grids={set(ds.grid_shapes)}")
        except Exception as e:  # noqa: BLE001
            print(f"ref={ref}: {type(e).__name__}: {e}")
