"""
concat_dataset.py

Ensemble-agnostic training data for the temporal closure: concatenate any list
of packed datasets (one per ensemble member / regime) and pull a configurable
number of temporal snapshots from each.

Why this exists
---------------
Step 3 trains ONE general model over many regimes (different beta, nu, mu,
forcing, grid). The N-derivative map is regime-universal, so generality is
structural -- the model just needs to see all regimes pooled. Two requirements:

  1. ensemble-agnostic: the training set is `list(root_dirs)`; its size and
     membership are the length of that list. Add/remove a member -> change the
     list, nothing else.
  2. snapshot count is a single knob. The 7-snapshot builder packs 14 input
     columns [omega_0..omega_m6, psi_0..psi_m6]; PackedClosureDataset maps
     fields BY NAME, so pulling n lags is just selecting the first n omega_ and
     psi_ columns. Pass n_snapshots=4 to train the 4-lag stencil, =7 for the
     7-lag stencil -- no field lists to hand-edit. (The diagnostic + corrector
     test decide which n; this turns that decision into one integer.)

Regime metadata
---------------
Each member carries (Delta_T, beta, nu, mu). Delta_T is what the closure-weighted
loss needs (cs_k = coef_k * Delta_T^p_k), and it can differ across members, so it
must travel per-sample. With return_regime=True, __getitem__ returns
(x, y, regime) where regime = tensor([Delta_T, beta, nu, mu]); default collate
stacks it to (B,4). Default (return_regime=False) returns (x, y) -- a drop-in for
the current train.py loop.

Mixed grids
-----------
The ensemble mixes 256^2 and 512^2 members. A batch must be grid-homogeneous or
default collate cannot stack it. GridHomogeneousBatchSampler buckets indices by
(Ny,Nx) and only ever emits single-grid batches; the conv model is size-agnostic
so nothing else changes. make_concat_loaders auto-uses it when >1 grid is present.

Usage
-----
    from concat_dataset import make_concat_loaders
    roots = ['.../FRC-b0', '.../FRC-b1', '.../DEC-base', ...]   # held out: FRC-b075
    tl, vl, te, *ds = make_concat_loaders(
        roots, batch_size=4, n_snapshots=4,          # or 7
        target_fields=('N_dot_0_anal','N_ddot_0_anal','N_3dot_0_anal'),
        compute_dtype='float64', return_regime=True)
"""

from __future__ import annotations

import bisect
import json
import math
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset, Sampler

from dataset import PackedClosureDataset


def snapshot_input_fields(n_snapshots: int):
    """The n-lag input-field list: [omega_0..omega_m(n-1), psi_0..psi_m(n-1)]."""
    if n_snapshots < 1:
        raise ValueError(f"n_snapshots must be >=1, got {n_snapshots}")
    om = ['omega_0'] + [f'omega_m{k}' for k in range(1, n_snapshots)]
    ps = ['psi_0'] + [f'psi_m{k}' for k in range(1, n_snapshots)]
    return tuple(om + ps)


class ConcatClosureDataset(Dataset):
    """Concatenate PackedClosureDataset over root_dirs, pulling n_snapshots lags.

    Each member must have at least n_snapshots packed omega_/psi_ lags (the
    7-snapshot build packs 7; the old 4-snapshot build packs 4 -> n_snapshots<=4).
    Members with an empty split are skipped (e.g. a root used only for test).
    """

    def __init__(self, root_dirs: Sequence,
                 split: str = 'train', n_snapshots: int = 4,
                 target_fields=('N_dot_0_anal', 'N_ddot_0_anal', 'N_3dot_0_anal'),
                 packed_subdir: str = 'packed', compute_dtype: str = 'float32',
                 return_regime: bool = False):
        super().__init__()
        if isinstance(root_dirs, (str, Path)):
            root_dirs = [root_dirs]
        self.n_snapshots = int(n_snapshots)
        self.input_fields = snapshot_input_fields(self.n_snapshots)
        self.target_fields = tuple(target_fields)
        self.return_regime = bool(return_regime)

        self.subsets = []         # PackedClosureDataset per (non-empty) member
        self.regimes = []         # dict(Delta_T, beta, nu, mu) per member
        self.regime_vecs = []     # tensor([Delta_T, beta, nu, mu]) per member
        self.grid_shapes = []     # (Ny, Nx) per member
        self.roots = []           # resolved root per member
        self._cum = []            # cumulative lengths for global indexing

        total = 0
        for rd in root_dirs:
            ds = PackedClosureDataset(rd, split=split,
                                      input_fields=self.input_fields,
                                      target_fields=self.target_fields,
                                      packed_subdir=packed_subdir,
                                      compute_dtype=compute_dtype)
            if len(ds) == 0:
                continue
            reg = self._read_regime(Path(rd), ds)
            self.subsets.append(ds)
            self.regimes.append(reg)
            self.regime_vecs.append(torch.tensor(
                [reg['Delta_T'], reg['beta'], reg['nu'], reg['mu']],
                dtype=torch.float32))
            self.grid_shapes.append((ds.Ny, ds.Nx))
            self.roots.append(Path(rd))
            total += len(ds)
            self._cum.append(total)

        if not self.subsets:
            raise ValueError(f"no non-empty '{split}' split across {list(root_dirs)}")

    @staticmethod
    def _read_regime(root: Path, ds: PackedClosureDataset):
        # Delta_T from pack_meta; beta/nu/mu from manifest.json if present.
        dT = float(ds.meta.get('Delta_T', float('nan')))
        beta = nu = mu = float('nan')
        man = root / 'manifest.json'
        if man.exists():
            with open(man) as f:
                m = json.load(f)
            dT = float(m.get('Delta_T', dT))
            beta = float(m.get('beta', beta))
            nu = float(m.get('nu', nu))
            mu = float(m.get('mu', mu))
        return dict(Delta_T=dT, beta=beta, nu=nu, mu=mu)

    def __len__(self):
        return self._cum[-1] if self._cum else 0

    def _locate(self, i: int):
        """global index -> (member index, local index)."""
        s = bisect.bisect_right(self._cum, i)
        lo = self._cum[s - 1] if s > 0 else 0
        return s, i - lo

    def sample_grid_shape(self, i: int):
        s, _ = self._locate(i)
        return self.grid_shapes[s]

    def __getitem__(self, i):
        s, j = self._locate(i)
        x, y = self.subsets[s][j]
        if self.return_regime:
            return x, y, self.regime_vecs[s]
        return x, y


class GridHomogeneousBatchSampler(Sampler):
    """Yield batches whose samples share a grid (Ny,Nx), so default collate works.

    Buckets the dataset's global indices by grid shape, shuffles within each
    bucket, chunks into batch_size, then (optionally) shuffles batch order across
    buckets. Call set_epoch(e) each epoch for a fresh shuffle.
    """

    def __init__(self, dataset: ConcatClosureDataset, batch_size: int,
                 shuffle: bool = True, drop_last: bool = False, seed: int = 0):
        self.ds = dataset
        self.bs = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        buckets = {}
        for i in range(len(dataset)):
            buckets.setdefault(dataset.sample_grid_shape(i), []).append(i)
        self.buckets = buckets
        self._len = sum(
            (len(v) // self.bs) if self.drop_last else math.ceil(len(v) / self.bs)
            for v in buckets.values())

    def set_epoch(self, e: int):
        self.epoch = int(e)

    def __len__(self):
        return self._len

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        batches = []
        for idxs in self.buckets.values():
            order = (torch.randperm(len(idxs), generator=g).tolist()
                     if self.shuffle else list(range(len(idxs))))
            idxs = [idxs[p] for p in order]
            for k in range(0, len(idxs), self.bs):
                b = idxs[k:k + self.bs]
                if self.drop_last and len(b) < self.bs:
                    continue
                batches.append(b)
        if self.shuffle:
            perm = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[p] for p in perm]
        yield from batches


def make_concat_loaders(root_dirs, batch_size=4, num_workers=2, n_snapshots=4,
                        target_fields=('N_dot_0_anal', 'N_ddot_0_anal', 'N_3dot_0_anal'),
                        packed_subdir='packed', compute_dtype='float32',
                        return_regime=False, seed=0):
    """(train, val, test) loaders + datasets over a pooled ensemble.

    Mirrors dataset.make_loaders' return signature so it is a drop-in swap.
    Uses GridHomogeneousBatchSampler automatically when the pool mixes grids.
    """
    from torch.utils.data import DataLoader

    def make(split, shuffle):
        ds = ConcatClosureDataset(root_dirs, split=split, n_snapshots=n_snapshots,
                                  target_fields=target_fields,
                                  packed_subdir=packed_subdir,
                                  compute_dtype=compute_dtype,
                                  return_regime=return_regime)
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

    train_loader, train_ds = make('train', True)
    val_loader, val_ds = make('val', False)
    test_loader, test_ds = make('test', False)

    print(f"[concat] members(train)={len(train_ds.subsets)} "
          f"n_snapshots={n_snapshots} in_ch={2*n_snapshots} "
          f"grids={sorted(set(train_ds.grid_shapes))} "
          f"train/val/test={len(train_ds)}/{len(val_ds)}/{len(test_ds)} "
          f"regime={'on' if return_regime else 'off'}")
    if len(set(train_ds.grid_shapes)) > 1:
        print("[concat] mixed grids -> GridHomogeneousBatchSampler "
              "(call sampler.set_epoch(e) per epoch via the loader's batch_sampler)")
    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds


if __name__ == '__main__':
    import sys
    roots = sys.argv[1:]
    if not roots:
        print("usage: python concat_dataset.py ROOT [ROOT ...]")
        raise SystemExit(0)
    for n in (4, 7):
        try:
            ds = ConcatClosureDataset(roots, split='train', n_snapshots=n,
                                      return_regime=True)
            print(f"n_snapshots={n}: len={len(ds)} in_fields={ds.input_fields}")
            x, y, r = ds[0]
            print(f"   x={tuple(x.shape)} y={tuple(y.shape)} "
                  f"regime(dT,beta,nu,mu)={r.tolist()} grids={set(ds.grid_shapes)}")
        except Exception as e:  # noqa: BLE001 -- smoke test
            print(f"n_snapshots={n}: {type(e).__name__}: {e}")
