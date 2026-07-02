"""
dataset.py - PyTorch Dataset for the closure-NN training data.

Loads samples from a directory built by build_training_data.py:
    root_dir/
        manifest.json
        split.npz                   # train_idx, val_idx, test_idx
        samples/
            sample_NNNNNN.npz       # one record each

Usage:
    from dataset import ClosureDataset

    train_ds = ClosureDataset(root, split='train',
                              input_fields=('omega_0', 'psi_0'),
                              target_field='f_NN_target')
    val_ds   = ClosureDataset(root, split='val',   ...)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,
                              num_workers=2, pin_memory=True)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# Channels the build script saves and that we can use as NN inputs.
AVAILABLE_INPUT_FIELDS = (
    'omega_0',         # vorticity at t^0
    'psi_0',           # streamfunction at t^0
    'grad_psi_sq',     # |grad psi|^2 (proxy for kinetic energy density)
    'omega_x',         # d omega / dx at t^0
    'omega_y',         # d omega / dy at t^0
    'omega_m1',        # vorticity at t^{-1} (also useful for AB2 stencil reasoning)
    # NEW (Fix D): analytical scaffolding fields the closure formula needs.
    # The closure target (1/12)[L*Ndot - 5*Nddot] requires Ndot and Nddot,
    # which the NN cannot compose from raw (omega, psi) easily. Adding these
    # as input channels lets the NN combine them via local k-dependent
    # multipliers instead of having to derive them from convolutions.
    #
    # Inference-cost categorization:
    #   FREE:   N_0 (already computed for AB2 RHS at every step)
    #   CHEAP:  L_omega, L2_omega, L3_omega, L_N, L2_N (spectral multipliers)
    #   EXPENSIVE: N_dot_0_anal (2 Jacobians), N_ddot_0_anal (3 Jacobians)
    #
    # ALL fields are saved to npz for diagnostics, but production training
    # should NOT use the EXPENSIVE ones as inputs because computing them at
    # inference would defeat the purpose of replacing them with an NN.
    # We register them here only so that experiments using them as inputs
    # (e.g. for ablation studies) can still load the dataset.
    'N_0',             # nonlinear RHS at t^0:  N = -J(psi, omega) + F  [FREE]
    'N_dot_0_anal',    # analytical dN/dt  at t^0  [EXPENSIVE: 2 Jacobians]
    'N_ddot_0_anal',   # analytical d^2N/dt^2 at t^0  [EXPENSIVE: 3 Jacobians]
    'L_omega',         # L * omega^0  [CHEAP]
    'L2_omega',        # L^2 * omega^0  [CHEAP]
    'L3_omega',        # L^3 * omega^0  [CHEAP]
    'L_N',             # L * N(omega^0)  [CHEAP]
    'L2_N',            # L^2 * N(omega^0)  [CHEAP]
)

# Channels we typically use as targets (model regresses these).
AVAILABLE_TARGET_FIELDS = (
    'f_NN_target',     # NEW (Fix D): bare physics bracket (1/12)*[L*Ndot-5*Nddot]
    'e_NN_incr',       # the NN-residual increment (alternative target)
    'f_NN_target_from_e',  # diagnostic: numerical f_NN_target from -e_NN_incr/coef
)


class ClosureDataset(Dataset):
    """Loads (input_channels, target_channel) pairs from per-sample .npz files."""

    def __init__(
        self,
        root_dir: str | Path,
        split: str = 'train',
        input_fields: Sequence[str] = ('omega_0',),
        target_field: str = 'f_NN_target',
        normalize: bool = False,
        norm_stats_path: Optional[Path] = None,
    ):
        """
        Args:
            root_dir:      path to the dataset directory (contains manifest.json)
            split:         'train', 'val', or 'test' -- selects which indices to use
            input_fields:  tuple of field names to stack as input channels.
                           Default: just ('omega_0',). Useful alternatives:
                           ('omega_0', 'psi_0'), or all five.
            target_field:  which field to regress against. Default: 'f_NN_target'.
            normalize:     if True, normalize inputs and target by per-channel stats
                           computed once on the train set.
            norm_stats_path: optional path to load/save normalization stats. If
                             normalize=True and split='train', computes & saves
                             stats here. If split='val'/'test', loads them.
        """
        super().__init__()
        self.root = Path(root_dir)
        manifest_path = self.root / 'manifest.json'
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest not found: {manifest_path}")
        with open(manifest_path) as f:
            self.manifest = json.load(f)

        # Validate field names
        for f in input_fields:
            if f not in AVAILABLE_INPUT_FIELDS and f != target_field:
                raise ValueError(
                    f"input field '{f}' not in known set {AVAILABLE_INPUT_FIELDS}")
        if target_field not in AVAILABLE_TARGET_FIELDS:
            raise ValueError(
                f"target field '{target_field}' not in known set "
                f"{AVAILABLE_TARGET_FIELDS}")

        # Load split indices
        split_path = self.root / 'split.npz'
        with np.load(split_path) as sp:
            key = f'{split}_idx'
            if key not in sp.files:
                raise ValueError(f"split '{split}' not found; available: {sp.files}")
            self.indices = sp[key].astype(np.int64)

        self.input_fields = tuple(input_fields)
        self.target_field = target_field

        self.Nx = int(self.manifest['Nx'])
        self.Ny = int(self.manifest['Ny'])

        # Optional normalization
        self.normalize = bool(normalize)
        if self.normalize:
            if norm_stats_path is None:
                norm_stats_path = self.root / 'norm_stats.npz'
            self._setup_normalization(split, Path(norm_stats_path))
        else:
            self.input_mean = self.input_std = None
            self.target_mean = self.target_std = None

    def _setup_normalization(self, split: str, stats_path: Path) -> None:
        """If split=='train' and stats not on disk, compute & save. Else load."""
        if split == 'train' and not stats_path.exists():
            print(f"[ClosureDataset] computing normalization stats for {len(self.indices)} train samples...")
            in_means, in_sqsums, n = (
                np.zeros(len(self.input_fields), dtype=np.float64),
                np.zeros(len(self.input_fields), dtype=np.float64),
                0,
            )
            t_mean = 0.0
            t_sqsum = 0.0
            for idx in self.indices:
                rec = self._load_record(int(idx))
                for ch_i, fname in enumerate(self.input_fields):
                    a = rec[fname]
                    in_means[ch_i] += a.mean()
                    in_sqsums[ch_i] += np.mean(a * a)
                t = rec[self.target_field]
                t_mean += t.mean()
                t_sqsum += np.mean(t * t)
                n += 1
            in_means /= n
            in_sqsums /= n
            t_mean /= n
            t_sqsum /= n
            in_var = in_sqsums - in_means ** 2
            t_var = t_sqsum - t_mean ** 2
            in_std = np.sqrt(np.maximum(in_var, 1e-30))
            t_std = np.sqrt(max(t_var, 1e-30))
            np.savez(stats_path,
                     input_fields=np.array(self.input_fields),
                     input_mean=in_means.astype(np.float32),
                     input_std=in_std.astype(np.float32),
                     target_field=np.array([self.target_field]),
                     target_mean=np.array([t_mean], dtype=np.float32),
                     target_std=np.array([t_std], dtype=np.float32))
            print(f"[ClosureDataset] saved norm stats to {stats_path}")
        if not stats_path.exists():
            raise FileNotFoundError(
                f"normalization stats not found at {stats_path}; "
                f"run with split='train' first to generate them.")
        with np.load(stats_path, allow_pickle=False) as st:
            self.input_mean   = torch.from_numpy(st['input_mean'].astype(np.float32))
            self.input_std    = torch.from_numpy(st['input_std'].astype(np.float32))
            self.target_mean  = float(st['target_mean'][0])
            self.target_std   = float(st['target_std'][0])

    def _load_record(self, sample_idx: int) -> dict:
        path = self.root / 'samples' / f'sample_{sample_idx:06d}.npz'
        with np.load(path) as zf:
            return {k: np.array(zf[k]) for k in zf.files}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        sample_idx = int(self.indices[i])
        rec = self._load_record(sample_idx)
        # stack input channels: (C, Ny, Nx)
        x = np.stack([rec[f] for f in self.input_fields], axis=0).astype(np.float32)
        y = rec[self.target_field].astype(np.float32)[None]  # (1, Ny, Nx)
        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)
        if self.normalize:
            x_t = (x_t - self.input_mean[:, None, None]) / self.input_std[:, None, None]
            y_t = (y_t - self.target_mean) / self.target_std
        return x_t, y_t

    # Convenience: scalar metadata for diagnostics
    def get_meta(self, i: int) -> dict:
        sample_idx = int(self.indices[i])
        rec = self._load_record(sample_idx)
        return dict(
            seed_t=float(rec['seed_t']),
            seed_idx=int(rec['seed_idx']),
            batch_idx=int(rec['batch_idx']),
        )


def make_loaders(root_dir, batch_size=4, num_workers=2,
                 input_fields=('omega_0',), target_field='f_NN_target',
                 normalize=False):
    """Convenience constructor: returns (train, val, test) DataLoaders."""
    from torch.utils.data import DataLoader

    train_ds = ClosureDataset(root_dir, split='train',
                              input_fields=input_fields,
                              target_field=target_field,
                              normalize=normalize)
    val_ds   = ClosureDataset(root_dir, split='val',
                              input_fields=input_fields,
                              target_field=target_field,
                              normalize=normalize)
    test_ds  = ClosureDataset(root_dir, split='test',
                              input_fields=input_fields,
                              target_field=target_field,
                              normalize=normalize)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds


if __name__ == '__main__':
    import sys
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('./data/decaying_turbulence_dT_1em3')
    ds = ClosureDataset(root, split='train',
                        input_fields=('omega_0', 'psi_0'),
                        target_field='f_NN_target')
    print(f"Dataset size: {len(ds)}")
    x, y = ds[0]
    print(f"Sample 0: x.shape={x.shape}  y.shape={y.shape}")
    print(f"  x stats per channel: mean={x.mean(dim=(-1,-2)).tolist()}  "
          f"std={x.std(dim=(-1,-2)).tolist()}")
    print(f"  y stats:             mean={y.mean().item():.3e}  std={y.std().item():.3e}")
    print(f"  meta: {ds.get_meta(0)}")
