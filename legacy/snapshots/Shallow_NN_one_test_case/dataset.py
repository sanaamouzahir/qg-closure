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

    # multi-channel N-derivative target (cheap_deriv model):
    train_ds = ClosureDataset(root, split='train',
                              input_fields=('omega_0','omega_m1','omega_m2',
                                            'psi_0','psi_m1','psi_m2'),
                              target_fields=('N_dot_0_anal','N_ddot_0_anal',
                                             'N_3dot_0_anal'))

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
    # Core inputs for v2 minimal architecture
    'omega_0',         # vorticity at t^0
    'omega_m1',        # vorticity at t^{-1}
    'omega_m2',        # vorticity at t^{-2}        [v2 NEW]
    'psi_0',           # streamfunction at t^0
    'psi_m1',          # streamfunction at t^{-1}    [v2 NEW]
    'psi_m2',          # streamfunction at t^{-2}    [v2 NEW]
    # Backward-compat / old options (saved but generally not used by v2)
    'grad_psi_sq',     # |grad psi|^2 (proxy for kinetic energy density)
    'omega_x',         # d omega / dx at t^0
    'omega_y',         # d omega / dy at t^0
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
    # DERIVED on the fly (computed in __getitem__ from saved fields):
    #   N_m1 = -J(psi_m1, omega_m1) + F     where psi_m1 = inv_lap(omega_m1)
    # Cost: 1 Jacobian per sample-load. NOT saved to disk, but available as
    # if it were. At inference, N_m1 is FREE because it's just N^{n-1} which
    # AB2 already cached from the previous step. So including it as an input
    # is consistent with the "no extra inference cost" rule.
    'N_m1',            # nonlinear RHS at t^{-1}: N(omega_m1)  [DERIVED, free at inference]
)

# Fields computed on the fly in __getitem__ from saved fields. The build
# script never produces these directly; they're synthesized at load time.
DERIVED_FIELDS = ('N_m1',)

# Channels we typically use as targets (model regresses these).
AVAILABLE_TARGET_FIELDS = (
    # ---- OPERATIVE multi-channel target (cheap_deriv) -------------------- #
    # The model predicts the RAW local N-time-derivatives; the L^k brackets of
    # every truncation order are assembled spectrally at inference, never
    # learned. One target set serves all orders:
    #   R3 = (1/12)[L*Ndot - 5*Nddot]
    #   R4 = (1/24)[2*L^2*Ndot - 4*L*Nddot + N3dot]
    #   R5 = (1/240)[...Ndot..Nddot..N3dot.. - 7*N4dot]
    # so (Ndot, Nddot)           -> R3
    #    (Ndot, Nddot, N3dot)    -> R3 + R4   <-- current run (out_orders=3)
    #    (Ndot, Nddot, N3dot, N4dot) -> R3 + R4 + R5
    'N_dot_0_anal',    # dN/dt    at t^0   (R3, R4, R5)
    'N_ddot_0_anal',   # d2N/dt2  at t^0   (R3, R4, R5)
    'N_3dot_0_anal',   # d3N/dt3  at t^0   (R4, R5)
    'N_4dot_0_anal',   # d4N/dt4  at t^0   (R5)
    # ---- LEGACY / ablation / diagnostic (do NOT use for new runs) -------- #
    # Pre-combined single-channel R3 bracket. Superseded by the raw operators
    # above: it bakes in ONLY R3 and cannot be extended to R4+, so it is kept
    # solely to reload old single-channel runs.
    'f_NN_target',     # (1/12)*[L*Ndot - 5*Nddot]   [LEGACY, R3-only]
    'e_NN_incr',       # NN-residual increment        [diagnostic]
    'f_NN_target_from_e',  # numerical f_NN_target from -e_NN_incr/coef [diagnostic]
)


class ClosureDataset(Dataset):
    """Loads (input_channels, target_channels) pairs from per-sample .npz files."""

    def __init__(
        self,
        root_dir: str | Path,
        split: str = 'train',
        input_fields: Sequence[str] = ('omega_0',),
        target_field: str = 'f_NN_target',
        target_fields: Optional[Sequence[str]] = None,
        normalize: bool = False,
        norm_stats_path: Optional[Path] = None,
    ):
        """
        Args:
            root_dir:      path to the dataset directory (contains manifest.json)
            split:         'train', 'val', or 'test' -- selects which indices to use
            input_fields:  tuple of field names to stack as input channels.
                           Default: just ('omega_0',). Useful alternatives:
                           ('omega_0', 'psi_0'), or all six.
            target_field:  single field to regress against. Default: 'f_NN_target'.
            target_fields: optional list of fields for a MULTI-CHANNEL target
                           (e.g. ('N_dot_0_anal','N_ddot_0_anal','N_3dot_0_anal')).
                           Overrides target_field if given.
            normalize:     if True, normalize inputs and target by per-channel stats
                           computed once on the train set. NOT supported for
                           multi-channel targets (use a relative loss instead).
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

        # Resolve target: target_fields (plural) overrides target_field (single).
        self.target_fields = tuple(target_fields) if target_fields else (target_field,)
        self.target_field = self.target_fields[0]  # back-compat (norm stats etc.)

        # Validate field names
        for f in input_fields:
            if f not in AVAILABLE_INPUT_FIELDS and f not in self.target_fields:
                raise ValueError(
                    f"input field '{f}' not in known set {AVAILABLE_INPUT_FIELDS}")
        for t in self.target_fields:
            if t not in AVAILABLE_TARGET_FIELDS:
                raise ValueError(
                    f"target field '{t}' not in known set {AVAILABLE_TARGET_FIELDS}")

        # Load split indices
        split_path = self.root / 'split.npz'
        with np.load(split_path) as sp:
            key = f'{split}_idx'
            if key not in sp.files:
                raise ValueError(f"split '{split}' not found; available: {sp.files}")
            self.indices = sp[key].astype(np.int64)

        self.input_fields = tuple(input_fields)
        # self.target_field / self.target_fields already resolved above

        self.Nx = int(self.manifest['Nx'])
        self.Ny = int(self.manifest['Ny'])

        # If any input field is a DERIVED field, build the spectral helpers
        # (kx, ky, inv_lap) needed to compute it on the fly. We use numpy FFTs
        # so this works in CPU dataloader workers.
        self._derived_in_use = tuple(f for f in self.input_fields
                                      if f in DERIVED_FIELDS)
        if self._derived_in_use:
            Lx = float(self.manifest['Lx'])
            Ly = float(self.manifest['Ly'])
            # Wavenumbers matching numpy.fft.fft2 conventions.
            # rfft2 only returns the positive half of the last (x) axis.
            kx_full = 2 * np.pi * np.fft.fftfreq(self.Nx, d=Lx / self.Nx)
            ky_full = 2 * np.pi * np.fft.fftfreq(self.Ny, d=Ly / self.Ny)
            # broadcast (Ny, Nx)
            self._kx2d = kx_full[None, :].astype(np.float64)         # (1, Nx)
            self._ky2d = ky_full[:, None].astype(np.float64)         # (Ny, 1)
            k_sq = self._kx2d ** 2 + self._ky2d ** 2                  # (Ny, Nx)
            # Avoid divide-by-zero at k=0 (mean mode); psi mean is gauge-fixed.
            self._inv_k_sq = np.where(k_sq > 0.0, -1.0 / k_sq, 0.0)
            # Forcing F is None for the closure scenarios we care about
            # (decay, no forcing). For forced turbulence we'd need to load F
            # from somewhere -- punt on that case unless asked.
            scenario = str(self.manifest.get('scenario', ''))
            if scenario == 'forced_turbulence':
                # WARN but don't fail; user may not actually need N_m1.
                print(f"[ClosureDataset] WARN: derived field N_m1 requested for "
                      f"forced_turbulence scenario; F not handled in derived "
                      f"computation. Result will be N_m1 = -J(psi_m1, omega_m1) "
                      f"WITHOUT forcing term.")

        # Optional normalization
        self.normalize = bool(normalize)
        if self.normalize and len(self.target_fields) > 1:
            raise NotImplementedError(
                "per-channel target normalization is not implemented for "
                "multi-channel targets; use --loss rel_l2 instead of --normalize")
        if self.normalize:
            if norm_stats_path is None:
                norm_stats_path = self.root / 'norm_stats.npz'
            self._setup_normalization(split, Path(norm_stats_path))
        else:
            self.input_mean = self.input_std = None
            self.target_mean = self.target_std = None

    # -------------- Derived fields (Option A: compute on the fly) -------------- #

    def _compute_n_m1(self, rec: dict) -> np.ndarray:
        """
        Compute N(omega^{-1}) = -J(psi^{-1}, omega^{-1}) [+ F]  on the fly.

        Inputs:
          rec['omega_m1'] : (Ny, Nx) float32, the saved previous-step vorticity.

        Returns:
          (Ny, Nx) float32, the nonlinear RHS at omega^{-1}, in physical space.

        Sign convention matches build_training_data_fixD.py:
          N = -J(psi, omega) + F,  with J written in flux/divergence form
              J(psi, omega) = +d/dx(u*omega) + d/dy(v*omega),  u = -d_y psi, v = +d_x psi.

        Cost per call: 1 Jacobian = ~5 FFTs on a (Ny, Nx) grid. Using numpy
        FFTs so this runs cleanly in dataloader worker processes.
        """
        omega_m1 = rec['omega_m1']
        # Spectralize once
        omh = np.fft.fft2(omega_m1)
        # psi_hat = -omega_hat / k^2 (i.e. inv-Laplacian; sign in inv_k_sq)
        psih = self._inv_k_sq * omh
        # u = -d psi/dy = -i ky * psih,  v = +d psi/dx = +i kx * psih
        uh = -1j * self._ky2d * psih
        vh = +1j * self._kx2d * psih
        u = np.real(np.fft.ifft2(uh))
        v = np.real(np.fft.ifft2(vh))
        # J in flux form: d_x(u*omega) + d_y(v*omega)
        uq_h = np.fft.fft2(u * omega_m1)
        vq_h = np.fft.fft2(v * omega_m1)
        j_hat = 1j * self._kx2d * uq_h + 1j * self._ky2d * vq_h
        j_phys = np.real(np.fft.ifft2(j_hat))
        # N = -J(psi, omega) + F. For decay, F=0.
        return (-j_phys).astype(np.float32)

    def _get_field(self, rec: dict, fname: str) -> np.ndarray:
        """Return a saved field or a derived field, by name."""
        if fname in DERIVED_FIELDS:
            if fname == 'N_m1':
                return self._compute_n_m1(rec)
            raise ValueError(f"unknown derived field '{fname}'")
        return rec[fname]

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
                    a = self._get_field(rec, fname)
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
        x = np.stack([self._get_field(rec, f) for f in self.input_fields],
                     axis=0).astype(np.float32)
        y = np.stack([rec[t] for t in self.target_fields],
                     axis=0).astype(np.float32)  # (C_out, Ny, Nx)
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


def _packed_available(root_dir, packed_subdir='packed') -> bool:
    p = Path(root_dir) / packed_subdir
    return ((p / 'inputs.npy').exists() and (p / 'targets.npy').exists()
            and (p / 'pack_meta.json').exists())


class PackedClosureDataset(Dataset):
    """Fast path: contiguous float32 memmaps written by pack_dataset_mmap.py.

    No zip-opens, no DEFLATE, OS-cacheable. Layout under root/<packed_subdir>/:
        inputs.npy   (N, C_in,  Ny, Nx) float32
        targets.npy  (N, C_out, Ny, Nx) float32
        pack_meta.json   -- records the packed field order
        split.npz        -- (or falls back to root/split.npz)
    Requested fields are mapped to columns BY NAME, so they may be a subset or
    reordering of what was packed.
    """

    def __init__(self, root_dir, split='train',
                 input_fields=('omega_0',), target_field='f_NN_target',
                 target_fields=None, normalize=False, packed_subdir='packed',
                 compute_dtype='float32'):
        super().__init__()
        self.root = Path(root_dir)
        self.pdir = self.root / packed_subdir
        with open(self.pdir / 'pack_meta.json') as f:
            self.meta = json.load(f)
        self._np_dtype = np.float64 if compute_dtype == 'float64' else np.float32

        self.target_fields = tuple(target_fields) if target_fields else (target_field,)
        self.target_field = self.target_fields[0]
        self.input_fields = tuple(input_fields)

        if normalize:
            raise NotImplementedError(
                "normalize is not supported on the packed fast path; use "
                "--loss rel_l2 (no --normalize), or the unpacked ClosureDataset.")
        self.normalize = False

        in_packed = {n: c for c, n in enumerate(self.meta['input_fields'])}
        tg_packed = {n: c for c, n in enumerate(self.meta['target_fields'])}
        try:
            self._in_cols = [in_packed[f] for f in self.input_fields]
        except KeyError as e:
            raise ValueError(f"input field {e} not in packed set "
                             f"{self.meta['input_fields']}; repack with it.")
        try:
            self._tg_cols = [tg_packed[f] for f in self.target_fields]
        except KeyError as e:
            raise ValueError(f"target field {e} not in packed set "
                             f"{self.meta['target_fields']}; repack with it.")

        self.Ny, self.Nx = int(self.meta['Ny']), int(self.meta['Nx'])

        split_path = self.pdir / 'split.npz'
        if not split_path.exists():
            split_path = self.root / 'split.npz'
        with np.load(split_path) as sp:
            key = f'{split}_idx'
            if key not in sp.files:
                raise ValueError(f"split '{split}' not found; available: {sp.files}")
            self.indices = sp[key].astype(np.int64)

        # opened lazily so each dataloader worker mmaps in its own process
        self._inp = None
        self._tgt = None

    def _ensure_open(self):
        if self._inp is None:
            self._inp = np.load(self.pdir / 'inputs.npy', mmap_mode='r')
            self._tgt = np.load(self.pdir / 'targets.npy', mmap_mode='r')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        self._ensure_open()
        idx = int(self.indices[i])
        # fancy-index the channel dim -> contiguous copy in the compute dtype
        x = np.ascontiguousarray(self._inp[idx][self._in_cols], dtype=self._np_dtype)
        y = np.ascontiguousarray(self._tgt[idx][self._tg_cols], dtype=self._np_dtype)
        return torch.from_numpy(x), torch.from_numpy(y)

    def get_meta(self, i):
        return dict(index=int(self.indices[i]))


def make_loaders(root_dir, batch_size=4, num_workers=2,
                 input_fields=('omega_0',), target_field='f_NN_target',
                 target_fields=None, normalize=False, packed='auto',
                 packed_subdir='packed', compute_dtype='float32'):
    """Convenience constructor: returns (train, val, test) loaders + datasets.

    packed: 'auto' (use packed memmaps if present), True (force), False (force
    per-sample npz). Auto falls back to npz if --normalize is requested.
    compute_dtype: 'float32' or 'float64' (packed path only); float64 keeps the
    high-order time-FD clean and needs float64 packed inputs to be worthwhile.
    """
    from torch.utils.data import DataLoader

    use_packed = (packed is True) or (
        packed == 'auto' and _packed_available(root_dir, packed_subdir))
    if use_packed and normalize:
        print("[make_loaders] normalize requested -> falling back to unpacked "
              "dataset (packed path has no normalize).")
        use_packed = False

    def make(split):
        if use_packed:
            return PackedClosureDataset(root_dir, split=split,
                                        input_fields=input_fields,
                                        target_field=target_field,
                                        target_fields=target_fields,
                                        normalize=normalize,
                                        packed_subdir=packed_subdir,
                                        compute_dtype=compute_dtype)
        return ClosureDataset(root_dir, split=split,
                              input_fields=input_fields,
                              target_field=target_field,
                              target_fields=target_fields,
                              normalize=normalize)

    train_ds, val_ds, test_ds = make('train'), make('val'), make('test')
    src = (Path(root_dir) / packed_subdir) if use_packed else root_dir
    print(f"[make_loaders] using {'PACKED' if use_packed else 'per-sample npz'} "
          f"dataset at {src}"
          + (f"  (compute_dtype={compute_dtype})" if use_packed else ""))

    pw = num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=False, persistent_workers=pw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=pw)
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
