"""
ic_patch.py

Patch for ic.py to add a 'from_file' IC function.

This adds a new entry to ic_library that loads omega from a saved .npy file
on disk -- enabling restart-from-snapshot for dt-sensitivity sweeps.

To apply the patch:
  - Either copy the _init_from_file function below into your ic.py and add
    'from_file': _init_from_file to the ic_library dict.
  - OR have run_qg.py monkeypatch ic.py at import time with this module:
        from qg.solver.ic import ic_library
        from . import ic_patch  # registers _init_from_file
"""

import numpy as np
import torch

from qg.solver.opt.basis import to_spectral


def _init_from_file(grid, derivative,
                    path: str = None,
                    n_batch: int = 1,
                    persistent: bool = False,
                    **kwargs):
    """
    Initial condition loaded from a saved .npy file containing omega(x,y).

    Args:
        grid:       CartesianGrid (for shape + device)
        derivative: Derivative (unused but signature compat)
        path:       absolute path to a .npy of shape (Ny, Nx) or (B, Ny, Nx)
                    or (1, Ny, Nx).
        n_batch:    if path has only one batch, broadcast to this many.
        persistent: ignored (kept for API parity with _init_randn).

    Returns: spectral-space initial vorticity qh of shape (n_batch, Ny, Nx_half).
    """
    if path is None:
        raise ValueError("ic.from_file requires a 'path' argument")

    arr = np.load(path)
    if arr.ndim == 2:
        Ny, Nx = arr.shape
        arr = arr[None]  # (1, Ny, Nx)
    elif arr.ndim == 3:
        pass  # (B_or_1, Ny, Nx)
    else:
        raise RuntimeError(f"ic.from_file: expected 2D or 3D npy, got ndim={arr.ndim}")

    if arr.shape[1] != grid.Ny or arr.shape[2] != grid.Nx:
        raise RuntimeError(
            f"ic.from_file: snapshot shape {arr.shape[1:]} does not match "
            f"grid ({grid.Ny}, {grid.Nx}). Use the same Nx, Ny as the source DNS.")

    # Broadcast to n_batch if only one batch
    if arr.shape[0] == 1 and n_batch > 1:
        arr = np.broadcast_to(arr, (n_batch, grid.Ny, grid.Nx)).copy()
    elif arr.shape[0] != n_batch:
        raise RuntimeError(
            f"ic.from_file: snapshot has {arr.shape[0]} batches but "
            f"ic.n_batch={n_batch}. Either match them or save a single-batch IC "
            f"and let it broadcast.")

    omega = torch.tensor(arr, dtype=torch.float64).to(grid.device)
    qh = to_spectral(omega)  # (n_batch, Ny, Nx_half)
    return qh
