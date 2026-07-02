"""
build_training_data.py

Build a supervised-learning dataset for the temporal-closure NN, by repeatedly
applying the Test C procedure across many seed snapshots from a long DNS run.

Supports MULTI-BATCH ensembles: for chaotic scenarios like decaying turbulence,
each batch in the source npy is an independent realization of the same flow.
Training across batches gives the NN access to a much wider sampling of the
flow's state distribution, which would otherwise be highly correlated within
a single trajectory.

For each (batch b, seed t*):
  1) Ultra-fine RK4 forward by 2*Delta_T to build the AB2 stencil
     (omega_{-1} at offset Delta_T, omega_0 at offset 2*Delta_T).
  2) ONE coarse AB2CN2 step of size Delta_T from (omega_{-1}, omega_0)
     -> omega_1_coarse.
  3) K fine AB2CN2 steps of size h_fine = Delta_T / K from the same stencil
     -> omega_K_fine. This is "truth" by definition (c).
  4) e_total      = omega_K_fine - omega_1_coarse                  (full closure)
  5) e_anal_incr  = - (Delta_T^3) * (1 - 1/K^2) * (1/12) * [L^3 omega_0 + L^2 N_0]
                    (sign flipped vs predicted_local_error_phys, which returns
                     coarse - fine; here we want fine - coarse).
  6) e_NN_incr    = e_total - e_anal_incr   (training error increment)
  7) f_NN_target  = S(Delta_T)^{-1} * e_NN_incr  in spectral space, where
                    S(Delta_T) = Delta_T / (1 - 0.5*Delta_T*L_hat) is the AB2CN2
                    response operator to a constant added forcing.
                    -> S^{-1} = (1 - 0.5*Delta_T*L_hat) / Delta_T
     This is the field we ask the NN to predict. Adding f_NN to the QG RHS at
     coarse-step inference time recovers omega_K_fine to leading order.
  8) f_anal       = S(Delta_T)^{-1} * e_anal_incr  (also saved; computed from
                    omega_0 at inference time analytically, but precomputed here
                    for diagnostics).

Output: a directory at out_dir/<scenario>_dT_<tag>/ containing:
  manifest.json     -- dataset-level metadata (one place for attrs)
  split.npz         -- train_idx, val_idx, test_idx (int32 arrays of row indices)
  samples/
    sample_000000.npz, sample_000001.npz, ...
        each .npz contains:
          omega_0, psi_0, grad_psi_sq, omega_x, omega_y       (Ny, Nx) float32
          omega_m1, omega_1_coarse, omega_K_fine              (Ny, Nx) float32
          e_total, e_anal_incr, e_NN_incr                     (Ny, Nx) float32
          f_NN_target, f_anal                                 (Ny, Nx) float32
          seed_t, seed_idx, batch_idx                         scalars

Records are stored in the order [batch[0]:seeds, batch[1]:seeds, ...] -- so
indices [b*seeds_per_batch, (b+1)*seeds_per_batch) all come from one batch.

Pros of this layout:
  - No external dependency (h5py not required)
  - Each sample is one file, so a crashed job keeps all completed samples
  - PyTorch Dataset can just glob `samples/sample_*.npz`
  - Compressed by default (np.savez_compressed)

Usage examples:

  # Single-batch (original Test-C-like) build:
  python build_training_data.py \\
      --scenario decaying_turbulence \\
      --source-omega .../DNS_FR_omega.npy \\
      --source-times .../DNS_FR_times.npy \\
      --source-yaml  .../decaying_turbulence.yaml \\
      --out-dir      ./training_data/ \\
      --batch-index 0  --n-seeds 1000

  # Ensemble build, all 20 batches, 500 seeds each, by-batch split:
  python build_training_data.py \\
      --scenario decaying_turbulence \\
      ... \\
      --n-batches 20  --n-seeds 500
      # split-mode 'auto' will pick by_batch since n_batches > 1

The split (train/val/test) is time-block: first 70% of seeds train,
next 15% val, last 15% test.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import yaml

# QG operators
from qg.solver.grid.cartesian import CartesianGrid
from qg.solver.opt.derivative import Derivative
from qg.solver.opt.basis import to_spectral, to_physical


# --------------------------------------------------------------------------- #
# QG operator wrappers (lifted from Step 3, behavior-identical)               #
# --------------------------------------------------------------------------- #

def J_phys(psi_phys: torch.Tensor, omega_phys: torch.Tensor,
           derivative) -> torch.Tensor:
    """Jacobian J(psi, omega) = u dq/dx + v dq/dy with u=-dpsi/dy, v=+dpsi/dx."""
    psih = to_spectral(psi_phys)
    qh   = to_spectral(omega_phys)
    uh = -1 * derivative.dy * psih
    vh = +1 * derivative.dx * psih
    u = to_physical(uh)
    v = to_physical(vh)
    q = to_physical(qh)
    uq_h = to_spectral(u * q)
    vq_h = to_spectral(v * q)
    j_hat = derivative.dx * uq_h + derivative.dy * vq_h
    return to_physical(j_hat)


def L_op(omega_phys: torch.Tensor, L_hat: torch.Tensor) -> torch.Tensor:
    qh = to_spectral(omega_phys)
    return to_physical(L_hat * qh)


def build_L_hat(derivative, nu: float, mu: float, B: float) -> torch.Tensor:
    """L = nu * Lap - mu * I - i * B * dx * inv_lap. Spectral diagonal."""
    L_hat = nu * derivative.laplacian - mu
    if B != 0.0:
        L_hat = L_hat - B * derivative.dx * derivative.inv_laplacian
    return L_hat


def ab2cn2_step_spectral(qh_n: torch.Tensor, qh_nm1: torch.Tensor,
                         dt: float, derivative, L_hat: torch.Tensor,
                         F_phys: Optional[torch.Tensor]) -> torch.Tensor:
    """One AB2CN2 IMEX step. Matches Step 3's implementation exactly."""
    def N_at_qh(qh):
        psi_hat = derivative.inv_laplacian * qh
        psi = to_physical(psi_hat)
        omega = to_physical(qh)
        N_phys = -1.0 * J_phys(psi, omega, derivative)
        if F_phys is not None:
            N_phys = N_phys + F_phys
        return to_spectral(N_phys)

    Nh_n   = N_at_qh(qh_n)
    Nh_nm1 = N_at_qh(qh_nm1)
    AB2_Nh = 1.5 * Nh_n - 0.5 * Nh_nm1

    rhs_hat = qh_n + dt * (0.5 * L_hat * qh_n + AB2_Nh)
    denom_hat = 1.0 - 0.5 * dt * L_hat
    return rhs_hat / denom_hat


def rk4_step(omega: torch.Tensor, dt: float, derivative,
             L_hat: torch.Tensor, F_phys: Optional[torch.Tensor]) -> torch.Tensor:
    def rhs_phys(om):
        psi_hat = derivative.inv_laplacian * to_spectral(om)
        psi = to_physical(psi_hat)
        N = -1.0 * J_phys(psi, om, derivative)
        if F_phys is not None:
            N = N + F_phys
        return L_op(om, L_hat) + N
    k1 = rhs_phys(omega)
    k2 = rhs_phys(omega + 0.5 * dt * k1)
    k3 = rhs_phys(omega + 0.5 * dt * k2)
    k4 = rhs_phys(omega + dt * k3)
    return omega + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# --------------------------------------------------------------------------- #
# Analytical part of the closure: only L^3 omega + L^2 N (Ndot, Nddot left to NN)
# --------------------------------------------------------------------------- #

def compute_N_phys(omega: torch.Tensor, derivative,
                   F_phys: Optional[torch.Tensor]) -> torch.Tensor:
    """
    N(omega) = -J(psi, omega) + F  (physical space).
    Pulled out of E_analytical_phys so the same value can be saved as an
    NN input feature without recomputing the Jacobian.
    """
    psi_hat = derivative.inv_laplacian * to_spectral(omega)
    psi = to_physical(psi_hat)
    N = -1.0 * J_phys(psi, omega, derivative)
    if F_phys is not None:
        N = N + F_phys
    return N


def compute_N_dot_phys(omega: torch.Tensor, derivative,
                       L_hat: torch.Tensor,
                       F_phys: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Analytical N_dot via chain rule + bilinearity of J (same formula as in
    step3_validate_n_derivatives.py):

        omega_dot = L*omega + N(omega)
        psi_dot   = inv_lap * omega_dot
        N_dot     = -J(psi_dot, omega) - J(psi, omega_dot)

    Saved as an NN input feature so the model has direct access to dN/dt
    instead of having to recover it from omega via convolution stencils.
    """
    psi_hat = derivative.inv_laplacian * to_spectral(omega)
    psi = to_physical(psi_hat)

    Nval = -1.0 * J_phys(psi, omega, derivative)
    if F_phys is not None:
        Nval = Nval + F_phys

    omega_dot = L_op(omega, L_hat) + Nval

    psi_dot_hat = derivative.inv_laplacian * to_spectral(omega_dot)
    psi_dot = to_physical(psi_dot_hat)

    return (-1.0 * J_phys(psi_dot, omega, derivative)
            - 1.0 * J_phys(psi, omega_dot, derivative))


def E_analytical_phys(omega: torch.Tensor, derivative,
                      L_hat: torch.Tensor,
                      F_phys: Optional[torch.Tensor]) -> torch.Tensor:
    """
    The analytical portion of the closure E:
        E_anal = (1/12) * [L^3 omega  +  L^2 N(omega)]
    The L*N_dot - 5*N_ddot pieces are deferred to the NN.

    Sign convention matches Step 3's predicted_local_error_phys: this is the
    contribution to omega_coarse - omega_fine. To convert to omega_fine -
    omega_coarse (which is what we want for "the missing increment"), the
    caller should negate the result.
    """
    qh = to_spectral(omega)
    L3_omega_phys = to_physical(L_hat ** 3 * qh)

    psi_hat = derivative.inv_laplacian * qh
    psi = to_physical(psi_hat)
    N_phys = -1.0 * J_phys(psi, omega, derivative)
    if F_phys is not None:
        N_phys = N_phys + F_phys
    Nh = to_spectral(N_phys)
    L2_N_phys = to_physical(L_hat ** 2 * Nh)

    return (1.0 / 12.0) * (L3_omega_phys + L2_N_phys)


# --------------------------------------------------------------------------- #
# AB2CN2 forcing-response inverse: convert error increment to forcing field   #
# --------------------------------------------------------------------------- #

def invert_ab2cn2_response(e_phys: torch.Tensor, dT: float,
                           L_hat: torch.Tensor) -> torch.Tensor:
    """
    Given an AB2CN2-step increment e (in physical space), return the
    constant forcing f such that ONE AB2CN2 step with f added to the RHS
    (treated explicitly, AB2-style) would produce that increment.

    Derivation: AB2CN2 with constant added explicit forcing f over one step
    gives extra increment delta_omega_f = (dT) * f / (1 - 0.5*dT*L_hat) in
    spectral space. So
        f_hat = (1 - 0.5*dT*L_hat) / dT  *  e_hat.

    Diagonal in k, no FFT padding needed beyond the one round trip.
    """
    e_hat = to_spectral(e_phys)
    f_hat = e_hat * (1.0 - 0.5 * dT * L_hat) / dT
    return to_physical(f_hat)


# --------------------------------------------------------------------------- #
# Snapshot reader (mmap with fallback, matches Step 3)                        #
# --------------------------------------------------------------------------- #

class _OmegaLoader:
    """Read a single snapshot from a (T, Ny, Nx) or (B, T, Ny, Nx) array.

    Supports two source layouts:
      - .npy: a raw numpy array on disk (3D or 4D). Best with mmap.
      - .npz: a compressed archive with key 'omega_FR' (or 'omega') of shape
              (B, T, Ny, Nx) or (T, Ny, Nx). NPZ is loaded eagerly into RAM
              once -- ~80 MB for 256^2 x 1200 x float32, fine.

    For multi-batch sources: call set_batch(b) to choose which batch
    subsequent read(idx) calls return from. Batch switching is cheap.
    """
    def __init__(self, path: Path, batch_index: int = 0):
        self.path = Path(path)
        suffix = self.path.suffix.lower()

        if suffix == '.npz':
            # Eagerly load the npz once. Find the omega key.
            with np.load(self.path) as zf:
                key = None
                for candidate in ('omega_FR', 'omega', 'q', 'q_FR'):
                    if candidate in zf.files:
                        key = candidate
                        break
                if key is None:
                    raise RuntimeError(
                        f"no recognized omega key in {self.path.name}; "
                        f"keys = {zf.files}")
                arr = np.asarray(zf[key])
            self._dtype = arr.dtype
            shape = arr.shape

            if arr.ndim == 4:
                self._has_batch_dim = True
                self._n_batches, self._n_snapshots = shape[0], shape[1]
                self._spatial_shape = (shape[2], shape[3])
            elif arr.ndim == 3:
                self._has_batch_dim = False
                self._n_batches = 1
                self._n_snapshots = shape[0]
                self._spatial_shape = (shape[1], shape[2])
            else:
                raise RuntimeError(f"unexpected omega ndim={arr.ndim}")

            # Hold the full array. (Memory cost: ~80 MB for the typical
            # decaying-turb run.)
            self._array = arr
            self.mode = 'npz'
        else:
            # .npy path: parse header, try mmap, fall back to per-snapshot read.
            from numpy.lib import format as np_format
            with open(self.path, 'rb') as f:
                major, minor = np_format.read_magic(f)
                if (major, minor) == (1, 0):
                    shape, fortran_order, dtype = np_format.read_array_header_1_0(f)
                elif (major, minor) == (2, 0):
                    shape, fortran_order, dtype = np_format.read_array_header_2_0(f)
                else:
                    raise RuntimeError(f"unsupported .npy version {major}.{minor}")
                self._header_offset = f.tell()
            if fortran_order:
                raise RuntimeError("Fortran-order .npy not supported")
            self._dtype = dtype

            if len(shape) == 4:
                self._has_batch_dim = True
                B, T, Ny, Nx = shape
                self._n_batches = B
                self._n_snapshots = T
                self._spatial_shape = (Ny, Nx)
            elif len(shape) == 3:
                self._has_batch_dim = False
                self._n_batches = 1
                self._n_snapshots = shape[0]
                self._spatial_shape = (shape[1], shape[2])
            else:
                raise RuntimeError(f"unexpected omega shape {shape}")

            # Try mmap once on the full file
            try:
                self._mmap = np.load(self.path, mmap_mode='r')
                self.mode = 'mmap'
            except OSError:
                self._mmap = None
                self.mode = 'fallback'

        if batch_index >= self._n_batches:
            raise IndexError(f"batch_index={batch_index} >= n_batches={self._n_batches}")
        self._batch_index = batch_index

    @property
    def n_batches(self) -> int:
        return self._n_batches

    def set_batch(self, b: int) -> None:
        if b >= self._n_batches:
            raise IndexError(f"batch index {b} >= n_batches={self._n_batches}")
        self._batch_index = b

    @property
    def spatial_shape(self) -> Tuple[int, int]:
        return self._spatial_shape

    @property
    def n_snapshots(self) -> int:
        return self._n_snapshots

    def read(self, idx: int) -> np.ndarray:
        if self.mode == 'npz':
            if self._has_batch_dim:
                return np.asarray(self._array[self._batch_index, idx],
                                  dtype=self._dtype)
            return np.asarray(self._array[idx], dtype=self._dtype)
        if self.mode == 'mmap':
            if self._has_batch_dim:
                return np.asarray(self._mmap[self._batch_index, idx], dtype=self._dtype)
            return np.asarray(self._mmap[idx], dtype=self._dtype)
        # fallback (.npy seek+read)
        Ny, Nx = self._spatial_shape
        per_bytes = Ny * Nx * self._dtype.itemsize
        if self._has_batch_dim:
            absolute_idx = self._batch_index * self._n_snapshots + idx
        else:
            absolute_idx = idx
        offset = self._header_offset + absolute_idx * per_bytes
        with open(self.path, 'rb') as f:
            f.seek(offset)
            buf = f.read(per_bytes)
        return np.frombuffer(buf, dtype=self._dtype).reshape(Ny, Nx).copy()


# --------------------------------------------------------------------------- #
# Build features and increments for one seed                                  #
# --------------------------------------------------------------------------- #

def build_one_seed(
    seed_omega_np: np.ndarray,
    Delta_T: float,
    K: int,
    h_fine: float,
    h_ultrafine: float,
    derivative,
    L_hat: torch.Tensor,
    F_phys: Optional[torch.Tensor],
    dtype: torch.dtype,
    device: str,
) -> dict:
    """
    Run the Test-C procedure for a single seed. Returns a dict of (Ny, Nx)
    arrays keyed by name, all numpy float32 ready for HDF5 dump.
    """
    seed = torch.tensor(seed_omega_np, dtype=dtype, device=device)[None]  # (1, Ny, Nx)

    # ---- 1) Ultra-fine RK4 forward by 2*Delta_T to build (omega^{-1}, omega^0) ---- #
    n_uf_to_dT  = int(round(Delta_T / h_ultrafine))
    n_uf_to_2dT = 2 * n_uf_to_dT
    omega_cur = seed.clone()
    omega_m1 = None
    for step in range(n_uf_to_2dT):
        omega_cur = rk4_step(omega_cur, h_ultrafine, derivative, L_hat, F_phys)
        if step + 1 == n_uf_to_dT:
            omega_m1 = omega_cur.clone()
    omega_n0 = omega_cur

    # ---- 2) Coarse: 1 AB2CN2 step of size Delta_T ---- #
    qh_m1 = to_spectral(omega_m1)
    qh_n0 = to_spectral(omega_n0)
    qh_coarse = ab2cn2_step_spectral(qh_n0, qh_m1, Delta_T, derivative,
                                     L_hat, F_phys)
    om_coarse = to_physical(qh_coarse)

    # ---- 3) Fine: K AB2CN2 steps of size h_fine ---- #
    qh_minus = qh_m1
    qh_curr  = qh_n0
    for _ in range(K):
        qh_next = ab2cn2_step_spectral(qh_curr, qh_minus, h_fine,
                                       derivative, L_hat, F_phys)
        qh_minus = qh_curr
        qh_curr = qh_next
    om_fine = to_physical(qh_curr)

    # ---- 4) Increments ---- #
    # e_total: fine - coarse (the missing increment that closure should fill).
    e_total_phys = om_fine - om_coarse  # (1, Ny, Nx)

    # e_anal_increment: (Delta_T^3) * (1 - 1/K^2) * E_anal(omega^0)
    # NOTE: predicted_local_error_phys returns coarse - fine. We want
    #       fine - coarse, so we negate.
    K_factor = 1.0 - 1.0 / (K ** 2)
    coef = (Delta_T ** 3) * K_factor
    E_anal_phys = E_analytical_phys(omega_n0, derivative, L_hat, F_phys)
    e_anal_increment_phys = -coef * E_anal_phys  # negate to get fine - coarse sign

    # e_NN_increment: residual after subtracting analytical part
    e_NN_increment_phys = e_total_phys - e_anal_increment_phys

    # ---- 5) Convert error increments to forcings via S^{-1}(Delta_T) ---- #
    f_anal_phys = invert_ab2cn2_response(e_anal_increment_phys, Delta_T, L_hat)
    f_NN_target_phys = invert_ab2cn2_response(e_NN_increment_phys, Delta_T, L_hat)

    # ---- 6) Inputs / features for the NN ---- #
    psi_hat_n0 = derivative.inv_laplacian * qh_n0
    psi_n0 = to_physical(psi_hat_n0)
    # Velocity components (could be useful)
    u_n0 = to_physical(-1 * derivative.dy * psi_hat_n0)
    v_n0 = to_physical(+1 * derivative.dx * psi_hat_n0)
    grad_psi_sq = u_n0 ** 2 + v_n0 ** 2
    # Vorticity gradients
    omega_x = to_physical(derivative.dx * qh_n0)
    omega_y = to_physical(derivative.dy * qh_n0)
    # Nonlinear RHS at t^0 and its analytical time derivative.
    # These give the NN direct access to N(omega_0) and dN/dt evaluated at
    # omega_0 instead of forcing it to recover them from omega_0 via 3x3
    # convolution stencils -- which is the operation the closure formula
    # actually needs (since N_dot and N_ddot drive the truncation residual).
    N_0_phys = compute_N_phys(omega_n0, derivative, F_phys)
    N_dot_0_phys = compute_N_dot_phys(omega_n0, derivative, L_hat, F_phys)

    def cpu32(t: torch.Tensor) -> np.ndarray:
        return t[0].detach().to('cpu', dtype=torch.float32).numpy()

    return dict(
        omega_0          = cpu32(omega_n0),
        psi_0            = cpu32(psi_n0),
        grad_psi_sq      = cpu32(grad_psi_sq),
        omega_x          = cpu32(omega_x),
        omega_y          = cpu32(omega_y),
        omega_m1         = cpu32(omega_m1),
        N_0              = cpu32(N_0_phys),
        N_dot_0_anal     = cpu32(N_dot_0_phys),
        omega_1_coarse   = cpu32(om_coarse),
        omega_K_fine     = cpu32(om_fine),
        e_total          = cpu32(e_total_phys),
        e_anal_incr      = cpu32(e_anal_increment_phys),
        e_NN_incr        = cpu32(e_NN_increment_phys),
        f_NN_target      = cpu32(f_NN_target_phys),
        f_anal           = cpu32(f_anal_phys),
    )


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

PER_SAMPLE_FIELDS = [
    'omega_0', 'psi_0', 'grad_psi_sq', 'omega_x', 'omega_y',
    'omega_m1', 'N_0', 'N_dot_0_anal',
    'omega_1_coarse', 'omega_K_fine',
    'e_total', 'e_anal_incr', 'e_NN_incr',
    'f_NN_target', 'f_anal',
]


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scenario', type=str, required=True,
                   choices=['decaying_turbulence', 'flow_past_cylinder'])
    p.add_argument('--source-omega', type=Path, required=True,
                   help='path to DNS_FR_omega.npy of the reference (smallest dt) run')
    p.add_argument('--source-times', type=Path, default=None,
                   help='path to DNS_FR_times.npy (optional if source-omega is '
                        'a .npz containing a "times" key)')
    p.add_argument('--source-yaml', type=Path, required=True,
                   help='YAML config to extract Lx, Ly, nu, mu, beta')
    p.add_argument('--out-dir', type=Path, required=True,
                   help='directory to write the HDF5 file')

    p.add_argument('--Delta-T', type=float, default=1.0e-3,
                   help='coarse AB2CN2 step (default 1e-3)')
    p.add_argument('--h-fine', type=float, default=1.0e-5,
                   help='fine AB2CN2 step (default 1e-5)')
    p.add_argument('--h-ultrafine', type=float, default=5.0e-6,
                   help='RK4 ultra-fine step for stencil warmup (default 5e-6)')

    p.add_argument('--n-seeds', type=int, default=200,
                   help='number of seeds to sample from the trajectory PER BATCH '
                        '(default 200)')
    p.add_argument('--t-start', type=float, default=5.0,
                   help='earliest seed time (skip spinup, default 5.0)')
    p.add_argument('--t-end',   type=float, default=-1.0,
                   help='latest seed time; -1 -> source_times[-1] - 2*Delta_T '
                        '(leave room for stencil) (default -1)')

    # Batch handling. Two modes:
    #   --batch-index B            : single-batch mode (legacy default); B fixed.
    #   --batches "0,1,2,3,..."    : multi-batch mode; comma-separated list.
    #   --n-batches N              : multi-batch mode; uses batches [0, N).
    # If both --batches and --n-batches are given, --batches wins.
    p.add_argument('--batch-index', type=int, default=0,
                   help='single batch to read from source npy (default 0). '
                        'Ignored if --batches or --n-batches is given.')
    p.add_argument('--batches', type=str, default=None,
                   help='comma-separated list of batch indices to use, e.g. '
                        '"0,1,2,3" (overrides --batch-index and --n-batches).')
    p.add_argument('--n-batches', type=int, default=None,
                   help='use the first N batches of the source npy '
                        '(equivalent to --batches "0,1,...,N-1").')

    p.add_argument('--device', type=str, default='cuda',
                   help='cpu or cuda (default cuda)')
    p.add_argument('--dtype', type=str, default='float64',
                   choices=['float32', 'float64'],
                   help='solver precision (default float64; matches Step 3)')

    p.add_argument('--train-frac', type=float, default=0.70)
    p.add_argument('--val-frac',   type=float, default=0.15)
    # test_frac = 1 - train_frac - val_frac

    p.add_argument('--split-mode', type=str, default='auto',
                   choices=['auto', 'by_batch', 'by_time'],
                   help='train/val/test split policy. by_batch: held-out batch '
                        'indices for val/test (recommended for multi-batch). '
                        'by_time: time-block split within the seed-time range '
                        '(recommended for single-batch). auto: by_batch if '
                        '>1 batch, by_time otherwise.')

    args = p.parse_args()

    # ------------------------------- Setup ------------------------------- #
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[build_training_data] scenario: {args.scenario}")
    print(f"[build_training_data] source omega: {args.source_omega}")
    if args.source_times is not None:
        print(f"[build_training_data] source times (explicit): {args.source_times}")
    print(f"[build_training_data] source yaml:  {args.source_yaml}")

    # Load YAML to get physical parameters
    with open(args.source_yaml, 'r') as f:
        yaml_cfg = yaml.safe_load(f)

    # The YAML structure: {scenario: ..., qg: {grid: {Lx, Ly, Nx, Ny, ...},
    # pde: {nu, mu, B, ...}, ...}}.
    # Step 3 didn't load YAML at all, it relied on CLI Lx/Ly. Let's be defensive
    # and try a few paths.
    def _yaml_get(path, default=None):
        cur = yaml_cfg
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return default
        return cur

    def _grid_float(key, default):
        v = _yaml_get(['qg', 'grid', key])
        if v is None:
            v = _yaml_get(['grid', key])
        if v is None:
            return float(default)
        return float(v)

    def _pde_float(key, default=0.0):
        v = _yaml_get(['qg', 'pde', key])
        if v is None:
            v = _yaml_get(['pde', key])
        if v is None:
            return float(default)
        return float(v)

    Lx   = _grid_float('Lx', default=2.0 * math.pi)
    Ly   = _grid_float('Ly', default=2.0 * math.pi)
    nu   = _pde_float('nu')
    mu   = _pde_float('mu')
    beta = _pde_float('B')
    print(f"[build_training_data] Lx={Lx}, Ly={Ly}, nu={nu}, mu={mu}, beta={beta}")

    # Load the trajectory times. If --source-times is given, use that. Else,
    # try to find a 'times' key in the source .npz. Else, fail.
    if args.source_times is not None:
        times = np.load(args.source_times)
        times_origin = str(args.source_times)
    elif args.source_omega.suffix.lower() == '.npz':
        with np.load(args.source_omega) as zf:
            if 'times' in zf.files:
                times = np.asarray(zf['times'])
            else:
                sys.exit(f"ERROR: source npz {args.source_omega} has no 'times' "
                         f"key; pass --source-times to specify it explicitly. "
                         f"Available keys: {zf.files}")
        times_origin = f"npz['times'] in {args.source_omega}"
    else:
        sys.exit("ERROR: --source-times is required when --source-omega is .npy")
    print(f"[build_training_data] source times: {times_origin}")
    print(f"[build_training_data] source times: T_save={len(times)}, "
          f"t[0]={times[0]:.4f}, t[-1]={times[-1]:.4f}, "
          f"dt_save={times[1]-times[0]:.4f}")

    omega_loader = _OmegaLoader(args.source_omega, batch_index=0)
    Ny, Nx = omega_loader.spatial_shape
    n_avail_batches = omega_loader.n_batches
    print(f"[build_training_data] grid: ({Ny}, {Nx})  (mode={omega_loader.mode})")
    print(f"[build_training_data] available batches in source: {n_avail_batches}")

    # ----------------------------- Resolve batches ----------------------------- #
    if args.batches is not None:
        try:
            batches = [int(b.strip()) for b in args.batches.split(',') if b.strip()]
        except ValueError:
            sys.exit(f"ERROR: --batches '{args.batches}' is not a comma-list of ints")
    elif args.n_batches is not None:
        if args.n_batches < 1:
            sys.exit(f"ERROR: --n-batches must be >= 1 (got {args.n_batches})")
        batches = list(range(args.n_batches))
    else:
        batches = [args.batch_index]
    for b in batches:
        if b < 0 or b >= n_avail_batches:
            sys.exit(f"ERROR: batch index {b} out of [0, {n_avail_batches})")
    print(f"[build_training_data] using batches: {batches} ({len(batches)} total)")

    # Build QG grid + derivative on the chosen device
    dtype = torch.float64 if args.dtype == 'float64' else torch.float32
    device = args.device
    grid = CartesianGrid(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, device=device)
    derivative = Derivative(grid)
    # Derivative might not auto-move; ensure it's on device
    derivative.dx = derivative.dx.to(device)
    derivative.dy = derivative.dy.to(device)
    derivative.laplacian = derivative.laplacian.to(device)
    derivative.inv_laplacian = derivative.inv_laplacian.to(device)
    L_hat = build_L_hat(derivative, nu=nu, mu=mu, B=beta).to(device)
    # No forcing for decaying turbulence; could be added later for forced cases
    F_phys = None

    # ----------------------------- Pick seeds ----------------------------- #
    t_end = args.t_end if args.t_end > 0 else float(times[-1] - 2 * args.Delta_T)
    t_start = args.t_start
    if t_start >= t_end:
        sys.exit(f"ERROR: t_start ({t_start}) >= t_end ({t_end})")

    # Stencil warmup needs 2*Delta_T forward; require seed_t + 2*Delta_T <= times[-1]
    max_seed_t = float(times[-1] - 2 * args.Delta_T)
    if t_end > max_seed_t:
        print(f"[build_training_data] clipping t_end to {max_seed_t:.4f} "
              f"(need 2*Delta_T={2*args.Delta_T} headroom)")
        t_end = max_seed_t

    # Linearly space t* values in [t_start, t_end], find nearest snapshot indices
    t_targets = np.linspace(t_start, t_end, args.n_seeds)
    seed_indices = np.unique(np.searchsorted(times, t_targets))
    if len(seed_indices) < args.n_seeds:
        print(f"[build_training_data] note: only {len(seed_indices)} unique "
              f"snapshot indices for {args.n_seeds} requested seeds (snapshot "
              f"spacing too coarse).")
    print(f"[build_training_data] seeds per batch: {len(seed_indices)} in t range "
          f"[{times[seed_indices[0]]:.3f}, {times[seed_indices[-1]]:.3f}]")
    print(f"[build_training_data] Delta_T={args.Delta_T}, h_fine={args.h_fine}, "
          f"h_ultrafine={args.h_ultrafine}")
    K_int = int(round(args.Delta_T / args.h_fine))
    K_check = K_int * args.h_fine
    if abs(K_check - args.Delta_T) > 1e-12 * args.Delta_T:
        sys.exit(f"ERROR: Delta_T ({args.Delta_T}) is not an integer multiple "
                 f"of h_fine ({args.h_fine}); K = Delta_T/h_fine = {args.Delta_T/args.h_fine}")
    print(f"[build_training_data] K = Delta_T / h_fine = {K_int}")

    # ----------------------- Resolve split policy ----------------------- #
    if args.split_mode == 'auto':
        split_mode = 'by_batch' if len(batches) > 1 else 'by_time'
    else:
        split_mode = args.split_mode
        if split_mode == 'by_batch' and len(batches) < 3:
            print(f"[build_training_data] WARN: --split-mode by_batch with "
                  f"only {len(batches)} batches; need at least 3 for a "
                  f"meaningful train/val/test split. Continuing anyway.")
    print(f"[build_training_data] split-mode: {split_mode}")

    # ----------------------- Open output directory ----------------------- #
    dT_tag = f'{args.Delta_T:.0e}'.replace('e-0', 'em').replace('e+0', 'ep')
    out_dataset_dir = args.out_dir / f'{args.scenario}_dT_{dT_tag}'
    samples_dir = out_dataset_dir / 'samples'
    samples_dir.mkdir(parents=True, exist_ok=True)
    print(f"[build_training_data] writing to {out_dataset_dir}/")

    seeds_per_batch = len(seed_indices)
    N = seeds_per_batch * len(batches)
    print(f"[build_training_data] total samples: {N} "
          f"({seeds_per_batch} seeds * {len(batches)} batches)")

    # ----------------------- Outer loop: batches ----------------------- #
    # Records are stored in the order
    # [batch[0]:seeds, batch[1]:seeds, ..., batch[-1]:seeds]
    # so a row range [b*seeds_per_batch, (b+1)*seeds_per_batch] gives all
    # samples from batches[b].
    t_total_start = time.time()
    global_i = 0
    n_failed = 0
    sample_records = []  # list of dicts for manifest

    for b_pos, b in enumerate(batches):
        omega_loader.set_batch(b)
        print(f"\n[build_training_data] === batch {b} "
              f"({b_pos+1}/{len(batches)}) ===")
        for j, s_idx in enumerate(seed_indices):
            t_seed = float(times[s_idx])
            seed_np = omega_loader.read(int(s_idx))

            t0 = time.time()
            try:
                rec = build_one_seed(
                    seed_np, args.Delta_T, K_int, args.h_fine, args.h_ultrafine,
                    derivative, L_hat, F_phys, dtype, device,
                )
            except Exception as e:
                print(f"  b={b} [{j+1}/{seeds_per_batch}] s_idx={s_idx} "
                      f"FAILED: {e}")
                global_i += 1
                n_failed += 1
                continue
            elapsed = time.time() - t0

            # Save one .npz file for this sample
            sample_path = samples_dir / f'sample_{global_i:06d}.npz'
            np.savez_compressed(
                sample_path,
                # All field arrays (Ny, Nx) float32
                omega_0=rec['omega_0'],
                psi_0=rec['psi_0'],
                grad_psi_sq=rec['grad_psi_sq'],
                omega_x=rec['omega_x'],
                omega_y=rec['omega_y'],
                omega_m1=rec['omega_m1'],
                N_0=rec['N_0'],
                N_dot_0_anal=rec['N_dot_0_anal'],
                omega_1_coarse=rec['omega_1_coarse'],
                omega_K_fine=rec['omega_K_fine'],
                e_total=rec['e_total'],
                e_anal_incr=rec['e_anal_incr'],
                e_NN_incr=rec['e_NN_incr'],
                f_NN_target=rec['f_NN_target'],
                f_anal=rec['f_anal'],
                # Scalar metadata stored as 0-d arrays
                seed_t=np.float32(t_seed),
                seed_idx=np.int32(s_idx),
                batch_idx=np.int32(b),
            )
            sample_records.append(dict(
                index=global_i,
                file=f'samples/sample_{global_i:06d}.npz',
                seed_t=t_seed,
                seed_idx=int(s_idx),
                batch_idx=int(b),
            ))

            # Diagnostic norms
            l2_emp  = np.sqrt(np.mean(rec['e_total']     ** 2))
            l2_anal = np.sqrt(np.mean(rec['e_anal_incr'] ** 2))
            l2_NN   = np.sqrt(np.mean(rec['e_NN_incr']   ** 2))
            l2_fNN  = np.sqrt(np.mean(rec['f_NN_target'] ** 2))
            if (j+1) % 20 == 1 or (j+1) == seeds_per_batch:
                print(f"  b={b} [{j+1:4d}/{seeds_per_batch}] s_idx={s_idx:5d} "
                      f"t={t_seed:8.3f}  |e_tot|={l2_emp:.3e} "
                      f"|e_anal|={l2_anal:.3e} |e_NN|={l2_NN:.3e} "
                      f"|f_NN|={l2_fNN:.3e}  elapsed={elapsed:.2f}s")
            global_i += 1

    t_total = time.time() - t_total_start
    n_completed = N - n_failed
    print(f"\n[build_training_data] completed: {n_completed}/{N} "
          f"({n_failed} failures)")

    # ----------------------- Train/val/test split ----------------------- #
    idx = np.arange(N, dtype=np.int32)
    if split_mode == 'by_time':
        # Time-block split within each batch's row-block.
        if len(batches) == 1:
            n_train = int(round(N * args.train_frac))
            n_val   = int(round(N * args.val_frac))
            train_idx = idx[:n_train]
            val_idx   = idx[n_train:n_train + n_val]
            test_idx  = idx[n_train + n_val:]
        else:
            t_n_train = int(round(seeds_per_batch * args.train_frac))
            t_n_val   = int(round(seeds_per_batch * args.val_frac))
            tr, va, te = [], [], []
            for b_pos, b in enumerate(batches):
                base = b_pos * seeds_per_batch
                tr.extend(range(base, base + t_n_train))
                va.extend(range(base + t_n_train, base + t_n_train + t_n_val))
                te.extend(range(base + t_n_train + t_n_val,
                                base + seeds_per_batch))
            train_idx = np.array(tr, dtype=np.int32)
            val_idx   = np.array(va, dtype=np.int32)
            test_idx  = np.array(te, dtype=np.int32)
    elif split_mode == 'by_batch':
        n_b = len(batches)
        n_b_train = max(1, int(round(n_b * args.train_frac)))
        n_b_val   = max(1, int(round(n_b * args.val_frac)))
        n_b_test  = n_b - n_b_train - n_b_val
        if n_b_test < 1:
            if n_b_val > 1:
                n_b_val -= 1
                n_b_test = 1
            elif n_b_train > 1:
                n_b_train -= 1
                n_b_test = 1
        train_b = list(range(0, n_b_train))
        val_b   = list(range(n_b_train, n_b_train + n_b_val))
        test_b  = list(range(n_b_train + n_b_val, n_b))
        print(f"[build_training_data] train batches: "
              f"{[batches[k] for k in train_b]}")
        print(f"[build_training_data]   val batches: "
              f"{[batches[k] for k in val_b]}")
        print(f"[build_training_data]  test batches: "
              f"{[batches[k] for k in test_b]}")
        tr, va, te = [], [], []
        for k in train_b:
            base = k * seeds_per_batch
            tr.extend(range(base, base + seeds_per_batch))
        for k in val_b:
            base = k * seeds_per_batch
            va.extend(range(base, base + seeds_per_batch))
        for k in test_b:
            base = k * seeds_per_batch
            te.extend(range(base, base + seeds_per_batch))
        train_idx = np.array(tr, dtype=np.int32)
        val_idx   = np.array(va, dtype=np.int32)
        test_idx  = np.array(te, dtype=np.int32)
    else:
        raise RuntimeError(f"unknown split mode {split_mode}")

    # Save split as a single .npz
    np.savez(
        out_dataset_dir / 'split.npz',
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )

    # ----------------------- Manifest (JSON) ----------------------- #
    manifest = dict(
        scenario=args.scenario,
        Lx=float(Lx), Ly=float(Ly),
        Nx=int(Nx), Ny=int(Ny),
        nu=float(nu), mu=float(mu), beta=float(beta),
        Delta_T=float(args.Delta_T),
        K=int(K_int),
        h_fine=float(args.h_fine),
        h_ultrafine=float(args.h_ultrafine),
        source_omega_path=str(args.source_omega),
        source_times_path=str(args.source_times) if args.source_times else times_origin,
        source_yaml_path=str(args.source_yaml),
        batches_used=[int(b) for b in batches],
        n_batches_used=int(len(batches)),
        seeds_per_batch=int(seeds_per_batch),
        n_total=int(N),
        n_completed=int(n_completed),
        n_failed=int(n_failed),
        n_train=int(len(train_idx)),
        n_val=int(len(val_idx)),
        n_test=int(len(test_idx)),
        split_mode=split_mode,
        dtype=args.dtype,
        device=device,
        per_sample_fields=PER_SAMPLE_FIELDS,
        samples=sample_records,
    )
    with open(out_dataset_dir / 'manifest.json', 'w') as fjson:
        json.dump(manifest, fjson, indent=2)

    print(f"\n[build_training_data] split: train {len(train_idx)} / "
          f"val {len(val_idx)} / test {len(test_idx)}")
    print(f"[build_training_data] total time: {t_total:.1f}s "
          f"({t_total/max(N,1):.2f}s per sample)")
    # Compute total disk size of samples/
    total_bytes = sum(p.stat().st_size for p in samples_dir.glob('*.npz'))
    print(f"[build_training_data] dataset on disk: "
          f"{total_bytes / 1024**2:.1f} MiB across {n_completed} files")
    print(f"[build_training_data] manifest:  {out_dataset_dir / 'manifest.json'}")
    print(f"[build_training_data] split:     {out_dataset_dir / 'split.npz'}")


if __name__ == '__main__':
    main()