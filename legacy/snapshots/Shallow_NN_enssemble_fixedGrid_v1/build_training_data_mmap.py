"""
build_training_data_mmap.py

Build the temporal-closure training set DIRECTLY in mmap-optimal format, with a
SEVEN-snapshot stencil (so up to N6dot is representable in the time-FD span).

Differences vs build_training_data(_fixD_v2).py
------------------------------------------------
1) SEVEN time levels. The seed read from the DNS is the EARLIEST level; the
   warmup integrates it forward 6*dT, snapshotting at every dT mark (m5 at +dT,
   m4 at +2dT, ..., m1 at +5dT, 0 at +6dT), so omega_m6 := seed is free. Inputs
   are now 14 channels:
       [omega_0..omega_m6,  psi_0..psi_m6]
   This gives time-orders 0..6, widening the Jacobian-feature / time-FD span
   (N^(3), N^(4) and N^(5) targets are comfortably inside it). Targets remain the
   exact analytic chain-rule N-derivatives and do NOT depend on stencil depth;
   the extra snapshots only feed the model richer time-difference features.

2) Output is CONTIGUOUS float32 memmaps, not per-sample compressed npz:
       <out>/<scenario>_dT_<tag>/
           packed/inputs.npy    (N, 14, Ny, Nx) float32
           packed/targets.npy   (N, 5, Ny, Nx) float32  [Ndot,Nddot,N3dot,N4dot,N5dot]
           packed/pack_meta.json
           packed/split.npz
           split.npz
           manifest.json
           [packed/diag_f32.npy, packed/diag_f64.npy if --with-diagnostics]
   No zip-opens, no DEFLATE, OS-cacheable. dataset.py's PackedClosureDataset
   auto-detects packed/ -- no separate pack step.

3) Truth / diagnostics are OPTIONAL (--with-diagnostics). The analytic
   N-derivative TARGETS come from the chain rule at omega_0 and need no fine
   integration, so by default the expensive K-step truth loop is skipped. With
   the flag, the closure-validation fields (omega_1_coarse, omega_K_fine,
   e_total, e_anal_incr, e_NN_incr, f_NN_target_from_e, f_anal, f_NN_target,
   N_0) are written as separate memmaps; the residuals are kept float64.

float32 is used for the training memmaps on purpose: the build-time float64 was
to survive the e_total cancellation; the N-derivative targets are O(1)..O(1e10)
physics and lose nothing in float32. The solver/grid still runs in float64.

Usage (forced beta-plane, 5-target N-derivatives up to N^(5)):
    python build_training_data_mmap.py \\
        --scenario forced_turbulence \\
        --source-omega .../DNS_FR_omega.npy --source-times .../DNS_FR_times.npy \\
        --source-yaml  .../forced_turbulence.yaml \\
        --out-dir      .../training/data/ \\
        --n-batches 20 --n-seeds 500 --max-order 5
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
from numpy.lib.format import open_memmap

from qg.solver.grid.cartesian import CartesianGrid
from qg.solver.opt.derivative import Derivative
from qg.solver.opt.basis import to_spectral, to_physical


# ----- field layout (column order of the packed arrays) -------------------- #
def make_input_fields(M):
    """[omega_0..omega_m{M-1}, psi_0..psi_m{M-1}] -- 2M input channels for M marks."""
    om = ['omega_0'] + [f'omega_m{k}' for k in range(1, M)]
    ps = ['psi_0'] + [f'psi_m{k}' for k in range(1, M)]
    return om + ps


INPUT_FIELDS = make_input_fields(7)   # default; main rebuilds for --n-marks
TARGET_FIELDS = ['N_dot_0_anal', 'N_ddot_0_anal', 'N_3dot_0_anal',
                 'N_4dot_0_anal', 'N_5dot_0_anal']
DIAG_F32_FIELDS = ['omega_1_coarse', 'omega_K_fine', 'f_anal', 'f_NN_target', 'N_0']
DIAG_F64_FIELDS = ['e_total', 'e_anal_incr', 'e_NN_incr', 'f_NN_target_from_e']


# --------------------------------------------------------------------------- #
# QG operator wrappers (behavior-identical to the validated build)            #
# --------------------------------------------------------------------------- #

def J_phys(psi_phys, omega_phys, derivative):
    """Dealiased Jacobian J(psi, omega), flux form: d_x(u*omega)+d_y(v*omega),
    u=-d_y psi, v=+d_x psi. The u*q / v*q products are 2/3-rule dealiased
    (derivative.dealias, in-place) before assembly -- identical to the original
    build_training_data.py / fixD_v2 J_phys and the solver's RHS dealiasing."""
    psih = to_spectral(psi_phys)
    qh = to_spectral(omega_phys)
    u = to_physical(-1 * derivative.dy * psih)
    v = to_physical(+1 * derivative.dx * psih)
    q = to_physical(qh)
    uq_h = to_spectral(u * q).clone()
    vq_h = to_spectral(v * q).clone()
    derivative.dealias(uq_h)
    derivative.dealias(vq_h)
    j_hat = derivative.dx * uq_h + derivative.dy * vq_h
    return to_physical(j_hat)


def L_op(omega_phys, L_hat):
    return to_physical(L_hat * to_spectral(omega_phys))


def build_L_hat(derivative, nu, mu, B):
    L_hat = nu * derivative.laplacian - mu
    if B != 0.0:
        L_hat = L_hat - B * derivative.dx * derivative.inv_laplacian
    return L_hat


def ab2cn2_step_spectral(qh_n, qh_nm1, dt, derivative, L_hat, F_phys):
    def N_at_qh(qh):
        psi = to_physical(derivative.inv_laplacian * qh)
        omega = to_physical(qh)
        N_phys = -1.0 * J_phys(psi, omega, derivative)
        if F_phys is not None:
            N_phys = N_phys + F_phys
        return to_spectral(N_phys)
    AB2_Nh = 1.5 * N_at_qh(qh_n) - 0.5 * N_at_qh(qh_nm1)
    rhs_hat = qh_n + dt * (0.5 * L_hat * qh_n + AB2_Nh)
    return rhs_hat / (1.0 - 0.5 * dt * L_hat)


def rk4_step(omega, dt, derivative, L_hat, F_phys):
    def rhs_phys(om):
        psi = to_physical(derivative.inv_laplacian * to_spectral(om))
        N = -1.0 * J_phys(psi, om, derivative)
        if F_phys is not None:
            N = N + F_phys
        return L_op(om, L_hat) + N
    k1 = rhs_phys(omega)
    k2 = rhs_phys(omega + 0.5 * dt * k1)
    k3 = rhs_phys(omega + 0.5 * dt * k2)
    k4 = rhs_phys(omega + dt * k3)
    return omega + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def E_analytical_phys(omega, derivative, L_hat, F_phys):
    """Analytical closure part (1/12)[L^3 omega + L^2 N], coarse-fine sign."""
    qh = to_spectral(omega)
    L3_omega = to_physical(L_hat ** 3 * qh)
    psi = to_physical(derivative.inv_laplacian * qh)
    N_phys = -1.0 * J_phys(psi, omega, derivative)
    if F_phys is not None:
        N_phys = N_phys + F_phys
    L2_N = to_physical(L_hat ** 2 * to_spectral(N_phys))
    return (1.0 / 12.0) * (L3_omega + L2_N)


def compute_n_derivatives(omega, derivative, L_hat, F, max_order):
    """[N^(0..max_order)] along the trajectory at omega via the chain rule.

        omega^(k) = L omega^(k-1) + N^(k-1),  psi^(k) = inv_lap omega^(k)
        N^(m)     = -sum_j C(m,j) J(psi^(m-j), omega^(j))   (+F at m=0)
    Static forcing (F^(k)=0, k>=1).
    """
    omega_d = [omega]
    psi_d = [to_physical(derivative.inv_laplacian * to_spectral(omega))]
    N0 = -1.0 * J_phys(psi_d[0], omega_d[0], derivative)
    if F is not None:
        N0 = N0 + F
    N_d = [N0]
    for k in range(1, max_order + 1):
        omega_k = L_op(omega_d[k - 1], L_hat) + N_d[k - 1]
        omega_d.append(omega_k)
        psi_d.append(to_physical(derivative.inv_laplacian * to_spectral(omega_k)))
        Nk = torch.zeros_like(omega)
        for j in range(0, k + 1):
            Nk = Nk - math.comb(k, j) * J_phys(psi_d[k - j], omega_d[j], derivative)
        N_d.append(Nk)
    return N_d


# --------------------------------------------------------------------------- #
# Snapshot reader (mmap with fallback)                                        #
# --------------------------------------------------------------------------- #

class _OmegaLoader:
    """Read a snapshot from a (T,Ny,Nx) or (B,T,Ny,Nx) .npy/.npz source."""
    def __init__(self, path: Path, batch_index: int = 0):
        self.path = Path(path)
        suffix = self.path.suffix.lower()
        if suffix == '.npz':
            with np.load(self.path) as zf:
                key = next((c for c in ('omega_FR', 'omega', 'q', 'q_FR')
                            if c in zf.files), None)
                if key is None:
                    raise RuntimeError(f"no omega key in {self.path.name}; {zf.files}")
                arr = np.asarray(zf[key])
            self._dtype = arr.dtype
            if arr.ndim == 4:
                self._has_batch_dim = True
                self._n_batches, self._n_snapshots = arr.shape[0], arr.shape[1]
                self._spatial_shape = (arr.shape[2], arr.shape[3])
            elif arr.ndim == 3:
                self._has_batch_dim = False
                self._n_batches, self._n_snapshots = 1, arr.shape[0]
                self._spatial_shape = (arr.shape[1], arr.shape[2])
            else:
                raise RuntimeError(f"unexpected omega ndim={arr.ndim}")
            self._array = arr
            self.mode = 'npz'
        else:
            from numpy.lib import format as np_format
            with open(self.path, 'rb') as f:
                major, minor = np_format.read_magic(f)
                rd = (np_format.read_array_header_1_0 if (major, minor) == (1, 0)
                      else np_format.read_array_header_2_0)
                shape, fortran_order, dtype = rd(f)
                self._header_offset = f.tell()
            if fortran_order:
                raise RuntimeError("Fortran-order .npy not supported")
            self._dtype = dtype
            if len(shape) == 4:
                self._has_batch_dim = True
                _, self._n_snapshots = shape[0], shape[1]
                self._n_batches = shape[0]
                self._spatial_shape = (shape[2], shape[3])
            elif len(shape) == 3:
                self._has_batch_dim = False
                self._n_batches, self._n_snapshots = 1, shape[0]
                self._spatial_shape = (shape[1], shape[2])
            else:
                raise RuntimeError(f"unexpected omega shape {shape}")
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
    def n_batches(self): return self._n_batches
    @property
    def spatial_shape(self): return self._spatial_shape
    @property
    def n_snapshots(self): return self._n_snapshots

    def set_batch(self, b):
        if b >= self._n_batches:
            raise IndexError(f"batch {b} >= n_batches={self._n_batches}")
        self._batch_index = b

    def read(self, idx):
        if self.mode == 'npz':
            a = (self._array[self._batch_index, idx] if self._has_batch_dim
                 else self._array[idx])
            return np.asarray(a, dtype=self._dtype)
        if self.mode == 'mmap':
            a = (self._mmap[self._batch_index, idx] if self._has_batch_dim
                 else self._mmap[idx])
            return np.asarray(a, dtype=self._dtype)
        Ny, Nx = self._spatial_shape
        per = Ny * Nx * self._dtype.itemsize
        ai = (self._batch_index * self._n_snapshots + idx) if self._has_batch_dim else idx
        with open(self.path, 'rb') as f:
            f.seek(self._header_offset + ai * per)
            buf = f.read(per)
        return np.frombuffer(buf, dtype=self._dtype).reshape(Ny, Nx).copy()


# --------------------------------------------------------------------------- #
# One seed                                                                    #
# --------------------------------------------------------------------------- #

def _cpu(t, dt):
    return t[0].detach().to('cpu', dtype=dt).numpy()


def build_one_seed(seed_omega_np, Delta_T, K, h_fine, h_ultrafine,
                   derivative, L_hat, F_phys, dtype, device,
                   with_diagnostics=False, max_order=5, M=7):
    """M-mark stencil (dt spacing) + analytic N-derivative targets. Optional truth.

    M marks at t_seed + {0,1,...,M-1}*dt; seed = earliest (omega_m{M-1}), free.
    omega_0 (latest mark) is where the N-derivative targets live. M>7 lays down a
    deep trajectory so coarse Delta_T = j*dt slices (7-snapshot stencil + delta)
    come out of a single run -- see slice_delta_sweep.py.
    """
    seed = torch.tensor(seed_omega_np, dtype=dtype, device=device)[None]

    # ---- M levels at t_seed + {0..M-1}*dt; warmup integrates forward (M-1)*dt ---- #
    n_uf_to_dT = int(round(Delta_T / h_ultrafine))
    marks = [None] * M                               # marks[k] = state at t_seed + k*dt
    marks[0] = seed.clone()                          # earliest = omega_m{M-1}
    omega_cur = seed.clone()
    for step in range((M - 1) * n_uf_to_dT):
        omega_cur = rk4_step(omega_cur, h_ultrafine, derivative, L_hat, F_phys)
        if (step + 1) % n_uf_to_dT == 0:
            marks[(step + 1) // n_uf_to_dT] = omega_cur.clone()
    omega_n0 = marks[M - 1]                           # latest = omega_0

    # field order: omega_0 = marks[M-1], omega_m{k} = marks[M-1-k]
    om_marks = [marks[M - 1 - k] for k in range(M)]
    qh_marks = [to_spectral(o) for o in om_marks]
    psi_marks = [to_physical(derivative.inv_laplacian * qh) for qh in qh_marks]
    qh_n0 = qh_marks[0]
    qh_m1 = qh_marks[1]

    # ---- analytic N-derivative targets at omega_0 (no fine integration) ---- #
    # Exact chain-rule N-derivatives; independent of stencil depth. They are
    # only meaningful at the omega_0 anchor; the deep-trajectory + slicer path
    # computes the empirical delta at every interior anchor instead.
    N_d = compute_n_derivatives(omega_n0, derivative, L_hat, F_phys, max_order=max_order)
    Ndot, Nddot, N3dot, N4dot, N5dot = N_d[1], N_d[2], N_d[3], N_d[4], N_d[5]

    f32, f64 = torch.float32, torch.float64
    in_fields = make_input_fields(M)                 # [omega_0..m{M-1}, psi_0..m{M-1}]
    out = {}
    for k in range(M):
        out[in_fields[k]] = _cpu(om_marks[k], f64)          # float64 inputs (writer casts)
        out[in_fields[M + k]] = _cpu(psi_marks[k], f64)
    out.update(
        N_dot_0_anal=_cpu(Ndot, f32), N_ddot_0_anal=_cpu(Nddot, f32),
        N_3dot_0_anal=_cpu(N3dot, f32), N_4dot_0_anal=_cpu(N4dot, f32),
        N_5dot_0_anal=_cpu(N5dot, f32),
    )

    if with_diagnostics:
        f64 = torch.float64
        qh_coarse = ab2cn2_step_spectral(qh_n0, qh_m1, Delta_T, derivative, L_hat, F_phys)
        om_coarse = to_physical(qh_coarse)
        qh_minus, qh_curr = qh_m1, qh_n0
        for _ in range(K):
            qh_next = ab2cn2_step_spectral(qh_curr, qh_minus, h_fine,
                                           derivative, L_hat, F_phys)
            qh_minus, qh_curr = qh_curr, qh_next
        om_fine = to_physical(qh_curr)
        e_total = om_fine - om_coarse
        coef = (Delta_T ** 3) * (1.0 - 1.0 / (K ** 2))
        E_anal = E_analytical_phys(omega_n0, derivative, L_hat, F_phys)
        e_anal_incr = -coef * E_anal
        e_NN_incr = e_total - e_anal_incr
        f_NN_from_e = -e_NN_incr / coef
        f_NN_target = (1.0 / 12.0) * (L_op(Ndot, L_hat) - 5.0 * Nddot)
        out.update(
            omega_1_coarse=_cpu(om_coarse, f32), omega_K_fine=_cpu(om_fine, f32),
            f_anal=_cpu(E_anal, f32), f_NN_target=_cpu(f_NN_target, f32),
            N_0=_cpu(N_d[0], f32),
            e_total=_cpu(e_total, f64), e_anal_incr=_cpu(e_anal_incr, f64),
            e_NN_incr=_cpu(e_NN_incr, f64), f_NN_target_from_e=_cpu(f_NN_from_e, f64),
        )
    return out


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scenario', type=str, required=True,
                   choices=['decaying_turbulence', 'flow_past_cylinder',
                            'forced_turbulence'])
    p.add_argument('--source-omega', type=Path, required=True)
    p.add_argument('--source-times', type=Path, default=None)
    p.add_argument('--dt-save', type=float, default=None,
                   help='snapshot save interval. If set, the time axis is '
                        'synthesized as arange(n_snapshots)*dt_save instead of '
                        'reading a stored "times" (no --source-times / npz key '
                        'needed). Use when DNS_FR.npz holds only omega at a '
                        'known uniform save cadence.')
    p.add_argument('--source-yaml', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)

    p.add_argument('--Delta-T', type=float, default=1.0e-3,
                   help='mark spacing dt_base. For a deep build, this is the base; '
                        'slice coarse Delta_T = j*dt_base out of it (slice_delta_sweep.py).')
    p.add_argument('--n-marks', type=int, default=7,
                   help='number of saved snapshots M at dt spacing (default 7). '
                        'M>7 lays a deep trajectory so {5e-3,1e-2,1.5e-2}=j*5e-3 all '
                        'slice from ONE run (e.g. --Delta-T 5e-3 --n-marks 25).')
    p.add_argument('--k-fine', type=int, default=100,
                   help='fine-ref subcycling ratio: h_fine = Delta_T/k_fine (keeps '
                        'build cost constant across Delta_T).')
    p.add_argument('--k-ultrafine', type=int, default=200,
                   help='warmup subcycling ratio: h_ultrafine = Delta_T/k_ultrafine.')
    p.add_argument('--h-fine', type=float, default=None, help='override; else Delta_T/k_fine')
    p.add_argument('--h-ultrafine', type=float, default=None,
                   help='override; else Delta_T/k_ultrafine')

    p.add_argument('--n-seeds', type=int, default=200)
    p.add_argument('--t-start', type=float, default=5.0)
    p.add_argument('--t-end', type=float, default=-1.0)

    p.add_argument('--batch-index', type=int, default=0)
    p.add_argument('--batches', type=str, default=None)
    p.add_argument('--n-batches', type=int, default=None)

    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--dtype', type=str, default='float64', choices=['float64'])

    p.add_argument('--train-frac', type=float, default=0.70)
    p.add_argument('--val-frac', type=float, default=0.15)
    p.add_argument('--split-mode', type=str, default='auto',
                   choices=['auto', 'by_batch', 'by_time'])

    p.add_argument('--max-order', type=int, default=5,
                   help='highest N-derivative target order to save (default 5: '
                        'Ndot..N5dot). Targets are exact analytic chain-rule '
                        'derivatives, independent of stencil depth; the '
                        '7-snapshot stencil supplies time-FD features through '
                        'N^(6). N^(5) is needed to assemble R6 (AB2CN2).')
    p.add_argument('--input-dtype', type=str, default='float32',
                   choices=['float32', 'float64'],
                   help='dtype of packed inputs.npy. float64 preserves high-order '
                        'time-FD accuracy (clean N3dot, matches the float64 '
                        'rollout) at ~2x I/O; float32 is smaller/cacheable but '
                        'caps N3dot via FD cancellation. Targets stay float32.')
    p.add_argument('--with-diagnostics', action='store_true',
                   help='also run the K-step truth integration and write the '
                        'closure-validation memmaps (omega_K_fine, e_total, ...).')
    p.add_argument('--flush-every', type=int, default=200)
    args = p.parse_args()

    if args.max_order < 5:
        sys.exit(f"ERROR: --max-order must be >=5 to fill all 5 target channels "
                 f"({TARGET_FIELDS}); got {args.max_order}")

    M = int(args.n_marks)
    if M < 2:
        sys.exit(f"ERROR: --n-marks must be >=2; got {M}")
    global INPUT_FIELDS
    INPUT_FIELDS = make_input_fields(M)
    h_fine = args.h_fine if args.h_fine is not None else args.Delta_T / args.k_fine
    h_ultrafine = (args.h_ultrafine if args.h_ultrafine is not None
                   else args.Delta_T / args.k_ultrafine)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[build] scenario={args.scenario}  M={M}-mark mmap build  "
          f"dt={args.Delta_T:g} h_fine={h_fine:g} (K={round(args.Delta_T/h_fine)}) "
          f"h_ultrafine={h_ultrafine:g}")

    with open(args.source_yaml) as f:
        yaml_cfg = yaml.safe_load(f)

    def _yget(path, default=None):
        cur = yaml_cfg
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return default
        return cur

    def _grid_float(key, default):
        v = _yget(['qg', 'grid', key]) or _yget(['grid', key])
        return float(v) if v is not None else float(default)

    def _pde_float(key, default=0.0):
        v = _yget(['qg', 'pde', key])
        if v is None:
            v = _yget(['pde', key])
        return float(v) if v is not None else float(default)

    Lx = _grid_float('Lx', 2.0 * math.pi)
    Ly = _grid_float('Ly', 2.0 * math.pi)
    nu, mu, beta = _pde_float('nu'), _pde_float('mu'), _pde_float('B')
    print(f"[build] Lx={Lx} Ly={Ly} nu={nu} mu={mu} beta={beta}")

    if args.dt_save is not None:
        # synthesize a uniform time axis from the snapshot count -- no stored times.
        _probe = _OmegaLoader(args.source_omega, batch_index=0)
        n_snap = _probe.n_snapshots
        times = np.arange(n_snap, dtype=np.float64) * float(args.dt_save)
        times_origin = f"synthesized: arange({n_snap})*{args.dt_save}"
        print(f"[build] times synthesized from --dt-save={args.dt_save} "
              f"(n_snapshots={n_snap}, t in [0, {times[-1]:.4f}])")
    elif args.source_times is not None:
        times = np.load(args.source_times)
        times_origin = str(args.source_times)
    elif args.source_omega.suffix.lower() == '.npz':
        with np.load(args.source_omega) as zf:
            if 'times' not in zf.files:
                sys.exit(f"ERROR: no 'times' in {args.source_omega}; "
                         f"pass --source-times or --dt-save")
            times = np.asarray(zf['times'])
        times_origin = f"npz['times'] in {args.source_omega}"
    else:
        sys.exit("ERROR: --source-times or --dt-save required when "
                 "--source-omega is .npy")

    loader = _OmegaLoader(args.source_omega, batch_index=0)
    Ny, Nx = loader.spatial_shape
    n_avail = loader.n_batches
    print(f"[build] grid=({Ny},{Nx})  mode={loader.mode}  avail_batches={n_avail}")

    if args.batches is not None:
        batches = [int(b) for b in args.batches.split(',') if b.strip()]
    elif args.n_batches is not None:
        batches = list(range(args.n_batches))
    else:
        batches = [args.batch_index]
    for b in batches:
        if not (0 <= b < n_avail):
            sys.exit(f"ERROR: batch {b} out of [0,{n_avail})")
    print(f"[build] batches={batches}")

    dtype = torch.float64
    device = args.device
    grid = CartesianGrid(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, device=device, precision=args.dtype)
    derivative = Derivative(grid)
    derivative.dx = derivative.dx.to(device)
    derivative.dy = derivative.dy.to(device)
    derivative.laplacian = derivative.laplacian.to(device)
    derivative.inv_laplacian = derivative.inv_laplacian.to(device)
    L_hat = build_L_hat(derivative, nu=nu, mu=mu, B=beta).to(device)

    fc = _yget(['qg', 'forcing']) or _yget(['forcing'])
    F_phys = forcing_meta = None
    if fc and isinstance(fc, dict) and fc.get('function') == 'unscaled_cosine':
        A = float(fc.get('A', 0.0)); Bk = float(fc.get('B', 0.0)); Cc = float(fc.get('C', 0.0))
        D = float(fc.get('D', 0.0)); E = float(fc.get('E', 0.0)); Ff = float(fc.get('F', 0.0))
        if Cc != 0.0 or Ff != 0.0:
            sys.exit("ERROR: time-dependent forcing incompatible with static-forcing closure.")
        x = torch.linspace(0, Lx, Nx, device=device, dtype=dtype)
        y = torch.linspace(0, Ly, Ny, device=device, dtype=dtype)
        F_phys = (A * torch.cos(Bk * x[None, :]) + D * torch.cos(E * y[:, None]))[None]
        forcing_meta = dict(function='unscaled_cosine', A=A, B=Bk, C=Cc, D=D, E=E, F=Ff)
        print(f"[build] static forcing {A}*cos({Bk}x)+{D}*cos({E}y)")
    else:
        print("[build] no forcing (F=0)")

    # seeds
    t_end = args.t_end if args.t_end > 0 else float(times[-1] - 2 * args.Delta_T)
    if args.t_start >= t_end:
        sys.exit(f"ERROR: t_start {args.t_start} >= t_end {t_end}")
    max_seed_t = float(times[-1] - 2 * args.Delta_T)
    t_end = min(t_end, max_seed_t)
    t_targets = np.linspace(args.t_start, t_end, args.n_seeds)
    seed_indices = np.unique(np.searchsorted(times, t_targets))
    seeds_per_batch = len(seed_indices)
    K_int = int(round(args.Delta_T / h_fine))
    if abs(K_int * h_fine - args.Delta_T) > 1e-12 * args.Delta_T:
        sys.exit(f"ERROR: Delta_T not integer multiple of h_fine")
    print(f"[build] seeds/batch={seeds_per_batch}  K={K_int}  max_order={args.max_order}")

    split_mode = ('by_batch' if len(batches) > 1 else 'by_time') \
        if args.split_mode == 'auto' else args.split_mode

    dT_tag = f'{args.Delta_T:.0e}'.replace('e-0', 'em').replace('e+0', 'ep')
    out_dir = args.out_dir / f'{args.scenario}_dT_{dT_tag}'
    pdir = out_dir / 'packed'
    pdir.mkdir(parents=True, exist_ok=True)

    N = seeds_per_batch * len(batches)
    n_tgt = len(TARGET_FIELDS)
    in_np_dtype = np.float32 if args.input_dtype == 'float32' else np.float64
    in_isz = np.dtype(in_np_dtype).itemsize
    print(f"[build] N={N}  input_dtype={args.input_dtype}  "
          f"inputs ~{N*len(INPUT_FIELDS)*Ny*Nx*in_isz/1024**3:.1f} GiB  "
          f"targets ~{N*n_tgt*Ny*Nx*4/1024**3:.1f} GiB"
          + ("  (+diagnostics)" if args.with_diagnostics else ""))

    inputs_mm = open_memmap(pdir / 'inputs.npy', mode='w+', dtype=in_np_dtype,
                            shape=(N, len(INPUT_FIELDS), Ny, Nx))
    targets_mm = open_memmap(pdir / 'targets.npy', mode='w+', dtype=np.float32,
                             shape=(N, n_tgt, Ny, Nx))
    diag32_mm = diag64_mm = None
    if args.with_diagnostics:
        diag32_mm = open_memmap(pdir / 'diag_f32.npy', mode='w+', dtype=np.float32,
                                shape=(N, len(DIAG_F32_FIELDS), Ny, Nx))
        diag64_mm = open_memmap(pdir / 'diag_f64.npy', mode='w+', dtype=np.float64,
                                shape=(N, len(DIAG_F64_FIELDS), Ny, Nx))

    t0 = time.time()
    gi = 0
    failed = []
    sample_records = []
    for b_pos, b in enumerate(batches):
        loader.set_batch(b)
        print(f"\n[build] === batch {b} ({b_pos+1}/{len(batches)}) ===")
        for j, s_idx in enumerate(seed_indices):
            t_seed = float(times[s_idx])
            seed_np = loader.read(int(s_idx))
            try:
                rec = build_one_seed(seed_np, args.Delta_T, K_int, h_fine,
                                     h_ultrafine, derivative, L_hat, F_phys,
                                     dtype, device,
                                     with_diagnostics=args.with_diagnostics,
                                     max_order=args.max_order, M=M)
            except Exception as e:
                print(f"  b={b} [{j+1}/{seeds_per_batch}] s_idx={s_idx} FAILED: {e}")
                failed.append(gi); gi += 1
                continue
            inputs_mm[gi] = np.stack([rec[f] for f in INPUT_FIELDS], axis=0)
            targets_mm[gi] = np.stack([rec[f] for f in TARGET_FIELDS], axis=0)
            if args.with_diagnostics:
                diag32_mm[gi] = np.stack([rec[f] for f in DIAG_F32_FIELDS], axis=0)
                diag64_mm[gi] = np.stack([rec[f] for f in DIAG_F64_FIELDS], axis=0)
            sample_records.append(dict(index=gi, seed_t=t_seed,
                                       seed_idx=int(s_idx), batch_idx=int(b)))
            if (gi + 1) % args.flush_every == 0:
                inputs_mm.flush(); targets_mm.flush()
                if args.with_diagnostics:
                    diag32_mm.flush(); diag64_mm.flush()
            if (j + 1) % 20 == 1 or (j + 1) == seeds_per_batch:
                ndot = np.sqrt(np.mean(rec['N_dot_0_anal'] ** 2))
                n3 = np.sqrt(np.mean(rec['N_3dot_0_anal'] ** 2))
                n5 = np.sqrt(np.mean(rec['N_5dot_0_anal'] ** 2))
                print(f"  b={b} [{j+1:4d}/{seeds_per_batch}] s_idx={s_idx:5d} "
                      f"t={t_seed:8.3f}  |Ndot|={ndot:.3e} |N3dot|={n3:.3e} "
                      f"|N5dot|={n5:.3e}")
            gi += 1

    inputs_mm.flush(); targets_mm.flush()
    del inputs_mm, targets_mm
    if args.with_diagnostics:
        diag32_mm.flush(); diag64_mm.flush(); del diag32_mm, diag64_mm

    n_completed = N - len(failed)
    print(f"\n[build] completed {n_completed}/{N}  ({len(failed)} failures)")

    # ---- split (failed rows excluded) ---- #
    idx = np.arange(N, dtype=np.int32)
    if split_mode == 'by_time':
        tr, va, te = [], [], []
        t_n_train = int(round(seeds_per_batch * args.train_frac))
        t_n_val = int(round(seeds_per_batch * args.val_frac))
        for b_pos in range(len(batches)):
            base = b_pos * seeds_per_batch
            tr += list(range(base, base + t_n_train))
            va += list(range(base + t_n_train, base + t_n_train + t_n_val))
            te += list(range(base + t_n_train + t_n_val, base + seeds_per_batch))
        train_idx, val_idx, test_idx = map(lambda L: np.array(L, np.int32), (tr, va, te))
    else:  # by_batch
        n_b = len(batches)
        n_b_train = max(1, int(round(n_b * args.train_frac)))
        n_b_val = max(1, int(round(n_b * args.val_frac)))
        if n_b_train + n_b_val >= n_b:
            n_b_val = max(1, n_b - n_b_train - 1)
        sel = lambda ks: np.array([i for k in ks
                                   for i in range(k * seeds_per_batch,
                                                  (k + 1) * seeds_per_batch)], np.int32)
        train_idx = sel(range(0, n_b_train))
        val_idx = sel(range(n_b_train, n_b_train + n_b_val))
        test_idx = sel(range(n_b_train + n_b_val, n_b))

    if failed:
        fset = set(failed)
        flt = lambda a: a[~np.isin(a, list(fset))]
        train_idx, val_idx, test_idx = flt(train_idx), flt(val_idx), flt(test_idx)

    for tgt in (out_dir / 'split.npz', pdir / 'split.npz'):
        np.savez(tgt, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    pack_meta = dict(
        source_root=str(out_dir), N=N, Ny=Ny, Nx=Nx,
        input_fields=INPUT_FIELDS, target_fields=TARGET_FIELDS,
        dtype='float32', input_dtype=args.input_dtype, target_dtype='float32',
        n_missing=len(failed),
        Lx=float(Lx), Ly=float(Ly), Delta_T=float(args.Delta_T),
        diag_f32_fields=DIAG_F32_FIELDS if args.with_diagnostics else [],
        diag_f64_fields=DIAG_F64_FIELDS if args.with_diagnostics else [],
    )
    with open(pdir / 'pack_meta.json', 'w') as f:
        json.dump(pack_meta, f, indent=2)

    manifest = dict(
        scenario=args.scenario, Lx=float(Lx), Ly=float(Ly), Nx=int(Nx), Ny=int(Ny),
        nu=float(nu), mu=float(mu), beta=float(beta),
        Delta_T=float(args.Delta_T), K=int(K_int),
        h_fine=float(h_fine), h_ultrafine=float(h_ultrafine),
        n_snapshots_per_sample=M, max_order=int(args.max_order),
        input_fields=INPUT_FIELDS, target_fields=TARGET_FIELDS,
        source_omega_path=str(args.source_omega),
        source_times_path=(str(args.source_times) if args.source_times else times_origin),
        source_yaml_path=str(args.source_yaml),
        batches_used=[int(b) for b in batches], seeds_per_batch=int(seeds_per_batch),
        n_total=int(N), n_completed=int(n_completed), n_failed=int(len(failed)),
        n_train=int(len(train_idx)), n_val=int(len(val_idx)), n_test=int(len(test_idx)),
        split_mode=split_mode, dtype='float32 (packed) / float64 (solver)',
        has_forcing=bool(F_phys is not None), forcing=forcing_meta, device=device,
        with_diagnostics=bool(args.with_diagnostics),
        format='packed_mmap', packed_subdir='packed', samples=sample_records,
    )
    with open(out_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    dt = (time.time() - t0) / 60
    print(f"\n[build] split: train {len(train_idx)} / val {len(val_idx)} / test {len(test_idx)}")
    print(f"[build] done in {dt:.1f} min  ({dt*60/max(N,1):.2f}s/sample)")
    print(f"[build] packed dir: {pdir}")
    print(f"[build] train with --root-dir {out_dir} "
          f"--input-fields {' '.join(INPUT_FIELDS)} "
          f"--target-fields {' '.join(TARGET_FIELDS)}")


if __name__ == '__main__':
    main()