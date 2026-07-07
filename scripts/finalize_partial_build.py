"""
finalize_partial_build.py -- salvage an interrupted build_training_data_mmap.py run.

The deep 28-mark builders preallocate packed/inputs.npy and packed/targets.npy at
the full N (numpy open_memmap w+ writes the header immediately; the file is
sparse until samples are written) and only write split.npz / pack_meta.json /
manifest.json at the very end. A qdel therefore leaves fully-valid samples
0..n-1 on disk with no metadata and a possibly torn trailing record.

This tool, per member deep dir:
  1. parses the LAST "[mmap-build] cmd:" line of the member's build log to
     recover the exact generator arguments (argparse last-wins semantics);
  2. finds n_complete by scanning the memmaps: a sample counts only if EVERY
     input channel and EVERY target channel is finite with nonzero amplitude
     (omega/psi/N^(m) fields of developed or decaying turbulence are never
     identically zero; a torn record has zero-filled trailing channels);
     n = min(n_inputs, n_targets);
  3. truncates BOTH .npy files in place: the header is rewritten with the new
     shape padded to the byte-identical header length (data offset unchanged),
     then the file is truncated to header + n rows;
  4. reconstructs sample_records from the generator's deterministic seed logic
     (linspace(t_start, t_end, n_seeds) -> searchsorted(times) -> unique),
     reading `times` from the source DNS npz;
  5. writes split.npz (out_dir AND packed/), pack_meta.json, manifest.json with
     EXACTLY the schema build_training_data_mmap.py writes (n_total = n_completed
     = n, n_failed = 0 -- the log is grepped for FAILED lines and the count is
     asserted zero, else the tool refuses: interleaved failure indices cannot be
     reconstructed after truncation);
  6. writes salvage_info.json (provenance sidecar: original N, salvaged n,
     killed job id if given, scan verdicts) -- NOT part of the generator schema,
     kept in a separate file on purpose;
  7. validates by loading 3 random samples end-to-end through
     dataset.PackedClosureDataset (train split, all input/target fields).

Usage (from anywhere; needs numpy+torch+yaml, CPU only):
    python finalize_partial_build.py \
        --deep-dir  .../data/ensemble_N5_7lag/FRC-b0/forced_turbulence_dT_5em3 \
        --log       .../logs/build_mmap_FRC-b0.log \
        [--training-dir .../src/qg/training] [--job-id 1825507] [--dry-run]

Exit code 0 iff metadata written AND validation PASS (or --dry-run).
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from numpy.lib import format as npf

TARGET_FIELDS = ['N_dot_0_anal', 'N_ddot_0_anal', 'N_3dot_0_anal',
                 'N_4dot_0_anal', 'N_5dot_0_anal']


def make_input_fields(M):
    om = ['omega_0'] + [f'omega_m{k}' for k in range(1, M)]
    ps = ['psi_0'] + [f'psi_m{k}' for k in range(1, M)]
    return om + ps


# --------------------------------------------------------------------------- #
# 1. build-log command recovery                                               #
# --------------------------------------------------------------------------- #

def parse_build_cmd(log_path: Path) -> dict:
    """Recover the generator args from the LAST '[mmap-build] cmd:' line.
    Last-wins for repeated flags, matching argparse (the submitter appends
    overrides, e.g. --Delta-T appears twice in the new8 builds)."""
    cmd_line = None
    with open(log_path) as f:
        for line in f:
            if '[mmap-build] cmd:' in line:
                cmd_line = line.split('cmd:', 1)[1].strip()
    if cmd_line is None:
        raise RuntimeError(f'no "[mmap-build] cmd:" line in {log_path}')
    toks = shlex.split(cmd_line)
    args = {}
    i = 0
    while i < len(toks):
        if toks[i].startswith('--'):
            key = toks[i][2:].replace('-', '_')
            if i + 1 < len(toks) and not toks[i + 1].startswith('--'):
                args[key] = toks[i + 1]
                i += 2
            else:
                args[key] = True   # store_true flag
                i += 1
        else:
            i += 1
    n_failed_in_log = 0
    with open(log_path) as f:
        for line in f:
            if 'FAILED:' in line:
                n_failed_in_log += 1
    args['_n_failed_in_log'] = n_failed_in_log
    args['_cmd_line'] = cmd_line
    return args


# --------------------------------------------------------------------------- #
# 2/3. memmap scan + in-place truncation                                       #
# --------------------------------------------------------------------------- #

def read_npy_header(path: Path):
    with open(path, 'rb') as f:
        version = npf.read_magic(f)
        npf._check_version(version)
        shape, fortran, dtype = npf._read_array_header(f, version)
        return shape, fortran, dtype, f.tell(), version


def _channel_ok(arr2d) -> bool:
    m = np.abs(arr2d, dtype=np.float64).max()
    return np.isfinite(m) and m > 0.0


def sample_complete(mm, i) -> bool:
    return all(_channel_ok(mm[i, c]) for c in range(mm.shape[1]))


def find_n_complete(path: Path) -> int:
    """Largest n such that samples 0..n-1 are fully written (every channel
    finite-nonzero). Binary search on channel 0, then walk down verifying
    ALL channels (catches a torn record whose early channels landed)."""
    mm = np.load(path, mmap_mode='r')
    N = mm.shape[0]
    lo, hi = -1, N - 1          # invariant: ch0(lo) nonzero (or lo=-1), ch0(hi+1..) zero
    if _channel_ok(mm[N - 1, 0]):
        lo = N - 1
    else:
        a, b = 0, N - 1          # find last i with ch0 nonzero in [a,b)
        if not _channel_ok(mm[0, 0]):
            return 0
        while b - a > 1:
            mid = (a + b) // 2
            if _channel_ok(mm[mid, 0]):
                a = mid
            else:
                b = mid
        lo = a
    # walk down from lo until a fully-complete sample; everything below a
    # complete sample was written strictly earlier (sequential gi order).
    i = lo
    while i >= 0 and not sample_complete(mm, i):
        i -= 1
    del mm
    return i + 1


def truncate_npy_inplace(path: Path, n: int, dry_run=False):
    """Rewrite header shape[0]=n at IDENTICAL header byte length, ftruncate."""
    shape, fortran, dtype, offset, version = read_npy_header(path)
    if fortran:
        raise RuntimeError(f'{path}: fortran order unsupported')
    if n > shape[0]:
        raise RuntimeError(f'{path}: n={n} > allocated {shape[0]}')
    new_shape = (n,) + tuple(shape[1:])
    d = npf.dtype_to_descr(dtype)
    header = {'descr': d, 'fortran_order': False, 'shape': new_shape}
    row_bytes = int(np.prod(shape[1:])) * dtype.itemsize
    new_size = offset + n * row_bytes
    if dry_run:
        return new_size
    # build the header bytes exactly offset long
    import io
    buf = io.BytesIO()
    npf._write_array_header(buf, header, version=version)
    raw = bytearray(buf.getvalue())
    if len(raw) > offset:
        raise RuntimeError(f'{path}: new header ({len(raw)}B) exceeds old '
                           f'({offset}B); cannot truncate in place')
    if len(raw) < offset:
        # re-pad: header dict string ends with spaces + '\n' before data;
        # extend the padding so the data offset is unchanged.
        pad = offset - len(raw)
        assert raw[-1:] == b'\n'
        raw = raw[:-1] + b' ' * pad + b'\n'
        # fix the little-endian header-length field (bytes 8:10 for v1, 8:12 for v2)
        hlen = len(raw) - (10 if version == (1, 0) else 12)
        if version == (1, 0):
            if hlen > 65535:
                raise RuntimeError('padded v1 header too long')
            raw[8:10] = int(hlen).to_bytes(2, 'little')
        else:
            raw[8:12] = int(hlen).to_bytes(4, 'little')
    with open(path, 'r+b') as f:
        f.write(bytes(raw))
        f.truncate(new_size)
    # verify
    shape2, _, dtype2, offset2, _ = read_npy_header(path)
    assert shape2 == new_shape and offset2 == offset and dtype2 == dtype, \
        f'{path}: post-truncation header mismatch'
    return new_size


# --------------------------------------------------------------------------- #
# 4. deterministic seed reconstruction (mirrors the generator)                 #
# --------------------------------------------------------------------------- #

def reconstruct_seeds(args: dict):
    src = Path(args['source_omega'])
    Delta_T = float(args['Delta_T'])
    n_seeds = int(args.get('n_seeds', 200))
    t_start = float(args.get('t_start', 5.0))
    t_end_arg = float(args.get('t_end', -1.0))
    if 'dt_save' in args:
        with np.load(src) as zf:
            key = next(c for c in ('omega_FR', 'omega', 'q', 'q_FR') if c in zf.files)
            n_snap = zf[key].shape[-3] if zf[key].ndim >= 3 else zf[key].shape[0]
        times = np.arange(n_snap, dtype=np.float64) * float(args['dt_save'])
        times_origin = f"synthesized: arange({n_snap})*{args['dt_save']}"
    elif 'source_times' in args:
        times = np.load(args['source_times'])
        times_origin = str(args['source_times'])
    else:
        with np.load(src) as zf:
            times = np.asarray(zf['times'])
        times_origin = f"npz['times'] in {src}"
    t_end = t_end_arg if t_end_arg > 0 else float(times[-1] - 2 * Delta_T)
    t_end = min(t_end, float(times[-1] - 2 * Delta_T))
    t_targets = np.linspace(t_start, t_end, n_seeds)
    seed_indices = np.unique(np.searchsorted(times, t_targets))
    return times, times_origin, seed_indices


# --------------------------------------------------------------------------- #
# 5. metadata writers (schema-exact copies of the generator's dicts)           #
# --------------------------------------------------------------------------- #

def yaml_physics(source_yaml: Path):
    with open(source_yaml) as f:
        cfg = yaml.safe_load(f)

    def yget(path, default=None):
        cur = cfg
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return default
        return cur

    def grid_float(key, default):
        v = yget(['qg', 'grid', key])
        if v is None:
            v = yget(['grid', key])
        return float(v) if v is not None else float(default)

    def pde_float(key, default=0.0):
        v = yget(['qg', 'pde', key])
        if v is None:
            v = yget(['pde', key])
        return float(v) if v is not None else float(default)

    import math
    Lx = grid_float('Lx', 2.0 * math.pi)
    Ly = grid_float('Ly', 2.0 * math.pi)
    nu, mu, beta = pde_float('nu'), pde_float('mu'), pde_float('B')

    fc = yget(['qg', 'forcing']) or yget(['forcing'])
    forcing_meta = None
    has_forcing = False
    if fc and isinstance(fc, dict) and fc.get('function') == 'unscaled_cosine':
        A = float(fc.get('A', 0.0)); Bk = float(fc.get('B', 0.0)); Cc = float(fc.get('C', 0.0))
        D = float(fc.get('D', 0.0)); E = float(fc.get('E', 0.0)); Ff = float(fc.get('F', 0.0))
        forcing_meta = dict(function='unscaled_cosine', A=A, B=Bk, C=Cc, D=D, E=E, F=Ff)
        has_forcing = True
    return Lx, Ly, nu, mu, beta, has_forcing, forcing_meta


def by_time_split(n, train_frac, val_frac):
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    train_idx = np.arange(0, n_train, dtype=np.int32)
    val_idx = np.arange(n_train, n_train + n_val, dtype=np.int32)
    test_idx = np.arange(n_train + n_val, n, dtype=np.int32)
    return train_idx, val_idx, test_idx


# --------------------------------------------------------------------------- #
# 7. validation through the dataset class                                      #
# --------------------------------------------------------------------------- #

def validate(deep_dir: Path, training_dir: Path, input_fields, seed=0):
    sys.path.insert(0, str(training_dir))
    from dataset import PackedClosureDataset  # noqa: E402
    ds = PackedClosureDataset(root_dir=str(deep_dir), split='train',
                              input_fields=tuple(input_fields),
                              target_fields=tuple(TARGET_FIELDS))
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(ds), size=min(3, len(ds)), replace=False)
    for i in picks:
        x, y = ds[int(i)]
        for name, t in (('inputs', x), ('targets', y)):
            a = np.asarray(t)
            if not np.all(np.isfinite(a)):
                return False, f'sample {i}: non-finite {name}'
            if np.abs(a).max() == 0.0:
                return False, f'sample {i}: all-zero {name}'
    return True, f'{len(picks)} samples loaded, finite, nonzero (indices {sorted(picks.tolist())})'


# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--deep-dir', type=Path, required=True,
                   help='the <scenario>_dT_<tag> dir containing packed/')
    p.add_argument('--log', type=Path, required=True,
                   help='the build_mmap_<member>.log of the killed job')
    p.add_argument('--training-dir', type=Path,
                   default=Path('/gdata/projects/ml_scope/Closure_modeling/QG-closure/'
                                'qg-simple-package-stable/src/qg/training'))
    p.add_argument('--job-id', type=str, default=None)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()

    deep = a.deep_dir.resolve()
    pdir = deep / 'packed'
    inputs_p, targets_p = pdir / 'inputs.npy', pdir / 'targets.npy'
    for f in (inputs_p, targets_p):
        if not f.exists():
            sys.exit(f'ERROR: missing {f}')
    if (deep / 'manifest.json').exists():
        sys.exit(f'ERROR: {deep}/manifest.json already exists -- this build '
                 f'completed (or was already finalized); refusing to touch it.')

    args = parse_build_cmd(a.log)
    if args['_n_failed_in_log'] != 0:
        sys.exit(f"ERROR: {args['_n_failed_in_log']} FAILED sample(s) in the log; "
                 f"interleaved failure indices cannot be reconstructed -- salvage "
                 f"by hand.")

    M = int(args.get('n_marks', 7))
    input_fields = make_input_fields(M)

    in_shape, _, in_dtype, _, _ = read_npy_header(inputs_p)
    tg_shape, _, tg_dtype, _, _ = read_npy_header(targets_p)
    N_alloc, C_in, Ny, Nx = in_shape
    assert C_in == 2 * M, f'inputs channels {C_in} != 2*M={2*M}'
    assert tg_shape[:2] == (N_alloc, len(TARGET_FIELDS)) and tg_shape[2:] == (Ny, Nx)

    n_in = find_n_complete(inputs_p)
    n_tg = find_n_complete(targets_p)
    n = min(n_in, n_tg)
    print(f'[finalize] {deep.parent.name}: alloc N={N_alloc}  complete inputs={n_in} '
          f'targets={n_tg}  -> n={n}')
    if n == 0:
        sys.exit('ERROR: no complete samples -- nothing to salvage.')

    times, times_origin, seed_indices = reconstruct_seeds(args)
    if len(seed_indices) < n:
        sys.exit(f'ERROR: reconstructed only {len(seed_indices)} seeds < n={n}; '
                 f'seed logic mismatch -- do not trust this salvage.')
    sample_records = [dict(index=i, seed_t=float(times[int(seed_indices[i])]),
                           seed_idx=int(seed_indices[i]), batch_idx=0)
                      for i in range(n)]

    Delta_T = float(args['Delta_T'])
    h_fine = float(args.get('h_fine', Delta_T / float(args.get('k_fine', 100))))
    h_ultrafine = float(args.get('h_ultrafine', Delta_T / float(args.get('k_ultrafine', 200))))
    K_int = int(round(Delta_T / h_fine))
    train_frac = float(args.get('train_frac', 0.70))
    val_frac = float(args.get('val_frac', 0.15))
    Lx, Ly, nu, mu, beta, has_forcing, forcing_meta = yaml_physics(Path(args['source_yaml']))

    if a.dry_run:
        print(f'[finalize] DRY RUN: would truncate to n={n}, split '
              f'{int(round(n*train_frac))}/{int(round(n*val_frac))}/'
              f'{n-int(round(n*train_frac))-int(round(n*val_frac))}')
        return

    truncate_npy_inplace(inputs_p, n)
    truncate_npy_inplace(targets_p, n)

    train_idx, val_idx, test_idx = by_time_split(n, train_frac, val_frac)
    for tgt in (deep / 'split.npz', pdir / 'split.npz'):
        np.savez(tgt, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    pack_meta = dict(
        source_root=str(deep), N=n, Ny=int(Ny), Nx=int(Nx),
        input_fields=input_fields, target_fields=TARGET_FIELDS,
        dtype='float32', input_dtype=str(args.get('input_dtype', 'float32')),
        target_dtype='float32',
        n_missing=0,
        Lx=float(Lx), Ly=float(Ly), Delta_T=Delta_T,
        diag_f32_fields=[], diag_f64_fields=[],
    )
    with open(pdir / 'pack_meta.json', 'w') as f:
        json.dump(pack_meta, f, indent=2)

    manifest = dict(
        scenario=args['scenario'], Lx=float(Lx), Ly=float(Ly), Nx=int(Nx), Ny=int(Ny),
        nu=float(nu), mu=float(mu), beta=float(beta),
        Delta_T=Delta_T, K=K_int,
        h_fine=h_fine, h_ultrafine=h_ultrafine,
        n_snapshots_per_sample=M, max_order=int(args.get('max_order', 5)),
        input_fields=input_fields, target_fields=TARGET_FIELDS,
        source_omega_path=str(args['source_omega']),
        source_times_path=(str(args.get('source_times')) if args.get('source_times')
                           else times_origin),
        source_yaml_path=str(args['source_yaml']),
        batches_used=[0], seeds_per_batch=int(n),
        n_total=int(n), n_completed=int(n), n_failed=0,
        n_train=int(len(train_idx)), n_val=int(len(val_idx)), n_test=int(len(test_idx)),
        split_mode=str(args.get('split_mode', 'by_time')),
        dtype='float32 (packed) / float64 (solver)',
        has_forcing=bool(has_forcing), forcing=forcing_meta,
        device=str(args.get('device', 'cuda')),
        with_diagnostics=False,
        format='packed_mmap', packed_subdir='packed', samples=sample_records,
    )
    with open(deep / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    salvage = dict(
        salvaged_at=datetime.now(timezone.utc).isoformat(),
        killed_job_id=a.job_id, original_alloc_N=int(N_alloc),
        n_complete_inputs=int(n_in), n_complete_targets=int(n_tg), n=int(n),
        torn_records_dropped=int(max(n_in, n_tg) - n),
        build_cmd=args['_cmd_line'], build_log=str(a.log),
        method='per-channel finite-nonzero scan; in-place npy header rewrite + ftruncate',
    )
    with open(deep / 'salvage_info.json', 'w') as f:
        json.dump(salvage, f, indent=2)

    ok, msg = validate(deep, a.training_dir, input_fields)
    verdict = 'PASS' if ok else 'FAIL'
    print(f'[finalize] {deep.parent.name}: n={n}  split '
          f'{len(train_idx)}/{len(val_idx)}/{len(test_idx)}  validation {verdict}: {msg}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
