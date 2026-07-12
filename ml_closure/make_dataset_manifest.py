"""
Step-0 canonical-artifact generator for one SGS-closure run dir (CP-ML-1 plan S2).

For <run_dir> (e.g. .../SGS_closure_ensemble/FPC-const) this writes, idempotently:

  <run_dir>/DNS_LES_s<scale>.npz   canonicalized from the EXISTING piff product
                                   piff/DNS_s<scale>_LES.npz (approved default 2:
                                   NO recompute of Pi_FF from DNS_FR). Adds:
                                     * times prepended with t=0 (the packaged product
                                       stores 445 frames incl. IC vs 444 save times —
                                       the audit-A off-by-one, fixed here once);
                                     * ubar/vbar computed ONCE from psi = inv_lap(omega_bar)
                                       with the solver's own convention (basis.puv):
                                       u = -d(psi)/dy, v = +d(psi)/dx, and the k=0 mode
                                       of u set to U(t_n) exactly as bc.Flow.const_x_flow
                                       does (state.uh[...,0,0] = U); v k=0 mode = 0;
                                     * U_snap / Re_snap / zeta_snap at each frame from the
                                       inlet table at step n = round(t/dt) (never from
                                       field statistics);
                                     * meta json (conventions, source, ranges).
  <run_dir>/U_of_t.npz             byte copy of the run's inlet table (config qg.bc.inlet_table).
  <run_dir>/DATASET_MANIFEST.md    human sections + a fenced ```yaml machine block that
                                   dataset_piff.py parses (paths, shapes, grid, dt_save,
                                   sponge extent, mask params, filter, seeds, NaN check,
                                   caveats) — BRANCH_LOG 2026-07-07 template list.

Float32 storage / float64 compute (spec S0 precision policy for this branch).

Usage:
    python make_dataset_manifest.py <run_dir> [--scale 4] [--force]
"""

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

MANIFEST_NAME = 'DATASET_MANIFEST.md'
MACHINE_BLOCK_BEGIN = '<!-- machine-block -->'


def _fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _sha256(path, max_bytes=None):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def compute_uv_from_omega(omega_bar, Lx, Ly, U_snap):
    """u,v from psi = inv_laplacian(omega) per solver convention (basis.puv):
    uh = -dy*ph, vh = +dx*ph; then u k0-mode := U(t_n), v k0-mode := 0
    (bc.Flow.const_x_flow). f64 compute, f32 return."""
    T, Ny, Nx = omega_bar.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=Ly / Ny)
    KX, KY = np.meshgrid(kx, ky)            # (Ny, Nx), row=y col=x like the fields
    ksq = KX**2 + KY**2
    inv_lap = np.zeros_like(ksq)
    nz = ksq > 0
    inv_lap[nz] = -1.0 / ksq[nz]            # laplacian = -k^2
    ubar = np.empty((T, Ny, Nx), dtype=np.float32)
    vbar = np.empty((T, Ny, Nx), dtype=np.float32)
    for t in range(T):
        qh = np.fft.fft2(omega_bar[t].astype(np.float64))
        ph = inv_lap * qh
        u = np.real(np.fft.ifft2(-1j * KY * ph))   # zero-mean by construction
        v = np.real(np.fft.ifft2(1j * KX * ph))
        ubar[t] = (u + U_snap[t]).astype(np.float32)
        vbar[t] = v.astype(np.float32)
    return ubar, vbar


def measured_strip(mask2d, axis, thresh):
    """Support of a filtered sponge strip along one axis: indices where the
    max over the other axis exceeds thresh. Returns (lo, hi) or None."""
    prof = mask2d.max(axis=axis)
    nz = np.where(prof > thresh)[0]
    if len(nz) == 0:
        return None
    return int(nz.min()), int(nz.max())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('run_dir')
    ap.add_argument('--scale', type=int, default=4)
    ap.add_argument('--force', action='store_true', help='overwrite existing canonical artifacts')
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    s = args.scale
    if not run_dir.is_dir():
        _fail(f"run dir not found: {run_dir}")

    # ---- inputs ----------------------------------------------------------- #
    cfg_path = run_dir / 'config.yaml'
    if not cfg_path.exists():
        _fail(f"missing {cfg_path}")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)['qg']

    src_npz = run_dir / 'piff' / f'DNS_s{s}_LES.npz'
    if not src_npz.exists():
        alt = run_dir / f'piff_s{s}' / 'DNS_LES.npz'
        if alt.exists():
            src_npz = alt
        else:
            _fail(f"piff product not found: {run_dir}/piff/DNS_s{s}_LES.npz nor {alt}")
    src_npz = src_npz.resolve()

    table_src = Path(cfg['bc']['inlet_table'])
    if not table_src.exists():
        _fail(f"inlet table not found: {table_src}")

    out_npz = run_dir / f'DNS_LES_s{s}.npz'
    out_table = run_dir / 'U_of_t.npz'
    out_manifest = run_dir / MANIFEST_NAME
    for p in (out_npz, out_manifest):
        if p.exists() and not args.force:
            _fail(f"{p} exists — rerun with --force to overwrite")

    print(f"[manifest] run {run_dir.name}: source {src_npz}")
    z = np.load(src_npz)
    omega_bar = z['omega_bar']          # (B, T, Ny, Nx) f32
    pi_ff = z['pi_ff']
    times = np.asarray(z['times'], dtype=np.float64)
    chi_obs_bar = z['chi_obs_bar']
    chi_sponge_bar = z['chi_sponge_bar']
    scale = int(z['_scale'][0])
    alpha = float(z['_alpha'][0])
    if scale != s:
        _fail(f"npz _scale={scale} != requested --scale {s}")
    B, T, Ny, Nx = omega_bar.shape
    if B != 1:
        _fail(f"expected single-member product, got B={B}")

    # ---- time axis: fix the packaging off-by-one (frames incl. IC) -------- #
    if T == len(times) + 1:
        times_full = np.concatenate([[0.0], times])
        print(f"[manifest] times off-by-one fixed: {len(times)} save times -> {T} frames (t=0 IC prepended)")
    elif T == len(times):
        times_full = times
    else:
        _fail(f"frames ({T}) vs times ({len(times)}): neither equal nor off-by-one")

    dt = float(cfg['time']['dt'])
    save_rate = int(cfg['time']['save_rate'])
    dt_save = dt * save_rate
    Lx = float(cfg['grid']['Lx']); Ly = float(cfg['grid']['Ly'])
    nu = float(cfg['pde']['nu'])
    r = float(cfg['mask']['r'])
    D = 2.0 * r
    # mask.circular IGNORES config x_center/y_center: center is the domain center.
    x_c, y_c = Lx / 2.0, Ly / 2.0
    width = float(cfg['bc']['width'])
    penalty = float(cfg['pde']['penalty'])
    sponge_factor = float(cfg['bc']['sponge'])

    # ---- inlet table + per-frame U/Re/zeta -------------------------------- #
    tab = np.load(table_src)
    t_tab = np.asarray(tab['t'], dtype=np.float64)
    U_tab = np.asarray(tab['U'], dtype=np.float64)
    Re_tab = np.asarray(tab['Re'], dtype=np.float64)
    dt_tab = float(t_tab[1] - t_tab[0])
    if abs(dt_tab - dt) > 1e-12:
        _fail(f"table dt {dt_tab} != solver dt {dt}")
    steps = np.rint(times_full / dt).astype(np.int64)
    if steps.max() >= len(U_tab):
        _fail(f"snapshot step {steps.max()} beyond table length {len(U_tab)}")
    if np.max(np.abs(t_tab[steps] - times_full)) > 1e-9:
        _fail("snapshot times do not sit on table time grid")
    U_snap = U_tab[steps]
    Re_snap = Re_tab[steps]

    # ---- ubar/vbar (computed ONCE, at build) ------------------------------ #
    print(f"[manifest] computing ubar/vbar for {T} frames ({Ny}x{Nx}) ...")
    ubar, vbar = compute_uv_from_omega(omega_bar[0], Lx, Ly, U_snap)

    # ---- NaN check --------------------------------------------------------- #
    nan_counts = {k: int(np.isnan(a).sum()) for k, a in
                  [('omega_bar', omega_bar), ('pi_ff', pi_ff), ('ubar', ubar), ('vbar', vbar)]}
    if any(nan_counts.values()):
        _fail(f"NaN in canonical fields: {nan_counts}")

    # ---- sponge extents: analytic strips + measured filtered support ------ #
    sp = chi_sponge_bar[0]
    thresh = 0.05
    # as implemented (bc.Region via outlet_mask_rtd, single): vertical outlet strip
    # x/Lx in (1-w, 1); horizontal bidirectional strip y/Ly in (1-2w, 1).
    sponge_x_frac = [1.0 - width, 1.0]
    sponge_y_frac = [1.0 - 2.0 * width, 1.0]
    # measure each strip away from the other (else the profile max picks up the
    # perpendicular strip and reports full-domain support)
    iy_top = int(Ny * sponge_y_frac[0]) - 4
    ix_out = int(Nx * sponge_x_frac[0]) - 4
    x_meas = measured_strip(sp[:iy_top, :], axis=0, thresh=thresh)   # x columns
    y_meas = measured_strip(sp[:, :ix_out], axis=1, thresh=thresh)   # y rows

    # ---- write canonical npz ---------------------------------------------- #
    meta = {
        'created': datetime.now(timezone.utc).isoformat(),
        'source': str(src_npz),
        'generator': 'ml_closure/make_dataset_manifest.py',
        'canonicalized': 'rename of piff product + ubar/vbar + times t=0 prepend; Pi_FF NOT recomputed (approved default 2)',
        'uv_convention': 'u=-dpsi/dy, v=+dpsi/dx, psi=inv_lap(omega_bar) (solver basis.puv); u k0-mode = U(t_n), v k0-mode = 0 (bc.Flow.const_x_flow)',
        'pi_ff_convention': 'Pi_FF = filter[J+Brinkman+Sponge]_FR - [J+Brinkman+Sponge]_LES (spatial_closure/compute_pi_ff.py:181, Eq.8 arXiv 2508.06678)',
        'times_convention': 'times[i] is frame i; frame 0 = IC at t=0 (off-by-one fixed at build)',
        'U_snap_convention': 'U_of_t.npz U[round(t/dt)] — never per-snapshot field statistics',
        'precision': 'f64 compute, f32 storage',
        'D': D, 'nu': nu, 'Lx': Lx, 'Ly': Ly, 'x_c': x_c, 'y_c': y_c,
        'scale': scale, 'alpha': alpha,
    }
    print(f"[manifest] writing {out_npz} ...")
    np.savez_compressed(
        out_npz,
        omega_bar=omega_bar, pi_ff=pi_ff, ubar=ubar[None], vbar=vbar[None],
        times=times_full, U_snap=U_snap, Re_snap=Re_snap,
        chi_obs_bar=chi_obs_bar, chi_sponge_bar=chi_sponge_bar,
        _scale=np.array([scale], dtype=np.int32),
        _alpha=np.array([alpha], dtype=np.float32),
        meta=np.array(json.dumps(meta)),
    )

    if out_table.exists() and not args.force:
        print(f"[manifest] {out_table} exists — kept")
    else:
        shutil.copy2(table_src, out_table)
        print(f"[manifest] copied inlet table -> {out_table}")

    # ---- manifest ---------------------------------------------------------- #
    tab_meta = json.loads(str(tab['meta'])) if 'meta' in tab.files else {}
    machine = {
        'run': run_dir.name,
        'run_dir': str(run_dir),
        'files': {
            'dataset': out_npz.name,
            'u_table': out_table.name,
            'source_piff': str(src_npz),
            'source_table': str(table_src),
        },
        'shapes': {
            'omega_bar': list(omega_bar.shape), 'pi_ff': list(pi_ff.shape),
            'ubar': [1, T, Ny, Nx], 'vbar': [1, T, Ny, Nx],
            'times': [T], 'chi_obs_bar': list(chi_obs_bar.shape),
            'chi_sponge_bar': list(chi_sponge_bar.shape),
        },
        'dtypes': {'fields': 'float32', 'times': 'float64'},
        'grid': {
            'N_FR': [Ny * scale, Nx * scale], 'N_LES': [Ny, Nx],
            'Lx': Lx, 'Ly': Ly, 'dx_LES': Lx / Nx, 'dy_LES': Ly / Ny,
        },
        'time': {
            'dt': dt, 'save_rate': save_rate, 'dt_save': dt_save,
            'n_frames': T, 't_first': float(times_full[0]), 't_last': float(times_full[-1]),
            'uniform_t': bool(np.allclose(np.diff(times_full), dt_save, atol=1e-9)),
        },
        'table': {'dt': dt_tab, 'len': int(len(U_tab)), 'meta': tab_meta},
        'physics': {
            'nu': nu, 'D': D, 'r': r, 'x_c': x_c, 'y_c': y_c,
            'penalty_factor': penalty, 'sponge_factor': sponge_factor,
            'eta_penalty_phys': penalty * dt, 'eta_sponge_phys': sponge_factor * dt,
            'U_range': [float(U_snap.min()), float(U_snap.max())],
            'Re_range': [float(Re_snap.min()), float(Re_snap.max())],
        },
        'filter': {'scale': scale, 'alpha': alpha,
                   'operator': 'LESFilter (Gaussian + sharp cutoff + avg-pool), qg._output.filter'},
        'sponge': {
            'bc_function': cfg['bc']['function'], 'width_frac': width,
            'x_strip_frac': sponge_x_frac, 'y_strip_frac': sponge_y_frac,
            'measured_x_cols': list(x_meas) if x_meas else None,
            'measured_y_rows': list(y_meas) if y_meas else None,
            'measured_thresh': thresh,
        },
        'mask': {'function': cfg['mask']['function'], 'r': r,
                 'config_x_center_IGNORED_by_code': cfg['mask']['x_center'],
                 'config_y_center_IGNORED_by_code': cfg['mask']['y_center']},
        'window': {'t_usable_lo': 30.0, 't_usable_hi': float(times_full[-1])},
        'seeds': {'run_seed': cfg.get('seed'), 'ic_seed': cfg['ic'].get('seed'),
                  'table_seed': tab_meta.get('seed')},
        'nan_check': {'pass': True, 'counts': nan_counts},
        'sha256': {'dataset': _sha256(out_npz), 'u_table': _sha256(out_table)},
    }

    md = f"""# DATASET_MANIFEST — {run_dir.name} (scale {scale})

Generated {meta['created']} by ml_closure/make_dataset_manifest.py. Zero-archaeology
handoff per BRANCH_LOG 2026-07-07 template. The fenced yaml block below is the
machine-readable source of truth for dataset_piff.py — edit NOTHING by hand.

## Files
- `{out_npz.name}` — canonical training product (omega_bar, pi_ff, ubar, vbar, times,
  U_snap, Re_snap, chi_obs_bar, chi_sponge_bar, meta). f32 fields, {T} frames {Ny}x{Nx}.
- `{out_table.name}` — inlet table copy (t, Re, U at solver dt={dt}).

## Conventions (as implemented — code refs)
- Pi_FF sign: `filter[J+Brinkman+Sponge](FR) - [J+Brinkman+Sponge](LES)`,
  spatial_closure/compute_pi_ff.py line 181 (Eq. 8, arXiv 2508.06678). NOT recomputed here.
- Velocity: u = -dpsi/dy, v = +dpsi/dx with psi = inv_laplacian(omega_bar)
  (qg/solver/opt/basis.py puv); u mean mode set to U(t_n), v mean 0
  (qg/_input/sources/bc.py Flow.const_x_flow). Computed once at build.
- times[i] = time of frame i; frame 0 is the IC at t=0 (the 445-frames-vs-444-times
  packaging off-by-one is fixed HERE, once).
- U(t_n) for normalization comes from `{out_table.name}` at step n = round(t/dt) —
  never from per-snapshot field statistics (spec S1.2).
- CAVEAT: mask.circular IGNORES config x_center/y_center; the cylinder sits at the
  domain center (x_c, y_c) = ({x_c:.6f}, {y_c:.6f}); D = {D:.6f}.
- Sponge (bc `{cfg['bc']['function']}`, width {width}): right-outlet x-strip
  x/Lx in [{sponge_x_frac[0]}, 1] and top y-strip y/Ly in [{sponge_y_frac[0]}, 1];
  measured filtered support (thresh {thresh}): x cols {x_meas}, y rows {y_meas} of {Nx}.
- Usable window t >= 30 (spec S1.2); record ends t = {times_full[-1]:.2f}.

{MACHINE_BLOCK_BEGIN}
```yaml
{yaml.safe_dump(machine, sort_keys=False)}```
"""
    with open(out_manifest, 'w') as f:
        f.write(md)
    print(f"[manifest] wrote {out_manifest}")
    print(f"[manifest] DONE: dataset {out_npz.name} sha256 {machine['sha256']['dataset'][:16]}..., "
          f"{T} frames, U range {machine['physics']['U_range']}, Re range {machine['physics']['Re_range']}")


if __name__ == '__main__':
    main()
