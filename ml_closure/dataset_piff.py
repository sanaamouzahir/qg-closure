"""
Manifest-driven Pi_FF crop loader (ML SPEC 01 S1).

Reads every path/shape/convention from <run_dir>/DATASET_MANIFEST.md (the fenced
yaml machine block written by make_dataset_manifest.py) and hard-fails loudly on
any mismatch — shapes are NEVER hard-coded. Snapshot times come from the dataset
`times` array (approved default 3: never assume uniform t). U(t_n) comes from
U_of_t.npz at step n = round(t/dt) — never from per-snapshot field statistics;
NO per-sample standardization anywhere.

Sample = 64x64 crop (config), channels [omega*, u*, v*, SDF*], target Pi*,
conditioning zeta = (Re(t_n) - 3900)/1700, loss mask excluding sponge + body.
Wake-biased sampling 80/20 (config). Time split train [30,100) / val [100,120];
LOMO split via data.lomo_holdout. Deterministic given (seed, epoch).

float32 throughout (spec S0 precision policy — intentional, differs from the
temporal-closure branch).
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset

MACHINE_BLOCK_BEGIN = '<!-- machine-block -->'


def load_conf(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _f(x):
    """Explicit float cast (repo YAML rule)."""
    return float(x)


def parse_manifest(run_dir):
    p = Path(run_dir) / 'DATASET_MANIFEST.md'
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing — run make_dataset_manifest.py on {run_dir} first (Step 0)")
    text = p.read_text()
    if MACHINE_BLOCK_BEGIN not in text:
        raise ValueError(f"{p}: no machine block marker {MACHINE_BLOCK_BEGIN!r}")
    block = text.split(MACHINE_BLOCK_BEGIN, 1)[1]
    try:
        block = block.split('```yaml', 1)[1].split('```', 1)[0]
    except IndexError:
        raise ValueError(f"{p}: malformed fenced yaml machine block")
    return yaml.safe_load(block)


def normalize_fields(omega, u, v, pi, U, D):
    """Spec S1.2/S1.3 nondimensionalization. U from the inlet table, never stats."""
    U = _f(U); D = _f(D)
    return omega * (D / U), u / U, v / U, pi * (D * D / (U * U))


def signed_distance(chi_obs, dx, dy):
    """SDF from the filtered obstacle mask: >0 outside, <0 inside, ~0 boundary.
    Physical units. Computed once per run, cached by RunData."""
    inside = chi_obs > 0.5
    if not inside.any():
        raise ValueError("obstacle mask empty at threshold 0.5 — wrong chi_obs_bar?")
    d_out = distance_transform_edt(~inside, sampling=(dy, dx))
    d_in = distance_transform_edt(inside, sampling=(dy, dx))
    return np.where(inside, -d_in, d_out).astype(np.float64)


class RunData:
    """One run's canonical arrays + static masks. Everything validated against
    the manifest at load; hard-fails with the mismatch spelled out."""

    def __init__(self, run_dir, conf):
        self.run_dir = Path(run_dir)
        self.name = self.run_dir.name
        dc = conf['data']
        self.scale = int(dc['scale'])
        man = parse_manifest(self.run_dir)
        self.man = man

        npz_path = self.run_dir / man['files']['dataset']
        if npz_path.name != f"DNS_LES_s{self.scale}.npz":
            raise ValueError(f"{self.name}: manifest dataset {npz_path.name} != scale {self.scale}")
        # data.variant (Sanaa 2026-07-13 filter ruling): '' = the manifest's
        # sharp-filter canonical; 'gaussian' = DNS_LES_s<scale>_gaussian.npz
        # (all-Gaussian rebuild, same folder, same keys/shapes — the manifest
        # stays the linkage/shape authority, only the payload file swaps).
        self.variant = str(dc.get('variant') or '')
        if self.variant:
            npz_path = self.run_dir / f"DNS_LES_s{self.scale}_{self.variant}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"{npz_path} missing — "
                                    f"{'Gaussian rebuild' if self.variant else 'Step 0'} incomplete")
        z = np.load(npz_path)

        for key in ('omega_bar', 'pi_ff', 'ubar', 'vbar'):
            want = tuple(man['shapes'][key])
            got = z[key].shape
            if got != want:
                raise ValueError(f"{self.name}: {key} shape {got} != manifest {want}")

        self.omega = z['omega_bar'][0]     # (T, Ny, Nx) f32
        self.pi = z['pi_ff'][0]
        self.u = z['ubar'][0]
        self.v = z['vbar'][0]
        self.times = np.asarray(z['times'], dtype=np.float64)
        self.U_snap = np.asarray(z['U_snap'], dtype=np.float64)
        self.Re_snap = np.asarray(z['Re_snap'], dtype=np.float64)
        self.chi_obs = z['chi_obs_bar'][0]
        self.chi_sponge = z['chi_sponge_bar'][0]
        self.T, self.Ny, self.Nx = self.omega.shape
        if len(self.times) != self.T:
            raise ValueError(f"{self.name}: times ({len(self.times)}) != frames ({self.T})")

        g = man['grid']
        if [self.Ny, self.Nx] != list(g['N_LES']):
            raise ValueError(f"{self.name}: grid {self.Ny}x{self.Nx} != manifest {g['N_LES']}")
        self.Lx, self.Ly = _f(g['Lx']), _f(g['Ly'])
        self.dx, self.dy = self.Lx / self.Nx, self.Ly / self.Ny
        ph = man['physics']
        self.D = _f(ph['D'])
        self.x_c, self.y_c = _f(ph['x_c']), _f(ph['y_c'])

        # U(t_n) cross-check against the table at step index (spec S1.2)
        tab = np.load(self.run_dir / man['files']['u_table'])
        t_tab = np.asarray(tab['t'], dtype=np.float64)
        dt_tab = float(t_tab[1] - t_tab[0])
        steps = np.rint(self.times / dt_tab).astype(np.int64)
        if steps.max() >= len(tab['U']):
            raise ValueError(f"{self.name}: snapshot beyond table end")
        U_from_tab = np.asarray(tab['U'], dtype=np.float64)[steps]
        if np.max(np.abs(U_from_tab - self.U_snap)) > 1e-9:
            raise ValueError(f"{self.name}: U_snap in npz disagrees with U_of_t.npz at step index")

        zc = conf['zeta']
        self.zeta_snap = ((self.Re_snap - _f(zc['re0'])) / _f(zc['re_scale'])).astype(np.float64)

        # zeta_dot = d(zeta)/dt from the table Re(t), boxcar-smoothed over
        # ~1 T_shed (ORDER 3c, 2026-07-13): the wake-state-lag coordinate.
        # O(N) running mean via cumsum (tables are ~1e6 rows), then centered
        # gradient at table resolution, sampled at the snapshot steps. The
        # ensemble-std normalization is a recorded MODEL buffer (zdot_sd),
        # not applied here — this stays a pure per-run physical quantity.
        tsm = _f(zc.get('tshed_smooth', 2.992))     # T_shed_mid, FPC ensemble
        Re_tab = np.asarray(tab['Re'], dtype=np.float64)
        w = max(1, int(round(tsm / dt_tab)))
        csum = np.cumsum(np.insert(Re_tab, 0, 0.0))
        idx = np.arange(len(Re_tab))
        lo = np.maximum(idx - w // 2, 0)
        hi = np.minimum(idx + (w - w // 2), len(Re_tab))
        Re_sm = (csum[hi] - csum[lo]) / (hi - lo)
        dRe_dt = np.gradient(Re_sm, dt_tab)
        self.zeta_dot_snap = (dRe_dt[steps] / _f(zc['re_scale'])).astype(np.float64)

        # |grad omega_bar| planes for the GP grad feature (ORDER 3c) — raw
        # physical magnitude, centered differences, periodic (the domain is;
        # obstacle/sponge pixels are excluded by the loss mask, not here).
        # Normalization D^2/U(t_n) applied at crop time with the frame's U.
        self.need_grad = bool(conf.get('model', {}).get('use_grad_feature', False))
        if self.need_grad:
            gm = np.empty_like(self.omega)
            for k in range(self.T):
                om = self.omega[k].astype(np.float64)
                gx = (np.roll(om, -1, axis=1) - np.roll(om, 1, axis=1)) / (2.0 * self.dx)
                gy = (np.roll(om, -1, axis=0) - np.roll(om, 1, axis=0)) / (2.0 * self.dy)
                gm[k] = np.sqrt(gx * gx + gy * gy).astype(np.float32)
            self.gradmag = gm

        # |laplacian omega_bar| planes for the GP lap feature — raw physical
        # magnitude, centered 5-point stencil, periodic (mirrors gradmag; same
        # mask/exclusion logic). Normalization D^3/U(t_n) applied at crop time
        # with the frame's U (one extra D vs the grad feature's D^2/U: the
        # laplacian of omega* in x/D coordinates).
        self.need_lap = bool(conf.get('model', {}).get('use_lap_feature', False))
        if self.need_lap:
            lm = np.empty_like(self.omega)
            for k in range(self.T):
                om = self.omega[k].astype(np.float64)
                lx = (np.roll(om, -1, axis=1) + np.roll(om, 1, axis=1)
                      - 2.0 * om) / (self.dx * self.dx)
                ly = (np.roll(om, -1, axis=0) + np.roll(om, 1, axis=0)
                      - 2.0 * om) / (self.dy * self.dy)
                lm[k] = np.abs(lx + ly).astype(np.float32)
            self.lapmag = lm

        # ---- static masks -------------------------------------------------- #
        # SDF cache keyed by variant too: the Gaussian chi_obs_bar is smoother
        # than the sharp one, so their distance maps differ
        vtag = f'_{self.variant}' if self.variant else ''
        sdf_cache = self.run_dir / f'SDF_s{self.scale}{vtag}.npy'
        if sdf_cache.exists():
            self.sdf = np.load(sdf_cache)
            if self.sdf.shape != (self.Ny, self.Nx):
                raise ValueError(f"{sdf_cache}: stale shape {self.sdf.shape}")
        else:
            self.sdf = signed_distance(self.chi_obs, self.dx, self.dy)
            np.save(sdf_cache, self.sdf)

        clipD = _f(dc['sdf_clip_D']) * self.D
        self.sdf_star = (np.clip(self.sdf, -clipD, clipD) / clipD).astype(np.float32)

        # sponge exclusion: manifest analytic strips UNION filtered-mask support
        sxf = [_f(v) for v in man['sponge']['x_strip_frac']]
        syf = [_f(v) for v in man['sponge']['y_strip_frac']]
        xf = (np.arange(self.Nx) + 0.5) / self.Nx
        yf = (np.arange(self.Ny) + 0.5) / self.Ny
        strip = (xf[None, :] >= sxf[0]) | (yf[:, None] >= syf[0])
        strip |= self.chi_sponge > _f(dc['sponge_thresh'])
        # upstream exclusion (Sanaa 2026-07-13 night): the inlet sponge was
        # undersized, so upstream of the obstacle carries reflection + spurious
        # v — mask pixels with x < x_c + upstream_mask_x_lo_D * D from loss and
        # metrics (true Pi ~ 0 there anyway). Default None = no change; the
        # boundary layer/incident window survive (e.g. -1.5 keeps ~1 D ahead
        # of the leading edge). Loss-side only — inputs never zeroed (S1.5).
        up_lo = dc.get('upstream_mask_x_lo_D')
        self.upstream_blend = None
        if up_lo is not None:
            xs = (np.arange(self.Nx) + 0.5) * self.dx
            x_edge = self.x_c + _f(up_lo) * self.D
            strip |= (xs[None, :] < x_edge)
            # INPUT REPAIR (Sanaa 2026-07-13 night follow-up: "does the model
            # see those inputs with the fake vorticity?"): upstream of the
            # same line, replace inputs with the KNOWN freestream (omega*=0,
            # u*=1, v*=0) via an 8-px linear blend — the model's receptive
            # field then sees clean truth, never the reflection artifact.
            # w = 1 -> real data, 0 -> analytic freestream; per-COLUMN weights.
            bw = 8.0 * self.dx
            self.upstream_blend = np.clip(
                (xs - (x_edge - bw)) / bw, 0.0, 1.0).astype(np.float64)
        self.valid = (~strip) & (self.sdf >= 0.0)          # (Ny, Nx) bool
        self.n_valid = int(self.valid.sum())
        if self.n_valid == 0:
            raise ValueError(f"{self.name}: empty valid mask")

    def signal_block_weights(self, block):
        """Per-frame block-sampling weights ∝ sum of Pi^2 over (block x block)
        tiles of VALID pixels (Sanaa 2026-07-13: Jacobian-only targets carry
        90% of their energy in ~0.07% of pixels — geometric wake boxes no
        longer guarantee signal in a batch; sample where the energy actually
        is). Computed once, cached on the instance; float64 sums, tiny table
        (T x Ny/b x Nx/b)."""
        if getattr(self, '_sbw', None) is not None and self._sbw_block == block:
            return self._sbw
        b = int(block)
        Ny, Nx = self.Ny - self.Ny % b, self.Nx - self.Nx % b
        w = np.empty((self.T, Ny // b, Nx // b), dtype=np.float64)
        vm = self.valid[:Ny, :Nx].reshape(Ny // b, b, Nx // b, b)
        for t in range(self.T):
            p2 = (self.pi[t][:Ny, :Nx].astype(np.float64) ** 2)
            w[t] = (p2.reshape(Ny // b, b, Nx // b, b) * vm).sum(axis=(1, 3))
        s = w.sum(axis=(1, 2), keepdims=True)
        self._sbw = w / np.maximum(s, 1e-300)
        self._sbw_block = b
        return self._sbw

    def frames_in(self, t_lo, t_hi):
        """Frame indices with t in [t_lo, t_hi). Times read from the record —
        no uniform-t assumption."""
        return np.where((self.times >= t_lo) & (self.times < t_hi))[0]

    def crop(self, frame, cy, cx, size):
        """Periodic (wrap) crop centered at (cy, cx). Returns x(4,s,s), y(s,s),
        mask(s,s), zeta, zeta_dot, g(s,s)|None, lap(s,s)|None — float32 torch
        tensors except mask (bool). g = |grad omega_bar|* = |grad omega_bar| *
        D^2/U (the gradient of omega* in x/D coordinates), None unless
        model.use_grad_feature. lap = |lap omega_bar|* = |lap omega_bar| *
        D^3/U (the laplacian of omega* in x/D coordinates), None unless
        model.use_lap_feature."""
        iy = (cy - size // 2 + np.arange(size)) % self.Ny
        ix = (cx - size // 2 + np.arange(size)) % self.Nx
        sl = np.ix_(iy, ix)
        U = self.U_snap[frame]
        om, u, v, pi = normalize_fields(
            self.omega[frame][sl].astype(np.float64), self.u[frame][sl].astype(np.float64),
            self.v[frame][sl].astype(np.float64), self.pi[frame][sl].astype(np.float64),
            U, self.D)
        if self.upstream_blend is not None:
            # blend to analytic freestream upstream (see __init__ note):
            # omega* -> 0, u* -> 1 (u = U(t)), v* -> 0. Column weights w=1
            # keep real data; w=0 = pure freestream. sdf channel untouched.
            w = self.upstream_blend[ix][None, :]           # (1, s) per column
            om = om * w
            u = u * w + (1.0 - w)
            v = v * w
        x = np.stack([om, u, v, self.sdf_star[sl]]).astype(np.float32)
        g = None
        if self.need_grad:
            gs = self.gradmag[frame][sl].astype(np.float64) * (self.D * self.D / _f(U))
            if self.upstream_blend is not None:
                gs = gs * self.upstream_blend[ix][None, :]
            g = torch.from_numpy(gs.astype(np.float32))
        lap = None
        if self.need_lap:
            ls = self.lapmag[frame][sl].astype(np.float64) * (
                self.D * self.D * self.D / _f(U))
            if self.upstream_blend is not None:
                ls = ls * self.upstream_blend[ix][None, :]
            lap = torch.from_numpy(ls.astype(np.float32))
        return (torch.from_numpy(x),
                torch.from_numpy(pi.astype(np.float32)),
                torch.from_numpy(self.valid[sl]),
                torch.tensor(self.zeta_snap[frame], dtype=torch.float32),
                torch.tensor(self.zeta_dot_snap[frame], dtype=torch.float32),
                g, lap)

    def full_frame(self, frame):
        """Whole-domain channels/target/mask for a priori eval (model is
        convolutional + pointwise; crops are a training device only)."""
        return self.crop(frame, self.Ny // 2, self.Nx // 2, max(self.Ny, self.Nx))


def build_runs(conf):
    return [RunData(rd, conf) for rd in conf['data']['runs']]


def split_frames(runs, split, conf):
    """[(run_idx, frame_idx), ...] for a split. Time split per spec S1.7;
    LOMO hook: data.lomo_holdout = run basename -> that run is val (full usable
    window), the others are train (train window)."""
    dc = conf['data']
    lomo = dc.get('lomo_holdout')
    out = []
    for ri, r in enumerate(runs):
        if lomo:
            if (r.name == lomo) != (split == 'val'):
                continue
            lo, hi = ((_f(r.man['window']['t_usable_lo']), _f(dc['t_val_hi']) + 1e-9)
                      if split == 'val' else
                      (_f(dc['t_train_lo']), _f(dc['t_train_hi'])))
        elif split == 'train':
            lo, hi = _f(dc['t_train_lo']), _f(dc['t_train_hi'])
        else:
            lo, hi = _f(dc['t_val_lo']), _f(dc['t_val_hi']) + 1e-9
        out += [(ri, int(fi)) for fi in r.frames_in(lo, hi)]
    if not out:
        raise ValueError(f"split {split!r}: no frames (check windows / lomo_holdout)")
    return out


def count_masked_pixels(runs, split, conf):
    """ELBO data count N = TOTAL masked (valid) pixels over all frames of the
    split — counted once at build, logged in every artifact (plan S4)."""
    return int(sum(runs[ri].n_valid for ri, _ in split_frames(runs, split, conf)))


def target_stats(runs, split, conf):
    """EXACT mean/var of the normalized target over every valid pixel of the
    split (float64 accumulation, deterministic — no sampling). Feeds the
    data-informed GP hyperparameter INIT (Sanaa ruling 2026-07-12): this is an
    initialization constant, NOT a normalization of the data — the spec-S1.2
    target definition (pi * D^2/U^2) is untouched."""
    n, s1, s2 = 0, 0.0, 0.0
    for ri, fi in split_frames(runs, split, conf):
        r = runs[ri]
        U = _f(r.U_snap[fi])
        y = r.pi[fi][r.valid].astype(np.float64) * (r.D * r.D / (U * U))
        n += y.size
        s1 += float(y.sum())
        s2 += float((y * y).sum())
    mean = s1 / n
    var = max(s2 / n - mean * mean, 0.0)
    return {'n': int(n), 'mean': float(mean), 'var': float(var)}


def conditioning_stats(runs, split, conf):
    """Recorded normalization constants for the ORDER-3 conditioning inputs,
    exact over the split (float64): ensemble std of zeta_dot over frames
    (frame-level — zeta_dot is constant per frame) and mean of the normalized
    |grad omega_bar|* feature over every valid pixel. Mirrors target_stats:
    initialization/normalization constants, never per-sample statistics."""
    frames = split_frames(runs, split, conf)
    zd = np.array([runs[ri].zeta_dot_snap[fi] for ri, fi in frames], dtype=np.float64)
    out = {'zdot_sd': float(zd.std()), 'zdot_n_frames': int(zd.size)}
    if all(getattr(r, 'need_grad', False) for r in runs):
        n, s1, s2 = 0, 0.0, 0.0
        for ri, fi in frames:
            r = runs[ri]
            U = _f(r.U_snap[fi])
            gv = r.gradmag[fi][r.valid].astype(np.float64) * (r.D * r.D / U)
            n += gv.size
            s1 += float(gv.sum())
            s2 += float((gv * gv).sum())
        out['g_scale'] = s1 / n
        out['g2_scale'] = s2 / n          # s_feat normalization (structural prior)
        out['g_n_pixels'] = int(n)
    if all(getattr(r, 'need_lap', False) for r in runs):
        n, s1 = 0, 0.0
        for ri, fi in frames:
            r = runs[ri]
            U = _f(r.U_snap[fi])
            lv = r.lapmag[fi][r.valid].astype(np.float64) * (r.D * r.D * r.D / U)
            n += lv.size
            s1 += float(lv.sum())
        out['lap_scale'] = s1 / n
        out['lap_n_pixels'] = int(n)
    return out


class PiffCropDataset(Dataset):
    """Deterministic crop sampler. call set_epoch(ep) each epoch — the crop
    table is a pure function of (seed, split, epoch)."""

    def __init__(self, runs, split, conf, seed, n_crops=None):
        self.runs, self.split, self.conf = runs, split, conf
        self.seed = int(seed)
        dc = conf['data']
        self.size = int(dc['crop'])
        self.frames = split_frames(runs, split, conf)
        self.n_crops = int(n_crops if n_crops is not None
                           else dc[f'crops_per_epoch_{split}'])
        self.wake_frac = _f(dc['wake_frac'])
        self.wxlo, self.wxhi = _f(dc['wake_x_lo_D']), _f(dc['wake_x_hi_D'])
        self.wyh = _f(dc['wake_y_half_D'])
        # signal-biased sampling (2026-07-13): fraction of TRAIN crops whose
        # centers are drawn from the per-frame Pi^2 block distribution. 0.0
        # (default) = EXACT legacy path, bit-identical crop tables. TRAIN-ONLY
        # (G4 finding 2): val crops keep the legacy distribution so the val
        # curve / T6 gate stay comparable across runs. In the train three-way
        # split the wake share is wake_frac*(1-signal_frac); remainder uniform.
        self.signal_frac = _f(dc.get('signal_frac', 0.0)) if split == 'train' else 0.0
        self.signal_block = int(dc.get('signal_block', 8))
        self.set_epoch(0)

    def set_epoch(self, epoch):
        # fixed split ids — never hash() (PYTHONHASHSEED randomization breaks T1)
        rng = np.random.default_rng([self.seed, {'train': 0, 'val': 1}[self.split], int(epoch)])
        tbl = np.empty((self.n_crops, 4), dtype=np.int64)
        pick = rng.integers(0, len(self.frames), size=self.n_crops)
        if self.signal_frac <= 0.0:
            # EXACT legacy path (signal_frac 0): same rng call sequence,
            # bit-identical crop tables to pre-2026-07-13 code
            wake = rng.random(self.n_crops) < self.wake_frac
            for i in range(self.n_crops):
                ri, fi = self.frames[pick[i]]
                r = self.runs[ri]
                if wake[i]:
                    x = r.x_c + rng.uniform(self.wxlo, self.wxhi) * r.D
                    y = r.y_c + rng.uniform(-self.wyh, self.wyh) * r.D
                    cy = int(round(y / r.dy)) % r.Ny
                    cx = int(round(x / r.dx)) % r.Nx   # window may pass the outlet: wrap; loss mask handles sponge
                else:
                    cy = int(rng.integers(0, r.Ny))
                    cx = int(rng.integers(0, r.Nx))
                tbl[i] = (ri, fi, cy, cx)
        else:
            # three-way split: signal-biased (Pi^2 block distribution) /
            # geometric wake box / uniform. Wake share scaled by
            # (1-signal_frac) so conf wake_frac keeps its legacy meaning.
            wake_hi = self.signal_frac + self.wake_frac * (1.0 - self.signal_frac)
            u = rng.random(self.n_crops)
            for i in range(self.n_crops):
                ri, fi = self.frames[pick[i]]
                r = self.runs[ri]
                if u[i] < self.signal_frac:
                    b = self.signal_block
                    w = r.signal_block_weights(b)[fi]
                    wsum = float(w.sum())
                    if wsum <= 0.999:
                        # quiescent/zero-energy frame (G4 finding 1): the
                        # normalized row cannot feed rng.choice — uniform crop
                        cy = int(rng.integers(0, r.Ny))
                        cx = int(rng.integers(0, r.Nx))
                        tbl[i] = (ri, fi, cy, cx)
                        continue
                    flat = rng.choice(w.size, p=w.ravel())
                    by, bx = divmod(int(flat), w.shape[1])
                    cy = (by * b + int(rng.integers(0, b))) % r.Ny
                    cx = (bx * b + int(rng.integers(0, b))) % r.Nx
                elif u[i] < wake_hi:
                    x = r.x_c + rng.uniform(self.wxlo, self.wxhi) * r.D
                    y = r.y_c + rng.uniform(-self.wyh, self.wyh) * r.D
                    cy = int(round(y / r.dy)) % r.Ny
                    cx = int(round(x / r.dx)) % r.Nx
                else:
                    cy = int(rng.integers(0, r.Ny))
                    cx = int(rng.integers(0, r.Nx))
                tbl[i] = (ri, fi, cy, cx)
        self.table = tbl

    def __len__(self):
        return self.n_crops

    def __getitem__(self, i):
        ri, fi, cy, cx = self.table[i]
        r = self.runs[ri]
        x, y, m, zeta, zeta_dot, g, lap = r.crop(fi, cy, cx, self.size)
        out = {'x': x, 'y': y, 'mask': m, 'zeta': zeta, 'zeta_dot': zeta_dot,
               'run': int(ri), 'frame': int(fi)}
        if g is not None:
            out['g'] = g
        if lap is not None:
            out['lap'] = lap
        return out

    def epoch_hash(self):
        """Regression hash of the full epoch's tensors (T1)."""
        h = hashlib.sha256()
        h.update(self.table.tobytes())
        for i in range(len(self)):
            s = self[i]
            for k in ('x', 'y', 'mask', 'zeta', 'zeta_dot', 'g', 'lap'):
                if k in s:
                    h.update(s[k].numpy().tobytes())
        return h.hexdigest()


def describe(runs, conf, seed):
    """Loggable dataset summary (seed logged in every artifact — spec S1.8)."""
    return {
        'seed': int(seed),
        'variant': conf['data'].get('variant') or 'sharp-canonical',
        'runs': [r.name for r in runs],
        'n_train_frames': len(split_frames(runs, 'train', conf)),
        'n_val_frames': len(split_frames(runs, 'val', conf)),
        'N_train_pixels': count_masked_pixels(runs, 'train', conf),
        'N_val_pixels': count_masked_pixels(runs, 'val', conf),
        'valid_per_frame': {r.name: r.n_valid for r in runs},
        'lomo_holdout': conf['data'].get('lomo_holdout'),
    }


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description="loader smoke: build splits, print summary")
    ap.add_argument('--config', default=str(Path(__file__).parent / 'conf_piff.yaml'))
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    conf = load_conf(args.config)
    runs = build_runs(conf)
    print(json.dumps(describe(runs, conf, args.seed), indent=2))
    ds = PiffCropDataset(runs, 'train', conf, args.seed)
    s = ds[0]
    print({k: (tuple(v.shape) if torch.is_tensor(v) else v) for k, v in s.items()})
    print('epoch_hash', ds.epoch_hash()[:16])
