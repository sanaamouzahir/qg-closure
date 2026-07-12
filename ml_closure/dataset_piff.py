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
        if not npz_path.exists():
            raise FileNotFoundError(f"{npz_path} missing — Step 0 incomplete")
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

        # ---- static masks -------------------------------------------------- #
        sdf_cache = self.run_dir / f'SDF_s{self.scale}.npy'
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
        self.valid = (~strip) & (self.sdf >= 0.0)          # (Ny, Nx) bool
        self.n_valid = int(self.valid.sum())
        if self.n_valid == 0:
            raise ValueError(f"{self.name}: empty valid mask")

    def frames_in(self, t_lo, t_hi):
        """Frame indices with t in [t_lo, t_hi). Times read from the record —
        no uniform-t assumption."""
        return np.where((self.times >= t_lo) & (self.times < t_hi))[0]

    def crop(self, frame, cy, cx, size):
        """Periodic (wrap) crop centered at (cy, cx). Returns x(4,s,s), y(s,s),
        mask(s,s), zeta — all float32 torch tensors except mask (bool)."""
        iy = (cy - size // 2 + np.arange(size)) % self.Ny
        ix = (cx - size // 2 + np.arange(size)) % self.Nx
        sl = np.ix_(iy, ix)
        U = self.U_snap[frame]
        om, u, v, pi = normalize_fields(
            self.omega[frame][sl].astype(np.float64), self.u[frame][sl].astype(np.float64),
            self.v[frame][sl].astype(np.float64), self.pi[frame][sl].astype(np.float64),
            U, self.D)
        x = np.stack([om, u, v, self.sdf_star[sl]]).astype(np.float32)
        return (torch.from_numpy(x),
                torch.from_numpy(pi.astype(np.float32)),
                torch.from_numpy(self.valid[sl]),
                torch.tensor(self.zeta_snap[frame], dtype=torch.float32))

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
        self.set_epoch(0)

    def set_epoch(self, epoch):
        # fixed split ids — never hash() (PYTHONHASHSEED randomization breaks T1)
        rng = np.random.default_rng([self.seed, {'train': 0, 'val': 1}[self.split], int(epoch)])
        tbl = np.empty((self.n_crops, 4), dtype=np.int64)
        pick = rng.integers(0, len(self.frames), size=self.n_crops)
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
        self.table = tbl

    def __len__(self):
        return self.n_crops

    def __getitem__(self, i):
        ri, fi, cy, cx = self.table[i]
        r = self.runs[ri]
        x, y, m, zeta = r.crop(fi, cy, cx, self.size)
        return {'x': x, 'y': y, 'mask': m, 'zeta': zeta,
                'run': int(ri), 'frame': int(fi)}

    def epoch_hash(self):
        """Regression hash of the full epoch's tensors (T1)."""
        h = hashlib.sha256()
        h.update(self.table.tobytes())
        for i in range(len(self)):
            s = self[i]
            for k in ('x', 'y', 'mask', 'zeta'):
                h.update(s[k].numpy().tobytes())
        return h.hexdigest()


def describe(runs, conf, seed):
    """Loggable dataset summary (seed logged in every artifact — spec S1.8)."""
    return {
        'seed': int(seed),
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
