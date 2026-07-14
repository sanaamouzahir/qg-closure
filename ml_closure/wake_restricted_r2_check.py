"""
wake_restricted_r2_check.py — STANDALONE garbage check (2026-07-13 night order).

Context: this morning a "good" global R2 on the full-target model turned out to
be hollow — the skill was carried by the obstacle-ring band of Pi (the
body-force/penalty commutator), not by wake physics. Standing rule since then:
always report R2 restricted to pixels FARTHER than 1*D from the body
(SDF > 1*D), where the ring cannot contribute, alongside the global number.

This script recomputes full-frame predictions on ALL validation frames of a
checkpoint (reusing eval_piff.predict_frame — identical math, conditioning
flags travel with the ckpt) and reports pointwise R2 on:
  - the full valid mask (same population as eval_piff / training val R2), and
  - the wake-restricted mask: valid AND sdf > sdf_mult * D  (default 1.0),
per ensemble member and pooled. float64 accumulators.

New standalone file — imports the frozen production modules, edits nothing.

Usage (via piff_tool_job.sh, GPU):
  python wake_restricted_r2_check.py --ckpt runs_piff/<run>/<frozen>.pt \
      --config conf_piff_<x>.yaml --out runs_piff/<run>/eval_snapshot_20260713/wake_restricted_r2.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from dataset_piff import load_conf, build_runs, split_frames
from model_piff import PiffModel
from eval_piff import predict_frame

HERE = Path(__file__).resolve().parent


def full_frame_slice(run):
    """Reproduce dataset_piff.RunData.full_frame's periodic crop indices
    (crop(frame, Ny//2, Nx//2, max(Ny, Nx))) so per-run 2D masks line up
    pixel-for-pixel with predict_frame outputs."""
    size = max(run.Ny, run.Nx)
    iy = (run.Ny // 2 - size // 2 + np.arange(size)) % run.Ny
    ix = (run.Nx // 2 - size // 2 + np.arange(size)) % run.Nx
    return np.ix_(iy, ix)


def r2_of(acc):
    n, sy, sy2, sse = acc
    if n == 0:
        return None
    sst = sy2 - sy * sy / n
    return float(1.0 - sse / max(sst, 1e-30))


def main():
    ap = argparse.ArgumentParser(description="global vs wake-restricted (SDF > mult*D) val R2")
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', default=str(HERE / 'conf_piff.yaml'))
    ap.add_argument('--out', default=None, help='output yaml (default: <ckpt dir>/wake_restricted_r2.yaml)')
    ap.add_argument('--sdf-mult', type=float, default=1.0)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    conf = load_conf(args.config)
    device = args.device or conf['train']['device']
    gp_chunk = int(conf['train']['gp_chunk'])

    model = PiffModel(ckpt['conf']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # same ckpt-driven plumbing as eval_piff (variant + conditioning + tshed)
    conf.setdefault('model', {})['use_grad_feature'] = model.use_grad_feature
    ck_var = ckpt['conf'].get('data', {}).get('variant')
    if ck_var and not conf['data'].get('variant'):
        conf['data']['variant'] = ck_var
    conf['zeta']['tshed_smooth'] = ckpt['conf'].get('zeta', {}).get('tshed_smooth', 2.992)

    runs = build_runs(conf)
    frames = split_frames(runs, 'val', conf)
    print(f"[wakeR2] ckpt {args.ckpt} epoch {int(ckpt['epoch'])}; "
          f"{len(frames)} val frames from {[r.name for r in runs]}")

    wake2d = {}
    for r in runs:
        sl = full_frame_slice(r)
        wake2d[r.name] = r.sdf[sl] > args.sdf_mult * r.D

    acc = {}   # (member, tag) -> [n, sum_y, sum_y2, sse]  float64
    for k, (ri, fi) in enumerate(frames):
        r = runs[ri]
        p = predict_frame(model, r, fi, device, gp_chunk)
        m = p['mask']
        for tag, mm in (('global_valid', m), ('wake_restricted', m & wake2d[r.name])):
            y = p['truth'][mm].astype(np.float64)
            mu = p['mu2d'][mm].astype(np.float64)
            a = acc.setdefault((r.name, tag), [0, 0.0, 0.0, 0.0])
            a[0] += y.size
            a[1] += y.sum()
            a[2] += (y ** 2).sum()
            a[3] += ((y - mu) ** 2).sum()
        if (k + 1) % 50 == 0:
            print(f"[wakeR2] {k + 1}/{len(frames)} frames")

    members = [r.name for r in runs]
    out = {
        'ckpt': str(Path(args.ckpt).resolve()),
        'epoch': int(ckpt['epoch']),
        'sdf_mult': float(args.sdf_mult),
        'D_per_member': {r.name: float(r.D) for r in runs},
        'n_val_frames': len(frames),
        'definition': ("global_valid = eval_piff population (valid mask: outside body, "
                       "outside sponge). wake_restricted = global_valid AND "
                       f"SDF > {args.sdf_mult}*D (obstacle-ring band excluded; the standing "
                       "rule after the 2026-07-13 morning ring-R2 incident)."),
        'per_member': {}, 'pooled': {},
    }
    for name in members:
        out['per_member'][name] = {}
        for tag in ('global_valid', 'wake_restricted'):
            a = acc.get((name, tag), [0, 0.0, 0.0, 0.0])
            out['per_member'][name][tag] = {'r2': r2_of(a), 'n_pixels': int(a[0])}
    for tag in ('global_valid', 'wake_restricted'):
        tot = [0, 0.0, 0.0, 0.0]
        for name in members:
            a = acc.get((name, tag), [0, 0.0, 0.0, 0.0])
            for i in range(4):
                tot[i] += a[i]
        out['pooled'][tag] = {'r2': r2_of(tot), 'n_pixels': int(tot[0])}

    outp = Path(args.out or (Path(args.ckpt).parent / 'wake_restricted_r2.yaml'))
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, 'w') as f:
        yaml.safe_dump(out, f, sort_keys=False)
    print(yaml.safe_dump({'pooled': out['pooled']}, sort_keys=False))
    print(f"[wakeR2] wrote {outp}")


if __name__ == '__main__':
    main()
