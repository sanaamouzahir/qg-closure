#!/usr/bin/env python
"""write_blended_manifest.py -- pair the two band specialists into one loadable
blended model (two-band SGS closure, Sanaa GO 2026-07-20).

Writes, into --outdir:
  blended_manifest.yaml   the spec artifact: near_ckpt, far_ckpt, overlap_lo,
                          overlap_hi, geometry (+ provenance)
  blended.pt              a torch-loadable handle carrying the same manifest,
                          so every diagnostic's `torch.load(--ckpt)` line works
                          unchanged and `ck['conf']` still drives the variant /
                          tshed_smooth plumbing (see piff_model_loader).

Validates before writing: both ckpts exist and load, their trained confs agree
on the fields that must match (variant, sdf_clip_D, runs, the conditioning
flags), their data.band blocks are the expected near/far pair, and the
overlap fits inside sdf_clip_D. Then constructs the BlendedPiffModel once as a
smoke test. Refuses to overwrite unless --force.

Usage:
  python write_blended_manifest.py --near runs_piff/piff_fpc_gjs_bandnear/best.pt \\
      --far runs_piff/piff_fpc_gjs_bandfar/best.pt --geometry fpc \\
      --outdir runs_piff/piff_fpc_gjs_blended [--overlap-lo 1.0 --overlap-hi 1.5]
"""
import argparse
import json
import socket
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from model_piff_blended import BlendedPiffModel
from model_piff import PiffModel

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--near', required=True, help='near specialist ckpt (best.pt)')
    ap.add_argument('--far', required=True, help='far specialist ckpt (best.pt)')
    ap.add_argument('--geometry', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--overlap-lo', type=float, default=1.0)
    ap.add_argument('--overlap-hi', type=float, default=1.5)
    ap.add_argument('--eval-config', default=None,
                    help='config the blended diagnostics must use (NO data.band, '
                         'use_wall_gate false); recorded in the manifest')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    near_p, far_p = Path(args.near).resolve(), Path(args.far).resolve()
    for p in (near_p, far_p):
        if not p.exists():
            raise SystemExit(f"MISSING specialist checkpoint: {p}")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    man_p, handle_p = outdir / 'blended_manifest.yaml', outdir / 'blended.pt'
    if (man_p.exists() or handle_p.exists()) and not args.force:
        raise SystemExit(f"EXISTS: {man_p} / {handle_p} — refusing to clobber "
                         f"(pass --force to replace)")

    near_ck = torch.load(near_p, map_location='cpu', weights_only=False)
    far_ck = torch.load(far_p, map_location='cpu', weights_only=False)

    # ---- validation ------------------------------------------------------- #
    nd, fd = near_ck['conf']['data'], far_ck['conf']['data']
    for k in ('variant', 'sdf_clip_D', 'runs', 'scale', 'upstream_mask_x_lo_D'):
        if nd.get(k) != fd.get(k):
            raise SystemExit(f"near/far trained with different data.{k}: "
                             f"{nd.get(k)!r} vs {fd.get(k)!r}")
    nm, fm = near_ck['conf']['model'], far_ck['conf']['model']
    for k in ('use_zeta_dot', 'use_grad_feature', 'use_lap_feature',
              'noise_prior', 'likelihood'):
        if nm.get(k) != fm.get(k):
            raise SystemExit(f"near/far trained with different model.{k}: "
                             f"{nm.get(k)!r} vs {fm.get(k)!r} — the two "
                             f"specialists must consume the same batch tensors")
    nb, fb = nd.get('band'), fd.get('band')
    if not nb or not fb:
        raise SystemExit(f"both ckpts must be BAND specialists "
                         f"(near band={nb}, far band={fb})")
    if nb.get('name') != 'near' or fb.get('name') != 'far':
        raise SystemExit(f"band names must be near/far (got {nb.get('name')!r}"
                         f" / {fb.get('name')!r}) — did you swap --near/--far?")
    clip = float(nd['sdf_clip_D'])
    if clip < args.overlap_hi:
        raise SystemExit(f"data.sdf_clip_D {clip} < overlap_hi {args.overlap_hi}: "
                         f"the sdf recovered from input channel 3 saturates "
                         f"inside the blend region")
    # the overlap must actually be covered by BOTH bands' training masks
    if nb.get('sdf_hi') is not None and float(nb['sdf_hi']) < args.overlap_hi:
        raise SystemExit(f"near band ends at {nb['sdf_hi']} D but the blend "
                         f"asks it to contribute out to {args.overlap_hi} D")
    if fb.get('sdf_lo') is not None and float(fb['sdf_lo']) > args.overlap_lo:
        raise SystemExit(f"far band starts at {fb['sdf_lo']} D but the blend "
                         f"asks it to contribute from {args.overlap_lo} D")

    # ---- smoke test: the pair must actually assemble ---------------------- #
    near = PiffModel(near_ck['conf']); near.load_state_dict(near_ck['model'])
    far = PiffModel(far_ck['conf']); far.load_state_dict(far_ck['model'])
    model = BlendedPiffModel(near, far, overlap_lo=args.overlap_lo,
                             overlap_hi=args.overlap_hi, sdf_clip_D=clip,
                             geometry=args.geometry)
    model.eval()

    man = {
        'kind': 'blended_two_band',
        'geometry': args.geometry,
        'near_ckpt': str(near_p),
        'far_ckpt': str(far_p),
        'overlap_lo': float(args.overlap_lo),
        'overlap_hi': float(args.overlap_hi),
        'sdf_clip_D': clip,
        'manifest_dir': str(outdir.resolve()),
        'eval_config': args.eval_config,
        'near_band': nb, 'far_band': fb,
        'near_epoch': int(near_ck.get('epoch', -1)),
        'far_epoch': int(far_ck.get('epoch', -1)),
        'near_val': near_ck.get('val'), 'far_val': far_ck.get('val'),
        'blend': model.describe(),
        'written_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'written_host': socket.gethostname(),
    }
    man_p.write_text(yaml.safe_dump(man, sort_keys=False))

    # the eval conf carried by the handle: the FAR specialist's trained conf
    # with the band removed (the blend is scored on the WHOLE field) — this is
    # what ck['conf'] feeds downstream (variant, tshed_smooth, runs)
    eval_conf = yaml.safe_load(yaml.safe_dump(far_ck['conf']))
    eval_conf['data'].pop('band', None)
    eval_conf.setdefault('model', {})['use_wall_gate'] = False
    torch.save({'conf': eval_conf, 'model': {}, 'blended': man,
                'epoch': -1, 'seed': int(far_ck.get('seed', 0)),
                'val': None}, handle_p)

    print(f"[blend] wrote {man_p}")
    print(f"[blend] wrote {handle_p}  (point diagnostics --ckpt here)")
    print('[blend] ' + json.dumps(model.describe()))


if __name__ == '__main__':
    main()
