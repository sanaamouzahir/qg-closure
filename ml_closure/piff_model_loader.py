"""
One entry point that returns EITHER a plain PiffModel or a BlendedPiffModel
(two-band SGS closure, Sanaa GO 2026-07-20), so every diagnostic becomes
blend-aware with a one-line change at its model-construction site.

CONTRACT -- the plain path must stay byte-identical. The diagnostics all do:

    ck    = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model = PiffModel(ck['conf']).to(device)          # <- the ONE line patched
    model.load_state_dict(ck['model'])
    model.eval()

`load_piff_model(ck, device)` replaces only the middle line. For a NORMAL
checkpoint it returns exactly `PiffModel(ck['conf']).to(device)` -- unloaded,
so the caller's own load_state_dict/eval lines keep their meaning and the
behaviour is identical to before.

For a BLENDED handle the two specialists are loaded here from their own
checkpoints (they cannot come from one state dict: different shapes, different
recorded scales), and the returned model's load_state_dict accepts the handle's
EMPTY 'model' dict as a no-op -- which is why the caller's next line still
works untouched.

THE HANDLE. write_blended_manifest.py writes, side by side:
  blended_manifest.yaml  the human/spec artifact (near_ckpt, far_ckpt,
                         overlap_lo, overlap_hi, geometry, ...)
  blended.pt             the same content inside a torch-loadable dict
                         {'conf': <eval conf>, 'model': {}, 'blended': {...},
                          'epoch': -1, 'seed': ...}
so that `torch.load(--ckpt)` in every diagnostic works unchanged and
`ck['conf']` still drives the variant / tshed_smooth plumbing downstream.
Point --ckpt at blended.pt; the yaml is the source of truth it was built from.

`load_piff_model` also accepts a PATH (str/Path) directly -- a .pt (plain or
blended) or a blended_manifest.yaml -- for callers that are not following the
torch.load-first pattern.
"""

from pathlib import Path

import torch
import yaml

from model_piff import PiffModel

MANIFEST_NAME = 'blended_manifest.yaml'
HANDLE_NAME = 'blended.pt'


def _resolve(base, p):
    p = Path(p)
    return p if p.is_absolute() else (Path(base) / p).resolve()


def is_blended(obj):
    """True for a loaded blended handle dict or a manifest-shaped dict."""
    return isinstance(obj, dict) and ('blended' in obj or 'near_ckpt' in obj)


def is_multitask(ck):
    """True for a plain checkpoint dict whose conf turns on the coregionalized
    2-task head (Sanaa GO 2026-07-21). A multitask ckpt is an ORDINARY
    single-state-dict PiffModel checkpoint -- it is NOT a blended handle -- so
    it already flows through the plain-path `PiffModel(ck['conf'])` construction
    below unchanged; this predicate only lets the loader LOG the head so a
    diagnostic's stdout names what it built."""
    return (isinstance(ck, dict) and not is_blended(ck)
            and bool(ck.get('conf', {}).get('model', {}).get('multitask', False)))


def load_blended_manifest(path):
    """Read blended_manifest.yaml (or a blended.pt handle) -> manifest dict
    with near_ckpt/far_ckpt resolved to absolute paths."""
    path = Path(path)
    if path.suffix == '.pt':
        man = torch.load(path, map_location='cpu', weights_only=False)['blended']
    else:
        man = yaml.safe_load(path.read_text())
    man = dict(man)
    for k in ('near_ckpt', 'far_ckpt'):
        if k not in man:
            raise ValueError(f"{path}: blended manifest missing {k!r}")
        man[k] = str(_resolve(path.parent, man[k]))
    return man


def build_from_manifest(man, device='cpu'):
    from model_piff_blended import build_blended
    # REQUIRED manifest keys (G4 MINOR, 2026-07-20): the blend geometry is
    # never silently defaulted — a manifest that forgot overlap_hi would
    # otherwise hand over at a distance nobody chose.
    for k in ('overlap_lo', 'overlap_hi', 'sdf_clip_D'):
        if man.get(k) is None:
            raise ValueError(
                f"blended manifest missing REQUIRED key {k!r} (have "
                f"{sorted(man)}) — the blend geometry must be explicit, never "
                f"defaulted; regenerate with write_blended_manifest.py")
    near_ck = torch.load(man['near_ckpt'], map_location='cpu', weights_only=False)
    far_ck = torch.load(man['far_ckpt'], map_location='cpu', weights_only=False)
    # sdf_clip_D governs the exactness of the sdf recovered from input channel
    # 3. The manifest must agree with BOTH trained confs, else the recovered
    # s = sdf_star * sdf_clip_D is scaled by the wrong constant.
    clips = {float(ck['conf']['data']['sdf_clip_D']) for ck in (near_ck, far_ck)}
    if len(clips) != 1:
        raise ValueError(f"near/far trained with different data.sdf_clip_D "
                         f"{sorted(clips)} — the blend weight would be derived "
                         f"from inconsistent sdf_star channels")
    clip_trained = clips.pop()
    if float(man['sdf_clip_D']) != clip_trained:
        raise ValueError(
            f"blended manifest sdf_clip_D={float(man['sdf_clip_D'])} != the "
            f"value BOTH specialists were TRAINED with ({clip_trained}) — "
            f"channel 3 is clip-normalized, so s = sdf_star * sdf_clip_D would "
            f"be wrong by {float(man['sdf_clip_D']) / clip_trained:g}x")
    m = build_blended(near_ck, far_ck,
                      overlap_lo=float(man['overlap_lo']),
                      overlap_hi=float(man['overlap_hi']),
                      sdf_clip_D=clip_trained,
                      geometry=man.get('geometry'), device=device)
    print(f"[loader] BLENDED two-band model: near={man['near_ckpt']} "
          f"far={man['far_ckpt']} overlap=[{m.overlap_lo}, {m.overlap_hi}] D",
          flush=True)
    print(f"[loader] REQUIRED eval config convention: NO data.band, "
          f"use_wall_gate false (i.e. conf_piff_<g>_gjs_lap.yaml) — the blend "
          f"is scored on the whole field and applies any near-side gate itself",
          flush=True)
    return m


def check_eval_conf(model, conf):
    """Hard-fail unless the EVAL config's data.sdf_clip_D matches the value the
    blended model recovers the pixel sdf with (G4 MAJOR 2, 2026-07-20).

    WHY THIS IS NOT REDUNDANT with the manifest/trained-conf check: channel 3
    is built by the DATASET at eval time (dataset_piff.RunData:
    sdf_star = clip(sdf, +/-clipD)/clipD, clipD = data.sdf_clip_D * D), so the
    EVAL conf — not the trained conf — decides its normalization. If an eval
    config carried a different sdf_clip_D, s = sdf_star * model.sdf_clip_D is
    silently wrong and the partition of unity hands over at the wrong physical
    distance, with no error anywhere. Exact compare: this is a config constant,
    not a measurement. No-op for plain PiffModels."""
    if conf is None or not hasattr(model, 'sdf_clip_D'):
        return
    try:
        eval_clip = float(conf['data']['sdf_clip_D'])
    except (KeyError, TypeError) as e:
        raise ValueError(f"blended model needs data.sdf_clip_D in the EVAL "
                         f"config to validate the blend geometry ({e})")
    if eval_clip != model.sdf_clip_D:
        raise ValueError(
            f"EVAL CONFIG MISMATCH: data.sdf_clip_D={eval_clip} in the eval "
            f"config, but the blended model recovers the pixel sdf assuming "
            f"sdf_clip_D={model.sdf_clip_D} (the value both specialists were "
            f"trained with). Channel 3 is clip-normalized, so the blend weight "
            f"would hand over at {eval_clip / model.sdf_clip_D:g}x the intended "
            f"distance — i.e. the overlap [{model.overlap_lo}, "
            f"{model.overlap_hi}] D would really be "
            f"[{model.overlap_lo * eval_clip / model.sdf_clip_D:g}, "
            f"{model.overlap_hi * eval_clip / model.sdf_clip_D:g}] D. "
            f"Use the config the specialists were trained with.")
    print(f"[loader] eval-config sdf_clip_D={eval_clip} matches the blend "
          f"geometry (overlap [{model.overlap_lo}, {model.overlap_hi}] D)",
          flush=True)


def load_piff_model(ck, device='cpu', conf=None):
    """Plain checkpoint dict -> an UNLOADED PiffModel on `device` (exactly the
    line it replaces; `conf` is IGNORED on this path, so it stays byte-
    identical). Blended handle / manifest / path -> a ready, loaded, eval-mode
    BlendedPiffModel, with the eval config's data.sdf_clip_D validated against
    the blend geometry when `conf` is supplied. See the module docstring."""
    if isinstance(ck, (str, Path)):
        p = Path(ck)
        if p.suffix in ('.yaml', '.yml') or p.name == MANIFEST_NAME:
            m = build_from_manifest(load_blended_manifest(p), device=device)
            check_eval_conf(m, conf)
            return m
        ck = torch.load(p, map_location='cpu', weights_only=False)
    if not isinstance(ck, dict):
        raise TypeError(f"load_piff_model: expected a checkpoint dict or path, "
                        f"got {type(ck).__name__}")
    if is_blended(ck):
        man = ck['blended'] if 'blended' in ck else ck
        # resolve relative sub-ckpt paths against the manifest dir recorded at
        # write time (the handle may be read from anywhere)
        base = man.get('manifest_dir', '.')
        man = dict(man)
        for k in ('near_ckpt', 'far_ckpt'):
            man[k] = str(_resolve(base, man[k]))
        m = build_from_manifest(man, device=device)
        check_eval_conf(m, conf)
        return m
    if is_multitask(ck):
        # coregionalized 2-task head: still an ORDINARY plain PiffModel ckpt, so
        # the construction is identical to the plain return below (the caller
        # then load_state_dict/eval() as usual). Branch exists only to name the
        # head on stdout; plain non-multitask ckpts skip it and stay byte-
        # identical, blended handles were dispatched above.
        print("[loader] MULTITASK (coregionalized 2-task) PiffModel: task 0 "
              "near / task 1 far, per-pixel task from input channel 3; scored "
              "on the whole field", flush=True)
        return PiffModel(ck['conf']).to(device)
    return PiffModel(ck['conf']).to(device)
