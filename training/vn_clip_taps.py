#!/usr/bin/env python
"""
vn_clip_taps.py -- Wiener OPTION 6: OFFLINE von Neumann symbol clipping of the
learned base stencil taps (Sanaa GO 2026-07-14).

Testbed: deriv7_cond_local_v2 -- a-priori accurate, rollout-BLOWS-UP 6/6 at
dT 1e-2 / 1.5e-2 (four-way verdict). Goal: minimally modify the model's
learned SPATIAL stencil taps, OFFLINE (no retraining), so the frozen-
coefficient von Neumann amplification of the closed AB2CN2 scheme satisfies
|G_eff| <= 1 - eps on all VALID shells (|dt*sigma| <= 0.5) at the failing dTs,
with a symbol-change leash for accuracy preservation.

Machinery reused (read wiener_certificate.py's header for the G_eff math):
  * wiener_certificate.assemble_geff / vn_penalty / tap_symbol -- the P1(b)
    certificate, differentiable through the taps.
  * train_deriv_rollout.RootCtx + geff_shell_ctx -- deep-root loading and the
    (kappa_sh, L_hat_sh) shell context at REL_SHELLS.
  * The (sig, delta_taps) plumbing mirrors train_deriv_rollout.
    geff_window_penalty EXACTLY (context_feats_from_spectral -> cond head ->
    (dT/dT_ref)^(S-k) amp, k=0 zeroed).

DESIGN CHOICES (binding, documented):
  1. FROZEN delta_taps: the cond-head tap deltas are computed ONCE per
     (root, stride, window) with the ORIGINAL model and kept FIXED. We clip
     only the BASE taps. (The cond head reads sigma-hat features of the data,
     not the taps, so this is exact, not an approximation; freezing makes the
     "no retraining of the conditioning" contract explicit.)
  2. CENTRAL-ROW taps feed the certificate. assemble_geff's own read is
     model.grad.wx.reshape(-1, W)[0] == wx[0, 0, 0, :] -- the TOP row of the
     (C, 1, W, W) kernel, which is ZERO at physics init (the FD taps live in
     the central row W//2; rho == 1 calibration only holds there, and the
     cond-head deltas modulate exactly the central row/column). We therefore
     pass a shim whose grad.wx is the 1-D central-row taps (+ the trainable
     clip delta), so assemble_geff's [0] picks up exactly those taps and the
     certificate measures the operator that actually acts on the fields.
     The row-0 reading is printed as a flagged diagnostic (certificate
     row-selection discrepancy -- report upstream, do not edit the file).
  3. PER-CHANNEL 1-D tap deltas d (C, W), channel ch applied to the central
     ROW of wx[ch] and (x/y ISOTROPICALLY TIED) the central COLUMN of wy[ch].
     wx/wy are separate Parameters in SpatialGrad (not tied), but the
     certificate evaluates the two directions only through their isotropic
     MEAN symbol (assemble_geff's .mean(dim=2)), so an untied x/y split is
     an unobservable dof -- we tie them. Per-channel freedom is REQUIRED:
     v1 of this tool used one channel-shared row and plateaued (vn 5.85e-3
     -> 5.58e-3 over 500 iters, max|G| 1.089 -> 1.089 at s3, job 1833782)
     because the excess enters through the channel-specific cond-delta
     symbols; a shared shift only cancels the pooled mean. The per-channel
     base deltas reach the certificate by FOLDING into the delta_taps
     argument (the channel-resolved differentiable path assemble_geff
     already provides) -- exact, because base and cond deltas add in the
     same central-row tap space and base taps are dt-independent.
     Channel-to-channel trained differences are preserved (additive).
  4. Leash: lam_sym * mean_theta |sym(taps) - sym(taps0)|^2 over a uniform
     theta grid on (0, 2*pi/3] (the dealiased band). By linearity of
     tap_symbol this equals mean_theta |sym(d)|^2 -- computed that way.

Loss:  sum_{(root,stride) groups} vn_penalty(assemble_geff(...), lam=1,
       eps=--eps, valid=|dt*sigma|<=0.5)  +  lam_sym * mean|sym(d)|^2,
Adam on d only. Everything float64, CPU (5 shells x few windows is tiny).

Output: --out-ckpt = deep copy of the original ckpt with grad.wx/grad.wy
clipped and ckpt['vn_clip'] metadata; verdict table (per (root, stride):
max|G| before -> after over valid shells) on stdout.

Usage (from training/):
    python vn_clip_taps.py \
        --ckpt data/ensemble_N5_7lag/training_runs/deriv7_cond_local_v2/best.pt \
        --deep-roots data/ensemble_N5/FRC-kf4/forced_turbulence_dT_5em3 \
                     data/ensemble_N5_7lag/FRC-256/forced_turbulence_dT_5em3 \
        --strides 2 3 --n-windows 4 --eps 0.02 --lam-sym 1.0 --iters 500 \
        --out-ckpt .../best_vnclip.pt --device cpu
"""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import wiener_certificate as wc
from train_deriv_rollout import RootCtx, geff_shell_ctx, DT_TAGS
from rollout_aposteriori import load_deriv_model
from model_cond_local import REL_SHELLS


# --------------------------------------------------------------------------- #
# context collection (mirrors train_deriv_rollout.geff_window_penalty)         #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def collect_group(rc: RootCtx, model, stride: int, wins, device: str):
    """One (root, stride) group: batched (sig, dt, dx, delta_taps, valid)
    over the windows, plus the shared shell context. All FROZEN (design
    choice 1): the cond-head delta_taps are computed once with the original
    model and never re-evaluated during the clip."""
    from qg.solver.opt.basis import to_spectral
    sigs, deltas = [], []
    dtv = torch.tensor([stride * rc.dt_deep], dtype=torch.float64,
                       device=device)
    for w in wins:
        omega_stack, _ = rc.window_tensors(w, stride, 1, device)
        qh0 = to_spectral(omega_stack[0])
        qh1 = to_spectral(omega_stack[1])
        feats = model.context_feats_from_spectral(qh0, qh0 - qh1, dtv,
                                                  rc.Lx, rc.Ly, rc.Ny, rc.Nx)
        sig = feats[:, :len(REL_SHELLS)].to(torch.float64) / dtv.view(-1, 1)
        delta = model.cond(feats.to(model.cond.head.weight.dtype))
        kvec = model.k_of_channel.to(delta.dtype)
        amp = ((dtv.to(delta.dtype) / model.dt_ref_cond.to(delta.dtype))
               ** (model.S - kvec).view(1, -1)) \
            * (kvec > 0).to(delta.dtype).view(1, -1)
        delta = delta * amp.view(1, -1, 1, 1)
        sigs.append(sig.detach().cpu())
        deltas.append(delta.detach().cpu())
    sig = torch.cat(sigs)                                    # (B, n_sh)
    delta = torch.cat(deltas)                                # (B, C, 2, W)
    B = sig.shape[0]
    dt_b = torch.full((B,), stride * rc.dt_deep, dtype=torch.float64)
    dx_b = torch.full((B,), rc.dx, dtype=torch.float64)
    # linearization validity: |dt*sigma| <= 0.5 (see vn_penalty docstring)
    valid = (dt_b.view(-1, 1) * sig.abs()) <= 0.5
    ksh, Lsh = geff_shell_ctx(rc, 'cpu')
    return dict(member=rc.member, stride=stride, wins=list(wins),
                sig=sig, dt=dt_b, dx=dx_b, delta=delta, valid=valid,
                ksh=ksh.cpu(), Lsh=Lsh.cpu())


def geff_group(taps_row: torch.Tensor, mix_weight: torch.Tensor, n_ord: int,
               grp: dict, d_pc: torch.Tensor = None) -> torch.Tensor:
    """assemble_geff through a shim whose grad.wx IS the 1-D tap row (design
    choice 2): reshape(-1, W)[0] then returns exactly taps_row, so the
    certificate is differentiable in it. mix frozen (detached).

    d_pc (C, W): per-channel base-tap deltas, folded into the delta_taps
    argument (design choice 3) -- broadcast over windows and both directions
    (the isotropic direction-mean makes an x/y split unobservable here)."""
    shim = SimpleNamespace(n_ord=n_ord,
                           mix=SimpleNamespace(weight=mix_weight),
                           grad=SimpleNamespace(wx=taps_row))
    delta = grp['delta']
    if d_pc is not None:
        B, C, _, W = delta.shape
        delta = delta + d_pc.view(1, C, 1, W).expand(B, C, 2, W)
    return wc.assemble_geff(shim, grp['sig'], grp['dt'], grp['Lsh'],
                            grp['ksh'], grp['dx'], delta)


def group_maxg(g: torch.Tensor, valid: torch.Tensor) -> float:
    """max|G| over valid shells of the whole group (0.0 if none valid)."""
    if not bool(valid.any()):
        return 0.0
    return float(torch.where(valid, g, torch.zeros_like(g)).amax())


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', type=Path, required=True)
    p.add_argument('--deep-roots', type=Path, nargs='+', required=True,
                   help='deep 28-mark dirs (forced_turbulence_dT_*), NOT the '
                        'sliced sweep_dT_* dirs')
    p.add_argument('--strides', type=str, nargs='+', default=['2', '3'],
                   help='mark strides = dT/5e-3 (2 3 -> 1e-2, 1.5e-2); '
                        'comma or space separated')
    p.add_argument('--n-windows', type=int, default=4,
                   help='val windows per (root, stride)')
    p.add_argument('--eps', type=float, default=0.02,
                   help='stability margin: enforce |G_eff| <= 1 - eps')
    p.add_argument('--lam-sym', type=float, default=1.0,
                   help='symbol-change leash weight (accuracy preservation)')
    p.add_argument('--iters', type=int, default=500)
    p.add_argument('--lr', type=float, default=1e-3, help='Adam lr on d')
    p.add_argument('--n-theta', type=int, default=96,
                   help='uniform theta grid points on (0, 2pi/3]')
    p.add_argument('--roughness-min', type=float, default=1e-4,
                   help='RootCtx quiescent-window screen (rule 15)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out-ckpt', type=Path, required=True)
    p.add_argument('--device', type=str, default='cpu')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = (args.device if (args.device == 'cpu'
                              or torch.cuda.is_available()) else 'cpu')
    strides = [int(s) for tok in args.strides for s in str(tok).split(',')]

    # ---- roots + model (reused loaders) ----
    rcs = [RootCtx(r, device, args.roughness_min, args.seed)
           for r in args.deep_roots]
    model, model_name, n_snap = load_deriv_model(
        args.ckpt, rcs[0].man, strides[0] * rcs[0].dt_deep, device)
    if not hasattr(model, 'cond'):
        raise SystemExit(f"[vn_clip] model {model_name!r} has no cond head -- "
                         f"this tool targets cond_local ckpts")
    model.eval()

    wx = model.grad.wx.detach().to(torch.float64).cpu()      # (C, 1, W, W)
    wy = model.grad.wy.detach().to(torch.float64).cpu()
    C, _, W, _ = wx.shape
    c = W // 2
    base_row = wx[0, 0, c, :].clone()                        # central row (taps)
    row0 = wx[0, 0, 0, :].clone()                            # assemble_geff's read
    mixw = model.mix.weight.detach().to(torch.float64).cpu()
    n_ord = model.n_ord
    print(f"\n[vn_clip] wx shape {tuple(wx.shape)}  W={W}  central row c={c}")
    print(f"[vn_clip] ||central-row taps|| = {float(base_row.norm()):.6e}   "
          f"||row-0|| = {float(row0.norm()):.6e}")
    print("[vn_clip] NOTE (flag upstream, file NOT edited): assemble_geff's "
          "internal base-tap read is reshape(-1,W)[0] == the TOP kernel row, "
          "zero at physics init; this tool feeds it the CENTRAL row via a "
          "shim -- the row the FD taps and the cond deltas actually live in.")

    # ---- collect frozen contexts ----
    groups = []
    for rc in rcs:
        wins = (rc.val_idx or rc.train_idx)[:args.n_windows]
        if not wins:
            print(f"[vn_clip] {rc.member}: NO usable windows, skipped")
            continue
        for s in strides:
            if rc.m_max(s) < 1:
                print(f"[vn_clip] {rc.member} stride {s}: M_max<1, skipped")
                continue
            grp = collect_group(rc, model, s, wins, device)
            groups.append(grp)
            nv = int(grp['valid'].sum())
            print(f"[ctx] {rc.member:10s} stride {s} (dT {DT_TAGS[s]:>7s})  "
                  f"windows={len(wins)}  valid shells={nv}/"
                  f"{grp['valid'].numel()}  "
                  f"max|dt*sig|={float((grp['dt'].view(-1,1)*grp['sig'].abs()).max()):.3f}")
    if not groups:
        raise SystemExit("[vn_clip] no (root, stride) contexts collected")

    # ---- before ----
    with torch.no_grad():
        for grp in groups:
            g = geff_group(base_row, mixw, n_ord, grp)
            grp['g_before'] = group_maxg(g, grp['valid'])
            grp['n_unstable_before'] = int(((g > 1.0) & grp['valid']).sum())
            # flagged diagnostic: what the training-time certificate (row-0
            # read) would have reported on the same context
            grp['g_before_row0'] = group_maxg(
                geff_group(row0, mixw, n_ord, grp), grp['valid'])

    thr = 1.0 - args.eps
    print(f"\n[vn_clip] target |G_eff| <= {thr:.3f} on valid shells")
    already = all(grp['g_before'] <= thr for grp in groups)
    if already:
        print("[vn_clip] already certified at the original taps -- the clip "
              "delta will stay at ~0 (leash-only); writing out-ckpt anyway.")

    # ---- optimize the per-channel 1-D tap deltas (x/y tied) ----
    d = torch.zeros(C, W, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([d], lr=args.lr)
    theta = torch.linspace(0.0, 2.0 * math.pi / 3.0, args.n_theta + 1,
                           dtype=torch.float64)[1:]          # (0, 2pi/3]
    last_vn = None
    for it in range(args.iters):
        opt.zero_grad()
        vn_total = torch.zeros((), dtype=torch.float64)
        gmax_stride = {}
        for grp in groups:
            g = geff_group(base_row, mixw, n_ord, grp, d_pc=d)
            l, _ = wc.vn_penalty(g, 1.0, eps=args.eps, valid=grp['valid'])
            vn_total = vn_total + l
            s = grp['stride']
            gmax_stride[s] = max(gmax_stride.get(s, 0.0),
                                 group_maxg(g.detach(), grp['valid']))
        # leash == mean_{ch,theta} |sym(taps_ch)-sym(taps0_ch)|^2
        #       == mean|sym(d_ch)|^2  (tap_symbol is linear in the taps)
        leash = (wc.tap_symbol(d, theta).abs() ** 2).mean()
        loss = vn_total + args.lam_sym * leash
        loss.backward()
        opt.step()
        last_vn = float(vn_total)
        if it % 50 == 0 or it == args.iters - 1 or last_vn == 0.0:
            msg = '  '.join(f"s{s}:max|G|={v:.6f}"
                            for s, v in sorted(gmax_stride.items()))
            print(f"[iter {it:4d}] vn={last_vn:.3e}  "
                  f"sym-leash={float(leash):.3e}  ||d||={float(d.norm()):.3e}  {msg}")
        if last_vn == 0.0 and it > 0:
            print(f"[vn_clip] penalty exactly zero at iter {it} -- "
                  f"early stop (certified with margin)")
            break

    d_final = d.detach().clone()                             # (C, W)
    with torch.no_grad():
        sym_change = wc.tap_symbol(d_final, theta).abs()     # (C, n_theta)
        sym_l2 = float((sym_change ** 2).mean().sqrt())
        sym_sup = float(sym_change.max())
        for grp in groups:
            g = geff_group(base_row, mixw, n_ord, grp, d_pc=d_final)
            grp['g_after'] = group_maxg(g, grp['valid'])
            grp['n_unstable_after'] = int(((g > 1.0) & grp['valid']).sum())
        per_ch = '  '.join(f"ch{i}:{float(d_final[i].norm()):.2e}"
                           for i in range(C))
        print(f"\n[vn_clip] per-channel ||d_ch||_2: {per_ch}")

    # ---- verdict table ----
    print("\n================ VN-CLIP VERDICT (valid shells only) ================")
    print(f"{'member':12s} {'stride':>6s} {'dT':>8s} {'valid':>6s} "
          f"{'max|G| before':>14s} {'after':>10s} {'[row0-read]':>12s} "
          f"{'shells|G|>1':>12s}")
    per_stride = {}
    for grp in groups:
        nv = int(grp['valid'].sum())
        s = grp['stride']
        per_stride.setdefault(s, []).append(grp['g_after'])
        print(f"{grp['member']:12s} {s:>6d} {DT_TAGS[s]:>8s} "
              f"{nv:>3d}/{grp['valid'].numel():<3d} "
              f"{grp['g_before']:>13.6f} -> {grp['g_after']:>9.6f} "
              f"{grp['g_before_row0']:>12.6f} "
              f"{grp['n_unstable_before']:>5d} -> {grp['n_unstable_after']:<4d}")
    final_maxg = {s: max(v) for s, v in per_stride.items()}
    ok = all(v <= thr + 1e-12 for v in final_maxg.values())
    strict_stable = all(grp['n_unstable_after'] == 0 for grp in groups)
    print(f"\nfinal max|G| per stride: "
          + '  '.join(f"s{s}(dT {DT_TAGS[s]})={v:.6f}"
                      for s, v in sorted(final_maxg.items()))
          + f"   target <= {thr:.3f}  ->  {'CERTIFIED' if ok else 'NOT MET'}")
    print(f"|G|<=1 milestone (marginal von Neumann stability, the bare-"
          f"scheme level): {'MET' if strict_stable else 'NOT MET'} on all "
          f"valid shells")
    print(f"symbol change on (0, 2pi/3]: rms={sym_l2:.3e}  sup={sym_sup:.3e}  "
          f"||d||_2={float(d_final.norm()):.3e}  "
          f"(||base taps||_2={float(base_row.norm()):.3e})")

    # ---- write clipped ckpt ----
    state = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    new = copy.deepcopy(state)
    sd = new.get('model', new.get('model_state', new.get('state_dict', new)))
    wx_keys = [k for k in sd if k.endswith('grad.wx')]
    wy_keys = [k for k in sd if k.endswith('grad.wy')]
    if len(wx_keys) != 1 or len(wy_keys) != 1:
        raise SystemExit(f"[vn_clip] ambiguous grad tap keys: {wx_keys} {wy_keys}")
    kx, ky = wx_keys[0], wy_keys[0]
    tap_dtype = sd[kx].dtype
    sd[kx] = sd[kx].clone()
    sd[ky] = sd[ky].clone()
    sd[kx][:, :, c, :] += d_final.unsqueeze(1).to(tap_dtype)  # central ROW, per ch
    sd[ky][:, :, :, c] += d_final.unsqueeze(1).to(tap_dtype)  # central COL, per ch
    new['vn_clip'] = {
        'args': {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
        'strides': strides,
        'delta_taps_per_channel': d_final.tolist(),
        'final_max_geff_per_stride': {str(s): v
                                      for s, v in final_maxg.items()},
        'before_after': [
            {'member': grp['member'], 'stride': grp['stride'],
             'windows': grp['wins'],
             'n_valid_shells': int(grp['valid'].sum()),
             'max_geff_before': grp['g_before'],
             'max_geff_after': grp['g_after'],
             'n_shells_gt1_before': grp['n_unstable_before'],
             'n_shells_gt1_after': grp['n_unstable_after'],
             'max_geff_before_row0_read': grp['g_before_row0']}
            for grp in groups],
        'geff_le_1_all_valid_shells': bool(strict_stable),
        'symbol_change_rms': sym_l2,
        'symbol_change_sup': sym_sup,
        'certified': bool(ok),
        'note': ('offline von Neumann clip (Wiener option 6): PER-CHANNEL '
                 '1-D deltas (x/y isotropically tied) added to the central '
                 'row of grad.wx and central column of grad.wy; cond-head '
                 'delta_taps frozen; certificate fed CENTRAL-row taps via '
                 'shim, per-channel deltas folded through the delta_taps '
                 'path (assemble_geff row-0 read flagged as discrepancy).'),
    }
    args.out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new, args.out_ckpt)
    print(f"\n[vn_clip] wrote {args.out_ckpt}")

    # ---- write-back sanity: reload and verify the taps ----
    chk = torch.load(args.out_ckpt, map_location='cpu', weights_only=False)
    sd2 = chk.get('model', chk.get('model_state', chk.get('state_dict', chk)))
    err_x = float((sd2[kx].to(torch.float64)[:, 0, c, :]
                   - (wx[:, 0, c, :] + d_final)).abs().max())
    err_y = float((sd2[ky].to(torch.float64)[:, 0, :, c]
                   - (wy[:, 0, :, c] + d_final)).abs().max())
    print(f"[vn_clip] write-back check: max|wx row err|={err_x:.3e}  "
          f"max|wy col err|={err_y:.3e}  "
          f"{'OK' if max(err_x, err_y) < 1e-12 else 'MISMATCH'}")


if __name__ == '__main__':
    main()
