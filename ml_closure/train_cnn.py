"""
CNN-only Pi_FF training (Sanaa order 2026-07-22): FiLM-CNN + 1x1 head, no GP.

Loss (per batch, masked pixels only):
    L = mean_p [ (yhat_std(p) - Pi*(p)/sigma_loc(p))^2 ]
sigma_loc = FROZEN train-split rms of Pi* binned by sdf/D (computed exactly in
f64 here, recorded as model buffers). Equal to physical-space MSE weighted by
1/sigma_loc^2. NOTHING in the weighting is learnable (arm C/D/E lessons: robust
losses reject the wake signal; learnable noise buys it out).

Model selection: lowest val loss (same standardized MSE on the FIXED val crop
table). Reporting per region (near/far at data.eval_split_D) in PHYSICAL units;
the eval script (eval_cnn.py) adds the per-member x per-pixel err/|truth| plots.

Epoch stdout follows the monitor_piff.py grammar:
    [ep NNN] ... val NLL <val_loss> RMSE <rmse> R2 <r2> ... zeta_ls <film|dg|>
(NLL slot = standardized val MSE, the selection scalar; zeta_ls slot = FiLM
|dgamma| — the CNN's conditioning-activity signal. The CNN baseline card
diagnostics/baseline_cards/SGS_piff_cnn.json documents both substitutions.)

NaN policy (Sanaa 2026-07-19, mandatory): two consecutive non-finite epochs
=> NAN_ABORT.txt + last_nan_abort.pt + exit 9.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset_piff import (load_conf, build_runs, PiffCropDataset, describe,
                          split_frames, count_masked_pixels,
                          conditioning_stats, _f)
from model_cnn import PiffCNN

HERE = Path(__file__).resolve().parent


def batches(ds, batch_crops):
    idx = np.arange(len(ds))
    for i0 in range(0, len(idx), batch_crops):
        sel = idx[i0:i0 + batch_crops]
        items = [ds[int(i)] for i in sel]
        keys = [k for k in ('x', 'y', 'mask', 'zeta', 'zeta_dot', 'lap', 'psi') if k in items[0]]
        yield {k: torch.stack([it[k] for it in items]) for k in keys}


def sigma_profile(runs, split, conf, n_bins):
    """Train-split rms of the normalized target Pi* = pi * D^2/U^2 per sdf/D
    bin over [0, sdf_clip_D] (valid pixels; exact f64 accumulation, no
    sampling — mirrors dataset_piff.target_stats). Empty bins are filled from
    the NEAREST non-empty bin; a floor at 1e-3 * max rms guards the division.
    Returns (rms[n_bins], counts[n_bins], edges[n_bins+1])."""
    clip_D = _f(conf['data']['sdf_clip_D'])
    edges = np.linspace(0.0, clip_D, n_bins + 1)
    s2 = np.zeros(n_bins, dtype=np.float64)
    cnt = np.zeros(n_bins, dtype=np.int64)
    per_run_bin = {}          # run -> per-frame-invariant bin index of valid px
    for ri, fi in split_frames(runs, split, conf):
        r = runs[ri]
        if ri not in per_run_bin:
            s = np.clip(r.sdf[r.valid] / r.D, 0.0, clip_D)
            b = np.minimum((s / clip_D * n_bins).astype(np.int64), n_bins - 1)
            per_run_bin[ri] = b
        b = per_run_bin[ri]
        U = _f(r.U_snap[fi])
        y = r.pi[fi][r.valid].astype(np.float64) * (r.D * r.D / (U * U))
        s2 += np.bincount(b, weights=y * y, minlength=n_bins)
        cnt += np.bincount(b, minlength=n_bins)
    rms = np.sqrt(np.divide(s2, cnt, out=np.zeros_like(s2), where=cnt > 0))
    nz = np.flatnonzero(cnt > 0)
    if nz.size == 0:
        raise ValueError("sigma_profile: no valid pixel in the split")
    for i in np.flatnonzero(cnt == 0):
        rms[i] = rms[nz[np.argmin(np.abs(nz - i))]]
    rms = np.maximum(rms, 1.0e-3 * rms.max())
    return rms, cnt, edges


@torch.no_grad()
def evaluate(model, ds, device, batch_crops, eval_split_D):
    """Fixed val-crop metrics: standardized val loss (selection scalar) +
    physical-unit RMSE / R2, globally and per region (near: sdf <= split_D,
    far: sdf > split_D)."""
    model.eval()
    sse_std, n_std = 0.0, 0
    acc = {k: dict(sse=0.0, sy=0.0, sy2=0.0, n=0)
           for k in ('all', 'near', 'far')}
    for b in batches(ds, batch_crops):
        x, y = b['x'].to(device), b['y'].to(device)
        mask, zeta = b['mask'].to(device), b['zeta'].to(device)
        zd = b['zeta_dot'].to(device) if model.use_zeta_dot else None
        lp = b['lap'].to(device) if model.use_lap_input else None
        ps = b['psi'].to(device) if model.use_psi_input else None
        yhat_std = model(x, zeta, zd, lp, ps)
        sig = model.sigma_loc(x)
        r_std = (yhat_std - y / sig)[mask]
        sse_std += float((r_std.double() ** 2).sum())
        n_std += int(r_std.numel())
        err = (yhat_std * sig - y)
        s = x[:, 3] * model.sdf_clip_D
        for key, sel in (('all', mask),
                         ('near', mask & (s <= eval_split_D)),
                         ('far', mask & (s > eval_split_D))):
            e, yy = err[sel].double(), y[sel].double()
            acc[key]['sse'] += float((e * e).sum())
            acc[key]['sy'] += float(yy.sum())
            acc[key]['sy2'] += float((yy * yy).sum())
            acc[key]['n'] += int(yy.numel())
    out = {'val_loss': sse_std / max(n_std, 1)}
    for key, a in acc.items():
        if a['n'] == 0:
            out[f'rmse_{key}'], out[f'r2_{key}'] = float('nan'), float('nan')
            continue
        var = a['sy2'] / a['n'] - (a['sy'] / a['n']) ** 2
        out[f'rmse_{key}'] = float(np.sqrt(a['sse'] / a['n']))
        out[f'r2_{key}'] = float(1.0 - (a['sse'] / a['n']) / max(var, 1e-30))
    return out


def main():
    ap = argparse.ArgumentParser(description="Pi_FF CNN-only training (no GP head)")
    ap.add_argument('--config', default=str(HERE / 'conf_piff_fpc_cnn.yaml'))
    ap.add_argument('--run-name', required=True)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--weight-decay', type=float, default=None)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--amp-weight-alpha', type=float, default=0.0,
                    help='amplitude-importance loss weighting (Sanaa 2026-07-22): '
                         'w = eps + |truth_std|^alpha, loss = sum(w*r^2)/sum(w). '
                         'Weights from the TRUTH only (frozen data - no '
                         'prediction-shrinking channel); 0 = exact legacy loss. '
                         'Val metric and best.pt selection stay UNWEIGHTED so '
                         'arms remain comparable.')
    ap.add_argument('--amp-weight-eps', type=float, default=0.2,
                    help='weight floor: quiet-truth pixels keep this weight or '
                         'the prediction there is unconstrained (hallucination '
                         'guard). Only used when alpha > 0.')
    ap.add_argument('--enscon-beta', type=float, default=0.0,
                    help='EnsCon (Guan et al. 2022): loss = (1-beta)*MSE + '
                         'beta*mean_crops[(<om* Pi_hat*> - <om* Pi*>)^2] - '
                         'global SGS enstrophy-transfer conservation, frozen '
                         'physics target (no learnable weighting). 0 = off.')
    ap.add_argument('--init-ckpt-grow', default=None,
                    help='warm start from a ckpt with FEWER input channels: '
                         'the first conv is zero-padded for the new channels '
                         '(exact function preservation at init); rest strict.')
    ap.add_argument('--init-ckpt', default=None,
                    help='warm start: load a PiffCNN checkpoint (strict). The '
                         'recorded sigma_loc/zdot_sd buffers are recomputed '
                         'from the same pool then overwritten by the ckpt '
                         'values — identical by construction; a conf mismatch '
                         'fails loudly on state_dict shapes')
    ap.add_argument('--freeze-film', action='store_true',
                    help='Re-blind ablation (spec S2.1)')
    ap.add_argument('--device', default=None)
    ap.add_argument('--outdir', default=None)
    args = ap.parse_args()

    conf = load_conf(args.config)
    tc = conf['train']
    lr = _f(args.lr if args.lr is not None else tc['lr'])
    wd = _f(args.weight_decay if args.weight_decay is not None else tc['weight_decay'])
    epochs = int(args.epochs if args.epochs is not None else tc['epochs'])
    seed = int(args.seed if args.seed is not None else tc['seed'])
    device = args.device or tc['device']
    if args.freeze_film:
        conf['model']['film'] = False
    eval_split_D = _f(conf['data'].get('eval_split_D', 1.25))
    outdir = Path(args.outdir or (HERE / tc['outdir'])) / args.run_name
    outdir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    runs = build_runs(conf)
    train_ds = PiffCropDataset(runs, 'train', conf, seed)
    val_ds = PiffCropDataset(runs, 'val', conf, seed)   # FIXED for all epochs
    info = describe(runs, conf, seed)
    info.update({'lr': lr, 'weight_decay': wd, 'epochs': epochs,
                 'film': bool(conf['model']['film']), 'device': device,
                 'head': 'cnn_1x1', 'eval_split_D': eval_split_D,
                 'amp_weight_alpha': float(args.amp_weight_alpha),
                 'amp_weight_eps': float(args.amp_weight_eps),
                 'N_train_pixels': count_masked_pixels(runs, 'train', conf)})

    model = PiffCNN(conf).to(device)
    consts = {}
    if model.use_zeta_dot or model.use_lap_input:
        cstats = conditioning_stats(runs, 'train', conf)
        if model.use_zeta_dot:
            consts.update(model.set_zdot_sd(cstats['zdot_sd']))
        if model.use_lap_input:
            consts.update(model.set_lap_scale(cstats['lap_scale']))
    rms, cnt, edges = sigma_profile(runs, 'train', conf,
                                    int(model.sig_rms.numel()))
    consts.update(model.set_sigma_profile(rms))
    consts['sigma_loc_bin_counts'] = [int(c) for c in cnt]
    if args.init_ckpt:
        wck = torch.load(args.init_ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(wck['model'])   # strict: any conf drift fails loudly
        info['init_ckpt'] = str(args.init_ckpt)
        info['init_ckpt_epoch'] = int(wck.get('epoch', -1))
        print(f"[train] warm start from {args.init_ckpt} "
              f"(epoch={wck.get('epoch')}, val={wck.get('val', {})})")
    if args.init_ckpt_grow:
        wck = torch.load(args.init_ckpt_grow, map_location='cpu',
                         weights_only=False)
        sd = wck['model']
        w = sd['cnn.convs.0.weight']
        want = model.cnn.convs[0].weight.shape[1]
        have = w.shape[1]
        if have < want:
            pad = torch.zeros(w.shape[0], want - have, w.shape[2], w.shape[3],
                              dtype=w.dtype)
            sd['cnn.convs.0.weight'] = torch.cat([w, pad], dim=1)
            print(f"[train] channel-grow warm start: conv0 {have}->{want} "
                  f"in-ch (new channels ZERO = function preserved at init)")
        model.load_state_dict(sd)
        info['init_ckpt_grow'] = str(args.init_ckpt_grow)
        info['init_ckpt_epoch'] = int(wck.get('epoch', -1))
        print(f"[train] grown warm start from {args.init_ckpt_grow} "
              f"(epoch={wck.get('epoch')}, val={wck.get('val', {})})")
    info['recorded_constants'] = consts
    nparams = sum(p.numel() for p in model.parameters())
    info['n_params'] = int(nparams)
    print('[train]', json.dumps(info, indent=2))
    with open(outdir / 'run_info.yaml', 'w') as f:
        yaml.safe_dump(info, f, sort_keys=False)
    print(f"[train] sigma_loc rms: near(bin0) {rms[0]:.4e} -> far(bin-1) "
          f"{rms[-1]:.4e} (ratio {rms[0] / rms[-1]:.1f}x); params {nparams:,}")

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=int(tc['t0_restart']))

    bc = int(conf['data']['batch_crops'])
    hist = {k: [] for k in ('train_loss', 'val_loss', 'val_rmse', 'val_r2',
                            'r2_near', 'r2_far', 'rmse_near', 'rmse_far',
                            'film_dgamma', 'film_beta', 'lr')}
    best_val = np.inf
    nan_streak = 0

    for ep in range(epochs):
        t0 = time.time()
        train_ds.set_epoch(ep)
        model.train()
        losses = []
        ens_log = []
        for b in batches(train_ds, bc):
            x, y = b['x'].to(device), b['y'].to(device)
            mask, zeta = b['mask'].to(device), b['zeta'].to(device)
            zd = b['zeta_dot'].to(device) if model.use_zeta_dot else None
            lp = b['lap'].to(device) if model.use_lap_input else None
            ps = b['psi'].to(device) if model.use_psi_input else None
            yhat_std = model(x, zeta, zd, lp, ps)
            sig = model.sigma_loc(x)
            yst = y / sig
            r = (yhat_std - yst)[mask]
            if r.numel() == 0:
                continue
            if args.amp_weight_alpha > 0.0:
                w = args.amp_weight_eps + yst[mask].abs() ** args.amp_weight_alpha
                loss = (w * r * r).sum() / w.sum()
            else:
                loss = (r * r).mean()
            if args.enscon_beta > 0.0:
                # per-crop global SGS enstrophy transfer <om* Pi*> over valid
                # pixels; frozen physics target (Guan et al. 2022 EnsCon)
                mk = mask.float()
                npx = mk.sum(dim=(1, 2)).clamp_min(1.0)
                that = (x[:, 0] * yhat_std * sig * mk).sum(dim=(1, 2)) / npx
                ttru = (x[:, 0] * y * mk).sum(dim=(1, 2)) / npx
                ens = ((that - ttru) ** 2).mean()
                loss = (1.0 - args.enscon_beta) * loss + args.enscon_beta * ens
                ens_log.append(float(ens))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        sched.step()

        vm = evaluate(model, val_ds, device, bc, eval_split_D)
        tr = float(np.mean(losses)) if losses else float('nan')
        dg, bnorm = model.film_norms()
        for k, v in (('train_loss', tr), ('val_loss', vm['val_loss']),
                     ('val_rmse', vm['rmse_all']), ('val_r2', vm['r2_all']),
                     ('r2_near', vm['r2_near']), ('r2_far', vm['r2_far']),
                     ('rmse_near', vm['rmse_near']), ('rmse_far', vm['rmse_far']),
                     ('film_dgamma', dg), ('film_beta', bnorm),
                     ('lr', opt.param_groups[0]['lr'])):
            hist[k].append(v)

        # monitor_piff grammar (see module docstring for the slot mapping)
        print(f"[ep {ep:03d}] loss {tr:.4e}  val NLL {vm['val_loss']:.4e} "
              f"RMSE {vm['rmse_all']:.4e} R2 {vm['r2_all']:.4f} "
              f"sigma 0.000e+00  zeta_ls {dg:.3f}  "
              f"r2near {vm['r2_near']:.4f} r2far {vm['r2_far']:.4f} "
              f"rmse_near {vm['rmse_near']:.3e} rmse_far {vm['rmse_far']:.3e} "
              f"film |dg|={dg:.3e} |b|={bnorm:.3e}"
              + (f"  ens {np.mean(ens_log):.3e}" if ens_log else "")
              + f"  ({time.time() - t0:.0f}s)",
              flush=True)

        # HARD NaN GUARD (Sanaa mandate 2026-07-19): in-process, unforgettable
        ep_bad = not (np.isfinite(tr) and np.isfinite(vm['val_loss']))
        if ep_bad and nan_streak >= 1:
            (outdir / 'NAN_ABORT.txt').write_text(
                f"epoch {ep}: train_loss={tr} val_loss={vm['val_loss']}\n"
                f"STOP>CHECK>FIX>RESUBMIT: diagnose the first non-finite "
                f"statistic before resubmission.\n")
            torch.save({'model': model.state_dict(), 'conf': conf,
                        'epoch': ep, 'nan_abort': True},
                       outdir / 'last_nan_abort.pt')
            print(f"[NAN-ABORT] two consecutive non-finite epochs (ep {ep}); "
                  f"exiting 9", flush=True)
            sys.exit(9)
        nan_streak = 1 if ep_bad else 0

        state = {'model': model.state_dict(), 'conf': conf, 'seed': seed,
                 'epoch': ep, 'val': vm, 'lr': lr, 'weight_decay': wd,
                 'recorded_constants': consts}
        torch.save(state, outdir / 'last.pt')
        if vm['val_loss'] < best_val:
            best_val = vm['val_loss']
            torch.save(state, outdir / 'best.pt')
        np.savez(outdir / 'metrics.npz', seed=seed,
                 **{k: np.array(v) for k, v in hist.items()})

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    panels = [('train_loss', 'train loss (standardized MSE)'),
              ('val_loss', 'val loss (standardized MSE)'),
              ('val_r2', 'val R2 (physical, all)'),
              ('r2_near', 'val R2 near (sdf<=%.2fD)' % eval_split_D),
              ('r2_far', 'val R2 far'),
              ('film_dgamma', 'FiLM |dgamma| (conditioning activity)')]
    for ax, (k, ttl) in zip(axs.ravel(), panels):
        ax.plot(hist[k]); ax.set_title(ttl); ax.set_xlabel('epoch'); ax.grid(alpha=0.3)
    fig.suptitle(f"{args.run_name}  lr={lr:g} wd={wd:g} seed={seed}  "
                 f"best val loss={best_val:.4e}")
    fig.tight_layout()
    fig.savefig(outdir / 'curves.png', dpi=130)
    plt.close(fig)

    final = {'best_val_loss': float(best_val), 'epochs': epochs, 'seed': seed,
             'last': {k: hist[k][-1] for k in hist if hist[k]}}
    with open(outdir / 'final.yaml', 'w') as f:
        yaml.safe_dump(final, f, sort_keys=False)
    print(f"[train] done; best val loss {best_val:.4e}; artifacts in {outdir}")


if __name__ == '__main__':
    main()
