"""
Pi_FF closure ELBO training (ML SPEC 01 S3). One config = one job; the 6-run
grid is driven by scripts/sge/submit_piff_grid.sh passing --lr/--weight-decay.

ELBO SCALING (exact formula, CP-ML-1 plan S4 — document verbatim):

    ELBO = sum_{i=1..N} E_{q(f_i)}[log p(y_i | f_i)]  -  KL(q(u) || p(u)),
    N = TOTAL masked pixels in the training split, counted once at build
        (dataset_piff.count_masked_pixels) and logged in every artifact.

    Minibatch estimate per step:
        (N/B) * sum_{i in batch} E_q[log p(y_i|f_i)]  -  KL,
    B = masked pixels actually present in the minibatch (masked pixels only;
    crop count never enters). Implemented as gpytorch VariationalELBO(
    num_data=N) fed per-pixel; gpytorch returns the PER-DATUM value (divided
    by N), so logged losses are comparable across grid points.

Optimizer: Adam; weight decay on CNN parameters ONLY, never on GP hypers /
variational params / likelihood (spec S3.2). LR: cosine annealing with warm
restarts, T0 = 5 epochs. Model selection: lowest val NLL over the whole
schedule; checkpoint every epoch, keep best + last (spec S3.3).

PLAN B (pre-authorized, spec S3.4) is NOT auto-switched: its trigger symptoms
(val NLL improving while val RMSE worsens AND feature-space pairwise median
distance shrinking > 10x) are computed and logged every epoch; switching is a
human decision, reported always.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset_piff import (load_conf, build_runs, PiffCropDataset,
                          count_masked_pixels, describe, _f)
from model_piff import PiffModel, gpytorch

HERE = Path(__file__).resolve().parent


def batches(ds, batch_crops):
    idx = np.arange(len(ds))
    for i0 in range(0, len(idx), batch_crops):
        sel = idx[i0:i0 + batch_crops]
        items = [ds[int(i)] for i in sel]
        yield {k: torch.stack([it[k] for it in items]) for k in ('x', 'y', 'mask', 'zeta')}


def gaussian_nll(y, mu, var):
    return 0.5 * (np.log(2 * np.pi * var) + (y - mu) ** 2 / var)


@torch.no_grad()
def evaluate(model, ds, device, gp_chunk):
    """Predictive metrics on a (fixed) crop dataset: per-pixel NLL, RMSE, R2,
    mean predictive sigma, and the per-datum val ELBO surrogate (NLL)."""
    model.eval()
    ys, mus, vars_ = [], [], []
    for b in batches(ds, 8):
        x, y = b['x'].to(device), b['y'].to(device)
        mask, zeta = b['mask'].to(device), b['zeta'].to(device)
        gpin = model.masked_gp_inputs(x, zeta, mask)
        yt = y[mask]
        for i0 in range(0, gpin.shape[0], gp_chunk):
            pred = model.likelihood(model.gp(gpin[i0:i0 + gp_chunk]))
            mus.append(pred.mean.cpu().numpy())
            vars_.append(pred.variance.cpu().numpy())
        ys.append(yt.cpu().numpy())
    y = np.concatenate(ys); mu = np.concatenate(mus); var = np.concatenate(vars_)
    rmse = float(np.sqrt(np.mean((y - mu) ** 2)))
    r2 = float(1.0 - np.sum((y - mu) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-30))
    nll = float(np.mean(gaussian_nll(y, mu, var)))
    return {'nll': nll, 'rmse': rmse, 'r2': r2,
            'mean_sigma': float(np.mean(np.sqrt(var))), 'n_pixels': int(y.size)}


@torch.no_grad()
def probe_feature_spread(model, probe, device):
    """Median pairwise feature distance on a fixed probe batch (Plan-B
    collapse diagnostic)."""
    model.eval()
    f = model.masked_gp_inputs(probe['x'].to(device), probe['zeta'].to(device),
                               probe['mask'].to(device))
    n = min(f.shape[0], 2048)
    f = f[:n]
    d = torch.pdist(f.float())
    return float(d.median())


@torch.no_grad()
def residual_kurtosis(model, ds, device, n_batches=64):
    """Excess kurtosis of (target - predictive mean) after warmup (spec S2.2);
    > 5 -> raise a B-item proposing heteroscedastic/Student-t. Logged, never
    acted on silently."""
    model.eval()
    res = []
    for k, b in enumerate(batches(ds, 8)):
        if k >= n_batches:
            break
        gpin = model.masked_gp_inputs(b['x'].to(device), b['zeta'].to(device),
                                      b['mask'].to(device))
        mu = model.gp(gpin).mean.cpu().numpy()
        res.append(b['y'].numpy()[b['mask'].numpy()] - mu)
    r = np.concatenate(res)
    r = r - r.mean()
    s2 = np.mean(r ** 2)
    return float(np.mean(r ** 4) / max(s2 ** 2, 1e-30) - 3.0)


def main():
    ap = argparse.ArgumentParser(description="Pi_FF SVGP training (one grid point)")
    ap.add_argument('--config', default=str(HERE / 'conf_piff.yaml'))
    ap.add_argument('--run-name', required=True)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--weight-decay', type=float, default=None)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--freeze-film', action='store_true', help='Re-blind ablation (spec S2.1)')
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
    outdir = Path(args.outdir or (HERE / tc['outdir'])) / args.run_name
    outdir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    runs = build_runs(conf)
    train_ds = PiffCropDataset(runs, 'train', conf, seed)
    val_ds = PiffCropDataset(runs, 'val', conf, seed)   # epoch-0 table, FIXED for all epochs
    N = count_masked_pixels(runs, 'train', conf)        # ELBO data count (see module docstring)
    info = describe(runs, conf, seed)
    info.update({'lr': lr, 'weight_decay': wd, 'epochs': epochs,
                 'film': bool(conf['model']['film']), 'ELBO_num_data': N,
                 'device': device})
    print('[train]', json.dumps(info, indent=2))
    with open(outdir / 'run_info.yaml', 'w') as f:
        yaml.safe_dump(info, f, sort_keys=False)

    model = PiffModel(conf).to(device)
    npix = model.init_inducing_kmeans(train_ds, int(conf['model']['kmeans_pixels']),
                                      int(conf['model']['kmeans_iters']), seed, device=device)
    print(f"[train] inducing k-means init on {npix} pixels; M = {int(conf['model']['n_inducing'])}")

    mll = gpytorch.mlls.VariationalELBO(model.likelihood, model.gp, num_data=N)
    gp_named = {id(p) for p in model.gp.parameters()} | {id(p) for p in model.likelihood.parameters()}
    opt = torch.optim.Adam([
        {'params': list(model.cnn.parameters()), 'weight_decay': wd},   # CNN only
        {'params': [p for p in model.parameters() if id(p) in gp_named],
         'weight_decay': 0.0},                                          # never GP hypers
    ], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=int(tc['t0_restart']))

    bc = int(conf['data']['batch_crops'])
    gp_chunk = int(tc['gp_chunk'])
    probe = next(batches(val_ds, 8))                     # fixed collapse probe
    spread0, kurt = None, None
    hist = {k: [] for k in ('train_elbo', 'val_nll', 'val_rmse', 'val_r2',
                            'val_sigma', 'zeta_ls', 'film_dgamma', 'film_beta',
                            'feat_spread', 'lr')}
    best_nll = np.inf

    for ep in range(epochs):
        t0 = time.time()
        train_ds.set_epoch(ep)
        model.train()
        elbos = []
        for b in batches(train_ds, bc):
            x, y = b['x'].to(device), b['y'].to(device)
            mask, zeta = b['mask'].to(device), b['zeta'].to(device)
            gpin = model.masked_gp_inputs(x, zeta, mask)
            yt = y[mask]
            if yt.numel() == 0:
                continue
            opt.zero_grad(set_to_none=True)
            loss = -mll(model.gp(gpin), yt)              # per-datum (num_data=N)
            loss.backward()
            opt.step()
            elbos.append(-float(loss))
        sched.step()

        vm = evaluate(model, val_ds, device, gp_chunk)
        spread = probe_feature_spread(model, probe, device)
        if spread0 is None:
            spread0 = spread
        dg, bnorm = model.film_norms()
        zls = model.zeta_ard_lengthscale()
        hist['train_elbo'].append(float(np.mean(elbos)))
        hist['val_nll'].append(vm['nll']); hist['val_rmse'].append(vm['rmse'])
        hist['val_r2'].append(vm['r2']); hist['val_sigma'].append(vm['mean_sigma'])
        hist['zeta_ls'].append(zls); hist['film_dgamma'].append(dg)
        hist['film_beta'].append(bnorm); hist['feat_spread'].append(spread)
        hist['lr'].append(opt.param_groups[0]['lr'])

        collapse = spread < spread0 / 10.0
        print(f"[ep {ep:03d}] elbo/datum {hist['train_elbo'][-1]:+.4e}  "
              f"val NLL {vm['nll']:.4e} RMSE {vm['rmse']:.4e} R2 {vm['r2']:.4f} "
              f"sigma {vm['mean_sigma']:.3e}  zeta_ls {zls:.3f}  "
              f"film |dg|={dg:.3e} |b|={bnorm:.3e}  spread {spread:.3e}"
              f"{'  [PLAN-B SYMPTOM: feature collapse >10x]' if collapse else ''}"
              f"  ({time.time()-t0:.0f}s)")

        if ep == int(tc['kurtosis_after_epoch']):
            kurt = residual_kurtosis(model, train_ds, device)
            print(f"[train] residual-PDF excess kurtosis after warmup: {kurt:.3f}"
                  f"{'  -> RAISE B-ITEM (heteroscedastic / Student-t)' if kurt > 5 else ''}")

        state = {'model': model.state_dict(), 'conf': conf, 'seed': seed,
                 'epoch': ep, 'val': vm, 'ELBO_num_data': N,
                 'lr': lr, 'weight_decay': wd, 'kurtosis': kurt}
        torch.save(state, outdir / 'last.pt')
        if vm['nll'] < best_nll:
            best_nll = vm['nll']
            torch.save(state, outdir / 'best.pt')

        np.savez(outdir / 'metrics.npz', kurtosis=np.float64(kurt if kurt is not None else np.nan),
                 ELBO_num_data=N, seed=seed, **{k: np.array(v) for k, v in hist.items()})

    # curves
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    panels = [('train_elbo', 'train ELBO/datum'), ('val_nll', 'val NLL'),
              ('val_rmse', 'val RMSE'), ('val_r2', 'val R2'),
              ('val_sigma', 'mean predictive sigma'), ('feat_spread', 'feature spread (median pdist)')]
    for ax, (k, ttl) in zip(axs.ravel(), panels):
        ax.plot(hist[k]); ax.set_title(ttl); ax.set_xlabel('epoch'); ax.grid(alpha=0.3)
    fig.suptitle(f"{args.run_name}  lr={lr:g} wd={wd:g} seed={seed}  best val NLL={best_nll:.4e}")
    fig.tight_layout()
    fig.savefig(outdir / 'curves.png', dpi=130)
    plt.close(fig)

    with open(outdir / 'final.yaml', 'w') as f:
        yaml.safe_dump({'best_val_nll': float(best_nll), 'kurtosis': kurt,
                        'epochs': epochs, 'seed': seed,
                        'training_path': 'joint (Plan A); Plan-B symptoms logged per epoch',
                        'zeta_ard_lengthscale': hist['zeta_ls'][-1]}, f, sort_keys=False)
    print(f"[train] done; best val NLL {best_nll:.4e}; artifacts in {outdir}")


if __name__ == '__main__':
    main()
