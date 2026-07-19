#!/usr/bin/env python
"""diag_wallv2_nan_probe2.py -- stage-by-stage finiteness trace of the wallv2
cold-start GP init + ONE ELBO step (probe1 cleared the data/stat pipeline;
the NaN is born inside model init or the first ELBO). Replicates train_piff's
cold path verbatim, printing finiteness + conditioning at every stage. CPU,
same dtypes as training. Read-only outside a throwaway model instance.

Usage: python diag_wallv2_nan_probe2.py --config conf_piff_fpc_gjs_wallv2.yaml
"""
import argparse
import json

import numpy as np
import torch

from dataset_piff import (load_conf, build_runs, PiffCropDataset,
                          count_masked_pixels, target_stats, _f,
                          conditioning_stats)
from model_piff import PiffModel, gpytorch
from train_piff import batches, cond_kwargs
from pathlib import Path

HERE = Path(__file__).resolve().parent


def fin(name, t):
    t = t if torch.is_tensor(t) else torch.as_tensor(t)
    nf = int((~torch.isfinite(t)).sum())
    print(f"  [{ 'BAD' if nf else 'ok '}] {name:<28} shape={tuple(t.shape)} "
          f"nonfinite={nf} min={t.min().item():.3e} max={t.max().item():.3e}",
          flush=True)
    return nf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    conf = load_conf(HERE / args.config)
    tc = conf['train']
    seed = int(conf.get('seed', 0))
    torch.manual_seed(seed)
    runs = build_runs(conf)
    train_ds = PiffCropDataset(runs, 'train', conf, seed)
    device = 'cpu'
    model = PiffModel(conf).to(device)

    print("== stage 1: conditioning stats -> model consts", flush=True)
    cstats = conditioning_stats(runs, 'train', conf)
    print(json.dumps({k: float(v) for k, v in cstats.items()
                      if np.isscalar(v)}, indent=1), flush=True)
    model.set_conditioning_stats(zdot_sd=cstats.get('zdot_sd'),
                                 g_scale=cstats.get('g_scale'),
                                 lap_scale=cstats.get('lap_scale'))
    if model.noise_prior == 'structural':
        model.set_noise_feature_scale(cstats['g2_scale'])

    print("== stage 2: y standardization + hyper init", flush=True)
    ystats = target_stats(runs, 'train', conf)
    print(f"  y mean {ystats['mean']:.6e} var {ystats['var']:.6e} "
          f"n {ystats['n']}", flush=True)
    model.set_y_standardization(ystats['mean'], ystats['var'])
    model.init_hyperparams_from_stats(0.0, 1.0,
                                      noise_frac=_f(tc['init_noise_frac']))

    print("== stage 3: k-means inducing init", flush=True)
    npix = model.init_inducing_kmeans(
        train_ds, int(conf['model']['kmeans_pixels']),
        int(conf['model']['kmeans_iters']), seed, device=device)
    ip = model.gp.variational_strategy.inducing_points.detach()
    print(f"  kmeans pixels {npix}; M={ip.shape[0]} d={ip.shape[1]}",
          flush=True)
    fin('inducing_points', ip)
    with torch.no_grad():
        pd = torch.cdist(ip.double(), ip.double())
        pd = pd + torch.eye(pd.shape[0]).double() * 1e9
        print(f"  min pairwise inducing dist = {pd.min().item():.6e} "
              f"(duplicates => singular K_zz)", flush=True)
        n_dup = int((pd < 1e-6).any(dim=1).sum())
        print(f"  near-duplicate rows (<1e-6): {n_dup}", flush=True)
        for d in range(ip.shape[1]):
            col = ip[:, d]
            print(f"    dim {d:2d}: spread {float(col.max()-col.min()):.4e}",
                  flush=True)
        K = model.gp.covar_module(ip).evaluate()
        fin('K_zz', K)
        ev = torch.linalg.eigvalsh(K.double())
        print(f"  K_zz eig range [{ev.min().item():.3e}, "
              f"{ev.max().item():.3e}]  cond={ev.max().item()/max(ev.min().item(),1e-300):.3e}",
              flush=True)
        try:
            torch.linalg.cholesky(K.double())
            print("  K_zz cholesky (f64): OK", flush=True)
        except Exception as e:
            print(f"  K_zz cholesky (f64) FAILED: {e!r}", flush=True)
        try:
            torch.linalg.cholesky(K.float())
            print("  K_zz cholesky (f32): OK", flush=True)
        except Exception as e:
            print(f"  K_zz cholesky (f32) FAILED: {e!r}", flush=True)

    print("== stage 4: one ELBO step (no optimizer)", flush=True)
    N = count_masked_pixels(runs, 'train', conf)
    mll = gpytorch.mlls.VariationalELBO(model.likelihood, model.gp,
                                        num_data=N)
    model.train()
    b = next(batches(train_ds, int(conf['data']['batch_crops'])))
    x, y = b['x'].to(device), b['y'].to(device)
    mask, zeta = b['mask'].to(device), b['zeta'].to(device)
    fin('batch x', x)
    for k in ('g', 'lap', 'zeta', 'zeta_dot'):
        if k in b:
            fin(f'batch {k}', b[k])
    gpin = model.masked_gp_inputs(x, zeta, mask, **cond_kwargs(model, b, device))
    fin('gp inputs (features)', gpin)
    for d in range(gpin.shape[1]):
        col = gpin[:, d]
        if not torch.isfinite(col).all():
            print(f"    << feature column {d} NON-FINITE", flush=True)
    yt = y[mask]
    fin('targets (masked)', yt)
    out = model.gp(gpin)
    fin('gp posterior mean', out.mean)
    fin('gp posterior variance', out.variance)
    if model.noise_prior == 'structural':
        loss = -mll(out, model.standardize_y(yt),
                    noise=model.het_noise(b['g'].to(device)[mask]))
        fin('het noise', model.het_noise(b['g'].to(device)[mask]))
    else:
        loss = -mll(out, model.standardize_y(yt))
    fin('standardized y', model.standardize_y(yt))
    print(f"  ELBO loss = {float(loss):.6e}  finite={np.isfinite(float(loss))}",
          flush=True)
    print("[probe2] done", flush=True)


if __name__ == '__main__':
    main()
