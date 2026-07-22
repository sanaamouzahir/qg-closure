"""
Residual-GP head on a FROZEN plateaued CNN (Sanaa GO 2026-07-22).

    r(p) = ( Pi*(p) - Pi_hat_CNN(p) ) / sigma_loc(p)      [standardized residual]
    GP:   z(p) = [16 frozen CNN features, Pi_hat_std, zeta, zeta_dot/zdot_sd,
                  sdf/D]  ->  r(p)  with variance          [20-dim RBF-ARD SVGP]
    Pi_final = Pi_hat_CNN + sigma_loc * m(p);   Var = sigma_loc^2 * (v + noise)

Only the GP trains (kernel hypers, variational params, ONE homoscedastic noise).
The CNN, its head, sigma_loc, zdot_sd are frozen checkpoint state — the mean
burden is not learnable here, so the ELBO noise channel cannot buy it out (the
arm C/D/E + mtgp failure family is structurally closed). zeta/zeta_dot/Pi_hat
are information-redundant with the FiLM-conditioned features (Sanaa 2026-07-22
discussion) — kept as explicit ARD axes; their trained lengthscales are the
verdict on whether the kernel wanted them.

ELBO num_data = N total valid train pixels (same contract as train_piff).
Selection: lowest val NLL (spec S3.3). Assembled val_loss (standardized MSE of
Pi_final vs truth) is logged for direct comparability with the CNN blocks.
Monitor grammar preserved; zeta_ls slot = the TRUE zeta ARD lengthscale.
NaN policy: 2-strike in-process abort, exit 9.
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

import gpytorch

from dataset_piff import (load_conf, build_runs, PiffCropDataset,
                          count_masked_pixels, describe, _f)
from model_cnn import PiffCNN
from model_piff import _kmeans

HERE = Path(__file__).resolve().parent
GP_DIM = 20            # 16 features + pred_std + zeta + zdot_n + sdf/D


class ResidualSVGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        m = inducing_points.shape[0]
        var_dist = gpytorch.variational.CholeskyVariationalDistribution(m)
        strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, var_dist, learn_inducing_locations=True)
        super().__init__(strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=inducing_points.shape[-1]))

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))


def batches(ds, batch_crops):
    idx = np.arange(len(ds))
    for i0 in range(0, len(idx), batch_crops):
        sel = idx[i0:i0 + batch_crops]
        items = [ds[int(i)] for i in sel]
        keys = [k for k in ('x', 'y', 'mask', 'zeta', 'zeta_dot') if k in items[0]]
        yield {k: torch.stack([it[k] for it in items]) for k in keys}


@torch.no_grad()
def gp_inputs_and_residual(cnn, b, device):
    """One frozen-CNN forward per batch -> (z (P,20), r (P,)). The head output
    is derived from the SAME feature tensor (no second CNN pass)."""
    x, y = b['x'].to(device), b['y'].to(device)
    mask, zeta = b['mask'].to(device), b['zeta'].to(device)
    zd = b['zeta_dot'].to(device) if cnn.use_zeta_dot else None
    f = cnn.cnn(x, cnn._cond(zeta, zd))              # (B,16,H,W)
    pred_std = cnn.head(f).squeeze(1)                # (B,H,W)
    fh = f.permute(0, 2, 3, 1)                       # (B,H,W,16)
    B, H, W, _ = fh.shape
    zcol = zeta.reshape(-1, 1, 1, 1).expand(B, H, W, 1)
    zdn = (zd / cnn.zdot_sd if cnn.use_zeta_dot
           else torch.zeros_like(zeta)).reshape(-1, 1, 1, 1).expand(B, H, W, 1)
    s = (x[:, 3] * cnn.sdf_clip_D).unsqueeze(-1)     # (B,H,W,1) sdf/D clipped
    z = torch.cat([fh, pred_std.unsqueeze(-1), zcol, zdn, s], dim=-1)
    r = y / cnn.sigma_loc(x) - pred_std
    return z[mask], r[mask], pred_std[mask]


@torch.no_grad()
def evaluate(gp, lik, cnn, ds, device, bc, chunk):
    gp.eval(); lik.eval()
    nll_s, n = 0.0, 0
    sse_std = 0.0
    cov68 = cov95 = 0
    sig_s = 0.0
    for b in batches(ds, bc):
        z, r, _ = gp_inputs_and_residual(cnn, b, device)
        if r.numel() == 0:
            continue
        for i0 in range(0, z.shape[0], chunk):
            zz, rr = z[i0:i0 + chunk], r[i0:i0 + chunk]
            post = gp(zz)
            pred = lik(post)
            mu, var = pred.mean, pred.variance
            nll_s += float((0.5 * (torch.log(2 * np.pi * var)
                                   + (rr - mu) ** 2 / var)).sum())
            sse_std += float(((rr - mu) ** 2).double().sum())
            zsc = (rr - mu).abs() / var.sqrt()
            cov68 += int((zsc <= 1.0).sum()); cov95 += int((zsc <= 2.0).sum())
            sig_s += float(var.sqrt().sum())
            n += int(rr.numel())
    # assembled standardized val loss = mean (r - mu)^2  (mean-part metric)
    out = {'nll': nll_s / max(n, 1), 'val_loss': sse_std / max(n, 1),
           'cov68': cov68 / max(n, 1), 'cov95': cov95 / max(n, 1),
           'mean_sigma_std': sig_s / max(n, 1), 'n': n}
    return out


def main():
    ap = argparse.ArgumentParser(description="residual-GP head on a frozen CNN")
    ap.add_argument('--cnn-ckpt', required=True,
                    help='plateaued PiffCNN checkpoint (frozen)')
    ap.add_argument('--run-name', required=True)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--lr', type=float, default=1.0e-3)
    ap.add_argument('--n-inducing', type=int, default=512)
    ap.add_argument('--kmeans-pixels', type=int, default=10000)
    ap.add_argument('--init-noise-frac', type=float, default=0.1)
    ap.add_argument('--gp-chunk', type=int, default=65536)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--outdir', default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    wck = torch.load(args.cnn_ckpt, map_location='cpu', weights_only=False)
    conf = wck['conf']
    cnn = PiffCNN(conf).to(args.device)
    cnn.load_state_dict(wck['model'])
    cnn.eval()
    for p in cnn.parameters():
        p.requires_grad_(False)

    runs = build_runs(conf)
    train_ds = PiffCropDataset(runs, 'train', conf, args.seed)
    val_ds = PiffCropDataset(runs, 'val', conf, args.seed)
    N = count_masked_pixels(runs, 'train', conf)
    outdir = Path(args.outdir or (HERE / conf['train']['outdir'])) / args.run_name
    outdir.mkdir(parents=True, exist_ok=True)
    info = describe(runs, conf, args.seed)
    info.update({'cnn_ckpt': str(args.cnn_ckpt),
                 'cnn_ckpt_epoch': int(wck.get('epoch', -1)),
                 'gp_dim': GP_DIM, 'n_inducing': args.n_inducing,
                 'ELBO_num_data': N, 'lr': args.lr, 'epochs': args.epochs})

    # ---- k-means inducing init + residual stats (sampled, f64) ------------ #
    rng = np.random.default_rng(args.seed)
    feats, resids, have = [], [], 0
    for i in rng.permutation(len(train_ds)):
        z, r, _ = gp_inputs_and_residual(cnn, batches_one(train_ds, int(i)),
                                         args.device)
        if r.numel() == 0:
            continue
        take = min(r.shape[0], max(1, args.kmeans_pixels // 4))
        sel = torch.from_numpy(rng.choice(r.shape[0], size=take, replace=False))
        feats.append(z[sel].cpu()); resids.append(r[sel].double().cpu())
        have += take
        if have >= args.kmeans_pixels:
            break
    pts = torch.cat(feats)[:args.kmeans_pixels].to(torch.float32)
    rs = torch.cat(resids)
    r_mean, r_var = float(rs.mean()), float(rs.var())
    if pts.shape[0] < args.n_inducing:
        raise ValueError(f"only {pts.shape[0]} pixels for k-means")
    centers = _kmeans(pts, args.n_inducing, iters=50, seed=args.seed)
    gp = ResidualSVGP(centers.to(args.device)).to(args.device)
    lik = gpytorch.likelihoods.GaussianLikelihood().to(args.device)
    nf = args.init_noise_frac
    gp.mean_module.constant.data.fill_(r_mean)
    gp.covar_module.outputscale = (1.0 - nf) * r_var
    # G4 LOW-MEDIUM 2026-07-22: GaussianLikelihood noise_constraint floor is
    # 1e-4; a very low CNN plateau (r_var <= 1e-3 at nf=0.1) would NaN the
    # raw_noise inverse-transform and burn the submission. Floor it.
    lik.noise = max(nf * r_var, 2.0e-4)
    info['residual_stats'] = {'mean': r_mean, 'var': r_var,
                              'n_sample': int(rs.numel())}
    print('[gp-res]', json.dumps(info, indent=2))
    print(f"[gp-res] residual sample: mean {r_mean:+.4e} var {r_var:.4e} "
          f"(CNN val_loss at ckpt = the residual's expected var scale)")
    with open(outdir / 'run_info.yaml', 'w') as f:
        yaml.safe_dump(info, f, sort_keys=False)

    mll = gpytorch.mlls.VariationalELBO(lik, gp, num_data=N)
    opt = torch.optim.Adam(list(gp.parameters()) + list(lik.parameters()),
                           lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=5)

    bc = int(conf['data']['batch_crops'])
    hist = {k: [] for k in ('train_elbo', 'val_nll', 'val_loss', 'cov68',
                            'cov95', 'mean_sigma_std', 'zeta_ls', 'lr')}
    best_nll = np.inf
    nan_streak = 0
    zeta_dim = 17            # 0-based index of the zeta column in z

    for ep in range(args.epochs):
        t0 = time.time()
        train_ds.set_epoch(ep)
        gp.train(); lik.train()
        elbos = []
        for b in batches(train_ds, bc):
            z, r, _ = gp_inputs_and_residual(cnn, b, args.device)
            if r.numel() == 0:
                continue
            opt.zero_grad(set_to_none=True)
            loss = -mll(gp(z), r)
            loss.backward()
            opt.step()
            elbos.append(-float(loss))
        sched.step()
        vm = evaluate(gp, lik, cnn, val_ds, args.device, bc, args.gp_chunk)
        tr = float(np.mean(elbos)) if elbos else float('nan')
        ls = gp.covar_module.base_kernel.lengthscale.detach().cpu().reshape(-1)
        zls = float(ls[zeta_dim])
        for k, v in (('train_elbo', tr), ('val_nll', vm['nll']),
                     ('val_loss', vm['val_loss']), ('cov68', vm['cov68']),
                     ('cov95', vm['cov95']),
                     ('mean_sigma_std', vm['mean_sigma_std']),
                     ('zeta_ls', zls), ('lr', opt.param_groups[0]['lr'])):
            hist[k].append(v)
        print(f"[ep {ep:03d}] elbo {tr:+.4e}  val NLL {vm['nll']:.4e} "
              f"RMSE {np.sqrt(max(vm['val_loss'], 0.0)):.4e} "
              f"R2 {1.0 - vm['val_loss'] / max(r_var, 1e-30):.4f} "
              f"(vs train-sampled r_var) "
              f"sigma {vm['mean_sigma_std']:.3e}  zeta_ls {zls:.3f}  "
              f"cov68 {vm['cov68']:.3f} cov95 {vm['cov95']:.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

        ep_bad = not (np.isfinite(tr) and np.isfinite(vm['nll']))
        if ep_bad and nan_streak >= 1:
            (outdir / 'NAN_ABORT.txt').write_text(
                f"epoch {ep}: elbo={tr} val_nll={vm['nll']}\n"
                f"STOP>CHECK>FIX>RESUBMIT before rerunning.\n")
            torch.save({'gp': gp.state_dict(), 'lik': lik.state_dict(),
                        'conf': conf, 'epoch': ep, 'nan_abort': True},
                       outdir / 'last_nan_abort.pt')
            print(f"[NAN-ABORT] two consecutive non-finite epochs (ep {ep}); "
                  f"exiting 9", flush=True)
            sys.exit(9)
        nan_streak = 1 if ep_bad else 0

        state = {'gp': gp.state_dict(), 'lik': lik.state_dict(),
                 'cnn': cnn.state_dict(), 'conf': conf,
                 'cnn_ckpt': str(args.cnn_ckpt), 'epoch': ep, 'val': vm,
                 'gp_dim': GP_DIM, 'n_inducing': args.n_inducing,
                 'residual_stats': info['residual_stats'], 'seed': args.seed}
        torch.save(state, outdir / 'last.pt')
        if vm['nll'] < best_nll:
            best_nll = vm['nll']
            torch.save(state, outdir / 'best.pt')
        np.savez(outdir / 'metrics.npz', seed=args.seed,
                 **{k: np.array(v) for k, v in hist.items()})

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (k, ttl) in zip(axs.ravel(), [
            ('train_elbo', 'train ELBO/datum'), ('val_nll', 'val NLL'),
            ('val_loss', 'assembled val loss (std MSE)'),
            ('cov68', 'coverage @1sigma (target 0.68)'),
            ('mean_sigma_std', 'mean predictive sigma (std units)'),
            ('zeta_ls', 'zeta ARD lengthscale')]):
        ax.plot(hist[k]); ax.set_title(ttl); ax.set_xlabel('epoch'); ax.grid(alpha=0.3)
    fig.suptitle(f"{args.run_name}  best val NLL={best_nll:.4e}")
    fig.tight_layout()
    fig.savefig(outdir / 'curves.png', dpi=130)
    plt.close(fig)
    ard = gp.covar_module.base_kernel.lengthscale.detach().cpu().reshape(-1)
    final = {'best_val_nll': float(best_nll),
             'ard_lengthscales': {'features_0_15': [float(v) for v in ard[:16]],
                                  'pred_std': float(ard[16]),
                                  'zeta': float(ard[17]),
                                  'zeta_dot': float(ard[18]),
                                  'sdf_D': float(ard[19])}}
    with open(outdir / 'final.yaml', 'w') as f:
        yaml.safe_dump(final, f, sort_keys=False)
    print(f"[gp-res] done; best val NLL {best_nll:.4e}; ARD verdict on the "
          f"redundant coords: pred_std {ard[16]:.2f} zeta {ard[17]:.2f} "
          f"zdot {ard[18]:.2f} sdf {ard[19]:.2f}")


def batches_one(ds, i):
    s = ds[i]
    return {k: s[k].unsqueeze(0) for k in
            ('x', 'y', 'mask', 'zeta', 'zeta_dot') if k in s}


if __name__ == '__main__':
    main()
