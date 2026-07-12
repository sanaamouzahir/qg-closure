"""
t6_arm.py — 3-arm T6 discrimination (orchestrator ruling 2026-07-12 under
Sanaa's autonomy window; DECISIONS row of that date).

All arms start from the arm-2 baseline (recorded y-standardization + data-
informed init in standardized space, commit 5a75893) and replicate the T6
procedure exactly (500 fixed crops FPC-const s=4, seed 0, Adam lr 3e-3,
gate R2 > 0.95):

  A  --epochs 150                    "just needs time" (plateau +0.001-0.002/ep;
                                     fresh run — the pytest T6 saved no ckpt,
                                     so nothing is resumable)
  B  --n-inducing 1024               capacity (k-means init runs on the same
                                     10k sampled pixels, NOT 62M: cdist
                                     10000x1024 ~ 40 MB — checked, no fallback
                                     needed)
  C  --likelihood studentt           B-ITEM EXPERIMENT: StudentTLikelihood
                                     (non-conjugate; gpytorch VariationalELBO
                                     handles it via Gauss-Hermite quadrature
                                     of expected_log_prob — same mll class,
                                     same num_data). Tests the heavy-tail /
                                     noise-inflation hypothesis (kurtosis 395).

Writes per-epoch R2/RMSE curves to runs_piff/t6_arms/arm<X>.npz and prints a
final VERDICT line. Run from ml_closure/ (flat sibling imports).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import gpytorch

import dataset_piff as dp
from model_piff import PiffModel
from train_piff import batches, evaluate

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description="one T6 discrimination arm")
    ap.add_argument('--arm', required=True, choices=['A', 'B', 'C', 'D', 'E'])
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--n-inducing', type=int, default=512)
    ap.add_argument('--likelihood', default='gaussian',
                    choices=['gaussian', 'studentt', 'hetero'])
    ap.add_argument('--lr', type=float, default=3.0e-3)      # = T6 test recipe
    ap.add_argument('--crops', type=int, default=500)
    ap.add_argument('--warmup-epochs', type=int, default=0,
                    help='ARM E: epochs with the noise head FROZEN at the '
                         'homoscedastic init before unfreezing (hetero only)')
    ap.add_argument('--sigma-cap-logrange', type=float, default=2.0,
                    help='ARM E: clamp log sigma^2 to +/- this around the '
                         'homoscedastic init once unfrozen')
    ap.add_argument('--outdir', default=str(HERE / 'runs_piff' / 't6_arms'))
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    conf = dp.load_conf(HERE / 'conf_piff.yaml')
    conf['data'].update(crops_per_epoch_train=int(args.crops), crops_per_epoch_val=16)
    conf['model']['n_inducing'] = int(args.n_inducing)

    torch.manual_seed(0)
    np.random.seed(0)
    runs = dp.build_runs(conf)
    ds = dp.PiffCropDataset(runs, 'train', conf, seed=0)     # fixed table: pure overfit

    model = PiffModel(conf).to(device)
    noise_head = None
    if args.likelihood == 'studentt':
        # B-ITEM EXPERIMENT (spec S2.2 kurtosis item): swap BEFORE hyper init
        # so init_hyperparams_from_stats sets THIS likelihood's noise.
        model.likelihood = gpytorch.likelihoods.StudentTLikelihood().to(device)
    elif args.likelihood == 'hetero':
        # ARM D — HETEROSCEDASTIC B-ITEM (orchestrator ruling 2 of 2026-07-12,
        # motivated by arm C: kurtosis 395 = signal heteroscedasticity).
        # Per-pixel noise sigma^2(x) from the model's OWN inputs (never y):
        # linear head on the F+1 GP inputs -> softplus + 1e-4 floor
        # (standardized space; floor sigma ~ 1% of target std).
        # Verified-supported gpytorch path: FixedGaussianNoise.forward accepts
        # a per-batch `noise=` kwarg and wraps the live tensor in a
        # DiagLinearOperator (gradients flow to the head); the approximate MLL
        # forwards kwargs to expected_log_prob. learn_additional_noise=False.
        # Init: zero weights + bias softplus^-1(0.1) => EXACTLY the arm-2
        # data-informed noise at init.
        model.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            noise=torch.full((1,), 0.1), learn_additional_noise=False).to(device)
        noise_head = torch.nn.Linear(model.gp_input_dim, 1).to(device)
        torch.nn.init.zeros_(noise_head.weight)
        with torch.no_grad():
            noise_head.bias.fill_(float(np.log(np.expm1(0.1))))   # softplus -> 0.1

        # ARM E (orchestrator ruling 3): sigma cap — clamp sigma^2 to
        # exp(+/-logrange) around the homoscedastic init (0.1). Clamp's zero
        # gradient outside the band is the point: the head cannot run away.
        s2_lo = 0.1 * float(np.exp(-args.sigma_cap_logrange))
        s2_hi = 0.1 * float(np.exp(+args.sigma_cap_logrange))

        def het_sigma2(gpin, capped):
            s2 = torch.nn.functional.softplus(noise_head(gpin)).squeeze(-1) + 1.0e-4
            return s2.clamp(s2_lo, s2_hi) if capped else s2
    npix_km = model.init_inducing_kmeans(ds, 10000, int(conf['model']['kmeans_iters']),
                                         seed=0, device=device)
    st = dp.target_stats(runs, 'train', conf)
    std_const = model.set_y_standardization(st['mean'], st['var'])
    hyper0 = model.init_hyperparams_from_stats(
        0.0, 1.0, noise_frac=dp._f(conf['train']['init_noise_frac']))
    print(f"[t6{args.arm}] device {device} M {args.n_inducing} lik {args.likelihood} "
          f"epochs {args.epochs} crops {args.crops} kmeans_pixels {npix_km}")
    print(f"[t6{args.arm}] y-standardization {std_const} + GP init {hyper0}")
    if args.likelihood == 'studentt':
        print(f"[t6{args.arm}] StudentT deg_free init = {float(model.likelihood.deg_free):.3f} "
              f"(trainable), noise init = {float(model.likelihood.noise):.3f}; "
              f"VariationalELBO handles the non-conjugate likelihood by quadrature")

    n_pix = int(sum(ds[i]['mask'].sum() for i in range(len(ds))))
    mll = gpytorch.mlls.VariationalELBO(model.likelihood, model.gp, num_data=n_pix)
    warmup = int(args.warmup_epochs) if noise_head is not None else 0
    use_cap = warmup > 0            # ARM E; arm D (warmup 0) = uncapped legacy
    params = list(model.parameters())
    if noise_head is not None and warmup == 0:
        params += list(noise_head.parameters())     # arm-D joint-from-scratch
    opt = torch.optim.Adam(params, lr=float(args.lr))
    if noise_head is not None and warmup > 0:
        print(f"[t6{args.arm}] mean-warmup: head FROZEN (detached) for {warmup} ep, "
              f"then unfrozen with sigma^2 cap [{s2_lo:.4f}, {s2_hi:.4f}] (std space)")

    @torch.no_grad()
    def hetero_evaluate(gp_chunk):
        """evaluate() twin for arm D: predictive var = f.var + sigma^2(x),
        both unstandardized to physical units; also returns sigma stats."""
        model.eval()
        ys, mus, vars_, sds = [], [], [], []
        y_sd, y_mu = float(model.y_sd), float(model.y_mu)
        for b in batches(ds, 8):
            gpin = model.masked_gp_inputs(b['x'].to(device), b['zeta'].to(device),
                                          b['mask'].to(device))
            yt = b['y'].to(device)[b['mask'].to(device)]
            for i0 in range(0, gpin.shape[0], gp_chunk):
                ch = gpin[i0:i0 + gp_chunk]
                f = model.gp(ch)
                s2 = het_sigma2(ch, use_cap)
                mus.append((f.mean * y_sd + y_mu).cpu().numpy())
                vars_.append(((f.variance + s2) * y_sd * y_sd).cpu().numpy())
                sds.append((s2.sqrt() * y_sd).cpu().numpy())
            ys.append(yt.cpu().numpy())
        y = np.concatenate(ys); mu = np.concatenate(mus)
        sd = np.concatenate(sds)
        rmse = float(np.sqrt(np.mean((y - mu) ** 2)))
        r2 = float(1.0 - np.sum((y - mu) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-30))
        return {'r2': r2, 'rmse': rmse,
                'sig_min': float(sd.min()), 'sig_med': float(np.median(sd)),
                'sig_max': float(sd.max())}

    r2s, rmses = [], []
    verdict = 'FAIL'
    for ep in range(int(args.epochs)):
        t0 = time.time()
        if noise_head is not None and warmup > 0 and ep == warmup:
            # ARM E unfreeze: head joins the optimizer with fresh (zero) grads;
            # during warmup its sigma was DETACHED, so no stale accumulation.
            opt.add_param_group({'params': list(noise_head.parameters())})
            print(f"[t6{args.arm}] ep {ep:03d}: noise head UNFROZEN "
                  f"(sigma^2 cap [{s2_lo:.4f}, {s2_hi:.4f}])")
        head_live = noise_head is not None and (warmup == 0 or ep >= warmup)
        model.train()
        for b in batches(ds, int(conf['data']['batch_crops'])):
            gpin = model.masked_gp_inputs(b['x'].to(device), b['zeta'].to(device),
                                          b['mask'].to(device))
            yt = b['y'].to(device)[b['mask'].to(device)]
            opt.zero_grad(set_to_none=True)
            if noise_head is not None:
                s2 = het_sigma2(gpin, use_cap)
                if not head_live:
                    s2 = s2.detach()          # frozen warmup: no grads to head
                loss = -mll(model.gp(gpin), model.standardize_y(yt), noise=s2)
            else:
                loss = -mll(model.gp(gpin), model.standardize_y(yt))
            loss.backward()
            opt.step()
        if noise_head is not None:
            m = hetero_evaluate(int(conf['train']['gp_chunk']))
            extra = (f"  sigma(phys) min/med/max "
                     f"{m['sig_min']:.2f}/{m['sig_med']:.2f}/{m['sig_max']:.2f}")
        else:
            m = evaluate(model, ds, device, int(conf['train']['gp_chunk']))
            extra = ""
        r2s.append(m['r2']); rmses.append(m['rmse'])
        print(f"[t6{args.arm}] ep {ep:03d} train R2 {m['r2']:.4f} "
              f"RMSE {m['rmse']:.3e}  ({time.time()-t0:.0f}s){extra}")
        np.savez(outdir / f'arm{args.arm}.npz', r2=np.array(r2s), rmse=np.array(rmses),
                 arm=args.arm, epochs=args.epochs, n_inducing=args.n_inducing,
                 likelihood=args.likelihood, lr=args.lr, crops=args.crops,
                 warmup_epochs=int(args.warmup_epochs),
                 sigma_cap_logrange=float(args.sigma_cap_logrange))
        if m['r2'] > 0.95:
            verdict = 'PASS'
            break
    print(f"[t6{args.arm}] VERDICT {verdict}: best R2 {max(r2s):.4f} at ep "
          f"{int(np.argmax(r2s))} / final {r2s[-1]:.4f} after {len(r2s)} epochs "
          f"(gate 0.95); curve in {outdir / f'arm{args.arm}.npz'}")


if __name__ == '__main__':
    main()
