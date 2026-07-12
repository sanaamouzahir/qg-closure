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
    ap.add_argument('--arm', required=True, choices=['A', 'B', 'C'])
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--n-inducing', type=int, default=512)
    ap.add_argument('--likelihood', default='gaussian', choices=['gaussian', 'studentt'])
    ap.add_argument('--lr', type=float, default=3.0e-3)      # = T6 test recipe
    ap.add_argument('--crops', type=int, default=500)
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
    if args.likelihood == 'studentt':
        # B-ITEM EXPERIMENT (spec S2.2 kurtosis item): swap BEFORE hyper init
        # so init_hyperparams_from_stats sets THIS likelihood's noise.
        model.likelihood = gpytorch.likelihoods.StudentTLikelihood().to(device)
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
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    r2s, rmses = [], []
    verdict = 'FAIL'
    for ep in range(int(args.epochs)):
        t0 = time.time()
        model.train()
        for b in batches(ds, int(conf['data']['batch_crops'])):
            gpin = model.masked_gp_inputs(b['x'].to(device), b['zeta'].to(device),
                                          b['mask'].to(device))
            yt = b['y'].to(device)[b['mask'].to(device)]
            opt.zero_grad(set_to_none=True)
            loss = -mll(model.gp(gpin), model.standardize_y(yt))
            loss.backward()
            opt.step()
        m = evaluate(model, ds, device, int(conf['train']['gp_chunk']))
        r2s.append(m['r2']); rmses.append(m['rmse'])
        print(f"[t6{args.arm}] ep {ep:03d} train R2 {m['r2']:.4f} "
              f"RMSE {m['rmse']:.3e}  ({time.time()-t0:.0f}s)")
        np.savez(outdir / f'arm{args.arm}.npz', r2=np.array(r2s), rmse=np.array(rmses),
                 arm=args.arm, epochs=args.epochs, n_inducing=args.n_inducing,
                 likelihood=args.likelihood, lr=args.lr, crops=args.crops)
        if m['r2'] > 0.95:
            verdict = 'PASS'
            break
    print(f"[t6{args.arm}] VERDICT {verdict}: best R2 {max(r2s):.4f} at ep "
          f"{int(np.argmax(r2s))} / final {r2s[-1]:.4f} after {len(r2s)} epochs "
          f"(gate 0.95); curve in {outdir / f'arm{args.arm}.npz'}")


if __name__ == '__main__':
    main()
