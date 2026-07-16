"""
diag_lap_probe.py — why did the lap init-exactness probe miss at 1.16e-5 (fpc)
/ 2.34e-5 (cape), f64, tol 1e-5, on jobs 1836094/1836097?

Hypothesis under test: the surgery is EXACT; the miss is the lengthscale-1e6
INERTNESS LEAKAGE — kernel entries perturbed by (dlap/1e6)^2 ~ 1e-10 —
amplified through the K_zz solve by the kernel matrix's condition number,
NOT a wrong column/tensor. Discriminator: variant D zeroes the lap coordinate
on data AND inducing points, making the lap dim's squared-distance term
IDENTICALLY zero (kernel factor exp(0) = 1, independent of lengthscale and of
conditioning). D <~ 1e-8 => surgery exact, gate mechanism miscalibrated;
D >> 1e-8 => genuine surgery bug (STOP, tensor-level report).

Measures (f64, fixed val batch, same construction as train_piff.lap_init_probe):
  A. max|gpin_new[:, :D-1] - gpin_ref|          (shared-feature bit-identity)
  B. rel dmu/dvar at lap raw lengthscale 1e6    (the shipped gate — reproduce)
  C. rel dmu/dvar at lap raw lengthscale 1e12   (leakage /1e12 in sq-dist)
  D. rel dmu/dvar with lap coords zeroed        (EXACT inertness)
  E. cond(K_zz) of the ref model                (the amplifier)
GREEN diagnostic (charter analysis carve-out); CPU-only; touches no training.
Run from ml_closure:
  python diag_lap_probe.py --config conf_piff_fpc_gjs_lap.yaml \
      --ckpt runs_piff/piff_fpc_gjs_ylp75/best.pt
"""

import argparse
import json

import numpy as np
import torch

from dataset_piff import (load_conf, build_runs, PiffCropDataset,
                          conditioning_stats)
from model_piff import PiffModel
from train_piff import batches, lap_expand_state_dict


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


def main():
    ap = argparse.ArgumentParser(description="lap init-probe miss diagnosis")
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    conf = load_conf(args.config)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    runs = build_runs(conf)
    train_ds = PiffCropDataset(runs, 'train', conf, args.seed)
    val_ds = PiffCropDataset(runs, 'val', conf, args.seed)

    wck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    sd = wck['model']
    ref = PiffModel(wck['conf'])
    ref.load_state_dict(sd)
    model = PiffModel(conf)
    cstats = conditioning_stats(runs, 'train', conf)
    model.lap_scale.fill_(float(cstats['lap_scale']))
    info = lap_expand_state_dict(model, sd, cstats['lap_scale'], train_ds,
                                 args.seed)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert set(missing) <= {'lap_scale'} and not unexpected, (missing, unexpected)

    model.double().eval()
    ref.double().eval()
    b = next(batches(val_ds, 8))
    x, zeta, mask = b['x'].double(), b['zeta'].double(), b['mask']
    zd, g, lap = b['zeta_dot'].double(), b['g'].double(), b['lap'].double()
    out = {'ckpt': args.ckpt, 'surgery': info}
    with torch.no_grad():
        gin_ref = ref.masked_gp_inputs(x, zeta, mask, zeta_dot=zd, g=g)
        gin_new = model.masked_gp_inputs(x, zeta, mask, zeta_dot=zd, g=g,
                                         lap=lap)
        out['A_shared_feat_absdiff'] = float(
            (gin_new[:, :gin_ref.shape[1]] - gin_ref).abs().max())
        p = ref.gp(gin_ref)
        mu_r, var_r = p.mean, p.variance

        ls = model.gp.covar_module.base_kernel.raw_lengthscale
        ip = model.gp.variational_strategy.inducing_points
        saved_ls = ls.data[..., -1].clone()

        def latent(gin):
            model.train()
            model.eval()                    # drop gpytorch eval caches
            with torch.no_grad():
                q = model.gp(gin)
                return {'dmu': rel(q.mean, mu_r),
                        'dvar': rel(q.variance, var_r)}

        ls.data[..., -1] = 1.0e6
        out['B_ls1e6'] = latent(gin_new)    # the shipped gate — reproduce
        ls.data[..., -1] = 1.0e12
        out['C_ls1e12'] = latent(gin_new)
        ls.data[..., -1] = saved_ls
        saved_ip = ip.data[..., -1].clone()
        ip.data[..., -1] = 0.0
        gin_zero = model.masked_gp_inputs(x, zeta, mask, zeta_dot=zd, g=g,
                                          lap=torch.zeros_like(lap))
        out['D_zerocoord'] = latent(gin_zero)   # exact inertness
        ip.data[..., -1] = saved_ip

        try:
            Klazy = ref.gp.covar_module(
                ref.gp.variational_strategy.inducing_points)
            K = (Klazy.to_dense() if hasattr(Klazy, 'to_dense')
                 else Klazy.evaluate())
            out['E_Kzz_cond'] = float(torch.linalg.cond(K))
        except Exception as e:              # non-essential evidence
            out['E_Kzz_cond'] = f'unavailable: {e}'
    out['verdict'] = (
        'SURGERY-EXACT (gate mechanism miscalibrated: conditioning-amplified '
        'lengthscale leakage)' if out['D_zerocoord']['dmu'] < 1.0e-8
        else 'REAL-BUG (zero-coord inertness violated) — STOP')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
