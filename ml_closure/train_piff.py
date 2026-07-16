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
                          count_masked_pixels, target_stats, describe, _f,
                          conditioning_stats)
from model_piff import PiffModel, gpytorch

HERE = Path(__file__).resolve().parent


def batches(ds, batch_crops):
    idx = np.arange(len(ds))
    for i0 in range(0, len(idx), batch_crops):
        sel = idx[i0:i0 + batch_crops]
        items = [ds[int(i)] for i in sel]
        keys = [k for k in ('x', 'y', 'mask', 'zeta', 'zeta_dot', 'g', 'lap') if k in items[0]]
        yield {k: torch.stack([it[k] for it in items]) for k in keys}


def cond_kwargs(model, b, device):
    """Conditioning tensors for the model forward, from a batch dict (None
    when the corresponding ORDER-3 flag is off — exact legacy path)."""
    return {'zeta_dot': b['zeta_dot'].to(device) if model.use_zeta_dot else None,
            'g': b['g'].to(device) if model.use_grad_feature else None,
            'lap': b['lap'].to(device) if model.use_lap_feature else None}


def gaussian_nll(y, mu, var):
    return 0.5 * (np.log(2 * np.pi * var) + (y - mu) ** 2 / var)


# ---- lap warm-start surgery (option b, Sanaa GO 2026-07-16) --------------- #
def lap_expand_state_dict(model, sd, lap_scale, train_ds, seed):
    """Expand a pre-lap ckpt state dict to the +lap GP input dim. ONLY the new
    column/lengthscale initialize fresh; every other tensor loads 1:1.
    - inducing_points gain a lap column = the M evenly-spaced quantiles of the
      normalized lap feature over a seeded train-pixel sample (k-means-
      consistent: init_inducing_kmeans places centers over the empirical
      distribution of the normalized GP inputs, so the fresh dim spreads the
      same way; the assignment order is arbitrary — the ARD kernel only sees
      per-dim coordinate values).
    - raw_lengthscale gains one entry = the mean of the loaded raw ARD values
      (data-informed init, mobile from step one).
    Mutates sd in place; returns the manifest dict."""
    ip_key = 'gp.variational_strategy.inducing_points'
    ls_key = 'gp.covar_module.base_kernel.raw_lengthscale'
    d_ck = int(sd[ip_key].shape[-1])
    if d_ck != model.gp_input_dim - 1:
        raise RuntimeError(f"lap surgery: ckpt GP input dim {d_ck} != model "
                           f"{model.gp_input_dim} - 1 — not the +lap case")
    rng = np.random.default_rng([int(seed), 99])
    vals, have = [], 0
    for i in rng.permutation(len(train_ds)):
        s = train_ds[int(i)]
        v = s['lap'][s['mask']].numpy().astype(np.float64) / float(lap_scale)
        vals.append(v)
        have += v.size
        if have >= 20000:
            break
    vals = np.concatenate(vals)
    M = int(sd[ip_key].shape[0])
    q = np.quantile(vals, (np.arange(M) + 0.5) / M)
    lap_col = torch.from_numpy(q).to(sd[ip_key].dtype).reshape(M, 1)
    sd[ip_key] = torch.cat([sd[ip_key], lap_col], dim=-1)
    raw_new = sd[ls_key].mean(dim=-1, keepdim=True)
    sd[ls_key] = torch.cat([sd[ls_key], raw_new], dim=-1)
    return {'ckpt_gp_dim': d_ck, 'new_gp_dim': int(model.gp_input_dim),
            'lap_scale': float(lap_scale),
            'lap_col_q_range': [float(q[0]), float(q[-1])],
            'lap_raw_lengthscale_init': float(raw_new),
            'n_lap_sample_pixels': int(vals.size)}


@torch.no_grad()
def lap_init_probe(model, ref, val_ds, device, tol=1.0e-5):
    """I8-spirit init-exactness gate for the lap surgery: with the lap ARD raw
    lengthscale pushed to 1.0e6 (softplus is identity out there, so the lap
    dim contributes (dlap/1e6)^2 ~ 1e-11 to the kernel distance = round-off),
    the expanded GP must reproduce the pre-lap reference model's latent
    posterior mean AND variance on a fixed val batch. Run in float64 so the
    K_zz Cholesky's conditioning neither launders a genuine surgery bug into
    'round-off' nor amplifies round-off into a false failure. Hard-fails
    otherwise; the training-init lengthscale is restored either way."""
    b = next(batches(val_ds, 8))
    model.double().eval()
    ref.double().eval()
    try:
        x = b['x'].to(device).double()
        zeta = b['zeta'].to(device).double()
        mask = b['mask'].to(device)
        zd = b['zeta_dot'].to(device).double() if ref.use_zeta_dot else None
        g = b['g'].to(device).double() if ref.use_grad_feature else None
        lap = b['lap'].to(device).double()
        p_ref = ref.gp(ref.masked_gp_inputs(x, zeta, mask, zeta_dot=zd, g=g))
        ls = model.gp.covar_module.base_kernel.raw_lengthscale
        saved = ls.data[..., -1].clone()
        ls.data[..., -1] = 1.0e6
        p_new = model.gp(model.masked_gp_inputs(x, zeta, mask, zeta_dot=zd,
                                                g=g, lap=lap))
        ls.data[..., -1] = saved
        dmu = float((p_new.mean - p_ref.mean).abs().max()
                    / p_ref.mean.abs().max().clamp_min(1e-30))
        dvar = float((p_new.variance - p_ref.variance).abs().max()
                     / p_ref.variance.abs().max().clamp_min(1e-30))
    finally:
        model.float()
    if not (dmu < tol and dvar < tol):
        raise RuntimeError(
            f"lap init-exactness probe FAILED: rel dmu={dmu:.3e} "
            f"dvar={dvar:.3e} (tol {tol:.1e}) — the expanded GP does not "
            f"reproduce the ckpt with the lap dim inert; NOT training (I8)")
    return {'probe_rel_dmu': dmu, 'probe_rel_dvar': dvar,
            'probe_tol': float(tol), 'probe_n_pixels': int(p_ref.mean.numel())}


# ---- Student-t observation model (B-item, Sanaa 2026-07-14) --------------- #
# scipy is an existing pipeline dependency (dataset_piff / calibrate_piff /
# diagnose_piff); scipy.stats.t gives the exact per-pixel NLL and the central-
# coverage quantile. NLL/coverage under the t use the physical SCALE field
# (sqrt(het_noise)*y_sd) + the scalar learned nu — NOT the Gaussian z=1.
def student_t_nll(y, mu, scale, nu):
    """Per-pixel negative log-likelihood under StudentT(df=nu, loc=mu,
    scale=scale) in physical units."""
    from scipy.stats import t as _t
    return -_t.logpdf(y, df=float(nu), loc=mu, scale=scale)


def t_central_halfwidth(prob, nu):
    """Half-width (in SCALE units) of the central `prob` interval of the
    Student-t at nu: q such that P(|X| <= q) = prob."""
    from scipy.stats import t as _t
    return float(_t.ppf(0.5 + 0.5 * float(prob), df=float(nu)))


def t_coverage(y, mu, scale, nu, prob):
    """Empirical coverage of the central `prob` t-interval mu +/- q*scale."""
    q = t_central_halfwidth(prob, nu)
    return float(np.mean(np.abs(y - mu) <= q * scale))


@torch.no_grad()
def evaluate(model, ds, device, gp_chunk):
    """Predictive metrics on a (fixed) crop dataset: per-pixel NLL, RMSE, R2,
    mean predictive sigma, and the per-datum val ELBO surrogate (NLL)."""
    model.eval()
    student = model.is_student_t()
    ys, mus, vars_, scales, pvars, nvars = [], [], [], [], [], []
    for b in batches(ds, 8):
        x, y = b['x'].to(device), b['y'].to(device)
        mask, zeta = b['mask'].to(device), b['zeta'].to(device)
        gpin = model.masked_gp_inputs(x, zeta, mask, **cond_kwargs(model, b, device))
        yt = y[mask]
        gm = (b['g'].to(device)[mask] if model.noise_prior == 'structural'
              else None)
        for i0 in range(0, gpin.shape[0], gp_chunk):
            gm_c = gm[i0:i0 + gp_chunk] if gm is not None else None
            mu_p, var_p = model.predict_physical(gpin[i0:i0 + gp_chunk], g_masked=gm_c)
            mus.append(mu_p.cpu().numpy())
            vars_.append(var_p.cpu().numpy())
            if student:
                scales.append(model.student_scale(gm_c).cpu().numpy())
                # ratio diagnostic (std/GP space, consistent units): GP
                # posterior var vs the het_noise scale^2 the t-NLL uses. Verifies
                # the objective's noise-only scale omits a negligible post-var.
                pvars.append(model.gp(gpin[i0:i0 + gp_chunk]).variance.cpu().numpy())
                nvars.append(model.het_noise(gm_c).cpu().numpy())
        ys.append(yt.cpu().numpy())
    y = np.concatenate(ys); mu = np.concatenate(mus); var = np.concatenate(vars_)
    rmse = float(np.sqrt(np.mean((y - mu) ** 2)))
    r2 = float(1.0 - np.sum((y - mu) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-30))
    out = {'rmse': rmse, 'r2': r2, 'mean_sigma': float(np.mean(np.sqrt(var))),
           'n_pixels': int(y.size)}
    if student:
        scale = np.concatenate(scales); nu = model.student_nu()
        out['nll'] = float(np.mean(student_t_nll(y, mu, scale, nu)))   # model-selection metric
        out['coverage68'] = t_coverage(y, mu, scale, nu, 0.682689492)
        out['nu'] = float(nu); out['mean_scale'] = float(np.mean(scale))
        out['pv_ratio'] = float(np.mean(np.concatenate(pvars))
                                / max(np.mean(np.concatenate(nvars)), 1e-30))
    else:
        out['nll'] = float(np.mean(gaussian_nll(y, mu, var)))
    return out


@torch.no_grad()
def probe_feature_spread(model, probe, device):
    """Median pairwise feature distance on a fixed probe batch (Plan-B
    collapse diagnostic)."""
    model.eval()
    f = model.masked_gp_inputs(probe['x'].to(device), probe['zeta'].to(device),
                               probe['mask'].to(device),
                               **cond_kwargs(model, probe, device))
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
                                      b['mask'].to(device),
                                      **cond_kwargs(model, b, device))
        mu_t = model.gp(gpin).mean
        mu = (mu_t * model.y_sd + model.y_mu).cpu().numpy()   # physical units
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
    ap.add_argument('--init-ckpt', default=None,
                    help='warm start: load model state (strict=False; the ONLY '
                         'tolerated missing keys are the structural-noise '
                         'params/buffer — anything else hard-fails)')
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
    cstats = None
    lap_info = None
    if args.init_ckpt:
        wck = torch.load(args.init_ckpt, map_location='cpu', weights_only=False)
        sd = wck['model']
        # option-b lap warm-start surgery (Sanaa GO 2026-07-16): warm-starting
        # a +lap model from a pre-lap ckpt expands the two GP input-dim
        # tensors (inducing_points, raw_lengthscale) in place instead of
        # crashing on the shape mismatch. The pre-lap REFERENCE model is built
        # from the UNmutated dict first (init-probe target). lap_scale is
        # fresh (the ckpt predates the feature) and must be set BEFORE
        # sampling normalized lap values for the new inducing column.
        ck_lap = bool(wck.get('conf', {}).get('model', {})
                      .get('use_lap_feature', False))
        ref = None
        if model.use_lap_feature and not ck_lap:
            # G4 LOW-1 guard: the dim check alone would accept a ckpt whose
            # SHARED columns misalign (same count, different flags) — refuse
            ck_mc = wck.get('conf', {}).get('model', {})
            for fl in ('use_zeta_dot', 'use_grad_feature'):
                if bool(ck_mc.get(fl, False)) != bool(getattr(model, fl)):
                    raise RuntimeError(
                        f"lap surgery: ckpt {fl}={bool(ck_mc.get(fl, False))} "
                        f"!= model {bool(getattr(model, fl))} — shared GP "
                        f"columns would misalign")
            ref = PiffModel(wck['conf']).to(device)
            ref.load_state_dict(sd)
            cstats = conditioning_stats(runs, 'train', conf)
            model.lap_scale.fill_(float(cstats['lap_scale']))
            lap_info = lap_expand_state_dict(model, sd, cstats['lap_scale'],
                                             train_ds, seed)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # the ONLY tolerated missing keys: the structural-noise head (when warm-
        # starting a structural model from a non-structural ckpt), the fresh
        # lap_scale buffer (lap surgery path — set above, absent from pre-lap
        # ckpts) and, for the gaussian->student_t swap, the fresh Student-t
        # dof (the head itself — noise_a/noise_b/g2_scale — carries over 1:1
        # from the gaussian ckpt).
        allowed = {'noise_a', 'noise_b', 'g2_scale', 'lap_scale',
                   'likelihood.noise_covar.noise', 'likelihood.raw_nu'}
        bad_m = [k for k in missing if k not in allowed]
        bad_u = [k for k in unexpected
                 if k not in {'likelihood.noise_covar.raw_noise',
                              'likelihood.raw_noise',
                              'likelihood.noise_covar.raw_noise_constraint.lower_bound',
                              'likelihood.noise_covar.raw_noise_constraint.upper_bound'}]
        if bad_m or bad_u:
            raise RuntimeError(f"warm-start mismatch beyond the structural-"
                               f"noise keys: missing={bad_m} unexpected={bad_u}")
        print(f"[train] warm start from {args.init_ckpt} (epoch="
              f"{wck.get('epoch')}); fresh keys: {sorted(missing)}")
        if ref is not None:
            # I8-spirit gate: hard-fails the job (no training) on mismatch
            lap_info.update(lap_init_probe(model, ref, val_ds, device))
            del ref
            print(f"[train] lap surgery + init-exactness probe PASS: "
                  f"{json.dumps(lap_info)}")
    # conditioning normalization MUST precede the inducing k-means (G4 finding
    # 2026-07-13): the centers live in feature space, and the zeta_dot/grad
    # columns are built through the zdot_sd/g_scale buffers — identity buffers
    # at kmeans time would place the centers in raw units on those dims
    if args.init_ckpt:
        # WARM PATH (2026-07-13 night): the trained GP/inducing/standardization
        # ARE the model — no k-means re-init, no hyper reset, keep the ckpt's
        # recorded y_mu/y_sd/zdot_sd/g_scale (part of the weight contract; the
        # upstream mask changes the pool stats, the model must not re-scale).
        # Only the FRESH structural-noise feature scale is computed here.
        hyper0 = {'warm_start': str(args.init_ckpt)}
        if lap_info is not None:
            hyper0['lap_surgery'] = lap_info
        if model.noise_prior == 'structural':
            if cstats is None:
                cstats = conditioning_stats(runs, 'train', conf)
            hyper0.update(model.set_noise_feature_scale(cstats['g2_scale']))
            print(f"[train] structural-noise s_feat scale (fresh): "
                  f"g2_scale={cstats['g2_scale']:.6e}")
        if model.is_student_t():
            # Student-t dof init from a residual-kurtosis moment match
            # (excess-kurtosis of the t = 6/(nu-4) => nu = 4 + 6/kurt); the
            # warm GP gives an honest residual PDF. Clamp to [4.5, 8] (nu>2
            # already guaranteed by softplus+2); fallback 5.0 if kurt<=0.
            k = residual_kurtosis(model, train_ds, device)
            nu_init = float(np.clip(4.0 + 6.0 / k, 4.5, 8.0)) if k > 0 else 5.0
            model.set_student_nu(nu_init)
            hyper0['residual_excess_kurtosis'] = k
            hyper0['student_nu_init'] = nu_init
            print(f"[train] Student-t: residual excess kurtosis {k:.3e} -> "
                  f"nu_init {nu_init:.3f} (learnable)")
    else:
        if model.use_zeta_dot or model.use_grad_feature or model.use_lap_feature:
            cstats = conditioning_stats(runs, 'train', conf)
            cond_const = model.set_conditioning_stats(
                zdot_sd=cstats.get('zdot_sd'), g_scale=cstats.get('g_scale'),
                lap_scale=cstats.get('lap_scale'))
            if model.noise_prior == 'structural':
                cond_const.update(
                    model.set_noise_feature_scale(cstats['g2_scale']))
            print(f"[train] ORDER-3 conditioning stats: {json.dumps(cstats)}")
        npix = model.init_inducing_kmeans(train_ds, int(conf['model']['kmeans_pixels']),
                                          int(conf['model']['kmeans_iters']), seed, device=device)
        print(f"[train] inducing k-means init on {npix} pixels; M = {int(conf['model']['n_inducing'])}")

        # 2026-07-12 ruling: exact train-target stats at build -> recorded
        # invertible y-standardization (buffers, in every ckpt) + data-informed
        # hyperparameter init in the standardized space (var=1 -> conditioned K_zz;
        # the raw-space variant NaN'd on the float32 Cholesky, job 1830733)
        ystats = target_stats(runs, 'train', conf)
        std_const = model.set_y_standardization(ystats['mean'], ystats['var'])
        hyper0 = model.init_hyperparams_from_stats(
            0.0, 1.0, noise_frac=_f(tc['init_noise_frac']))
        hyper0.update(std_const)
        hyper0['stats_n_pixels'] = ystats['n']
        if model.use_zeta_dot or model.use_grad_feature or model.use_lap_feature:
            hyper0.update(cond_const)
            hyper0['conditioning_stats'] = cstats
    info['init_hyperparams'] = hyper0
    with open(outdir / 'run_info.yaml', 'w') as f:
        yaml.safe_dump(info, f, sort_keys=False)
    print(f"[train] y-standardization + GP init: {json.dumps(hyper0)}")

    mll = gpytorch.mlls.VariationalELBO(model.likelihood, model.gp, num_data=N)
    gp_named = {id(p) for p in model.gp.parameters()} | {id(p) for p in model.likelihood.parameters()}
    groups = [
        {'params': list(model.cnn.parameters()), 'weight_decay': wd},   # CNN only
        {'params': [p for p in model.parameters() if id(p) in gp_named],
         'weight_decay': 0.0},                                          # never GP hypers
    ]
    if model.noise_prior == 'structural':
        # a at base lr; b MOBILE at 10x lr (arm-F reviewer fix: b must be able
        # to move or the verdict is confounded by softplus saturation)
        groups.append({'params': [model.noise_a], 'weight_decay': 0.0})
        groups.append({'params': [model.noise_b], 'weight_decay': 0.0,
                       'lr': 10.0 * lr})
    opt = torch.optim.Adam(groups, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=int(tc['t0_restart']))

    bc = int(conf['data']['batch_crops'])
    gp_chunk = int(tc['gp_chunk'])
    probe = next(batches(val_ds, 8))                     # fixed collapse probe
    spread0, kurt = None, None
    hist = {k: [] for k in ('train_elbo', 'val_nll', 'val_rmse', 'val_r2',
                            'val_sigma', 'zeta_ls', 'zdot_ls', 'grad_ls',
                            'lap_ls', 'film_dgamma', 'film_beta', 'feat_spread',
                            'lr', 'val_cov68', 'val_nu', 'val_pv_ratio')}
    best_nll = np.inf

    for ep in range(epochs):
        t0 = time.time()
        train_ds.set_epoch(ep)
        model.train()
        elbos = []
        for b in batches(train_ds, bc):
            x, y = b['x'].to(device), b['y'].to(device)
            mask, zeta = b['mask'].to(device), b['zeta'].to(device)
            gpin = model.masked_gp_inputs(x, zeta, mask, **cond_kwargs(model, b, device))
            yt = y[mask]
            if yt.numel() == 0:
                continue
            opt.zero_grad(set_to_none=True)
            if model.noise_prior == 'structural':
                loss = -mll(model.gp(gpin), model.standardize_y(yt),
                            noise=model.het_noise(b['g'].to(device)[mask]))
            else:
                loss = -mll(model.gp(gpin), model.standardize_y(yt))  # per-datum (num_data=N), GP space
            loss.backward()
            opt.step()
            elbos.append(-float(loss))
        sched.step()

        vm = evaluate(model, val_ds, device, gp_chunk)
        spread = probe_feature_spread(model, probe, device)
        if spread0 is None:
            spread0 = spread
        dg, bnorm = model.film_norms()
        ards = model.ard_lengthscales()
        zls = ards['zeta']
        hist['train_elbo'].append(float(np.mean(elbos)))
        hist['val_nll'].append(vm['nll']); hist['val_rmse'].append(vm['rmse'])
        hist['val_r2'].append(vm['r2']); hist['val_sigma'].append(vm['mean_sigma'])
        hist['zeta_ls'].append(zls)
        hist['zdot_ls'].append(ards.get('zeta_dot', np.nan))
        hist['grad_ls'].append(ards.get('grad', np.nan))
        hist['lap_ls'].append(ards.get('lap', np.nan))
        hist['film_dgamma'].append(dg)
        hist['film_beta'].append(bnorm); hist['feat_spread'].append(spread)
        hist['lr'].append(opt.param_groups[0]['lr'])
        hist['val_cov68'].append(vm.get('coverage68', np.nan))
        hist['val_nu'].append(vm.get('nu', np.nan))
        hist['val_pv_ratio'].append(vm.get('pv_ratio', np.nan))

        collapse = spread < spread0 / 10.0
        extra_ls = ''
        if model.use_zeta_dot:
            extra_ls += f" zdot_ls {ards['zeta_dot']:.3f}"
        if model.use_grad_feature:
            extra_ls += f" grad_ls {ards['grad']:.3f}"
        if model.use_lap_feature:
            extra_ls += f" lap_ls {ards['lap']:.3f}"
        if model.noise_prior == 'structural':
            import torch.nn.functional as _F
            extra_ls += (f" a {float(_F.softplus(model.noise_a)):.4f}"
                         f" b {float(_F.softplus(model.noise_b)):.4f}")
        if model.is_student_t():
            extra_ls += (f" nu {vm['nu']:.3f} cov68 {vm['coverage68']:.4f}"
                         f" pv/nv {vm['pv_ratio']:.3e}")
        print(f"[ep {ep:03d}] elbo/datum {hist['train_elbo'][-1]:+.4e}  "
              f"val NLL {vm['nll']:.4e} RMSE {vm['rmse']:.4e} R2 {vm['r2']:.4f} "
              f"sigma {vm['mean_sigma']:.3e}  zeta_ls {zls:.3f}{extra_ls}  "
              f"film |dg|={dg:.3e} |b|={bnorm:.3e}  spread {spread:.3e}"
              f"{'  [PLAN-B SYMPTOM: feature collapse >10x]' if collapse else ''}"
              f"  ({time.time()-t0:.0f}s)")

        if ep == int(tc['kurtosis_after_epoch']):
            kurt = residual_kurtosis(model, train_ds, device)
            print(f"[train] residual-PDF excess kurtosis after warmup: {kurt:.3f}"
                  f"{'  -> RAISE B-ITEM (heteroscedastic / Student-t)' if kurt > 5 else ''}")

        state = {'model': model.state_dict(), 'conf': conf, 'seed': seed,
                 'epoch': ep, 'val': vm, 'ELBO_num_data': N,
                 'lr': lr, 'weight_decay': wd, 'kurtosis': kurt,
                 'init_hyperparams': hyper0}
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
                        'zeta_ard_lengthscale': hist['zeta_ls'][-1],
                        'ard_lengthscales_final': model.ard_lengthscales()}, f, sort_keys=False)
    print(f"[train] done; best val NLL {best_nll:.4e}; artifacts in {outdir}")


if __name__ == '__main__':
    main()
