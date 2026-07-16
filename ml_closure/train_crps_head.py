#!/usr/bin/env python
"""train_crps_head.py -- sigma master plan STAGE 2 (Sanaa full approval
2026-07-14 night): a RICHER aleatoric variance head trained with CRPS on the
FROZEN mean model's residuals.

Why (the finding stack): the structural head sigma^2 = sp(a) + sp(b)*g^2 is a
2-dof affine in g^2 -- it responds 2.7-4.2x in the top gradient decile while
|error| rises 8.8x (saturates), and the 3-param recal drove tau_pv to ~0.01,
i.e. intervals should come from the ALEATORIC head, not the GP posterior.
Stage 2 replaces the affine head with a small MLP on per-pixel features
[g, sdf*, zeta] trained with the closed-form Gaussian CRPS -- a proper scoring
rule that is far more robust to the heavy-tailed residuals than NLL (NLL's
optimum over-widens the bulk to pay for the tails; CRPS grows linearly in the
tail, not quadratically). Stage-1 stratified conformal
(sigma_conformal_prototype.py) then composes ON TOP of this head's sigma.

Freeze contract: the gjs checkpoint (mean + GP + FiLM-CNN + structural head)
is loaded eval() + requires_grad_(False); residuals and features are collected
under torch.no_grad() ONCE and cached in memory -- no gradient can reach the
base model by construction. best.pt is NEVER modified; the head is a separate
sidecar ckpt crps_head.pt.

Head: MLP 3 -> 32 -> 32 -> 1, softplus output + floor 1.0e-3, in STANDARDIZED
sigma units. Features (raw, transform inside forward):
    s_feat = g^2 / g2_scale   (the SAME standardization the structural head
                               uses; log1p inside the head conditions the
                               heavy tail -- monotone, so the post-hoc
                               monotonicity-in-g check is unaffected)
    sdf*                      (the clipped SDF plane, x channel 3)
    zeta                      (frame conditioning scalar)
Monotonicity in g is NOT hard-constrained (Sanaa spec); it is checked
post-hoc in the eval report.

Loss: CRPS of N(0, sigma^2) at residual r = y_std - gp_mean, closed form
    CRPS = sigma * ( z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi) ),  z = r/sigma
mean over TRAIN-split valid pixels (same masking as training). float32
throughout (spec S0 -- this pipeline, unlike the temporal branch, has no
catastrophic cancellation).

Honesty convention (same as recalibrate/calibrate/conformal): the val window
is split at t_mid; the FIT half (t < t_mid) is the per-epoch monitoring set;
the honesty half (t >= t_mid) is touched ONLY by the final eval report.

Outputs (next to the base ckpt; nothing existing is overwritten):
    runs_piff/<run>/crps_head.pt        head state + feature spec + y_sd + log
    runs_piff/<run>/crps_head_eval.yaml honesty-half report (arm-F staircase)
    pngs/crps_head/<run>/               staircase figure
--run-smoke suffixes every artifact with _smoke so a later production run
collides with nothing.

Usage (GPU -- model forwards for residual collection):
  cd ml_closure && qsub -q ibgpu.q -l gpu=1 -N crps_<g> \
      -o ../logs/crps_<g>.\$JOB_ID.log -j y -cwd -V \
      ../scripts/sge/piff_tool_job.sh train_crps_head.py \
      --ckpt runs_piff/piff_<g>_gjs/best.pt --config conf_piff_<g>_gjs.yaml
  --eval-only reuses a saved head (no retraining);
  --self-test runs the CPU formula/shape checks, no data needed.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset_piff import load_conf, build_runs, split_frames, _f
from model_piff import PiffModel

HERE = Path(__file__).resolve().parent
FLOOR = 1.0e-3                      # sigma floor, standardized units (spec)
CLAMP_LO, CLAMP_HI = 1.0e-3, 10.0   # arm-F VARIANCE band (baseline head only)


# ---------------------------------------------------------------- CRPS ----- #
def crps_gaussian(r, sigma):
    """Closed-form CRPS of N(0, sigma^2) evaluated at r (torch, elementwise):
    sigma * ( z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi) ),  z = r/sigma."""
    z = r / sigma
    Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    phi = torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


def gauss_nll_std(r, sigma):
    """Per-pixel Gaussian NLL in standardized units (numpy). Physical NLL =
    this + log(y_sd) -- the recalibrate_structural_sigma.py convention."""
    v = sigma ** 2
    return 0.5 * (np.log(2.0 * np.pi * v) + r ** 2 / v)


# ---------------------------------------------------------------- head ----- #
class CRPSHead(nn.Module):
    """sigma_std(x) = softplus(MLP([log1p(s_feat), sdf*, zeta])) + FLOOR.
    Input is the RAW feature triple [s_feat = g^2/g2_scale, sdf*, zeta];
    the log1p lives inside forward so the saved feature spec is unambiguous."""

    def __init__(self, width=32, floor=FLOOR):
        super().__init__()
        self.floor = float(floor)
        self.width = int(width)
        self.net = nn.Sequential(
            nn.Linear(3, self.width), nn.SiLU(),
            nn.Linear(self.width, self.width), nn.SiLU(),
            nn.Linear(self.width, 1))

    def forward(self, feats):
        x = torch.stack([torch.log1p(feats[..., 0].clamp_min(0.0)),
                         feats[..., 1], feats[..., 2]], dim=-1)
        return nn.functional.softplus(self.net(x).squeeze(-1)) + self.floor

    @torch.no_grad()
    def init_output_scale(self, sigma0):
        """Final-layer bias so sigma(init) ~ sigma0 (default-init weights are
        small, so the pre-softplus output ~ bias). Keeps epoch 0 sane instead
        of starting at softplus(0)+floor ~ 0.69."""
        s = max(float(sigma0) - self.floor, 1.0e-4)
        self.net[-1].bias.fill_(math.log(math.expm1(s)))


# ------------------------------------------------------------- collect ----- #
@torch.no_grad()
def collect(model, runs, frames, device, gp_chunk, per_frame_cap=None,
            seed=0, tag=''):
    """Walk frames through the FROZEN model (collect_decomposed conventions:
    full_frame, masked pixels, standardized residual r = y_std - gp_mean).
    Returns float32 numpy dict: sfeat (g^2/g2_scale), sdf (sdf* plane), zeta,
    r, t. per_frame_cap subsamples pixels per frame (seeded) BEFORE the GP
    call -- bounds both memory and GP cost for the big train split."""
    rng = np.random.default_rng([int(seed), 7])
    y_mu, y_sd = float(model.y_mu), float(model.y_sd)
    g2s = float(model.g2_scale)
    out = {k: [] for k in ('sfeat', 'sdf', 'zeta', 'r', 't')}
    t0 = time.time()
    for n, (ri, fi) in enumerate(frames):
        run = runs[ri]
        x, y, mask, zeta, zeta_dot, g, lap = run.full_frame(fi)
        xm, mk = x[None].to(device), mask[None].to(device)
        gpin = model.masked_gp_inputs(
            xm, zeta[None].to(device), mk,
            zeta_dot=(zeta_dot[None].to(device) if model.use_zeta_dot else None),
            g=g[None].to(device),
            lap=(lap[None].to(device) if getattr(model, 'use_lap_feature', False) else None))
        gm = g[None].to(device)[mk]                       # (P,) masked grad
        sdf_m = xm[:, 3][mk]                              # (P,) masked sdf*
        y_std = (y.numpy()[mask.numpy()].astype(np.float64) - y_mu) / y_sd
        P = gpin.shape[0]
        if per_frame_cap is not None and P > per_frame_cap:
            sel = np.sort(rng.choice(P, size=int(per_frame_cap), replace=False))
            sel_t = torch.from_numpy(sel).to(device)
            gpin, gm, sdf_m = gpin[sel_t], gm[sel_t], sdf_m[sel_t]
            y_std = y_std[sel]
        mus = []
        for i0 in range(0, gpin.shape[0], gp_chunk):
            mus.append(model.gp(gpin[i0:i0 + gp_chunk]).mean
                       .double().cpu().numpy())
        out['r'].append((y_std - np.concatenate(mus)).astype(np.float32))
        out['sfeat'].append((gm.double().cpu().numpy() ** 2 / g2s)
                            .astype(np.float32))
        out['sdf'].append(sdf_m.float().cpu().numpy())
        out['zeta'].append(np.full(y_std.shape, float(zeta), dtype=np.float32))
        out['t'].append(np.full(y_std.shape, float(run.times[fi]),
                                dtype=np.float32))
    out = {k: np.concatenate(v) for k, v in out.items()}
    print(f"[collect{tag}] {len(frames)} frames -> {out['r'].size} px "
          f"({time.time() - t0:.0f}s)")
    return out


def to_feats(d, device):
    """(P,3) raw feature tensor [s_feat, sdf*, zeta] on device."""
    return torch.from_numpy(
        np.stack([d['sfeat'], d['sdf'], d['zeta']], axis=1)).to(device)


@torch.no_grad()
def head_sigma(head, feats, batch=1 << 20):
    """Chunked head forward -> sigma_std numpy float64."""
    out = []
    for i0 in range(0, feats.shape[0], batch):
        out.append(head(feats[i0:i0 + batch]).double().cpu().numpy())
    return np.concatenate(out)


def structural_sigma(sfeat, spa, spb):
    """The CURRENT (stage-0) structural head, for the baseline column of the
    arm-F staircase: sigma = sqrt(clamp(sp(a) + sp(b)*s_feat, band))."""
    return np.sqrt(np.clip(spa + spb * sfeat.astype(np.float64),
                           CLAMP_LO, CLAMP_HI))


def coverage(r, sigma, k):
    return float(np.mean(np.abs(r) <= k * sigma))


def val_metrics(head, feats, r_np, y_sd):
    """Monitoring triple on a cached split: CRPS (std units), NLL (physical
    convention: std NLL + log y_sd), cov1."""
    sig = head_sigma(head, feats)
    r = r_np.astype(np.float64)
    z = torch.from_numpy(r)
    crps = float(crps_gaussian(z, torch.from_numpy(sig)).mean())
    return {'crps_std': crps,
            'nll': float(np.mean(gauss_nll_std(r, sig)) + np.log(y_sd)),
            'cov1': coverage(r, sig, 1.0),
            'sigma_med': float(np.median(sig))}


# ------------------------------------------------------------ self-test ---- #
def self_test():
    """CPU-only checks, no data: (1) closed-form CRPS == numerical quadrature;
    (2) head floor/shape; (3) CRPS is minimized near sigma ~ |r| spread."""
    torch.manual_seed(0)
    # (1) quadrature check: CRPS = int (Phi(x/s) - 1[x>=r])^2 dx
    for r0, s0 in [(0.3, 0.5), (-2.0, 1.3), (0.0, 0.05), (8.0, 0.2)]:
        xs = np.linspace(-60, 60, 2_000_001)
        F = 0.5 * (1 + np.vectorize(math.erf)(xs / (s0 * math.sqrt(2))))
        num = np.trapezoid((F - (xs >= r0)) ** 2, xs)
        cf = float(crps_gaussian(torch.tensor(r0, dtype=torch.float64),
                                 torch.tensor(s0, dtype=torch.float64)))
        # trapezoid error at the indicator's step is dx/2*|2Phi(z)-1| ~ 1.4e-5
        # at this grid -- the tolerance covers exactly that, nothing looser
        assert abs(num - cf) < 5e-5, (r0, s0, num, cf)
    # (2) head output positive, floored, right shape
    h = CRPSHead()
    h.init_output_scale(0.3)
    f = torch.randn(1000, 3).abs()
    s = h(f)
    assert s.shape == (1000,) and float(s.min()) >= FLOOR
    assert abs(float(s.mean()) - 0.3) < 0.2, float(s.mean())
    n_par = sum(p.numel() for p in h.parameters())
    # (3) CRPS optimum: for r ~ N(0, v) the minimizing constant sigma is
    # sqrt(v) -- check the loss curve dips there
    r = torch.randn(200_000)
    ls = [float(crps_gaussian(r, torch.full_like(r, s)).mean())
          for s in (0.5, 1.0, 2.0)]
    assert ls[1] < ls[0] and ls[1] < ls[2], ls
    print(f"[self-test] PASS  (closed-form CRPS == quadrature to step-"
          f"discontinuity tolerance 5e-5; head "
          f"params={n_par}, floor {FLOOR}, sigma>0; CRPS minimized at true "
          f"sigma)")


# ------------------------------------------------------------------ main --- #
def main():
    ap = argparse.ArgumentParser(description="STAGE-2 CRPS aleatoric head on a "
                                             "frozen gjs mean model")
    ap.add_argument('--ckpt', help='frozen base ckpt (runs_piff/<run>/best.pt)')
    ap.add_argument('--config', help='matching conf_piff_<g>_gjs.yaml')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--lr', type=float, default=1.0e-3)
    ap.add_argument('--batch-pixels', type=int, default=262144)
    ap.add_argument('--max-train-pixels', type=int, default=20_000_000,
                    help='seeded per-frame subsample cap over the train split')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default=None)
    ap.add_argument('--eval-only', action='store_true',
                    help='load crps_head.pt, honesty-half report only')
    ap.add_argument('--run-smoke', action='store_true',
                    help='few frames, few pixels, _smoke-suffixed artifacts')
    ap.add_argument('--self-test', action='store_true',
                    help='CPU formula/shape checks, no data, then exit')
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.ckpt or not args.config:
        raise SystemExit("--ckpt and --config are required (or --self-test)")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- frozen base model (no grads anywhere into it, by construction) --- #
    ckpt = torch.load(HERE / args.ckpt, map_location='cpu', weights_only=False)
    conf = load_conf(HERE / args.config)
    device = args.device or conf['train']['device']
    gp_chunk = int(conf['train']['gp_chunk'])
    model = PiffModel(ckpt['conf']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    model.requires_grad_(False)
    assert not any(p.requires_grad for p in model.parameters())
    if model.noise_prior != 'structural' or not model.use_grad_feature:
        raise SystemExit("STAGE 2 expects a structural-noise gjs ckpt "
                         "(needs g + g2_scale)")
    # same conf injections as sigma_conformal_prototype.py (must precede
    # build_runs: need_grad and zeta_dot smoothing are set at RunData load)
    conf.setdefault('model', {})['use_grad_feature'] = model.use_grad_feature
    conf['zeta']['tshed_smooth'] = ckpt['conf'].get('zeta', {}).get(
        'tshed_smooth', 2.992)

    spa = float(torch.nn.functional.softplus(model.noise_a))
    spb = float(torch.nn.functional.softplus(model.noise_b))
    y_sd = float(model.y_sd)
    g2s = float(model.g2_scale)

    run_dir = Path(HERE / args.ckpt).parent
    run_name = run_dir.name
    sfx = '_smoke' if args.run_smoke else ''
    head_path = run_dir / f'crps_head{sfx}.pt'
    eval_path = run_dir / f'crps_head_eval{sfx}.yaml'
    fig_dir = HERE / 'pngs' / 'crps_head' / run_name
    print(f"[crps] frozen base {args.ckpt} (epoch {ckpt.get('epoch')}); "
          f"sp(a)={spa:.4g} sp(b)={spb:.4g} y_sd={y_sd:.4g} g2_scale={g2s:.4g}")

    # ---- splits (honesty convention identical to recal/conformal) --------- #
    runs = build_runs(conf)
    train_frames = split_frames(runs, 'train', conf)
    val_frames = split_frames(runs, 'val', conf)
    t_lo, t_hi = _f(conf['data']['t_val_lo']), _f(conf['data']['t_val_hi'])
    t_mid = 0.5 * (t_lo + t_hi)
    fit_frames = [(ri, fi) for ri, fi in val_frames
                  if runs[ri].times[fi] < t_mid]
    hon_frames = [(ri, fi) for ri, fi in val_frames
                  if runs[ri].times[fi] >= t_mid]
    if not fit_frames or not hon_frames:
        raise SystemExit(f"val window does not straddle t_mid={t_mid:.2f}")

    max_train_px = int(args.max_train_pixels)
    if args.run_smoke:
        def stride(fr, cap):
            return fr[::max(1, len(fr) // cap)][:cap]
        train_frames = stride(train_frames, 8)
        fit_frames = stride(fit_frames, 4)
        hon_frames = stride(hon_frames, 4)
        max_train_px = min(max_train_px, 2_000_000)
        print(f"[smoke] frames: train {len(train_frames)}, fit "
              f"{len(fit_frames)}, honesty {len(hon_frames)}; "
              f"train px cap {max_train_px}")
    print(f"[crps] frames: train {len(train_frames)}  val-fit "
          f"{len(fit_frames)} (t<{t_mid:.1f})  honesty {len(hon_frames)} "
          f"(t>={t_mid:.1f})")

    head = CRPSHead().to(device)
    hist = {k: [] for k in ('train_crps', 'val_crps', 'val_nll', 'val_cov1',
                            'val_sigma_med')}
    best_ep = None

    if args.eval_only:
        if not head_path.exists():
            raise SystemExit(f"--eval-only: {head_path} missing")
        hck = torch.load(head_path, map_location='cpu', weights_only=False)
        head.load_state_dict(hck['head'])
        head.eval()
        best_ep = hck.get('best_epoch')
        print(f"[crps] eval-only: loaded {head_path} (best epoch {best_ep})")
    else:
        # ---- residual collection (frozen model, no_grad, cached once) ----- #
        cap = max(1, max_train_px // max(len(train_frames), 1))
        tr = collect(model, runs, train_frames, device, gp_chunk,
                     per_frame_cap=cap, seed=args.seed, tag='/train')
        vf = collect(model, runs, fit_frames, device, gp_chunk, tag='/val-fit')
        # ROBUST output init (smoke finding, job 1833850): the residuals are
        # so heavy-tailed that std(r) ~ 200x median|r| -- initializing at the
        # std parks the whole bulk 2 orders of magnitude too wide and 2-30
        # epochs of shared-bias travel can't recover. MAD-consistent scale
        # (1.4826*median|r|) starts the bulk right; the tail pixels' CRPS
        # gradient (-sigma/sqrt(pi) for z>>1) then widens sigma where s_feat
        # is large -- that direction has strong signal.
        r64 = tr['r'].astype(np.float64)
        sigma0 = float(np.clip(1.4826 * np.median(np.abs(r64)), 2e-3, 5.0))
        head.init_output_scale(sigma0)
        print(f"[crps] train residual std {r64.std():.4f}, MAD-scale "
              f"{sigma0:.5f} (std units) -> head output init; head params "
              f"{sum(p.numel() for p in head.parameters())}")

        feats_tr = to_feats(tr, device)
        r_tr = torch.from_numpy(tr['r']).to(device)
        feats_vf = to_feats(vf, device)
        opt = torch.optim.Adam(head.parameters(), lr=float(args.lr))
        best_crps, best_state = np.inf, None
        n = r_tr.shape[0]
        bp = int(args.batch_pixels)
        for ep in range(int(args.epochs)):
            te = time.time()
            head.train()
            perm = torch.randperm(n, device=device)
            tot, cnt = 0.0, 0
            for i0 in range(0, n, bp):
                sel = perm[i0:i0 + bp]
                opt.zero_grad(set_to_none=True)
                loss = crps_gaussian(r_tr[sel], head(feats_tr[sel])).mean()
                loss.backward()
                opt.step()
                tot += float(loss) * sel.numel()
                cnt += sel.numel()
            head.eval()
            vm = val_metrics(head, feats_vf, vf['r'], y_sd)
            hist['train_crps'].append(tot / cnt)
            hist['val_crps'].append(vm['crps_std'])
            hist['val_nll'].append(vm['nll'])
            hist['val_cov1'].append(vm['cov1'])
            hist['val_sigma_med'].append(vm['sigma_med'])
            print(f"[ep {ep:03d}] train CRPS(std) {tot / cnt:.5f}  val-fit "
                  f"CRPS {vm['crps_std']:.5f} NLL {vm['nll']:.4f} cov1 "
                  f"{vm['cov1']:.4f} sigma_med {vm['sigma_med']:.4f}  "
                  f"({time.time() - te:.0f}s)")
            if vm['crps_std'] < best_crps:
                best_crps, best_ep = vm['crps_std'], ep
                best_state = copy.deepcopy(head.state_dict())
        head.load_state_dict(best_state)
        head.eval()
        torch.save({'head': head.state_dict(),
                    'arch': {'in_dim': 3, 'width': head.width,
                             'hidden_layers': 2, 'floor': head.floor,
                             'act': 'silu',
                             'output': 'softplus + floor, sigma in '
                                       'STANDARDIZED units'},
                    'feature_spec': [
                        's_feat = g^2 / g2_scale (raw; log1p inside forward)',
                        'sdf_star (input channel 3)',
                        'zeta (frame conditioning scalar)'],
                    'y_sd': y_sd, 'g2_scale': g2s,
                    'softplus_a': spa, 'softplus_b': spb,
                    'base_ckpt': str(args.ckpt), 'config': str(args.config),
                    'loss': 'gaussian CRPS closed form',
                    'epochs': int(args.epochs), 'lr': float(args.lr),
                    'seed': int(args.seed), 'smoke': bool(args.run_smoke),
                    'n_train_pixels': int(n), 'sigma_init': sigma0,
                    'best_epoch': int(best_ep),
                    'best_val_fit_crps_std': float(best_crps),
                    'epoch_log': hist},
                   head_path)
        print(f"[crps] saved {head_path} (best epoch {best_ep}, val-fit "
              f"CRPS {best_crps:.5f}); base best.pt untouched")

    # ---- honesty-half report (the arm-F staircase) ------------------------ #
    hn = collect(model, runs, hon_frames, device, gp_chunk, tag='/honesty')
    feats_hn = to_feats(hn, device)
    sig = head_sigma(head, feats_hn)                       # std units
    sig_base = structural_sigma(hn['sfeat'], spa, spb)     # current head
    r = hn['r'].astype(np.float64)
    crps_std = float(crps_gaussian(torch.from_numpy(r),
                                   torch.from_numpy(sig)).mean())
    rep = {
        'base_ckpt': str(args.ckpt), 'head_ckpt': str(head_path),
        'smoke': bool(args.run_smoke), 'best_epoch': best_ep,
        'honesty_window_t': [float(t_mid), float(t_hi)],
        'n_pixels': int(r.size), 'y_sd': y_sd,
        'crps_std': crps_std, 'crps_physical': crps_std * y_sd,
        'nll': float(np.mean(gauss_nll_std(r, sig)) + np.log(y_sd)),
        'cov1': coverage(r, sig, 1.0), 'cov2': coverage(r, sig, 2.0),
        'cov3': coverage(r, sig, 3.0),
        'baseline_structural_head': {
            'nll': float(np.mean(gauss_nll_std(r, sig_base)) + np.log(y_sd)),
            'cov1': coverage(r, sig_base, 1.0),
            'cov2': coverage(r, sig_base, 2.0),
            'cov3': coverage(r, sig_base, 3.0),
            'crps_std': float(crps_gaussian(
                torch.from_numpy(r), torch.from_numpy(sig_base)).mean())},
    }

    # per-gradient-decile staircase (recal decile conventions; physical units)
    edges = np.quantile(hn['sfeat'], np.linspace(0, 1, 11))
    dec = {'grad_feature_hi': [], 'sigma_med_crps': [],
           'sigma_med_structural': [], 'abserr_med': [], 'cov1_crps': []}
    for i in range(10):
        m = (hn['sfeat'] >= edges[i]) & (
            hn['sfeat'] <= edges[i + 1] if i == 9 else hn['sfeat'] < edges[i + 1])
        if not m.any():                     # zero-inflated g -> dup edges
            continue
        dec['grad_feature_hi'].append(float(edges[i + 1]))
        dec['sigma_med_crps'].append(float(np.median(sig[m]) * y_sd))
        dec['sigma_med_structural'].append(float(np.median(sig_base[m]) * y_sd))
        dec['abserr_med'].append(float(np.median(np.abs(r[m])) * y_sd))
        dec['cov1_crps'].append(coverage(r[m], sig[m], 1.0))
    rep['honesty_gradient_deciles'] = dec
    sm = np.array(dec['sigma_med_crps'])
    rep['monotone_fraction_sigma_vs_grad_decile'] = float(
        np.mean(np.diff(sm) > 0)) if sm.size > 1 else None
    rep['sigma_response_top_over_bottom_decile'] = (
        float(sm[-1] / sm[0]) if sm.size > 1 else None)
    rep['abserr_top_over_bottom_decile'] = (
        float(dec['abserr_med'][-1] / dec['abserr_med'][0])
        if len(dec['abserr_med']) > 1 else None)
    try:
        from scipy.stats import spearmanr
        k = min(r.size, 2_000_000)
        idx = np.random.default_rng(args.seed).choice(r.size, k, replace=False)
        rep['spearman_sigma_abserr'] = float(
            spearmanr(sig[idx], np.abs(r[idx])).statistic)
    except Exception as e:                  # report must not die on scipy
        rep['spearman_sigma_abserr'] = f'unavailable: {e}'

    eval_path.write_text(yaml.safe_dump(rep, sort_keys=False))
    print(f"[crps] honesty half: CRPS(std) {crps_std:.5f}  NLL {rep['nll']:.4f}"
          f" (baseline {rep['baseline_structural_head']['nll']:.4f})  cov1/2/3 "
          f"{rep['cov1']:.4f}/{rep['cov2']:.4f}/{rep['cov3']:.4f}")
    print(f"[crps] staircase (physical): sigma_med "
          f"{['%.3g' % v for v in dec['sigma_med_crps']]}")
    print(f"[crps]                        abserr_med "
          f"{['%.3g' % v for v in dec['abserr_med']]}")
    print(f"[crps]              structural sigma_med "
          f"{['%.3g' % v for v in dec['sigma_med_structural']]}")
    print(f"[crps] sigma top/bottom decile response "
          f"{rep['sigma_response_top_over_bottom_decile']} vs abserr "
          f"{rep['abserr_top_over_bottom_decile']}; monotone frac "
          f"{rep['monotone_fraction_sigma_vs_grad_decile']}")
    print(f"[crps] wrote {eval_path}")

    # staircase figure (one axes, 3 series, distinct markers so identity is
    # not color-alone; recessive grid -- repo staircase-figure conventions)
    fig_dir.mkdir(parents=True, exist_ok=True)
    xs = np.arange(1, len(dec['sigma_med_crps']) + 1)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.semilogy(xs, dec['abserr_med'], color='black', ls='--', marker='^',
                label='median |error| (target)')
    ax.semilogy(xs, dec['sigma_med_structural'], color='#c26a2d', marker='o',
                label='median sigma, structural head (stage 0)')
    ax.semilogy(xs, dec['sigma_med_crps'], color='#2d6ac2', marker='s',
                label='median sigma, CRPS MLP head (stage 2)')
    ax.set_xlabel('|grad omega| feature decile (honesty half)')
    ax.set_ylabel('physical units')
    ax.set_title(f'sigma staircase vs error staircase, {run_name}'
                 f'{" [SMOKE]" if args.run_smoke else ""}')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fp = fig_dir / (f'sigma_staircase_crps_head_vs_structural_head_'
                    f'{run_name}{sfx}.png')
    fig.savefig(fp, dpi=130)
    plt.close(fig)
    print(f"[crps] wrote {fp.resolve()}")


if __name__ == '__main__':
    main()
