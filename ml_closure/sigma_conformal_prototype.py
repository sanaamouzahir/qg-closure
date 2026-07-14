#!/usr/bin/env python
"""sigma_conformal_prototype.py -- stage 1 of the sigma master plan
(Sanaa 2026-07-14 evening: "come up with a way to fix ALL the issues
related to the sigma").

The finding stack this answers: cov68 ~0.92-0.93 on both finals; still
~0.96-0.98 at 1-sigma AFTER the NLL-optimal 3-param recal (tau_pv ~ 0);
sigma flat in the top gradient decile while error rises 8.8x. Diagnosis:
the residuals are heavy-tailed, so NO Gaussian-NLL fit can deliver nominal
coverage -- the NLL optimum over-widens the bulk to pay for the tails.

Fix by construction: SPLIT-CONFORMAL calibration, stratified by the
gradient feature (the known failure axis). On the FIT half of the val
window, per stratum s and nominal level p, compute the empirical quantile

    q_s(p) = Quantile_p( |r_i| / sigma_recal_i ),  i in stratum s

and report TEST-half coverage of the intervals  q_s(p) * sigma_recal.
Split-conformal guarantees test coverage -> p (exchangeability), no
distributional assumption, no retraining -- it composes on top of the
3-param recal and turns its good sigma RANKING into honest intervals.

Output: <ckpt dir>/conformal_calibration.yaml (the q_s table = the
deployable artifact) + one reliability figure per model.

Run (model forwards -> GPU-appropriate, same as the recal jobs):
  cd ml_closure && qsub -q ibgpu.q -l gpu=1 -N confP_<g> -o ../logs/... \
      -j y -cwd -V ../scripts/sge/piff_tool_job.sh sigma_conformal_prototype.py \
      --ckpt runs_piff/piff_<g>_gjs/best.pt --config conf_piff_<g>_gjs.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset_piff import load_conf, build_runs, split_frames, _f
from model_piff import PiffModel
from recalibrate_structural_sigma import collect_decomposed, total_var

HERE = Path(__file__).resolve().parent
LEVELS = {'68': 0.6827, '95': 0.9545, '99.7': 0.9973}
N_STRATA = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    ckpt = torch.load(HERE / args.ckpt, map_location='cpu', weights_only=False)
    conf = load_conf(HERE / args.config)
    device = args.device or conf['train']['device']
    gp_chunk = int(conf['train']['gp_chunk'])
    model = PiffModel(ckpt['conf']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    conf.setdefault('model', {})['use_grad_feature'] = model.use_grad_feature
    conf['zeta']['tshed_smooth'] = ckpt['conf'].get('zeta', {}).get('tshed_smooth', 2.992)

    rc_path = Path(HERE / args.ckpt).parent / 'recalibration_structural.yaml'
    rc = yaml.safe_load(rc_path.read_text())
    spa, spb = float(rc['softplus_a']), float(rc['softplus_b'])
    s_a, s_b = float(rc['s_a']), float(rc['s_b'])
    tau = float(rc.get('tau_pv', 1.0))
    print(f"[conformal] recal from {rc_path}: s_a={s_a:.4g} s_b={s_b:.4g} "
          f"tau_pv={tau:.4g}")

    runs = build_runs(conf)
    frames = split_frames(runs, 'val', conf)
    r, vgp, sfeat, t = collect_decomposed(model, runs, frames, device, gp_chunk)
    sig = np.sqrt(total_var(vgp, sfeat, spa, spb, s_a, s_b, tau))
    z = np.abs(r) / sig                                      # conformity scores

    t_lo, t_hi = _f(conf['data']['t_val_lo']), _f(conf['data']['t_val_hi'])
    t_mid = 0.5 * (t_lo + t_hi)
    fit_m, test_m = t < t_mid, t >= t_mid

    # strata by gradient feature: quantile bins defined on the FIT half
    edges = np.quantile(sfeat[fit_m], np.linspace(0, 1, N_STRATA + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    stra_fit = np.digitize(sfeat[fit_m], edges[1:-1])
    stra_test = np.digitize(sfeat[test_m], edges[1:-1])

    out = {'recal_source': str(rc_path), 'n_strata': N_STRATA,
           'strata_edges_gradfeat': [float(e) for e in edges[1:-1]],
           'q': {}, 'coverage_test': {}}
    print(f"{'level':>6} {'stratum':>8} {'q_s':>8}  test-cov (recal-only -> conformal)")
    for lname, p in LEVELS.items():
        qs = []
        for s in range(N_STRATA):
            zs = z[fit_m][stra_fit == s]
            # split-conformal finite-sample quantile
            k = int(np.ceil((len(zs) + 1) * p)) if len(zs) else 1
            qs.append(float(np.sort(zs)[min(k, len(zs)) - 1]) if len(zs) else np.nan)
        out['q'][lname] = qs
        # gaussian z for the same nominal level (recal-only reference)
        from scipy.special import erfinv
        zg = np.sqrt(2.0) * erfinv(p)
        cov0, cov1, per_s = [], [], []
        zt = z[test_m]
        for s in range(N_STRATA):
            m = stra_test == s
            if not m.any():
                per_s.append(None)
                continue
            c0 = float(np.mean(zt[m] <= zg))
            c1 = float(np.mean(zt[m] <= qs[s]))
            per_s.append({'recal_only': c0, 'conformal': c1})
            cov0.append(c0); cov1.append(c1)
            print(f"{lname:>6} {s:>8d} {qs[s]:>8.3f}  {c0:.4f} -> {c1:.4f}")
        glob0 = float(np.mean(zt <= zg))
        glob1 = float(np.mean(np.concatenate(
            [zt[stra_test == s] <= qs[s] for s in range(N_STRATA)
             if (stra_test == s).any()])))
        out['coverage_test'][lname] = {
            'nominal': p, 'global_recal_only': glob0,
            'global_conformal': glob1, 'per_stratum': per_s}
        print(f"{lname:>6} {'GLOBAL':>8} {'':>8}  {glob0:.4f} -> {glob1:.4f} "
              f"(nominal {p:.4f})")

    ypath = Path(HERE / args.ckpt).parent / 'conformal_calibration.yaml'
    ypath.write_text(yaml.safe_dump(out, sort_keys=False))
    print(f"[conformal] wrote {ypath.resolve()}")

    # reliability figure: nominal vs empirical, recal-only vs conformal
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    noms = [LEVELS[k] for k in LEVELS]
    ax.plot(noms, noms, 'k--', label='perfect')
    ax.plot(noms, [out['coverage_test'][k]['global_recal_only'] for k in LEVELS],
            'o-', label='3-param recal only')
    ax.plot(noms, [out['coverage_test'][k]['global_conformal'] for k in LEVELS],
            's-', label='+ stratified conformal')
    ax.set_xlabel('nominal coverage'); ax.set_ylabel('empirical (held-out half)')
    ax.set_title('coverage fixed by construction (split-conformal)')
    ax.legend()
    fig.tight_layout()
    fdir = HERE / 'pngs' / 'sigma_conformal_prototype'
    fdir.mkdir(parents=True, exist_ok=True)
    tag = Path(args.ckpt).parent.name
    fp = fdir / f'conformal_reliability_{tag}.png'
    fig.savefig(fp, dpi=140)
    print(f"[conformal] wrote {fp.resolve()}")


if __name__ == '__main__':
    main()
