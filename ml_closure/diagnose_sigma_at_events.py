#!/usr/bin/env python
"""diagnose_sigma_at_events.py -- does the model KNOW it misses the near-wall
peaks? (Sanaa order 2026-07-16.) At the top-K extreme-event pixels: predicted
sigma vs |truth| and vs member RMSE.
    z_true = |truth - mu| / sigma   (how many predicted sigmas away truth is)
KNOWS-IT (inducing/coverage fix): median z_true <~ 3 and sigma >> RMSE there.
CONFIDENTLY-WRONG (feature blindness): median z_true >> 3 with sigma ~ RMSE.
CPU, one forward per unique frame. Outputs yaml + report push.
Usage: python diagnose_sigma_at_events.py --ckpt <best.pt> --config <conf>
    --member <M> --events <extreme_events.csv> [--report-run sigma_events_<g>]
"""
from __future__ import annotations
import argparse, csv, subprocess, sys
from pathlib import Path
import numpy as np, torch, yaml
from dataset_piff import load_conf, build_runs
from member_naming import modulation_name
from model_piff import PiffModel
from eval_piff import predict_frame

HERE = Path(__file__).resolve().parent
BRANCH_ROOT = HERE.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--member', required=True)
    ap.add_argument('--events', required=True)
    ap.add_argument('--top-k', type=int, default=50)
    ap.add_argument('--gp-chunk', type=int, default=200_000)
    ap.add_argument('--outdir', default=None,
                    help='output directory (STANDARD tree passthrough); '
                         'default: <ckpt dir>/sigma_at_events/')
    ap.add_argument('--plain-member-names', action='store_true',
                    help='write into <outdir>/<modulation>/{metrics.yaml,'
                         'per_event_rows.csv} per the STANDARD tree; default '
                         'keeps <outdir>/<member>.yaml + <member>_rows.csv')
    ap.add_argument('--pool-members', default='',
                    help='space-separated member codenames of the pool '
                         '(telS-A vs tel disambiguation, G4 finding 3)')
    ap.add_argument('--report-run', default=None)
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    conf = load_conf(HERE / args.config)
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    model = PiffModel(ck['conf'])
    model.load_state_dict(ck['model']); model.eval()
    conf.setdefault('model', {})['use_grad_feature'] = model.use_grad_feature
    v = ck['conf'].get('data', {}).get('variant')
    if v and not conf['data'].get('variant'):
        conf['data']['variant'] = v
    conf.setdefault('zeta', {})['tshed_smooth'] = float(
        ck['conf'].get('zeta', {}).get('tshed_smooth', 2.992))
    run = next(r for r in build_runs(conf) if r.name == args.member)

    with open(args.events) as f:
        evs = [{k: (int(x) if k == 'frame' else float(x)) for k, x in r.items()}
               for r in list(csv.DictReader(f))[:args.top_k]]
    frames = sorted({e['frame'] for e in evs})
    sig_all, z_all, rows = [], [], []
    rmse_ref = None
    with torch.no_grad():
        for fi in frames:
            p = predict_frame(model, run, fi, 'cpu', args.gp_chunk)
            if rmse_ref is None:
                e = p['mu'] - p['y']
                rmse_ref = float(np.sqrt(np.mean(e ** 2)))
            for ev in (e2 for e2 in evs if e2['frame'] == fi):
                iy = int(round(ev['y'] / run.dy)) % run.Ny
                ix = int(round(ev['x'] / run.dx)) % run.Nx
                s = float(p['sigma2d'][iy, ix])
                mu = float(p['mu2d'][iy, ix])
                z = abs(ev['truth'] - mu) / max(s, 1e-30)
                sig_all.append(s); z_all.append(z)
                rows.append({'frame': fi, 'truth': ev['truth'], 'mu': mu,
                             'sigma': s, 'z_true': z})
            print(f'[sigma] frame {fi} done', flush=True)
    med_s, med_z = float(np.median(sig_all)), float(np.median(z_all))
    verdict = ('KNOWS-IT: truth within ~3 predicted sigma at the events -- '
               'fix = inducing coverage / near-wall capacity'
               if med_z <= 3.0 else
               'CONFIDENTLY-WRONG: truth is %.0f sigma away (median) -- '
               'features cannot distinguish these states; fix = near-wall '
               'features' % med_z)
    out = {'member': args.member, 'n_events': len(rows),
           'rmse_frame_ref': rmse_ref,
           'sigma_median_at_events': med_s,
           'sigma_over_rmse_median': med_s / max(rmse_ref, 1e-30),
           'z_true_median': med_z,
           'z_true_p90': float(np.percentile(z_all, 90)),
           'verdict': verdict}
    od = Path(args.outdir) if args.outdir else ckpt.parent / 'sigma_at_events'
    if args.plain_member_names:      # STANDARD tree: modulation-named subdir
        out['member_modulation'] = modulation_name(args.member,
                                                   args.pool_members.split())
        od = od / out['member_modulation']
        y_p, c_p = od / 'metrics.yaml', od / 'per_event_rows.csv'
    else:
        y_p, c_p = od / f'{args.member}.yaml', od / f'{args.member}_rows.csv'
    od.mkdir(parents=True, exist_ok=True)
    with open(y_p, 'w') as f:
        yaml.safe_dump(out, f, sort_keys=False)
    with open(c_p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f'[sigma] VERDICT: {verdict}')
    print(f'[sigma] stats: {out}', flush=True)
    if args.report_run:
        rep = BRANCH_ROOT / 'reports' / args.report_run
        rep.mkdir(parents=True, exist_ok=True)
        (rep / 'summary.md').write_text(
            f'# sigma at events -- {args.member}\n\nverdict: **{verdict}**\n\n'
            '```yaml\n' + yaml.safe_dump(out, sort_keys=False) + '```\n')
        dw = BRANCH_ROOT / 'diagnostics' / 'digest_writer.py'
        if dw.exists():
            subprocess.run([sys.executable, str(dw), '--repo-dir',
                            str(BRANCH_ROOT), '--run-name', args.report_run,
                            '--event', 'done', '--note',
                            f'{args.member}: {verdict[:70]}'],
                           capture_output=True)


if __name__ == '__main__':
    main()
