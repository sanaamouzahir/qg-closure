"""consolidate_apost_cases.py -- pack each (checkpoint, variant, Delta_T) case
of a rollout_aposteriori.py matrix run into ONE .npz and delete the per-run
intermediates (Sanaa's output-discipline order, 2026-07-09: one file per case;
a single summary CSV on top as the index).

A matrix invocation of rollout_aposteriori.py with --ckpt (arm 'closure') and
--ckpt2 (arm 'closure2') produces one rollout_apost_<tag>.{npz,json,csv} plus
sigma_hat_<tag>_<arm>.csv. This script splits every tag into per-arm case
files:

  case_<ckptlabel>_<variant>_<dtlabel>.npz with keys:
    t, relL2_bare, relL2_closure          error-vs-t table (NaN past blowup)
    final_relL2_bare/closure, improvement_x, blowup_step (-1 = none), verdict
    E_t, E_bare, Z_bare, E_closure, Z_closure   scalar series
    sigma_steps, sigma_t, sigma_shells    sigma-hat(kappa) at checkpoints
    drift_corner, drift_low               median |sig(t,k)/sig(0,k)-1| over
                                          the corner band k in [184,240] and
                                          the healthy band k < 60
    bare_stack, closure_stack, cp_steps   checkpointed fields (float32)
    config_json, ckpt, ckpt_epoch, model  provenance

Usage (from the results dir or with --dir):
  python consolidate_apost_cases.py --dir <out_dir> \
      --tags full_1p5em2,drop_1p5em2,... \
      --arm-labels closure=uncond,closure2=cond [--delete-intermediates]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

CORNER = (184.0, 240.0)
LOWBAND = 60.0


def _sigma_csv(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        r = list(csv.reader(f))
    hdr, rows = r[0], r[1:]
    steps = np.array([int(x[0]) for x in rows], np.int64)
    t = np.array([float(x[1]) for x in rows], np.float64)
    sig = np.array([[float(v) for v in x[2:]] for x in rows], np.float64)
    return steps, t, sig, len(hdr) - 2


def _drift(sig, lo, hi):
    """median over shells in [lo,hi] of |sig(t,k)/sig(0,k) - 1| per row."""
    k = np.arange(sig.shape[1], dtype=np.float64)
    band = (k >= lo) & (k <= hi)
    s0 = sig[0][band]
    ok = s0 > 0
    if not ok.any():
        return np.full(sig.shape[0], np.nan)
    return np.median(np.abs(sig[:, band][:, ok] / s0[ok] - 1.0), axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dir', type=Path, required=True)
    ap.add_argument('--tags', type=str, required=True)
    ap.add_argument('--arm-labels', type=str,
                    default='closure=uncond,closure2=cond')
    ap.add_argument('--summary-csv', type=str,
                    default='ladder_matrix_summary.csv',
                    help='summary filename (override when several tag '
                         'groups of one dir are consolidated separately, '
                         'e.g. per-horizon)')
    ap.add_argument('--delete-intermediates', action='store_true')
    args = ap.parse_args()
    d = args.dir
    arm_lab = dict(kv.split('=') for kv in args.arm_labels.split(','))
    summary = []
    to_delete = []
    for tag in [t.strip() for t in args.tags.split(',') if t.strip()]:
        jpath = d / f'rollout_apost_{tag}.json'
        npath = d / f'rollout_apost_{tag}.npz'
        if not jpath.exists():
            print(f'[consolidate] MISSING {jpath.name} -- skipped')
            continue
        res = json.loads(jpath.read_text())
        run = np.load(npath)
        cfg = res.get('config', {})
        variant = 'dropnddot' if cfg.get('drop_nddot') == 'True' else 'full'
        if cfg.get('nn_project_radius') not in (None, 'None'):
            variant += '_proj'          # alias-safe-radius projection active
        if cfg.get('nn_dissipative_proj') == 'True':
            variant += '_dissproj'      # per-shell dissipative projection
        dtlab = f"{res['Delta_T']:g}".replace('.', 'p')
        for arm, lab in arm_lab.items():
            if f'{arm}_verdict' not in res:
                continue
            et = res.get('error_table', [])
            t = np.array([r['t'] for r in et], np.float64)
            rb = np.array([r.get('relL2_bare', np.nan) for r in et], np.float64)
            rc = np.array([r.get(f'relL2_{arm}', np.nan) for r in et],
                          np.float64)
            blow = res.get(f'{arm}_blowup_step')
            fin_b = res.get('final_relL2', {}).get('bare')
            fin_c = res.get('final_relL2', {}).get(arm)
            imp = res.get(f'improvement_x_{arm}')
            payload = dict(
                t=t, relL2_bare=rb, relL2_closure=rc,
                final_relL2_bare=np.float64(fin_b if fin_b else np.nan),
                final_relL2_closure=np.float64(fin_c if fin_c else np.nan),
                improvement_x=np.float64(imp if imp else np.nan),
                blowup_step=np.int64(-1 if blow is None else blow),
                verdict=np.str_(res[f'{arm}_verdict']),
                cp_steps=run['cp_steps'],
                bare_stack=run['bare_stack'] if 'bare_stack' in run else
                np.zeros(0, np.float32),
                closure_stack=run[f'{arm}_stack'],
                E_t=run[f'{arm}_t'],
                E_bare=run['bare_E'], Z_bare=run['bare_Z'],
                E_closure=run[f'{arm}_E'], Z_closure=run[f'{arm}_Z'],
                config_json=np.str_(json.dumps(res.get('config', {}))),
                ckpt=np.str_(cfg.get('ckpt2' if arm == 'closure2'
                                     else 'ckpt', '?')),
                model=np.str_(res.get('model2' if arm == 'closure2'
                                      else 'model', '?')),
                Delta_T=np.float64(res['Delta_T']), K=np.int64(res['K']),
                M=np.int64(res['M']),
            )
            if f'{arm}_proj_shell_count' in run:
                payload.update(
                    proj_shell_count=run[f'{arm}_proj_shell_count'],
                    proj_removed_frac=run[f'{arm}_proj_removed_frac'])
            sc = _sigma_csv(d / f'sigma_hat_{tag}_{arm}.csv')
            if sc is not None:
                steps, st, sig, n_sh = sc
                payload.update(sigma_steps=steps, sigma_t=st,
                               sigma_shells=sig,
                               drift_corner=_drift(sig, *CORNER),
                               drift_low=_drift(sig, 0.0, LOWBAND))
            case = f'case_{lab}_{variant}_{dtlab}.npz'
            np.savez(d / case, **payload)
            dc = payload.get('drift_corner')
            summary.append(dict(
                case=case.replace('.npz', ''), ckpt=lab, variant=variant,
                Delta_T=res['Delta_T'],
                blowup_step='' if blow is None else blow,
                verdict=res[f'{arm}_verdict'],
                final_relL2_bare='' if not fin_b else f'{fin_b:.4e}',
                final_relL2_closure='' if not fin_c else f'{fin_c:.4e}',
                improvement_x='' if not imp else f'{imp:.2f}',
                corner_drift_last='' if dc is None or not len(dc)
                else f'{dc[-1]:.4e}',
                low_drift_last='' if 'drift_low' not in payload
                else f'{payload["drift_low"][-1]:.4e}',
                proj_on='dissproj' in variant,
                proj_shells_mean=''
                if 'proj_shell_count' not in payload
                or not len(payload['proj_shell_count'])
                else f'{float(np.mean(payload["proj_shell_count"])):.1f}',
                proj_removed_frac_mean=''
                if 'proj_removed_frac' not in payload
                or not len(payload['proj_removed_frac'])
                else f'{float(np.mean(payload["proj_removed_frac"])):.3e}'))
            print(f'[consolidate] wrote {case}')
        run.close()
        to_delete += [jpath, npath, d / f'rollout_apost_{tag}.csv',
                      d / f'sigma_hat_{tag}_closure.csv',
                      d / f'sigma_hat_{tag}_closure2.csv',
                      d / f'sigma_hat_{tag}_r3anal.csv']
    if summary:
        spath = d / args.summary_csv
        with open(spath, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
        print(f'[consolidate] summary -> {spath}')
    if args.delete_intermediates:
        for p in to_delete:
            if p.exists():
                p.unlink()
        print(f'[consolidate] removed {sum(1 for _ in to_delete)} '
              f'intermediate paths (refs npz kept)')


if __name__ == '__main__':
    main()
