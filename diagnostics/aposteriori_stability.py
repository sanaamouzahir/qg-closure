"""
aposteriori_stability.py -- 3c: long-horizon stability verdicts.

Consumes one or more rollout_apost_<tag>.npz/.json pairs produced by
training/rollout_aposteriori.py run in LONG-HORIZON mode, e.g.:

    python rollout_aposteriori.py --root-dir <root> --ckpt <best.pt> \
        --ic-index 0 --no-truth --horizon-turnovers 100 \
        --arms bare,r3only,closure --scalars-every 5 --tag stab_<case>

(the driver already detects blowup per arm: non-finite Z or Z > 10x Z(0),
records the CFL max over checkpoints, and stamps STABLE/UNSTABLE verdicts
into the json; this diagnostic renders the evidence and writes the verdict
paragraphs).

Outputs (--out-dir): apost_stability_<tag>.png (E(t), Z(t) per arm with
blowup markers), apost_stability_<tag>.csv (per-arm verdict table), and a
one-paragraph verdict per configuration on stdout + verdicts .md.

Analysis-only; CPU is fine (qlogin/qrsh per the compute rule).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                     # noqa: E402

COLORS = {'bare': 'C0', 'r3only': 'C1', 'closure': 'C3'}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--npz', type=Path, nargs='+', required=True,
                   help='one or more rollout_apost_*.npz (long-horizon runs)')
    p.add_argument('--out-dir', type=Path, default=None)
    p.add_argument('--tag', type=str, default='stability')
    args = p.parse_args()

    out_dir = args.out_dir or args.npz[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(args.npz)
    fig, axes = plt.subplots(n, 2, figsize=(12.5, 4.2 * n), squeeze=False)
    rows, paragraphs = [], []

    for r, npz_path in enumerate(args.npz):
        d = np.load(npz_path)
        jpath = npz_path.with_suffix('.json')
        meta = json.loads(jpath.read_text()) if jpath.exists() else {}
        run_tag = npz_path.stem.replace('rollout_apost_', '')
        arms = [a for a in ('bare', 'r3only', 'closure')
                if f'{a}_t' in d.files]
        Delta_T = float(meta.get('Delta_T', 0.0))
        tau = float(meta.get('tau_eddy', np.nan))
        model = meta.get('model', '?')

        axE, axZ = axes[r]
        para_bits = []
        for a in arms:
            t, E, Z = d[f'{a}_t'], d[f'{a}_E'], d[f'{a}_Z']
            axE.plot(t, E, color=COLORS[a], lw=1.0, label=a)
            axZ.semilogy(t, np.maximum(Z, 1e-30), color=COLORS[a], lw=1.0,
                         label=a)
            blow = meta.get(f'{a}_blowup_step')
            verdict = meta.get(f'{a}_verdict',
                               'UNSTABLE' if blow is not None else 'STABLE')
            cfl = meta.get(f'{a}_cfl_max', float('nan'))
            if blow is not None:
                tb = blow * Delta_T
                axZ.axvline(tb, color=COLORS[a], ls=':', lw=1.5)
                axZ.annotate(f'{a} blowup', (tb, Z[-1]), color=COLORS[a],
                             fontsize=8, rotation=90, va='top')
            drift = (float(Z[-1] / max(Z[0], 1e-30)) if len(Z) else np.nan)
            horizon_turn = (t[-1] / tau) if (len(t) and np.isfinite(tau)
                                             and tau > 0) else np.nan
            rows.append(dict(run=run_tag, arm=a, verdict=verdict,
                             blowup_step=blow, cfl_max=cfl,
                             Z_final_over_Z0=drift,
                             horizon_turnovers=horizon_turn))
            para_bits.append(
                f"{a}: {verdict}"
                + (f" (blowup at t={blow*Delta_T:.2f})" if blow is not None
                   else f" through {horizon_turn:.0f} turnovers "
                        f"(Z_end/Z_0={drift:.2f}, CFL_max={cfl:.2f})"))
        axE.set_xlabel('t'); axE.set_ylabel('E')
        axE.set_title(f'{run_tag} (model {model}): energy')
        axE.legend(fontsize=8); axE.grid(alpha=0.3)
        axZ.set_xlabel('t'); axZ.set_ylabel('Z')
        axZ.set_title('enstrophy (log; dotted = blowup)')
        axZ.legend(fontsize=8); axZ.grid(alpha=0.3, which='both')

        para = (f"[3c verdict] {run_tag} (dT={Delta_T:g}, model {model}): "
                + "; ".join(para_bits) + ".")
        print(para)
        paragraphs.append(para)

    with open(out_dir / f'apost_stability_{args.tag}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    (out_dir / f'apost_stability_{args.tag}_verdicts.md').write_text(
        '\n\n'.join(paragraphs) + '\n')
    fig.tight_layout()
    fig.savefig(out_dir / f'apost_stability_{args.tag}.png', dpi=140)
    print(f'[3c] wrote apost_stability_{args.tag}.png/.csv/_verdicts.md '
          f'in {out_dir}')


if __name__ == '__main__':
    main()
