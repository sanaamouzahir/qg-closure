"""Per-(member,dT) Nddot comparison: w31 best.pt (ep32) vs cond_v2 best.
Nddot is the rollout floor (benign amplification 1:1) -- the number to watch.
Grouped bars per member, one panel per dT; ratio annotated. CPU plot."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

W = Path('/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning')
TR = W / 'training/data/ensemble_N5_7lag/training_runs'
OUT = W / 'diagnostics/Results/w31_eval_20260715'

def load(p):
    d = {}
    for r in csv.DictReader(open(p)):
        d[(r['member'], float(r['Delta_T']))] = {k: float(r[k]) for k in ('Ndot', 'Nddot', 'N3dot')}
    return d

v2 = load(TR / 'deriv7_cond_local_v2/eval_by_root_val.csv')
w31 = load(TR / 'deriv7_cond_local_w31/eval_by_root_val.csv')
members = sorted({m for m, _ in w31}, key=lambda s: (not s.startswith('DEC'), s))
dts = [0.005, 0.01, 0.015]

fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
x = np.arange(len(members))
for ax, dt in zip(axes, dts):
    a = [v2.get((m, dt), {}).get('Nddot', np.nan) for m in members]
    b = [w31.get((m, dt), {}).get('Nddot', np.nan) for m in members]
    ax.bar(x - 0.18, a, 0.36, label='cond_v2 (width 15)', color='0.6')
    ax.bar(x + 0.18, b, 0.36, label='w31 ep32 (width 31)', color='tab:orange')
    for xi, (ai, bi) in enumerate(zip(a, b)):
        if np.isfinite(ai) and np.isfinite(bi) and ai > 0:
            ax.text(xi, max(ai, bi) * 1.05, f'{bi/ai:.2f}x', ha='center', fontsize=7)
    ax.set_yscale('log'); ax.set_ylabel('val Nddot rel-L2')
    ax.set_title(f'dT = {dt}  (annotation: w31/cond_v2 ratio; <1 = w31 better)')
    ax.grid(alpha=0.3, axis='y', which='both'); ax.legend(fontsize=8)
axes[-1].set_xticks(x); axes[-1].set_xticklabels(members, rotation=45, ha='right')
fig.suptitle('Nddot (the rollout floor) per member x dT: width-31 vs cond_v2, val split', y=0.995)
fig.tight_layout()
fig.savefig(OUT / 'w31_vs_condv2_Nddot_by_root.png', dpi=150, bbox_inches='tight')

# compact summary table to txt
lines = ['member        dT      Nddot_v2  Nddot_w31  ratio']
rat = []
for m in members:
    for dt in dts:
        if (m, dt) in w31 and (m, dt) in v2:
            a, b = v2[(m, dt)]['Nddot'], w31[(m, dt)]['Nddot']
            rat.append(b / a)
            lines.append(f'{m:<13} {dt:<7} {a:.4f}    {b:.4f}     {b/a:.2f}')
lines.append(f'\nmedian ratio {np.median(rat):.3f} | mean {np.mean(rat):.3f} | worse-than-v2 cells: {sum(r>1 for r in rat)}/{len(rat)}')
(OUT / 'w31_vs_condv2_Nddot_table.txt').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines[-3:]))
print('wrote', OUT / 'w31_vs_condv2_Nddot_by_root.png')
