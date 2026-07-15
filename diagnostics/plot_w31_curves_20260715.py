"""Training curves for deriv7_cond_local_w31 vs its warm start (cond_v2).

Figure 1: pooled val + per-order medians over epochs, best epoch marked,
cond_v2 gate lines. N3dot excluded from the loss (weight 0) -> shown greyed.
CPU-only plotting job (standing rule)."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

W = Path('/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning')
RUN = W / 'training/data/ensemble_N5_7lag/training_runs/deriv7_cond_local_w31'
OUT = W / 'diagnostics/Results/w31_eval_20260715'

rows = list(csv.DictReader(open(RUN / 'log.csv')))
ep = [int(r['epoch']) for r in rows]
val = [float(r['val_relL2']) for r in rows]
best = [float(r['best_val']) for r in rows]
mNdot = [float(r['val_med_Ndot']) for r in rows]
mNddot = [float(r['val_med_Nddot']) for r in rows]
mN3dot = [float(r['val_med_N3dot']) for r in rows]
ibest = min(range(len(val)), key=lambda i: val[i])

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(ep, val, color='tab:blue', lw=1.2, label='pooled val rel-L2 (per epoch)')
ax[0].plot(ep, best, color='tab:blue', lw=2.2, ls='--', label='best so far')
ax[0].axhline(0.0968, color='tab:red', lw=1, ls=':', label='cond_v2 gate 0.0968')
ax[0].scatter([ep[ibest]], [val[ibest]], color='k', zorder=5,
              label=f'best.pt: ep{ep[ibest]}, {val[ibest]:.4f}')
ax[0].set_xlabel('epoch'); ax[0].set_ylabel('val rel-L2')
ax[0].set_title('w31: pooled validation (orders 1,2 weighted; N3dot weight-0)')
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

ax[1].plot(ep, mNdot, color='tab:green', lw=1.5, label='median Ndot')
ax[1].plot(ep, mNddot, color='tab:orange', lw=2.0, label='median Nddot (rollout floor)')
ax[1].plot(ep, mN3dot, color='0.7', lw=1.0, label='median N3dot (weight-0, untrained)')
ax[1].axhline(0.087, color='tab:red', lw=1, ls=':', label='cond_v2 med Nddot 0.087')
ax[1].set_xlabel('epoch'); ax[1].set_ylabel('median val rel-L2')
ax[1].set_yscale('log'); ax[1].set_title('w31: per-order medians')
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which='both')

fig.suptitle('deriv7_cond_local_w31 (width-31 conditioned, warm from cond_v2, 41 roots)', y=1.02)
fig.tight_layout()
fig.savefig(OUT / 'w31_training_curves.png', dpi=150, bbox_inches='tight')
print('wrote', OUT / 'w31_training_curves.png')
