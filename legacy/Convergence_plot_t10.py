"""
convergence_t10_plot_titled.py - colleague's plot, verbatim, with the
report-ready title and LaTeX-ified y-axis label.

Colleague's original logic (sorted-paths zip dt list, np.pow -> np.power
typo fix, RMS error against the reference dt). Only differences:
  - title:    'Convergence Analysis of Spectral AB2CN2 scheme'
  - y-axis:   r'$\|\omega_{\Delta t} - \omega\|$'

Run from $QG_DIR/outputs/colleague_t10_verbatim/:
    python convergence_t10_plot_titled.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# Sort by dirname (ascending: dt_0001, dt_0010, ...)
paths = sorted(d for d in os.listdir(".")
                if d.startswith("dt_") and os.path.isdir(d))

# Parse dt from each dir name (zero-padded ints / 10000.0)
dt_list = []
for p in paths:
    tag = p.replace("dt_", "")
    dt_list.append(int(tag) / 10000.0)
print("dirs / dts:")
for p, d in zip(paths, dt_list):
    print(f"  {p}  ->  dt = {d}")

# Load each run's qg_data.npy[0, -1, 0]
data = []
for p in paths:
    arr = np.load(os.path.join(p, "qg_data.npy"))
    data.append(arr[0, -1, 0])
    print(f"  loaded {p}/qg_data.npy: shape {arr.shape}, "
          f"|[-1]|={np.sqrt((arr[0,-1,0]**2).mean()):.3e}")

# Reference: smallest dt = first entry (paths sorted ascending)
true = data[0]
others_dt = dt_list[1:]
others    = data[1:]

# Compute RMS error
err = [np.sqrt(np.mean((d - true) ** 2)) for d in others]

print("\nPaired (dt, err):")
for d, e in zip(others_dt, err):
    print(f"  dt={d:.4f}   err={e:.4e}")

# Plot
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.loglog(others_dt, err, 'ko-', ms=7, lw=1.5, label='Error')

# Reference slopes (anchored to fit visually)
err_arr = np.array(err)
dt_arr  = np.array(others_dt)
anchor_y = err_arr[np.argmax(dt_arr)]
anchor_x = dt_arr [np.argmax(dt_arr)]

ref2 = anchor_y * (dt_arr / anchor_x) ** 2
ref1 = anchor_y * (dt_arr / anchor_x) ** 1
order = np.argsort(dt_arr)
ax.loglog(dt_arr[order], ref2[order], 'b--', lw=1.0, label='slope 2')
ax.loglog(dt_arr[order], ref1[order], 'r:',  lw=1.0, label='slope 1')

ax.set_xlabel(r'$\Delta t$', fontsize=13)
ax.set_ylabel(r'$\|\omega(T) - \omega_{\Delta t}(T)\|$', fontsize=13)
ax.set_title(r'AB2CN2 temporal convergence, Case #2 Decaying turbulence $\nu = 1.025\times 10^{-5}$, $T = 10$', fontsize=13)
ax.grid(True, which='both', alpha=0.3)
ax.legend(fontsize=11)

fig.tight_layout()
out_path = 'convergence.png'
fig.savefig(out_path, dpi=150)
print(f"\nwrote {out_path}")

# Print numerical slope
log_dt  = np.log(dt_arr)
log_err = np.log(err_arr)
slope, intercept = np.polyfit(log_dt, log_err, 1)
print(f"\nfitted slope: {slope:.3f}")
print(f"  (slope = 2.0 means 2nd-order convergence)")
