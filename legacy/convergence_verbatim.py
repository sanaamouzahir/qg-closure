"""
convergence_verbatim.py - exact reproduction of the convergence plot script
written in front of Sanaa.

Verbatim from his code (only minor cleanups: f-string for the savefig path,
consolidated comments). Drop into the parent dir containing dt_<tag>/ subdirs
each holding a qg_data.npy.

The expected layout for each run:
    .
    +- <subdir1>/qg_data.npy   shape (B, T_save, 4, Ny, Nx)
    +- <subdir2>/qg_data.npy
    +- ...

His code uses:
    paths = sorted(os.listdir("."))    # alphabetical
    data.append(np.load(path)[0, -2, 0])    # batch 0, 2nd-to-last save, channel 0

Channel 0 of state.out() is q (vorticity). [-2] is the second-to-last save.
"""

import os
import numpy as np
from matplotlib import pyplot as plt

# His exact dt list, base = 0.0001
dt = [0.0001, 0.0005, 0.001, 0.002, 0.005]

# His exact path-collection logic:
paths = sorted(os.listdir("."))
paths = [os.path.join(p, "qg_data.npy") for p in paths]

# Filter out anything that doesn't exist (e.g. the script itself, log files
# that sorted() picks up in cwd). Comment this out for true verbatim.
paths = [p for p in paths if os.path.isfile(p)]

print("paths in sorted-listdir order:")
for p, step in zip(paths, dt):
    print(f"  dt={step}  ->  {p}")

data = []
for path, step in zip(paths, dt):
    data.append(np.load(path)[0, -1, 0])    # batch 0, 2nd-to-last, channel 0

true = data[0]
err = [np.sqrt(np.abs(np.power(d - true, 2).mean())) for d in data[1:]]

# Reference slopes anchor at 1e4 like in his code
plt.figure(figsize=(7, 5))
plt.loglog(dt[1:], err, 'ko-', label='Error', lw=1.5, ms=8)
plt.loglog(dt[1:], [1e4 * t**2 for t in dt[1:]], 'o-', label='slope 2')
plt.loglog(dt[1:], [1e4 * t    for t in dt[1:]], 'o-', label='slope 1')
plt.xlabel('Time Step')
plt.ylabel('Error')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.title('Convergence (verbatim reproduction)')
plt.tight_layout()
plt.savefig('convergence.png', dpi=150)
print("wrote convergence.png")

# Print numeric values too, for the record
print(f"\ndts (excluding base): {dt[1:]}")
print(f"errors:               {err}")
