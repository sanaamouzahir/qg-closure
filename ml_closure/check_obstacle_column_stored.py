"""Decisive streak check on STORED targets: median |pi| over the obstacle's
x-column band vs the whole-field median (valid px), sharp vs gaussian files.
Triage reference (recompute): cylinder sharp 21.7 -> gaussian 0.09 on the
obstacle column. If the stored gaussian file shows the sharp-level column
excess -> the rebuild kept ringing -> FLAG."""
import sys, json
from pathlib import Path
import numpy as np

for p in sys.argv[1:]:
    m = Path(p)
    zg = np.load(m / 'DNS_LES_s4_gaussian.npz')
    zs = np.load(m / 'DNS_LES_s4.npz')
    chi_o = np.asarray(zg['chi_obs_bar']).squeeze()
    chi_s = np.asarray(zg['chi_sponge_bar']).squeeze()
    t = np.asarray(zs['times']); sel = np.where((t >= 100) & (t <= 120))[0][::4]
    colmask = (chi_o > 1e-3).any(axis=0)          # x-columns crossing the body
    Ny = chi_o.shape[0]
    body_rows = (chi_o > 1e-3).any(axis=1)
    far = ~body_rows                              # rows away from the body
    valid = (chi_o <= 1e-3) & (chi_s <= 1e-2)
    for lab, z in (('sharp', zs), ('gaussian', zg)):
        pi = np.asarray(z['pi_ff'][0][sel], dtype=np.float64)
        col = np.median(np.abs(pi[:, far][:, :, colmask]))   # obstacle columns, far rows
        allm = np.median(np.abs(pi[:, valid]))
        print(f"[colchk] {m.name} {lab}: col-median {col:.4g}  field-median {allm:.4g}  excess {col/max(allm,1e-300):.1f}x")
print('[colchk] done')
