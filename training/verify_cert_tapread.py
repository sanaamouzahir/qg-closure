#!/usr/bin/env python
"""verify_cert_tapread.py -- one-shot check of the 2026-07-14 assemble_geff
base-tap read fix: on the real cond_v2 ckpt, the fixed read (channel-0 row
sum = exact ky=0 x-response) must match the CENTRAL-row taps to within the
tiny trained off-row mass, and must differ decisively from the old top-row
read. Also drives assemble_geff end-to-end on a synthetic context with the
4-D model to prove the fixed branch executes and moves |G| off the old value.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rollout_aposteriori import load_deriv_model
import wiener_certificate as wc

ckpt = Path(sys.argv[1])
root = Path(sys.argv[2])                       # any deep root (manifest source)
man = json.loads((root / 'manifest.json').read_text())
model, _, _ = load_deriv_model(ckpt, man, 1.5e-2, 'cpu')
wx = model.grad.wx.detach().to(torch.float64)
C, _, W, _ = wx.shape
c = W // 2
central = wx[0, 0, c, :]
row0 = wx[0, 0, 0, :]
fixed = wx[0, 0].sum(dim=0)
print(f"wx shape {tuple(wx.shape)}  W={W}")
print(f"||central|| = {central.norm():.6e}")
print(f"||row0 (old read)|| = {row0.norm():.6e}")
print(f"||fixed read (row-sum)|| = {fixed.norm():.6e}")
print(f"||fixed - central|| = {(fixed - central).norm():.6e} "
      f"(= trained off-row mass; small)")
print(f"||fixed - row0||   = {(fixed - row0).norm():.6e} (= the bug magnitude)")

# end-to-end: synthetic single-shell context, 4-D model path
n_sh = 5
sig = torch.full((1, n_sh), 5.0, dtype=torch.float64)
dt = torch.tensor([1.5e-2], dtype=torch.float64)
Lsh = torch.full((n_sh,), -1.0, dtype=torch.complex128)
ksh = torch.linspace(10.0, 100.0, n_sh, dtype=torch.float64)
dx = torch.tensor([0.0245], dtype=torch.float64)
g_fixed = wc.assemble_geff(model, sig, dt, Lsh, ksh, dx)

class _Shim:
    pass
shim = _Shim()
shim.grad = _Shim(); shim.grad.wx = row0.clone()      # dim<4 -> old read path
shim.mix = model.mix; shim.n_ord = model.n_ord
g_old = wc.assemble_geff(shim, sig, dt, Lsh, ksh, dx)
shim.grad.wx = central.clone()
g_central = wc.assemble_geff(shim, sig, dt, Lsh, ksh, dx)

print("\nper-shell |G_eff| on the synthetic context:")
print(f"  fixed 4-D read : {[f'{v:.6f}' for v in g_fixed[0].tolist()]}")
print(f"  central-row 1-D: {[f'{v:.6f}' for v in g_central[0].tolist()]}")
print(f"  old row-0 1-D  : {[f'{v:.6f}' for v in g_old[0].tolist()]}")
ok = torch.allclose(g_fixed, g_central, rtol=2e-2) and \
    not torch.allclose(g_fixed, g_old, rtol=1e-3)
print(f"\nVERDICT: {'PASS' if ok else 'FAIL'} (fixed==central to 2%, != old)")
sys.exit(0 if ok else 3)
