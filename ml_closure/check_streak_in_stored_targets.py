"""Streak check on the STORED training targets (Sanaa question 2026-07-13):
apply the triage's streak-amplitude metric (std of the detrended column-mean
profile in the cleanest freestream band, normalized by the band's field std)
to pi_ff of DNS_LES_s4.npz (sharp) vs DNS_LES_s4_gaussian.npz (gaussian),
median over val-window frames. Also rel-L2(pi_sharp - pi_gauss)/||pi_sharp||.
Verdict: gaussian streak amplitude should be ~an order of magnitude below
sharp; if not, the rebuild did not remove the ringing -> FLAG."""
import sys
from pathlib import Path
import numpy as np
from triage_ab_filter import streak_amplitude, pick_freestream_rows

for p in sys.argv[1:]:
    m = Path(p)
    out = {}
    zs = np.load(m / 'DNS_LES_s4.npz'); zg = np.load(m / 'DNS_LES_s4_gaussian.npz')
    t = np.asarray(zs['times']); sel = np.where((t >= 100) & (t <= 120))[0][::4]
    chi_o = np.asarray(zg['chi_obs_bar']).squeeze(); chi_s = np.asarray(zg['chi_sponge_bar']).squeeze()
    rows = pick_freestream_rows(chi_o, chi_s)
    cols = np.arange(chi_o.shape[1])
    d = []
    for lab, z in (('sharp', zs), ('gaussian', zg)):
        amps = [streak_amplitude(np.asarray(z['pi_ff'][0][fi], dtype=np.float64), rows, cols)[0]
                for fi in sel]
        out[lab] = float(np.median(amps))
    a = np.asarray(zs['pi_ff'][0][sel], dtype=np.float64); b = np.asarray(zg['pi_ff'][0][sel], dtype=np.float64)
    rl2 = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-300))
    print(f"[streakchk] {m.name}: sharp {out['sharp']:.4f}  gaussian {out['gaussian']:.4f}  "
          f"ratio {out['sharp']/max(out['gaussian'],1e-12):.1f}x  relL2(sharp-gauss)/sharp {rl2:.3f}")
print('[streakchk] done')
