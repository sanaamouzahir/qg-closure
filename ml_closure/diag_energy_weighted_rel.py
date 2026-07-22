"""Energy-vs-pixel view of the per-pixel relative error (Sanaa 2026-07-22:
'most plots look like 100% error per pixel'). Per member, wake box, val
frames: what fraction of PIXELS vs of Pi^2 ENERGY sits below each relative
error level. If low-rel pixels carry most energy, the map's ~100% carpet is
the energetically-empty background."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dataset_piff import build_runs, split_frames, _f
from model_cnn import PiffCNN
from eval_cnn import ylp75_taper

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--frame-stride', type=int, default=4)
ap.add_argument('--device', default='cpu')
args = ap.parse_args()
ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
conf = ck['conf']; dc = conf['data']
model = PiffCNN(conf).to(args.device); model.load_state_dict(ck['model']); model.eval()
runs = build_runs(conf)
fbr = {}
for ri, fi in split_frames(runs, 'val', conf):
    fbr.setdefault(ri, []).append(fi)
LEV = [0.25, 0.5, 1.0, 2.0]
print(f"{'member':<12}{'lvl':>6}{'pixel%':>8}{'energy%':>9}   (share of wake pixels vs share of wake Pi^2 with rel err below lvl)")
for ri, frames in sorted(fbr.items()):
    r = runs[ri]
    xs = (np.arange(r.Nx)+0.5)*r.dx; ys = (np.arange(r.Ny)+0.5)*r.dy
    wake = (r.valid & (xs[None,:] >= r.x_c+_f(dc['wake_x_lo_D'])*r.D)
            & (xs[None,:] <= r.x_c+_f(dc['wake_x_hi_D'])*r.D)
            & (np.abs(ys[:,None]-r.y_c) <= _f(dc['wake_y_half_D'])*r.D))
    npix = np.zeros(len(LEV)); epix = np.zeros(len(LEV)); ntot = 0; etot = 0.0
    for fi in frames[::args.frame_stride]:
        x, y, m, zeta, zeta_dot, _, _ = r.full_frame(fi)
        with torch.no_grad():
            pred = model.predict_physical(x[None].to(args.device), zeta[None].to(args.device),
                zeta_dot[None].to(args.device) if model.use_zeta_dot else None)[0].cpu().numpy()
        pred = ylp75_taper(pred.astype(np.float64))
        t = y.numpy().astype(np.float64)
        tw, pw = t[wake], pred[wake]
        e2 = tw*tw
        rel = np.abs(pw-tw)/np.maximum(np.abs(tw), 1e-30)
        ntot += tw.size; etot += e2.sum()
        for i, L in enumerate(LEV):
            s = rel < L
            npix[i] += int(s.sum()); epix[i] += float(e2[s].sum())
    for i, L in enumerate(LEV):
        print(f"{r.name:<12}{L:>6.2f}{100*npix[i]/ntot:>7.1f}%{100*epix[i]/etot:>8.1f}%")
