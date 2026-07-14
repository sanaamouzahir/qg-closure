"""
upstream_effect_on_forces.py — Sanaa question 2026-07-14: were the
simulation-time force/shedding checks (C_d, C_l, f/St) off because of the
upstream reflection (undersized inlet sponge -> reflected wave + spurious v
ahead of the obstacle) that we later masked out of TRAINING?

The forces themselves were integrated by the solver at run time and cannot be
recomputed on "repaired" fields — but the solver ALSO logged U_cyl/Re_cyl
(the local velocity actually seen at the body) next to the table values
U_inlet/Re_inlet. If the reflection changed the effective inflow, the honest
normalization uses U_cyl, and the corrected numbers are:

    St_local  = f_sh * D / mean(U_cyl)          [vs St_inlet = f_sh*D/mean(U_inlet)]
    Cd_local  = mean(Cd_*) * (U_inlet/U_cyl)^2  [dynamic-pressure renormalization;
                                                 Fx is unchanged, only the reference
                                                 q = 0.5 rho U^2 D changes]

Also measures the contamination directly from the stored filtered fields
(DNS_LES_s4.npz ubar/vbar): in the upstream band x in [x_c-4D, x_c-1.5D]
(the region excluded from training by the mask), per-frame
mean(ubar)/U(t) (deficit) and rms(vbar)/U(t) (spurious v), medians over the
developed window t >= 30.

f_sh is REUSED from shedding/shedding_summary.yaml (Welch peak on Cl_mid) —
never recomputed. Reference context (never a validation target, charter):
cylinder Re 3900: Cd ~ 0.99, St ~ 0.21.

Usage (CPU, all.q via piff_tool_job.sh):
  python upstream_effect_on_forces.py --run-dir <member dir> --tag FPC-const \
      --xc 12.566371 --yc 12.566371 --D 1.256637 \
      --out-yaml <path> [--out-png <path>]
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--xc', type=float, required=True)
    ap.add_argument('--yc', type=float, required=True)
    ap.add_argument('--D', type=float, required=True)
    ap.add_argument('--t-min', type=float, default=30.0)
    ap.add_argument('--out-yaml', required=True)
    ap.add_argument('--out-png', default=None)
    args = ap.parse_args()
    rd = Path(args.run_dir)

    # ---- scalars: forces + local-vs-table velocity ------------------------ #
    sc = np.load(rd / 'scalars.npz', allow_pickle=True)
    t = np.asarray(sc['t'], dtype=np.float64)
    w = t >= args.t_min
    out = {'tag': args.tag, 'window_t': [float(args.t_min), float(t.max())],
           'n_scalar_samples': int(w.sum())}

    def stat(key):
        if key not in sc.files:
            return None
        v = np.asarray(sc[key], dtype=np.float64)[w]
        return v

    U_in, U_cyl = stat('U_inlet'), stat('U_cyl')
    ratio = (float(np.mean(U_cyl / U_in)) if U_in is not None and U_cyl is not None
             else None)
    out['U_local_over_U_inlet_mean'] = ratio
    if U_in is not None:
        out['U_inlet_mean'] = float(U_in.mean())
    if U_cyl is not None:
        out['U_local_mean'] = float(U_cyl.mean())
    for key in ('Cd_mid', 'Cd_inst', 'Cl_mid', 'Cl_inst'):
        v = stat(key)
        if v is None:
            continue
        out[f'{key}_mean'] = float(v.mean())
        out[f'{key}_rms'] = float(np.sqrt(np.mean((v - v.mean()) ** 2)))
        if key.startswith('Cd') and ratio is not None:
            out[f'{key}_mean_local_norm'] = float(v.mean() / ratio ** 2)

    # ---- shedding frequency: reuse the recorded Welch peak ---------------- #
    shy = rd / 'shedding' / 'shedding_summary.yaml'
    if shy.exists():
        sh = yaml.safe_load(shy.read_text())
        f_sh = float(sh['welch']['f_sh_peak_Cl_mid'])
        out['f_sh_peak_Cl_mid'] = f_sh
        if U_in is not None:
            out['St_inlet_norm'] = f_sh * args.D / float(U_in.mean())
        if U_cyl is not None:
            out['St_local_norm'] = f_sh * args.D / float(U_cyl.mean())

    # ---- upstream contamination measured from the stored filtered fields -- #
    les_path = next((rd / n for n in
                     ('DNS_LES_s4_gaussian_jonly.npz', 'DNS_LES_s4.npz')
                     if (rd / n).exists()), None)
    if les_path is not None:
        d = np.load(les_path, allow_pickle=True)
        times = np.asarray(d['times'], dtype=np.float64)
        fw = times >= args.t_min
        ny, nx = d['ubar'].shape[-2:]
        meta = d['meta'].item() if 'meta' in d.files else {}
        Lx = float(meta.get('Lx', 8 * np.pi))
        x = (np.arange(nx) + 0.5) * (Lx / nx)
        band = (x >= args.xc - 4.0 * args.D) & (x <= args.xc - 1.5 * args.D)
        Ut = np.asarray(d['U_snap'], dtype=np.float64)[fw]
        ub = np.asarray(d['ubar'], dtype=np.float64)[fw][:, :, band]
        vb = np.asarray(d['vbar'], dtype=np.float64)[fw][:, :, band]
        u_def = ub.mean(axis=(1, 2)) / Ut          # per-frame mean u / U(t)
        v_rms = np.sqrt((vb ** 2).mean(axis=(1, 2))) / Ut
        out['upstream_band_x_D'] = [-4.0, -1.5]
        out['upstream_u_over_U_median'] = float(np.median(u_def))
        out['upstream_v_rms_over_U_median'] = float(np.median(v_rms))
        out['les_source'] = les_path.name

        if args.out_png:
            prof = ub.mean(axis=(0, 1))            # time+y mean u(x) in band
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot((x[band] - args.xc) / args.D, prof / Ut.mean(), 'o-')
            ax.axhline(1.0, color='k', ls='--', label='clean inflow u = U(t)')
            ax.set_xlabel('(x - x_c)/D  (upstream of the obstacle)')
            ax.set_ylabel('mean u / U')
            ax.set_title(f'{args.tag}: upstream velocity deficit (reflection)')
            ax.legend()
            fig.tight_layout()
            Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.out_png, dpi=130)
            plt.close(fig)

    out['reference_context_never_targets'] = {'Cd_Re3900': 0.99, 'St_Re3900': 0.21}
    Path(args.out_yaml).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_yaml).write_text(yaml.safe_dump(out, sort_keys=False))
    print(yaml.safe_dump(out, sort_keys=False))
    print(f'[upstream] wrote {args.out_yaml}')


if __name__ == '__main__':
    main()
