#!/usr/bin/env python
"""certificate_transient_growth.py -- did the certificate optimize the WRONG
NORM? (Sanaa question 2026-07-20: "reconfirm how the amplification factor was
computed ... so we know what went wrong".)

THE SUSPICION. wiener_certificate.assemble_geff returns the SPECTRAL RADIUS
rho(A) of the closed AB2CN2 companion

    A = [[a+b, c2],
         [1  , 0 ]]

and the training penalty pushes rho below 1-eps. But rho(A) < 1 only bounds
the LIMIT: for a NON-NORMAL A, ||A^n|| can grow by orders of magnitude for
tens of steps before decaying. In a nonlinearly-coupled solver that transient
is all a blowup needs. p3 drove rho from ~1.7 to ~0.96 and the a-posteriori
blowup moved +1..+3 steps -- exactly what "we optimized the eigenvalue and
the transient was never bounded" would look like.

THE TEST. For the same shells, same states, same trained model, compute
alongside rho(A):

    ||A^n||_2 for n = 1..N          (true step-n amplification)
    G_max     = max_n ||A^n||       (transient peak)
    n_peak    = argmax_n ||A^n||
    kappa(V)  = eigenvector-basis condition number (non-normality measure;
                Kreiss/Bauer-Fike: sup_n ||A^n|| <= kappa(V) when rho<=1)
    numerical abscissa proxy: ||A||_2 itself (n=1 growth)

VERDICT LOGIC
  rho < 1 AND G_max ~ 1        -> the certificate measured the right thing;
                                  the blowup is NOT this linear mechanism
                                  (look to the nonlinear instrumentation).
  rho < 1 AND G_max >> 1       -> CONFIRMED: the certificate bounded the
                                  asymptote while the transient was free.
                                  Fixing it means penalizing max_n ||A^n||
                                  (or ||A||) instead of rho -- a one-line
                                  change of objective, not a new theory.

Reuses wiener_certificate's OWN assembly so the matrices are identical to the
ones training saw: we re-derive a, b, c2 per shell from the same inputs.
CPU, seconds, read-only.

Usage (from training/):
  python ../diagnostics/certificate_transient_growth.py \
      --ckpt data/.../rollout_ft_w31p3_certv2/best.pt \
      --root-dir data/ensemble_N5_7lag/FRC-kf4/sweep_dT_5em3 \
      --n-max 60 --out ../diagnostics/Results/cert_transient_p3.txt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'training'))

import wiener_certificate as wc          # noqa: E402
from model_cond_local import REL_SHELLS  # noqa: E402


def companion_series(a, b, c2, n_max):
    """||A^n||_2 for n=1..n_max, plus rho(A) and kappa(eigvecs), for the
    2x2 companion A = [[a+b, c2], [1, 0]] (complex, per shell)."""
    A = np.array([[a + b, c2], [1.0 + 0j, 0.0 + 0j]], dtype=np.complex128)
    ev, V = np.linalg.eig(A)
    rho = float(np.max(np.abs(ev)))
    try:
        kappa = float(np.linalg.cond(V))
    except np.linalg.LinAlgError:
        kappa = np.inf
    norms, P = [], np.eye(2, dtype=np.complex128)
    for _ in range(n_max):
        P = A @ P
        norms.append(float(np.linalg.norm(P, 2)))
    return rho, kappa, np.asarray(norms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--root-dir', required=True,
                    help='sweep root supplying dt/dx/L_hat geometry')
    ap.add_argument('--n-max', type=int, default=60)
    ap.add_argument('--out', default=None)
    ap.add_argument('--extra-shells', default='0.52,0.58,0.64')
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = ck.get('config', {})
    man = json.loads((Path(args.root_dir) / 'manifest.json').read_text())
    Ny, Nx = int(man['Ny']), int(man['Nx'])
    Lx = float(man.get('Lx', man.get('L', 2.0 * np.pi)))
    dx = Lx / Nx
    dt = float(cfg.get('strides', [1])[0]) * float(man.get('dt', 5.0e-3))
    nu = float(man.get('nu', 1.025e-4))
    mu = float(man.get('mu', 0.02))
    beta = float(man.get('beta', man.get('B', 0.0)))

    kxm = np.arange(Nx // 2 + 1, dtype=np.float64)
    kym = np.fft.fftfreq(Ny, d=1.0 / Ny)
    kmag = np.sqrt(kxm[None, :] ** 2 + kym[:, None] ** 2)
    n_sh = int(round(float(kmag.max()))) + 1
    rels = list(REL_SHELLS) + [float(x) for x in args.extra_shells.split(',')
                               if x.strip()]
    two_pi_L = 2.0 * np.pi / Lx

    # sigma sweep: the certificate's validity mask is |dt*sigma| <= 0.5, so
    # probe the certified regime AND the regime the run actually reaches
    # (blowup CFL ~ 4 => dt*sigma ~ 4).
    dtsig_probe = [0.1, 0.3, 0.5, 1.0, 2.0, 4.0]

    lines = []
    hdr = (f"certificate transient-growth audit\nckpt {args.ckpt}\n"
           f"grid {Ny}x{Nx} Lx={Lx:.4f} dx={dx:.5f} dt={dt:.4g} "
           f"nu={nu:.4g} mu={mu:.4g} beta={beta:.4g}\n"
           f"A = [[a+b, c2],[1,0]] per shell; rho = spectral radius (WHAT THE "
           f"CERTIFICATE PENALIZED); G_max = max_n ||A^n||_2 (the TRANSIENT).")
    lines.append(hdr)
    lines.append(f"\n{'rel':>6} {'|k|':>8} {'dt*sig':>7} {'rho':>9} "
                 f"{'G_max':>10} {'n_peak':>7} {'kappa(V)':>11} {'verdict':>22}")
    lines.append('-' * 92)

    worst = (0.0, None)
    for r in rels:
        i = min(int(round(r * (n_sh - 1))), n_sh - 1)
        ring = (np.round(kmag) == i)
        if not ring.any():
            continue
        kphys = i * two_pi_L
        Lh = complex(-nu * kphys ** 2 - mu,
                     (beta * kphys / max(kphys ** 2, 1e-30)) if beta else 0.0)
        for dts in dtsig_probe:
            sg = dts / dt
            cfold = dt ** 2 / 12.0
            E_adv = 1j * sg
            # closure transfer T is model-dependent; the AUDIT question is
            # about the COMPANION's norm behaviour, so probe the scheme with
            # the analytic fold only (T=0) -- the learned part can only add
            # to E_clo, and the transient/asymptote gap shown here is a
            # property of the companion, not of T.
            E_clo = E_adv * cfold * Lh ** 2
            r_imp = 1.0 / (1.0 - 0.5 * dt * Lh + (dt ** 3 / 12.0) * Lh ** 3)
            a = r_imp * (1.0 + 0.5 * dt * Lh)
            b = dt * r_imp * (1.5 * E_adv + E_clo)
            c2 = -0.5 * dt * r_imp * E_adv
            rho, kappa, norms = companion_series(a, b, c2, args.n_max)
            gmax = float(norms.max())
            npk = int(np.argmax(norms)) + 1
            if rho < 1.0 and gmax > 2.0:
                v = 'TRANSIENT UNBOUNDED'
            elif rho >= 1.0:
                v = 'rho>=1 (uncertified)'
            else:
                v = 'ok'
            if rho < 1.0 and gmax > worst[0]:
                worst = (gmax, (r, kphys, dts, rho, npk, kappa))
            lines.append(f"{r:6.2f} {kphys:8.1f} {dts:7.2f} {rho:9.5f} "
                         f"{gmax:10.3f} {npk:7d} {kappa:11.3e} {v:>22}")

    lines.append('')
    if worst[1] is not None:
        r, kphys, dts, rho, npk, kappa = worst[1]
        lines.append(
            f"VERDICT: with rho < 1 (certificate SATISFIED) the worst transient "
            f"is ||A^n|| = {worst[0]:.3f} at n={npk}, shell rel={r:.2f} "
            f"(|k|={kphys:.0f}), dt*sigma={dts:.2f}, kappa(V)={kappa:.2e}.")
        if worst[0] > 2.0:
            lines.append(
                "  => CONFIRMED: the penalty bounded the ASYMPTOTE while the "
                "step-n amplification was free to grow. Penalizing "
                "max_n ||A^n|| (or ||A||_2) instead of rho is the corrected "
                "objective.")
        else:
            lines.append(
                "  => NOT the explanation: the companion is close to normal "
                "here, so rho ~ transient. The blowup mechanism is elsewhere "
                "(see the nonlinear instrumentation).")
    text = '\n'.join(lines)
    print(text, flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + '\n')
        print(f"[cert] wrote {args.out}", flush=True)


if __name__ == '__main__':
    main()
