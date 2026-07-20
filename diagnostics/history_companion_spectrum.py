#!/usr/bin/env python
"""history_companion_spectrum.py -- THE DEAD BODY (Sanaa, 2026-07-20).

wiener_certificate builds a 2x2 companion -- AB2's two time levels:

    [w_{n+1}]   [a+b  c2] [w_n  ]
    [w_n    ] = [ 1    0] [w_{n-1}]

and penalizes its spectral radius. But the CLOSED system is not 2-level: the
network reads an S=7-snapshot history and forms Ndot/Nddot by TIME finite
differences of it, weights W_unit[k]/dt^k. So the real recurrence is
(1+S)-dimensional and its unstable eigenvalue can live entirely in the
history subspace the 2x2 never modeled.

MEASURED (blowup instrumentation, kf4 ic912, ckpt w31p3):
  - the NN correction grows at a CONSTANT factor ~2.85/step from step ~12
    while total enstrophy is flat to 0.1% (bare arm identical) => LINEAR
    mode, not turbulence;
  - it switches on only after ~S steps, i.e. once the history stack is
    fully SELF-GENERATED (before that it still holds true snapshots);
  - the 2x2 certificate read |G| = 0.96 on the same checkpoint.

THIS SCRIPT builds the FULL companion per shell and returns its spectral
radius. Per shell k, with the closure applied to the AB2CN2 step, the state
vector is

    x_n = [ w_n, w_{n-1}, w_{n-2}, ..., w_{n-S+1} ]   (S history levels)

The closure enters through Ndot/Nddot, which the model forms as
    N^(m)(t_n) ~ sum_{j=0}^{S-1} (W_unit[m,j] / dt^m) * N(t_{n-j}),
and in the Wiener freeze N(t) -> (i*sigma) * w(t) per shell (advection
symbol), so

    f_NN(k) = (1/12) [ L * Ndot - 5 * Nddot ]
            = (1/12) sum_j [ L*W1[j]/dt - 5*W2[j]/dt^2 ] * (i sigma) w_{n-j}

i.e. the correction is a LINEAR COMBINATION OF THE WHOLE HISTORY with
coefficients that carry 1/dt and 1/dt^2. Those are the entries the 2x2
companion silently dropped.

Row 0 of the full companion (folded application, r = 1/denom_clos):
    w_{n+1} = r*(1 + dt/2 L) w_n
            + r*dt*(1.5 E_adv) w_n - r*0.5*dt*E_adv w_{n-1}
            - r*dt^3 * f_NN_coeffs . [w_n ... w_{n-S+1}]
rows 1..S-1 shift.

Usage (from training/):
  python ../diagnostics/history_companion_spectrum.py --dt 5.0e-3 \
      --nu 1.025e-4 --mu 0.02 --Lx 12.5664 --Nx 512 [--out <file>]
"""
import argparse
from pathlib import Path

import numpy as np


def w_unit(S):
    """Backward-difference stencil at UNIT spacing: row k = order-k weights
    over [t0, t-1, ..., t-(S-1)]  (model_deriv_closure.TimeFD convention:
    W = inv(A).T with A[j, k] = (-j)^k / k!)."""
    j = np.arange(S, dtype=np.float64)
    A = np.zeros((S, S), dtype=np.float64)
    for k in range(S):
        A[:, k] = (-j) ** k / np.math.factorial(k)
    return np.linalg.inv(A).T          # (S, S): row k = order-k stencil


def companion(dt, L, sigma, S, W, coef_mode='folded'):
    """Full (S x S) companion of the closed step at one shell."""
    E_adv = 1j * sigma
    denom_bare = 1.0 - 0.5 * dt * L
    denom = denom_bare + (dt ** 3 / 12.0) * L ** 3 if coef_mode == 'folded' \
        else denom_bare
    r = 1.0 / denom
    # NN correction coefficients over the history: (1/12)[L*W1/dt - 5*W2/dt^2]
    # times the advection symbol i*sigma, times the applied DT^3 factor.
    c_hist = np.zeros(S, dtype=np.complex128)
    for jj in range(S):
        c_hist[jj] = (1.0 / 12.0) * (L * W[1, jj] / dt
                                     - 5.0 * W[2, jj] / dt ** 2) * E_adv
    A = np.zeros((S, S), dtype=np.complex128)
    # row 0: the update
    A[0, 0] = r * (1.0 + 0.5 * dt * L) + r * dt * 1.5 * E_adv
    A[0, 1] += r * (-0.5) * dt * E_adv
    A[0, :] += (-(dt ** 3)) * (r if coef_mode == 'folded' else 1.0) * c_hist
    # shift rows
    for i in range(1, S):
        A[i, i - 1] = 1.0
    return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dt', type=float, default=5.0e-3)
    ap.add_argument('--nu', type=float, default=1.025e-4)
    ap.add_argument('--mu', type=float, default=0.02)
    ap.add_argument('--Lx', type=float, default=12.5663706)
    ap.add_argument('--Nx', type=int, default=512)
    ap.add_argument('--S', type=int, default=7)
    ap.add_argument('--sigma-mode', default='cfl',
                    help="'cfl': sigma = U*k with U from --u-rms; "
                         "'dtsig': sweep dt*sigma directly")
    ap.add_argument('--u-rms', type=float, default=1.0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    S = args.S
    W = w_unit(S)
    two_pi_L = 2.0 * np.pi / args.Lx
    shells = [30, 60, 100, 150, 171, 200, 241, 300]

    lines = []
    lines.append("FULL HISTORY COMPANION vs the 2x2 certificate")
    lines.append(f"dt={args.dt:g} S={S} nu={args.nu:g} mu={args.mu:g} "
                 f"Lx={args.Lx:.4f} Nx={args.Nx} u_rms={args.u_rms:g}")
    lines.append("")
    lines.append(f"{'shell':>6} {'|k|':>8} {'dt*sig':>8} "
                 f"{'rho_2x2':>10} {'rho_FULL':>10} {'ratio':>9}  verdict")
    lines.append('-' * 78)
    worst = (0.0, None)
    for m in shells:
        kphys = m * two_pi_L
        L = complex(-args.nu * kphys ** 2 - args.mu, 0.0)
        sigma = args.u_rms * kphys
        dts = args.dt * sigma
        # --- the 2x2 the certificate used (no history) ---
        E_adv = 1j * sigma
        cfold = args.dt ** 2 / 12.0
        E_clo = E_adv * cfold * L ** 2
        r2 = 1.0 / (1.0 - 0.5 * args.dt * L + (args.dt ** 3 / 12.0) * L ** 3)
        a2 = r2 * (1.0 + 0.5 * args.dt * L)
        b2 = args.dt * r2 * (1.5 * E_adv + E_clo)
        c22 = -0.5 * args.dt * r2 * E_adv
        A2 = np.array([[a2 + b2, c22], [1.0, 0.0]], dtype=np.complex128)
        rho2 = float(np.max(np.abs(np.linalg.eigvals(A2))))
        # --- the FULL companion including the S-deep history ---
        AF = companion(args.dt, L, sigma, S, W)
        rhoF = float(np.max(np.abs(np.linalg.eigvals(AF))))
        v = 'UNSTABLE (2x2 blind)' if (rhoF > 1.0 and rho2 <= 1.0) else \
            ('unstable both' if rhoF > 1.0 else 'stable')
        if rhoF > worst[0]:
            worst = (rhoF, (m, kphys, dts, rho2))
        lines.append(f"{m:6d} {kphys:8.1f} {dts:8.3f} {rho2:10.5f} "
                     f"{rhoF:10.4f} {rhoF / max(rho2, 1e-30):9.2e}  {v}")
    lines.append('')
    if worst[1] is not None:
        m, kphys, dts, rho2 = worst[1]
        lines.append(
            f"WORST: shell {m} (|k|={kphys:.0f}) rho_FULL={worst[0]:.4f} "
            f"vs rho_2x2={rho2:.5f}")
        if worst[0] > 1.0 and rho2 <= 1.0:
            lines.append(
                "  => CONFIRMED: the closed system is LINEARLY UNSTABLE in the "
                "history subspace, and the 2x2 companion the certificate "
                "penalized cannot see it. The unstable eigenvalue enters "
                "through the time-FD weights W[1]/dt and W[2]/dt^2 acting on "
                "the S-deep self-generated history.")
            lines.append(
                "  => the certificate must be rebuilt on this (S x S) "
                "companion; penalizing rho_2x2 was structurally incapable of "
                "bounding the instability.")
    text = '\n'.join(lines)
    print(text, flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + '\n')


if __name__ == '__main__':
    main()
