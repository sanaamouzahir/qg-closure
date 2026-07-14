#!/usr/bin/env python
r"""modulation.py -- inlet-velocity modulation tables for the SGS-closure branch.

Charter: docs/briefs/SGS_closure_supervisor_brief.md (S3) + AMENDMENT_01 (SF, SH).

Generates the per-step inlet table U_of_t.npz consumed by the solver's
bc.inlet_table hook: Re(t) per one of five signals, U(t) = U_mid * Re(t)/Re_mid
(= 5.1282e-4 * Re to 5 digits; U(3900) = 2.0 exactly). The table is generated at
the SOLVER dt of the consuming run -- direct index lookup U[n] at step n, no
runtime interpolation. All signals hold Re = Re_mid for t < T_wait and are
continuous at t = T_wait, except telegraph which jumps by construction.

dt-consistency across the dt sweep (charter S3.2, Gate 1 check 3):
  * OU: the charter recursion Re_{n+1} = Re_mid + rho (Re_n - Re_mid)
    + sqrt(1 - rho^2) sigma z_n, rho = exp(-dt/tau_OU), is realized on a fixed
    micro-grid DT_MICRO = 1.25e-4 and subsampled to the requested dt (which must
    be an integer multiple of DT_MICRO). Subsampling a discrete OU is exact --
    the restriction to every k-th time is the same recursion with rho^k -- so
    tables generated at different dt from the same seed agree EXACTLY (bitwise)
    at shared times; the Gate 1 overlay is a strict identity check, not a
    statistical one. The hard clip to [Re_min, Re_max] is applied on the micro
    grid, so every dt table is the restriction of ONE clipped micro path.
  * telegraph: dwell times are drawn once from the seed as continuous
    Exponential(tau_dwell) variates; switch times, not step counts, define the
    signal, so any-dt tables agree exactly at shared times.

Physics constants are fixed by the charter S2 and are NOT CLI knobs:
  D = 2 * mask.r = 1.2566370614359172, St_ref = 0.21, U_mid = 2.0,
  Re_mid = 3900, Re_amp = 1700, Re range [2200, 5600],
  T_shed_mid = D/(St*U_mid) = 2.9920 (exact 2.99199300...).
Derived signal parameters (charter table, S3.1):
  sine period P = 5 T_shed_mid = 14.960; tau_OU = 5 T_shed_mid = 14.960,
  sigma_OU = 0.2 Re_amp = 340; tau_dwell = 4 T_shed_mid = 11.968.
The exact (unrounded) derived values are used; the charter's 4-digit roundings
are their display forms. AMENDMENT_01 SF (Gate D-1) needs a constant-Re table at
Re = 200: use `--signal const --re-const 200`.

Usage (from training/):
    python modulation.py --signal sine --dt 2.5e-4 --T 120 --t-wait 30 \
        --out U_of_t_sine.npz
    python modulation.py --signal ou --dt 1.25e-4 --T 15 --t-wait 3 \
        --out U_of_t_ou_gate1.npz --seed 20260707

Outputs: <out>.npz {t [N+1], Re [N+1], U [N+1], meta json-str} and <out>.png
(Re(t) + T_shed(t) twin axes, T_wait marked). float64 throughout.
"""
from __future__ import annotations
import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np

# --- charter S2 constants (fixed, non-negotiable; not CLI knobs) -------------
MASK_R = 0.628318530717959          # pi/5, from flow_past_cylinder_sponge.yaml
D = 2.0 * MASK_R                    # 1.2566370614359172
ST_REF = 0.21
U_MID = 2.0
RE_MID = 3900.0
RE_AMP = 1700.0
RE_MIN, RE_MAX = 2200.0, 5600.0
NU = U_MID * D / RE_MID             # 6.4443e-4 (YAML display form)
U_PER_RE = U_MID / RE_MID           # 5.1282e-4 (exact 2/3900); U = U_PER_RE*Re
T_SHED_MID = D / (ST_REF * U_MID)   # 2.9920 (exact 2.991993...)

# --- charter S3.1 signal parameters ------------------------------------------
P_SINE = 5.0 * T_SHED_MID           # 14.960
RAMP_LEN = 15.0                     # tau in [0, 15] -> hold at RE_MAX
TAU_OU = 5.0 * T_SHED_MID           # 14.960
SIGMA_OU = 0.2 * RE_AMP             # 340
TAU_DWELL = 4.0 * T_SHED_MID        # 11.968
CHIRP_F1_FACT = 3.0                 # chirp sweeps f: 1/P_SINE -> 3/P_SINE (OOD case, Sanaa 2026-07-14)
DEFAULT_SEED = 20260707

# OU micro-grid: finest dt in the whole campaign (Phase C dt study lower end).
DT_MICRO = 1.25e-4


def _steps(x, dt, what):
    n = int(round(x / dt))
    if abs(n * dt - x) > 1e-9 * max(1.0, abs(x)):
        raise SystemExit(f'{what}={x} is not an integer multiple of dt={dt}')
    return n


def signal_re(signal, t, n_wait, dt, seed, re_const, switch_smooth_steps=0):
    """Re(t_n) on the full table grid t (t_n = n*dt). Modulation for n >= n_wait."""
    re = np.full(t.shape, RE_MID, dtype=np.float64)
    tau = t[n_wait:] - t[n_wait]                      # 0, dt, 2dt, ...

    if signal == 'const':
        re[:] = re_const
        return re

    if signal == 'sine':
        re[n_wait:] = RE_MID + RE_AMP * np.sin(2.0 * np.pi * tau / P_SINE)
    elif signal == 'chirp':
        # OOD generalization case (Sanaa 2026-07-14): amplitude/band IDENTICAL
        # to sine (Re_mid +- Re_amp), frequency sweeps linearly from the trained
        # sine rate f0 = 1/P_SINE to CHIRP_F1_FACT*f0 across the modulation
        # horizon — out-of-distribution in TIME STRUCTURE only, not in Re range.
        # Deterministic (seed unused). f(tau) = f0 + (f1-f0)*tau/H;
        # phase = 2*pi*integral_0^tau f = 2*pi*tau*(f0 + 0.5*(f1-f0)*tau/H).
        horizon = t[-1] - t[n_wait]
        f0 = 1.0 / P_SINE
        f1 = CHIRP_F1_FACT / P_SINE
        re[n_wait:] = RE_MID + RE_AMP * np.sin(
            2.0 * np.pi * tau * (f0 + 0.5 * (f1 - f0) * tau / horizon))
    elif signal == 'ramp':
        re[n_wait:] = RE_MID + (RE_MAX - RE_MID) * np.minimum(tau / RAMP_LEN, 1.0)
    elif signal == 'ou':
        k = int(round(dt / DT_MICRO))
        if abs(k * DT_MICRO - dt) > 1e-12:
            raise SystemExit(f'OU: dt={dt} must be an integer multiple of DT_MICRO={DT_MICRO}')
        n_micro = (len(t) - 1 - n_wait) * k
        rho = math.exp(-DT_MICRO / TAU_OU)
        amp = math.sqrt(1.0 - rho * rho) * SIGMA_OU
        z = np.random.Generator(np.random.PCG64(seed)).standard_normal(n_micro)
        x = np.empty(n_micro + 1, dtype=np.float64)
        x[0] = RE_MID                                  # continuous at T_wait
        for j in range(n_micro):                       # hard clip on the micro grid
            x[j + 1] = min(max(RE_MID + rho * (x[j] - RE_MID) + amp * z[j], RE_MIN), RE_MAX)
        re[n_wait:] = x[::k]                           # exact OU subsampling
    elif signal == 'telegraph':
        rng = np.random.Generator(np.random.PCG64(seed))
        horizon = t[-1] - t[n_wait]
        switches = []
        acc = 0.0
        while acc <= horizon:                          # dwell draws depend only on the seed
            acc += rng.exponential(TAU_DWELL)
            switches.append(acc)
        n_flips = np.searchsorted(np.asarray(switches), tau, side='right')
        re[n_wait:] = np.where(n_flips % 2 == 0, RE_MAX, RE_MIN)   # starts at Re_max
        if switch_smooth_steps > 0:
            # FPC-tel rerun fix (Sanaa GO 2026-07-11): replace each instantaneous
            # level jump — INCLUDING the T_wait entry jump Re_mid -> Re_max, which
            # produced the first Cd~1378 penalty impulse at t=30.0 — by a linear
            # ramp over switch_smooth_steps solver steps. Switch TIMES (seed-drawn,
            # dt-consistent) are unchanged; only the jump shape is mollified, so
            # outside the ramp windows the table equals the hard-telegraph table
            # exactly at shared times.
            w = switch_smooth_steps * dt
            events = [(0.0, RE_MID, RE_MAX)]           # the T_wait entry jump
            lev = (RE_MAX, RE_MIN)
            for i, s in enumerate(switches):
                if s > horizon:
                    break
                events.append((s, lev[i % 2], lev[(i + 1) % 2]))
            gaps = np.diff([e[0] for e in events])
            if gaps.size and gaps.min() <= w:
                raise SystemExit(f'telegraph smoothing window {w} overlaps a dwell '
                                 f'(min gap {gaps.min()}) — reduce --switch-smooth-steps')
            if events[-1][0] + w > horizon:
                # a ramp truncated at t[-1] would end mid-jump SILENTLY otherwise
                raise SystemExit(
                    f'telegraph smoothing window {w} spans past the table end '
                    f'(last event {events[-1][0]}, horizon {horizon}) — '
                    f'reduce --switch-smooth-steps or extend T')
            re_mod = re[n_wait:]                        # view; writes land in re
            for s, lo, hi in events:
                sel = (tau >= s) & (tau < s + w)
                re_mod[sel] = lo + (hi - lo) * (tau[sel] - s) / w
    else:
        raise SystemExit(f'unknown signal {signal}')
    return re


def git_sha():
    for g in ('/opt/rocks/bin/git', 'git'):
        try:
            return subprocess.run([g, 'rev-parse', 'HEAD'], cwd=Path(__file__).parent,
                                  capture_output=True, text=True, timeout=10).stdout.strip() or 'unknown'
        except (OSError, subprocess.SubprocessError):
            continue
    return 'unknown'


def make_plot(out_png, t, re, u, t_wait, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, re, color='tab:blue', lw=1.0)
    ax.set_xlabel('t'); ax.set_ylabel('Re(t)', color='tab:blue')
    ax.axvline(t_wait, color='k', ls='--', lw=0.8)
    ax.text(t_wait, ax.get_ylim()[1], ' T_wait', va='top', fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(t, D / (ST_REF * u), color='tab:red', lw=1.0, alpha=0.7)
    ax2.set_ylabel(r'$T_{shed}(t) = D/(St\,U(t))$', color='tab:red')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--signal', required=True,
                   choices=['const', 'sine', 'ramp', 'ou', 'telegraph', 'chirp'])
    p.add_argument('--dt', type=float, required=True,
                   help='SOLVER dt of the consuming run (table is index-matched)')
    p.add_argument('--T', type=float, required=True)
    p.add_argument('--t-wait', type=float, required=True,
                   help='hold Re=Re_mid until this time; modulation starts here')
    p.add_argument('--out', required=True, help='output .npz path')
    p.add_argument('--seed', type=int, default=DEFAULT_SEED)
    p.add_argument('--re-const', type=float, default=RE_MID,
                   help='const signal only: constant Re value (Gate D-1 uses 200)')
    p.add_argument('--switch-smooth-steps', type=int, default=0,
                   help='telegraph only: linear ramp over this many solver steps at '
                        'each level switch, incl. the T_wait entry jump (0 = hard '
                        'jump, the pre-2026-07-11 behaviour)')
    args = p.parse_args()
    if args.switch_smooth_steps and args.signal != 'telegraph':
        raise SystemExit('--switch-smooth-steps only applies to --signal telegraph')

    dt, T, t_wait = float(args.dt), float(args.T), float(args.t_wait)
    n_total = _steps(T, dt, 'T')
    n_wait = _steps(t_wait, dt, 'T_wait')
    if n_wait > n_total:
        raise SystemExit('T_wait > T')

    t = np.arange(n_total + 1, dtype=np.float64) * dt
    re = signal_re(args.signal, t, n_wait, dt, args.seed, float(args.re_const),
                   args.switch_smooth_steps)
    u = U_PER_RE * re

    if args.signal != 'const':
        assert re.min() >= RE_MIN - 1e-9 and re.max() <= RE_MAX + 1e-9, \
            f'Re out of [{RE_MIN},{RE_MAX}]: [{re.min()},{re.max()}]'
    assert abs(U_PER_RE * RE_MID - U_MID) == 0.0   # U(3900) = 2.0 exactly

    meta = dict(signal=args.signal, dt=dt, T=T, T_wait=t_wait, seed=args.seed,
                re_const=float(args.re_const), D=D, St_ref=ST_REF, U_mid=U_MID,
                Re_mid=RE_MID, Re_amp=RE_AMP, Re_min=RE_MIN, Re_max=RE_MAX,
                nu=NU, U_per_Re=U_PER_RE, T_shed_mid=T_SHED_MID, P_sine=P_SINE,
                ramp_len=RAMP_LEN, tau_OU=TAU_OU, sigma_OU=SIGMA_OU,
                tau_dwell=TAU_DWELL, dt_micro=DT_MICRO,
                chirp_f1_fact=CHIRP_F1_FACT,
                switch_smooth_steps=args.switch_smooth_steps, git_sha=git_sha(),
                convention='U[n] is the inlet at step n (t_n = n*dt); '
                           'OU/telegraph paths are dt-consistent by micro-grid/'
                           'switch-time construction (see module docstring)')
    out = Path(args.out)
    np.savez(out, t=t, Re=re, U=u, meta=json.dumps(meta))
    make_plot(out.with_suffix('.png'), t, re, u, t_wait,
              f'{args.signal}  dt={dt:g}  T={T:g}  T_wait={t_wait:g}  seed={args.seed}')

    print(f'{out}  N={n_total + 1}  Re[{re.min():.1f},{re.max():.1f}]  '
          f'U[{u.min():.4f},{u.max():.4f}]  U(t=0)={u[0]:.6f}  '
          f'sha={meta["git_sha"][:9]}')


if __name__ == '__main__':
    main()
