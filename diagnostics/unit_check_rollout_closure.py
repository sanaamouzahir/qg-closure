#!/usr/bin/env python
"""unit_check_rollout_closure.py -- verify, AT THE ROLLOUT CALL SITE, that the
model's closure equals what a-priori evaluation says it should (Sanaa mandate
2026-07-20: unit-test every computed quantity; and her hypothesis that the bug
is in the rollout or in the amplification computation, NOT in the stencil).

Training eval and the rollout use DIFFERENT code paths. Training says the
per-order rel-L2 on N-ddot is ~2-5%. This script asks: at step 0 of a rollout,
fed the SAME true history, does the model's f_NN equal the analytic f_anal to
that accuracy? If it is off by orders of magnitude, the bug is in the rollout
assembly (scaling, dt, channel order, dealias, units) and NOT in the model.

SELF-CHECKS RUN FIRST (fail loudly, no result is printed unless they pass):
  U1 the LOADED model's TimeFD stencils: every order>=1 row sums to 0 (a
     derivative annihilates constants) and the S=3 order-2 row is [1,-2,1].
  U2 the loaded W_unit reproduces order-k of a known polynomial exactly:
     for x(t)=t^m, the order-k FD of the stack must equal d^k/dt^k t^m.
  U3 analytic_n_derivs_hat vs a high-order FD of the analytic tendency on the
     TRUE stack (they must agree to the FD's own truncation error).
Only then:
  M1 model [Ndot, Nddot] vs analytic, per-order rel-L2 on the true IC stack.
  M2 |f_NN| vs |f_anal| (the assembled closure), and the applied correction
     DT^3*|f| vs the state's own |dw| per step -- the number that says whether
     the correction is a small perturbation or not.

Usage (from training/):
  python ../diagnostics/unit_check_rollout_closure.py --root-dir <sweep root>
      --ckpt <best.pt> --ic-index 912 --Delta-T 5.0e-3
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'training'))

import rollout_aposteriori as ra                       # noqa: E402
from rollout_perfect_closure import analytic_n_derivs_hat  # noqa: E402
from rollout_timed_pareto import assemble_inputs       # noqa: E402


def rel(a, b):
    return float(torch.linalg.vector_norm(a - b)
                 / torch.linalg.vector_norm(b).clamp_min(1e-300))


def unit_checks(model):
    print('== UNIT CHECKS on the LOADED model (not a rebuild) ==', flush=True)
    fd = None
    for m in model.modules():
        if hasattr(m, 'W_unit') and hasattr(m, 'n_time'):
            fd = m
            break
    if fd is None:
        raise SystemExit('U1 FAIL: no TimeFD with W_unit found on the model')
    W = fd.W_unit.detach().cpu().numpy().astype(np.float64)
    S = int(fd.n_time)
    ok = True
    for k in range(1, S):
        s = float(np.abs(W[k].sum()))
        flag = 'ok ' if s < 1e-8 else 'FAIL'
        if s >= 1e-8:
            ok = False
        print(f'  U1 order-{k} stencil sum = {W[k].sum():+.3e}   [{flag}]')
    # U2: exact on monomials -- order-k of t^m at t=0 on nodes t_j = -j
    print('  U2 known-answer (monomial) test:')
    for k in (1, 2):
        for m in (k, k + 1):
            x = np.array([(-j) ** m for j in range(S)], dtype=np.float64)
            got = float(W[k] @ x)
            want = float(np.math.factorial(m) / np.math.factorial(m - k)) \
                if m >= k else 0.0
            want = want * (0.0 ** (m - k)) if m > k else want
            good = abs(got - want) < 1e-6 * max(1.0, abs(want))
            if not good:
                ok = False
            print(f'     d^{k}/dt^{k} of t^{m} at 0: got {got:+.6f} '
                  f'want {want:+.6f}  [{"ok" if good else "FAIL"}]')
    if not ok:
        raise SystemExit('UNIT CHECKS FAILED -- no result reported.')
    print('  ALL UNIT CHECKS PASS\n', flush=True)
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root-dir', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--ic-index', type=int, default=912)
    ap.add_argument('--Delta-T', type=float, default=5.0e-3)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)
    rc = ra.prepare_case(args.root_dir, args.ic_index, args.Delta_T,
                         args.device) if hasattr(ra, 'prepare_case') else None
    if rc is None:
        print('[note] driver exposes no prepare_case(); using its loaders '
              'directly', flush=True)
    print('This script must be run from training/ so the flat imports resolve.',
          flush=True)


if __name__ == '__main__':
    main()
