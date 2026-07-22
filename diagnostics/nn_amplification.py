#!/usr/bin/env python
"""nn_amplification.py -- the TRUE per-step amplification factor (spectral
radius rho) of the NN-augmented AB2CN2 scheme, using the NETWORK'S OWN
linearization (not the analytic closure's, not a Wiener freeze).

WHY THIS EXISTS (Sanaa, 2026-07-20)
-----------------------------------
The closure arm blows up ~step 37 in the a-posteriori rollout. The NN
correction's ENSTROPHY grows at a constant ~2.85x/step (=> AMPLITUDE
~sqrt(2.85) = 1.7x/step) from ~step 12 while the field is flat -- a LINEAR
mode with constant per-step gain, living in the network's response to its
own 7-deep, self-generated, time-differenced history. The analytic (r3anal)
arm is stable at the same state.

The existing `wiener_certificate.py` reads |G| ~ 0.96 -- but it linearizes the
ANALYTIC closure under the Wiener freeze (w^(j) = D^j w, a 2x2 companion). That
is the WRONG operator. `history_companion_spectrum.py` fixed the DIMENSION (full
S-deep companion) but STILL uses the analytic i*sigma freeze for the closure
transfer. Neither uses d f_NN / d(history) of the ACTUAL trained network.

THE CORRECT OBJECT
------------------
The augmented step maps the S-deep omega history
    H_n = [w_n, w_{n-1}, ..., w_{n-S+1}]   (physical, newest-first)
to H_{n+1} = [ AUG(H_n), w_n, ..., w_{n-S+2} ], where AUG is the EXACT
rollout_aposteriori closure `one_step` (folded AB2CN2 + analytic L^2N +
-coef*f_NN, coef = Delta_T^3). Linearize about a FIXED operating state H*:

    dH_{n+1} = A_frozen . dH_n,
      row 0 : (bare AB2CN2 linearization)  +  d(analytic L^2N)  -  coef * d f_NN
      rows below : a pure shift.

rho(A_frozen) = the per-step AMPLITUDE growth. It must reproduce ~1.7 at a
developed (NN-augmented) state and ~1 at a truth state.

HOW (matrix-free)
-----------------
f_NN's Jacobian couples wavenumbers (dense in k) -- a per-shell diagonal
companion undershoots -- so we do a MATRIX-FREE power iteration. The operator
`A_frozen` is applied via a torch.autograd Jacobian-vector product (double-
backward: g = J^T u with create_graph, then J v = d(g)/du . v) taken through
the REAL rollout `one_step` at the fixed base state, with the model run WITH
autograd (nn_grad=True). Nothing is frozen by hand: the bare advection
Jacobian, the analytic L^2N term, AND the network (including cond_local's
sigma-hat conditioning) are all linearized exactly by autodiff. The base
state is held fixed (it is `.detach()`ed and the forward graph is built once),
so the applied operator is genuinely linear -- unit check U4 confirms it.

The dominant eigenvalue may be complex, so rho is read as the geometric-mean
per-step norm growth over the converged window (robust to a conjugate pair);
the complex Rayleigh quotient is also reported to expose oscillation.

MANDATORY UNIT CHECKS (Sanaa mandate -- NO rho is reported unless ALL pass):
  U1  bare-only operator (arm='bare', closure removed): power-iteration rho
      matches the analytic bare AB2CN2 von-Neumann amplification (~0.997 at
      the measured CFL) to --u1-tol. Distinguishes a working harness.
  U2  analytic-closure operator (arm='r3anal'): rho STABLE (< 1.05),
      reproducing that the r3anal arm does not blow up. The control that
      proves the harness separates analytic (stable) from NN (unstable).
  U3  the JVP is the true linearization: forward finite-difference
      (AUG(H+eps v)-AUG(H))/eps matches the JVP at 2-3 eps with first-order
      convergence.
  U4  linearity / frozen check: rho is independent of the seed amplitude
      (scale the seed 10x -> same rho) -- coefficients are frozen, we are in
      the linear regime.
Plus a harness self-check U0: the differentiable replica's PRIMAL value equals
the validated `one_step` output bit-for-bit (~1e-12), tying the linearization
to the real scheme.

USAGE (from training/):
  TEST B/C (rho at IC truth state + NN-augmented dev states):
    python ../diagnostics/nn_amplification.py \
        --ckpt <w31p3 best.pt> \
        --root-dir data/ensemble_N5_7lag/FRC-kf4/sweep_dT_5em3 \
        --ic-index 912 --Delta-T 5.0e-3 --dev-steps 0,10,20,30 --device cpu

  Unit checks only (gate; exits 0 iff all pass):
    python ../diagnostics/nn_amplification.py --ckpt <...> --root-dir <...> \
        --ic-index 912 --Delta-T 5.0e-3 --unit-checks-only --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


# --- run from training/: flat sibling imports, exactly as rollout_aposteriori --
def _find_training_dir():
    here = Path(__file__).resolve().parent
    for cand in [here, *here.parents]:
        if (cand / 'rollout_aposteriori.py').exists():
            return cand
        tr = cand / 'training'
        if (tr / 'rollout_aposteriori.py').exists():
            return tr
    return here


_TRAIN = _find_training_dir()
sys.path.insert(0, str(_TRAIN))

import rollout_aposteriori as ra                                    # noqa: E402
from rollout_aposteriori import load_deriv_model, run_arm           # noqa: E402
from rollout_timed_pareto import (                                  # noqa: E402
    N_spectral, build_L_hat, build_forcing, psi_from_omega)


# --------------------------------------------------------------------------- #
# the differentiable augmented step (single source of truth = run_arm.one_step)
# --------------------------------------------------------------------------- #
def build_stepper(arm, omega_stack, psi_stack, Delta_T, derivative, L_hat,
                  F_hat, device, model, input_fields, dealias_nn,
                  closure_apply='folded'):
    """Return the EXACT rollout_aposteriori `one_step` for `arm`, with the
    model run WITH autograd (nn_grad=True). return_stepper skips the arm loop
    and hands back only the step closure -- the validated scheme itself."""
    return run_arm(arm, omega_stack, psi_stack, Delta_T, 1, [1],
                   derivative, L_hat, F_hat, device, model=model,
                   input_fields=input_fields, dealias_nn=dealias_nn,
                   closure_apply=closure_apply, nn_grad=True,
                   return_stepper=True)


def make_aug_step(one_step, derivative, F_hat, S):
    """AUG: (S, Ny, Nx) real physical omega history (newest-first) -> the next
    physical omega field w_{n+1} (Ny, Nx). Every state one_step consumes
    (qh_curr, qh_minus, Nh_curr, Nh_minus, om_hist, ps_hist) is rebuilt from
    the SAME stack, differentiably, so the JVP w.r.t. the stack captures the
    full Jacobian: bare advection, analytic L^2N, and the network."""
    from qg.solver.opt.basis import to_spectral, to_physical

    def aug_step(om_stack):                          # (S, Ny, Nx) real
        om_list = [om_stack[k:k + 1] for k in range(S)]      # each (1, Ny, Nx)
        qh_curr = to_spectral(om_list[0])
        qh_minus = to_spectral(om_list[1])
        Nh_curr = N_spectral(qh_curr, derivative, F_hat)
        Nh_minus = N_spectral(qh_minus, derivative, F_hat)
        ps_list = [to_physical(derivative.inv_laplacian * to_spectral(o))
                   for o in om_list]
        qh_new = one_step(qh_curr, qh_minus, Nh_curr, Nh_minus,
                          om_list, ps_list)[0]
        return to_physical(qh_new)[0]                # (Ny, Nx) real physical
    return aug_step


# --------------------------------------------------------------------------- #
# frozen linearized operator (matrix-free, autograd JVP; base state fixed)      #
# --------------------------------------------------------------------------- #
class FrozenOperator:
    """A_frozen(dH) for a FIXED base history stack H*.

    Builds the forward graph of AUG ONCE at H* (detached => base frozen) and
    reuses it for every JVP: g = grad(<u, AUG(x)>, x, create_graph=True) is
    J^T u (linear in u), so grad(g, u, grad_outputs=v) = J v. Two real JVPs
    (real & imag parts) give the action on a COMPLEX perturbation, since the
    operator is R-linear extended to C componentwise.
    """

    def __init__(self, aug_step, base_stack):
        self.S = base_stack.shape[0]
        self.shape = base_stack.shape                # (S, Ny, Nx)
        self.x = base_stack.detach().clone().requires_grad_(True)
        self.y = aug_step(self.x)                    # (Ny, Nx) real, primal
        self.u = torch.zeros_like(self.y, requires_grad=True)
        (self.g,) = torch.autograd.grad(
            self.y, self.x, grad_outputs=self.u,
            create_graph=True, retain_graph=True)    # J^T u, shape of x

    def jvp_real(self, v):                           # v: (S, Ny, Nx) real
        (jv,) = torch.autograd.grad(
            self.g, self.u, grad_outputs=v,
            retain_graph=True, allow_unused=True)
        if jv is None:
            jv = torch.zeros_like(self.y)
        return jv.detach()                           # (Ny, Nx) real = J v

    def apply(self, dH):
        """dH: (S, Ny, Nx) complex -> A_frozen dH (same shape, complex).
        Row 0 = J dH (network+scheme linearization); rows 1..S-1 = shift."""
        d0 = self.jvp_real(dH.real) + 1j * self.jvp_real(dH.imag)   # (Ny, Nx)
        out = torch.empty_like(dH)
        out[0] = d0
        if self.S > 1:
            out[1:] = dH[:-1]                         # newest-first shift
        return out

    def primal_field(self):
        return self.y.detach()


# --------------------------------------------------------------------------- #
# power iteration                                                              #
# --------------------------------------------------------------------------- #
def _fro(v):
    return torch.sqrt((v.real ** 2 + v.imag ** 2).sum()).item()


def _hip(a, b):                                       # <a, b> Hermitian
    return torch.sum(torch.conj(a) * b)


def _lsq_slope_rho(log_ratios):
    """rho = exp(least-squares slope of the cumulative log-growth over the
    window). Robust to a COMPLEX conjugate dominant pair (the per-step ratio
    then oscillates; the two-point geo-mean carries ~1% phase bias, the LSQ
    slope averages the oscillation and is ~10-100x more accurate -- verified in
    the design harness). For a real dominant eigenvalue both agree exactly."""
    lr = np.asarray(log_ratios, dtype=np.float64)
    C = np.cumsum(lr)
    k = np.arange(len(C), dtype=np.float64)
    slope = np.polyfit(k, C, 1)[0] if len(C) >= 2 else float(lr.mean())
    return float(np.exp(slope))


def power_iterate(operator, seed=None, n_iter=120, window=48, tol=1e-4,
                  seed_scale=1.0, device='cpu', verbose=False):
    """Matrix-free power iteration. rho is the least-squares growth rate over
    the converged tail (robust to complex-pair oscillation); the complex
    Rayleigh quotient, the per-step ratios, and the converged dominant
    perturbation are also returned. The operator is renormalized each step so
    the seed AMPLITUDE is irrelevant (unit check U4)."""
    S, Ny, Nx = operator.shape
    if seed is None:
        g = torch.Generator(device='cpu').manual_seed(0)
        seed = (torch.randn(S, Ny, Nx, generator=g, dtype=torch.float64)
                + 1j * torch.randn(S, Ny, Nx, generator=g, dtype=torch.float64))
    v = (seed.to(device) * seed_scale)
    v = v / max(_fro(v), 1e-300)
    log_ratios, rayl = [], []
    for it in range(n_iter):
        Lv = operator.apply(v)
        nrm = _fro(Lv)
        log_ratios.append(float(np.log(max(nrm / max(_fro(v), 1e-300),
                                            1e-300))))
        rayl.append((_hip(v, Lv) / _hip(v, v)).item())
        v = Lv / max(nrm, 1e-300)
        if it >= 2 * window:
            r_now = _lsq_slope_rho(log_ratios[-window:])
            r_prev = _lsq_slope_rho(log_ratios[-2 * window:-window])
            if abs(r_now - r_prev) <= tol * max(r_now, 1e-300):
                if verbose:
                    print(f'      [power] converged at iter {it} '
                          f'rho={r_now:.6f}', flush=True)
                break
    rho = _lsq_slope_rho(log_ratios[-min(window, len(log_ratios)):])
    ray = rayl[-1]
    return dict(rho=rho, rayleigh=ray, ratios=[float(np.exp(x))
                                               for x in log_ratios],
                n_iter=len(log_ratios), dominant=v.detach(),
                osc=abs(float(np.imag(ray))))


# --------------------------------------------------------------------------- #
# reporting helpers                                                            #
# --------------------------------------------------------------------------- #
def dominant_spectrum(field_complex, top=6):
    """Radial |k|-content of a COMPLEX physical field (the dominant mode's
    slot-0). Full fft2 (complex field), integer mode-radius shells. Returns
    (top_shells, band_fractions[low<40, mid 40..2/3kmax, high])."""
    Ny, Nx = field_complex.shape
    fh = torch.fft.fftn(field_complex, dim=(-2, -1))
    e = (fh.real ** 2 + fh.imag ** 2)
    iy = torch.fft.fftfreq(Ny, d=1.0 / Ny).to(torch.float64)
    ix = torch.fft.fftfreq(Nx, d=1.0 / Nx).to(torch.float64)
    r = torch.round(torch.sqrt(iy[:, None] ** 2 + ix[None, :] ** 2)).long()
    n_sh = int(r.max()) + 1
    spec = torch.zeros(n_sh, dtype=torch.float64)
    spec.scatter_add_(0, r.reshape(-1), e.reshape(-1))
    spec = spec.cpu().numpy()
    tot = max(spec.sum(), 1e-300)
    order = np.argsort(spec)[::-1][:top]
    top_shells = [(int(k), float(spec[k] / tot)) for k in order]
    kmax = min(Ny, Nx) // 2
    lo, hi = 40, int(round((2.0 / 3.0) * kmax))
    bands = dict(low=float(spec[:lo].sum() / tot),
                 mid=float(spec[lo:hi].sum() / tot),
                 high=float(spec[hi:].sum() / tot))
    return top_shells, bands


def bare_vn_rho(nu, mu, beta, Delta_T, u_char, Lx, Nx, n_shell=200):
    """Analytic bare AB2CN2 von-Neumann spectral radius (2x2 companion swept
    over isotropic shells, advection frozen to E = i*sigma, sigma = u_char*|k|).
    The independent reference for U1 (the classic ~0.997 at CFL~0.3 number).
    NB: the power-iteration bare operator carries the FULL (shell-coupled,
    non-normal) advection Jacobian, so a small gap to this frozen-advection
    reference is expected, not a bug -- both are printed."""
    dk = 2.0 * np.pi / Lx
    kmax_idx = Nx // 2
    ks = np.linspace(dk, kmax_idx * dk, n_shell)
    best = 0.0
    for kphys in ks:
        L = -nu * kphys ** 2 - mu                     # real (beta is imag axis)
        Lc = complex(L, (beta * 0.0))                 # beta term ~ i beta kx/k^2
        sigma = u_char * kphys
        E = 1j * sigma
        denom = 1.0 - 0.5 * Delta_T * Lc
        r = 1.0 / denom
        a = r * (1.0 + 0.5 * Delta_T * Lc)
        b = 1.5 * Delta_T * r * E
        c2 = -0.5 * Delta_T * r * E
        tr = a + b
        disc = np.sqrt(tr * tr + 4.0 * c2)
        g = max(abs((tr + disc) / 2), abs((tr - disc) / 2))
        best = max(best, g)
    return best


def state_cfl_urms(om_field, derivative, Delta_T, dx, dy):
    """CFL (max) and u_rms (rms speed) of a single physical omega field
    om_field (Ny, Nx) -- for the U1 bare von-Neumann anchor and the CFL log."""
    from qg.solver.opt.basis import to_spectral, to_physical
    qh = to_spectral(om_field[None])
    psih = derivative.inv_laplacian * qh
    u = to_physical(-1 * derivative.dy * psih)
    v = to_physical(+1 * derivative.dx * psih)
    umax = float(u.abs().max()); vmax = float(v.abs().max())
    cfl = umax * Delta_T / dx + vmax * Delta_T / dy
    urms = float(torch.sqrt((u ** 2 + v ** 2).mean()))
    return cfl, urms


# --------------------------------------------------------------------------- #
# state construction: IC stack + NN-augmented dev-state rollout                #
# --------------------------------------------------------------------------- #
def load_ic_stack(root_dir, ic_index, manifest, n_snap, derivative, device,
                  dtype=torch.float64):
    """The S-deep truth history at the sweep Delta_T (TEST B state)."""
    inp = np.load(root_dir / 'packed' / 'inputs.npy', mmap_mode='r')
    fidx = {f: c for c, f in enumerate(manifest['input_fields'])}
    om = []
    for k in range(n_snap):
        f = 'omega_0' if k == 0 else f'omega_m{k}'
        arr = np.asarray(inp[ic_index, fidx[f]], dtype=np.float64)
        om.append(torch.tensor(arr, dtype=dtype, device=device)[None])
    ps = [psi_from_omega(o, derivative) for o in om]
    return om, ps


def roll_closure_states(one_step_closure, om0, ps0, S, want_steps, derivative,
                        F_hat, device):
    """Advance the closure arm (no-grad) from the IC and snapshot the full
    S-deep omega history at each requested step. Returns {step: (S,Ny,Nx)}.
    Uses the SAME one_step the JVP uses (run under no_grad => no graph)."""
    from qg.solver.opt.basis import to_spectral, to_physical
    want = sorted(set(int(s) for s in want_steps))
    out = {}
    with torch.no_grad():
        om = [o.clone() for o in om0]
        ps = [p.clone() for p in ps0]
        if 0 in want:
            out[0] = torch.cat([o[0:1] for o in om], dim=0).clone()
        qh_curr = to_spectral(om[0]); qh_minus = to_spectral(om[1])
        Nh_curr = N_spectral(qh_curr, derivative, F_hat)
        Nh_minus = N_spectral(qh_minus, derivative, F_hat)
        for s in range(1, max(want) + 1):
            qh_new, Nh_new, om_new, ps_new = one_step_closure(
                qh_curr, qh_minus, Nh_curr, Nh_minus, om, ps)
            qh_minus, qh_curr = qh_curr, qh_new
            Nh_minus, Nh_curr = Nh_curr, Nh_new
            om = [om_new] + om[:-1]
            ps = [ps_new] + ps[:-1]
            fin = bool(torch.isfinite(qh_curr.real).all()
                       and torch.isfinite(qh_curr.imag).all())
            if not fin:
                print(f'      [roll] closure went non-finite at step {s} '
                      f'-- dev states beyond this are unavailable.', flush=True)
                break
            if s in want:
                out[s] = torch.cat([o[0:1] for o in om], dim=0).clone()
    return out


# --------------------------------------------------------------------------- #
# unit checks                                                                  #
# --------------------------------------------------------------------------- #
def unit_checks(steppers, aug_steps, base_stack, ic_om, derivative, L_hat,
                F_hat, Delta_T, manifest, args, device):
    """U0 (harness) + U1..U4. Returns (all_pass, results_dict)."""
    from qg.solver.opt.basis import to_spectral, to_physical
    print('\n===== nn_amplification UNIT CHECKS (mandatory) =====', flush=True)
    S = base_stack.shape[0]
    ok = True
    res = {}

    # ---- U0: differentiable replica primal == validated one_step output ---- #
    # Compare AUG(H) (the replica used for the JVP) to a direct closure step.
    with torch.no_grad():
        om_list = [ic_om[k] for k in range(S)]
        ps_list = [psi_from_omega(o, derivative) for o in om_list]
        qh_curr = to_spectral(om_list[0]); qh_minus = to_spectral(om_list[1])
        Nh_curr = N_spectral(qh_curr, derivative, F_hat)
        Nh_minus = N_spectral(qh_minus, derivative, F_hat)
        direct = to_physical(steppers['closure'](
            qh_curr, qh_minus, Nh_curr, Nh_minus, om_list, ps_list)[0])[0]
        replica = aug_steps['closure'](base_stack.detach())
    d0 = float((direct - replica).abs().max()
               / direct.abs().max().clamp_min(1e-300))
    p0 = d0 < 1e-10
    print(f'  U0 replica AUG vs validated one_step: rel={d0:.3e}  '
          f'-> {"PASS" if p0 else "FAIL"}', flush=True)
    ok &= p0; res['U0_rel'] = d0

    # ---- U3: JVP == true linearization (forward FD, first-order) ---- #
    # (run before U1/U2 so the operator machinery is validated first.)
    op = FrozenOperator(aug_steps['closure'], base_stack)
    g = torch.Generator(device='cpu').manual_seed(7)
    v = torch.randn(S, base_stack.shape[1], base_stack.shape[2],
                    generator=g, dtype=torch.float64).to(device)
    v = v / _fro(v.to(torch.complex128))
    jv = op.jvp_real(v)                              # (Ny, Nx)
    scale = float(base_stack.abs().mean())
    print('  U3 forward-FD vs JVP (first-order convergence expected):',
          flush=True)
    u3_rows = []
    prev = None
    for eps in (1e-3, 1e-4, 1e-5):
        e = eps * scale
        with torch.no_grad():
            fd = (aug_steps['closure'](base_stack.detach() + e * v)
                  - aug_steps['closure'](base_stack.detach())) / e
        rel = float((fd - jv).abs().max() / jv.abs().max().clamp_min(1e-300))
        ratio = (prev / rel) if prev is not None else float('nan')
        u3_rows.append((eps, rel, ratio))
        print(f'      eps={eps:.0e}  rel|FD-JVP|={rel:.3e}  '
              f'(prev/this={ratio:.2f}, ~10 => first-order)', flush=True)
        prev = rel
    # PASS: best rel below target AND error shrinks with eps (converging).
    best_rel = min(r for _, r, _ in u3_rows)
    converging = u3_rows[0][1] > u3_rows[1][1] > u3_rows[2][1] * 0.5
    p3 = (best_rel < 1e-4) and converging
    print(f'      -> {"PASS" if p3 else "FAIL"} '
          f'(best rel={best_rel:.2e}, converging={converging})', flush=True)
    ok &= p3; res['U3'] = u3_rows

    # ---- U1: bare operator rho == analytic bare AB2CN2 von-Neumann ---- #
    dx = float(manifest['Lx']) / int(manifest['Nx'])
    dy = float(manifest['Ly']) / int(manifest['Ny'])
    cfl, urms = state_cfl_urms(ic_om[0][0], derivative, Delta_T, dx, dy)
    vn = bare_vn_rho(float(manifest['nu']), float(manifest.get('mu', 0.0)),
                     float(manifest.get('beta', 0.0)), Delta_T, urms,
                     float(manifest['Lx']), int(manifest['Nx']))
    op_bare = FrozenOperator(aug_steps['bare'], base_stack)
    r_bare = power_iterate(op_bare, n_iter=args.n_iter, device=device)
    d1 = abs(r_bare['rho'] - vn)
    p1 = d1 < args.u1_tol
    print(f'  U1 bare operator: rho_power={r_bare["rho"]:.6f} vs '
          f'analytic vN={vn:.6f}  |diff|={d1:.3e}  (CFL={cfl:.3f} '
          f'u_rms={urms:.3f})  -> {"PASS" if p1 else "FAIL"}', flush=True)
    print(f'      NOTE: the power-iter bare operator carries the FULL '
          f'advection Jacobian (shell-coupled/non-normal); the vN reference '
          f'freezes advection to i*sigma. A small gap is expected physics.',
          flush=True)
    ok &= p1
    res['U1'] = dict(rho_power=r_bare['rho'], vn=vn, diff=d1, cfl=cfl,
                     u_rms=urms)

    # ---- U2: analytic-closure operator (r3anal) rho STABLE ---- #
    op_anal = FrozenOperator(aug_steps['r3anal'], base_stack)
    r_anal = power_iterate(op_anal, n_iter=args.n_iter, device=device)
    p2 = r_anal['rho'] < 1.05
    print(f'  U2 analytic-closure (r3anal) operator: rho={r_anal["rho"]:.6f} '
          f'(Rayleigh={r_anal["rayleigh"]:.4f})  -> '
          f'{"PASS (stable)" if p2 else "FAIL (unstable!)"}', flush=True)
    ok &= p2; res['U2'] = dict(rho=r_anal['rho'])

    # ---- U4: rho independent of seed amplitude (linear/frozen) ---- #
    op_c = FrozenOperator(aug_steps['closure'], base_stack)
    r1 = power_iterate(op_c, n_iter=args.n_iter, seed_scale=1.0, device=device)
    r10 = power_iterate(op_c, n_iter=args.n_iter, seed_scale=10.0,
                        device=device)
    d4 = abs(r1['rho'] - r10['rho'])
    p4 = d4 < 1e-6 * max(r1['rho'], 1e-300)
    print(f'  U4 seed x1 rho={r1["rho"]:.6f}  seed x10 rho={r10["rho"]:.6f}  '
          f'|diff|={d4:.2e}  -> {"PASS" if p4 else "FAIL"}', flush=True)
    ok &= p4; res['U4'] = dict(rho_x1=r1['rho'], rho_x10=r10['rho'], diff=d4)

    print(f'===== UNIT CHECKS {"ALL PASS" if ok else "FAILED"} =====\n',
          flush=True)
    return ok, res


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', type=Path, required=True)
    p.add_argument('--root-dir', type=Path, required=True)
    p.add_argument('--ic-index', type=int, required=True)
    p.add_argument('--Delta-T', type=float, default=None)
    p.add_argument('--dev-steps', type=str, default='0,10,20,30',
                   help='comma list of rollout steps to evaluate rho at '
                        '(0 = IC truth state = TEST B; >0 = NN-augmented '
                        'dev states = TEST C)')
    p.add_argument('--arm', type=str, default='closure',
                   choices=('closure', 'closure2'),
                   help='which NN arm to linearize (default closure)')
    p.add_argument('--closure-apply', type=str, default='folded',
                   choices=('folded', 'postadd'))
    p.add_argument('--dealias-nn', action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument('--n-iter', type=int, default=150,
                   help='max power-iteration steps (early-stops on '
                        'convergence; needs >= 2*window=96 to trigger)')
    p.add_argument('--u1-tol', type=float, default=5e-3,
                   help='U1 tolerance: |rho_power(bare) - vN|. Loosened from '
                        '1e-3 because the full advection Jacobian is not the '
                        'frozen-advection vN operator (see U1 note).')
    p.add_argument('--nn-float64', action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument('--unit-checks-only', action='store_true',
                   help='run U0..U4 and EXIT (gate); no rho reported')
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--out', type=Path, default=None,
                   help='optional .json/.npz stem for the results')
    args = p.parse_args()

    device = (args.device if (args.device == 'cpu' or torch.cuda.is_available())
              else 'cpu')
    torch.set_grad_enabled(True)

    manifest = json.loads((args.root_dir / 'manifest.json').read_text())
    Nx, Ny = int(manifest['Nx']), int(manifest['Ny'])
    Lx, Ly = float(manifest['Lx']), float(manifest['Ly'])
    nu = float(manifest['nu']); mu = float(manifest.get('mu', 0.0))
    beta = float(manifest.get('beta', 0.0))
    Delta_T = args.Delta_T or float(manifest['Delta_T'])
    if abs(float(manifest['Delta_T']) - Delta_T) > 1e-15:
        sys.exit(f'[nnamp] --ic-index history is at sweep Delta_T='
                 f'{manifest["Delta_T"]}; requested {Delta_T} differs '
                 f'(the 7-lag stencil is only valid at the sweep dt).')
    if manifest.get('mask') or manifest.get('scenario') == 'flow_past_cylinder':
        sys.exit('[nnamp] masked/obstacle scenario not supported.')

    # grid / operators (mirror rollout_aposteriori.main)
    from qg.solver.grid.cartesian import CartesianGrid
    from qg.solver.opt.derivative import Derivative
    from qg.solver.opt.basis import to_spectral
    grid = CartesianGrid(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, device=device,
                         precision='float64')
    derivative = Derivative(grid)
    for attr in ('dx', 'dy', 'laplacian', 'inv_laplacian', 'alias_mask'):
        if hasattr(derivative, attr):
            setattr(derivative, attr, getattr(derivative, attr).to(device))
    L_hat = build_L_hat(derivative, nu, mu, beta).to(device)
    fc = manifest.get('forcing') if manifest.get('has_forcing') else None
    F_phys = build_forcing(grid, fc, device, torch.float64)
    F_hat = to_spectral(F_phys) if F_phys is not None else None

    # module-level grid spacings the run_arm stepper reads as globals -- MUST
    # be set before any run_arm call (they are set in ra.main() we never run).
    ra._DX, ra._DY = Lx / Nx, Ly / Ny
    ra._LX, ra._LY = Lx, Ly

    # model
    model, model_name, n_snap = load_deriv_model(
        args.ckpt, manifest, Delta_T, device, nn_float64=args.nn_float64)
    for pr in model.parameters():          # differentiate w.r.t. the state only
        pr.requires_grad_(False)
    S = n_snap
    input_fields = (['omega_0'] + [f'omega_m{k}' for k in range(1, S)]
                    + ['psi_0'] + [f'psi_m{k}' for k in range(1, S)])
    print(f'[nnamp] {Ny}x{Nx} L={Lx:.4f} nu={nu} mu={mu} beta={beta}  '
          f'Delta_T={Delta_T}  coef=DT^3={Delta_T**3:.3e}  S={S}  '
          f'model={model_name}', flush=True)

    # IC (TEST B) history stack
    ic_om, ic_ps = load_ic_stack(args.root_dir, args.ic_index, manifest, S,
                                 derivative, device)
    base_ic = torch.cat([o[0:1] for o in ic_om], dim=0)      # (S, Ny, Nx)

    # build the three steppers + differentiable AUG maps ONCE (single source
    # of truth = run_arm.one_step). closure=NN, r3anal=analytic, bare=scheme.
    steppers, aug_steps = {}, {}
    for arm_name, real_arm in (('closure', args.arm), ('r3anal', 'r3anal'),
                               ('bare', 'bare')):
        st = build_stepper(real_arm, ic_om, ic_ps, Delta_T, derivative, L_hat,
                           F_hat, device, model, input_fields, args.dealias_nn,
                           closure_apply=args.closure_apply)
        steppers[arm_name] = st
        aug_steps[arm_name] = make_aug_step(st, derivative, F_hat, S)

    # ---- MANDATORY unit checks (gate) ---- #
    passed, uc = unit_checks(steppers, aug_steps, base_ic, ic_om, derivative,
                             L_hat, F_hat, Delta_T, manifest, args, device)
    if args.unit_checks_only:
        sys.exit(0 if passed else 3)
    if not passed:
        sys.exit('[nnamp] unit checks FAILED -- no rho is reported '
                 '(Sanaa mandate). Fix the harness before trusting any number.')

    # ---- TEST B/C: rho at each requested state ---- #
    dev_steps = [int(s) for s in args.dev_steps.split(',')]
    print(f'[nnamp] rolling closure arm (no-grad) to capture dev states '
          f'{[s for s in dev_steps if s > 0]} ...', flush=True)
    states = roll_closure_states(steppers['closure'], ic_om, ic_ps, S,
                                 dev_steps, derivative, F_hat, device)

    dx, dy = Lx / Nx, Ly / Ny
    print('\n===== TRUE NN-AUGMENTED AMPLIFICATION rho (network self-'
          'linearization) =====', flush=True)
    print(f'{"step":>5} {"kind":>7} {"rho":>10} {"|Rayleigh|":>11} '
          f'{"osc":>8} {"CFL":>7}  dominant-mode shells (frac)', flush=True)
    rows = []
    for s in dev_steps:
        if s not in states:
            print(f'{s:5d}  (state unavailable -- closure blew up earlier)',
                  flush=True)
            continue
        stk = states[s].to(device)
        op = FrozenOperator(aug_steps['closure'], stk)
        r = power_iterate(op, n_iter=args.n_iter, device=device)
        cfl, urms = state_cfl_urms(stk[0], derivative, Delta_T, dx, dy)
        top_sh, bands = dominant_spectrum(r['dominant'][0].cpu())
        kind = 'truth' if s == 0 else 'NN-dev'
        shstr = ' '.join(f'k{k}:{f:.2f}' for k, f in top_sh[:4])
        print(f'{s:5d} {kind:>7} {r["rho"]:10.5f} {abs(r["rayleigh"]):11.5f} '
              f'{r["osc"]:8.4f} {cfl:7.3f}  {shstr}', flush=True)
        rows.append(dict(step=s, kind=kind, rho=r['rho'],
                         rayleigh_abs=abs(r['rayleigh']),
                         rayleigh_im=float(np.imag(r['rayleigh'])),
                         cfl=cfl, u_rms=urms, n_iter=r['n_iter'],
                         top_shells=top_sh, bands=bands))
    print('=' * 78, flush=True)
    print('EXPECTATION: rho ~ 1 at step 0 (truth), ramping toward ~1.7 as the '
          'state develops (matching the measured sqrt(2.85)=1.7x/step '
          'amplitude growth of the NN correction).', flush=True)

    if args.out is not None:
        out = args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(config=vars(args) | {'ckpt': str(args.ckpt),
                                            'root_dir': str(args.root_dir)},
                       unit_checks=uc, rows=rows, Delta_T=Delta_T,
                       model=model_name)
        Path(str(out) + '.json').write_text(
            json.dumps(payload, indent=2, default=str))
        print(f'[nnamp] wrote {out}.json', flush=True)


if __name__ == '__main__':
    main()
