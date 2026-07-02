"""
rollout_perfect_closure.py  (forced turbulence, R3 + R4 + R5 (+R6), analytic "perfect" derivatives)
====================================================================================================
Closure CEILING test. The configuration is IDENTICAL to rollout_timed_pareto.py --
forced turbulence, truth = fine RK4 reused from --load-refs, coarse AB2CN2, and the SAME
spectral assembly and dealiasing -- EXCEPT the time-derivatives of N that the
closure needs (Ndot, Nddot, N3dot, N4dot, N5dot) are computed EXACTLY by the chain rule
from the rollout's OWN current state, instead of being predicted by the NN.

Comparing this to the NN rollout isolates the network's contribution to the error:
if perfect derivatives give an improvement >> the NN's, the NN is the wall
(epsilon_NN-limited), not the R3/R4 truncation order. If the perfect-closure improvement
is ALSO modest, the formulation/truncation order is the limit and a better NN won't help.

Removing more orders of the AB2CN2 truncation series drives the closed coarse trajectory
onto the EXACT FLOW (= fine-RK4 truth) to higher order: the per-step residual after
removing through R_p is O(h^{p+1}), so the rollout floor drops as

    R3      -> O(h^3) removed, residual O(h^4)
    R3+R4   -> residual O(h^5)            (current ceiling: the R5 floor)
    R3+R4+R5    -> residual O(h^6)        (the R6 floor)
    R3+R4+R5+R6 -> residual O(h^7)        (~machine precision in float64)

Watching the final rel-L2 collapse toward machine precision as orders are added CONFIRMS
that AB2CN2 + perfect closure reproduces the fine-RK4 truth (i.e. the exact flow).

Verified operators (Closed-Form LTE of RK4/AB4CN2/AB2CN2; all coefficients symbolic):
  tau_AB2 = -(h^3/12) R3 -(h^4/24) R4 -(h^5/240) R5 -(h^6/1440) R6 + O(h^7)
  R3 =   L^3 w +   L^2 N +   L Ndot  -  5 Nddot
  R4 = 2 L^4 w + 2 L^3 N + 2 L^2 Ndot - 4 L Nddot +      N3dot
  R5 = 13 L^5 w + 13 L^4 N + 13 L^3 Ndot - 17 L^2 Nddot + 8 L N3dot -   7 N4dot
  R6 = 43 L^6 w + 43 L^5 N + 43 L^4 Ndot - 47 L^3 Nddot + 28 L^2 N3dot - 17 L N4dot + 4 N5dot
The closure ADDS delta = tau (correction to add to each bare AB2CN2 step).

Chain rule  (N = -J(psi, omega) + F ; F steady so all Fdot = 0):
  w^(k)  = L w^(k-1) + N^(k-1),     psi^(k) = inv_lap * w^(k)
  N^(m)  = -sum_{j=0}^m C(m,j) J(psi^(m-j), omega^(j))

Usage:
  # current ceiling (R3+R4):
  python rollout_perfect_closure.py --root-dir <root> --load-refs <refs.npz> \
      --r4 --r4-n3dot-coef 1 --dealias-nn --diag --pareto --device cuda
  # confirm "exactly RK4": add R5, then R6, and watch the floor crash:
  python rollout_perfect_closure.py --root-dir <root> --load-refs <refs.npz> \
      --r4 --r4-n3dot-coef 1 --r5 --r6 --dealias-nn --diag --device cuda
"""

from __future__ import annotations
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _find_training_dir():
    here = Path(__file__).resolve().parent
    for ancestor in [here, *here.parents]:
        if (ancestor / 'dataset.py').exists():
            return ancestor
    return here
sys.path.insert(0, str(_find_training_dir()))
sys.path.insert(0, str(Path(__file__).resolve().parent))


# =========================================================================== #
# Solver primitives -- verbatim from rollout_timed_pareto.py so the bare/truth #
# legs and the L^k / J operators are bit-for-bit the same.                     #
# =========================================================================== #
def J_phys(psi_phys, omega_phys, derivative):
    """Dealiased J(psi, omega) = d_x(u*omega) + d_y(v*omega), u=-d_y psi, v=+d_x psi."""
    from qg.solver.opt.basis import to_spectral, to_physical
    psih = to_spectral(psi_phys); qh = to_spectral(omega_phys)
    u = to_physical(-1 * derivative.dy * psih)
    v = to_physical(+1 * derivative.dx * psih)
    q = to_physical(qh)
    uq_h = to_spectral(u * q).clone()
    vq_h = to_spectral(v * q).clone()
    derivative.dealias(uq_h); derivative.dealias(vq_h)
    j_hat = derivative.dx * uq_h + derivative.dy * vq_h
    return to_physical(j_hat)


def L_op(omega_phys, L_hat):
    from qg.solver.opt.basis import to_spectral, to_physical
    return to_physical(L_hat * to_spectral(omega_phys))


def _dealias_mul(yh, derivative):
    keep = getattr(derivative, '_keep_mask', None)
    if keep is None or keep.device != yh.device:
        keep = (~derivative.alias_mask).to(device=yh.device, dtype=torch.float64)
        derivative._keep_mask = keep
    return yh * keep


def _N_core(qh, derivative, F_hat):
    """Minimal one-Jacobian N-eval (5 FFTs), N kept SPECTRAL. Returns (N_hat, omega_phys, psih)."""
    from qg.solver.opt.basis import to_spectral, to_physical
    psih = derivative.inv_laplacian * qh
    u = to_physical(-1 * derivative.dy * psih)
    v = to_physical(+1 * derivative.dx * psih)
    omega = to_physical(qh)
    uq_h = to_spectral(u * omega).clone()
    vq_h = to_spectral(v * omega).clone()
    uq_h = _dealias_mul(uq_h, derivative)
    vq_h = _dealias_mul(vq_h, derivative)
    j_hat = derivative.dx * uq_h + derivative.dy * vq_h
    N_hat = (F_hat - j_hat) if F_hat is not None else (-1.0 * j_hat)
    return N_hat, omega, psih


def N_spectral(qh, derivative, F_hat):
    return _N_core(qh, derivative, F_hat)[0]


def build_L_hat(derivative, nu, mu, beta):
    L_hat = nu * derivative.laplacian - mu
    if beta != 0.0:
        L_hat = L_hat - beta * derivative.dx * derivative.inv_laplacian
    return L_hat


def rk4_step(omega, dt, derivative, L_hat, F_phys):
    def rhs(om):
        from qg.solver.opt.basis import to_spectral, to_physical
        psi = to_physical(derivative.inv_laplacian * to_spectral(om))
        N = -1.0 * J_phys(psi, om, derivative)
        if F_phys is not None:
            N = N + F_phys
        return L_op(om, L_hat) + N
    k1 = rhs(omega); k2 = rhs(omega + 0.5 * dt * k1)
    k3 = rhs(omega + 0.5 * dt * k2); k4 = rhs(omega + dt * k3)
    return omega + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def psi_from_omega(omega_phys, derivative):
    from qg.solver.opt.basis import to_spectral, to_physical
    return to_physical(derivative.inv_laplacian * to_spectral(omega_phys))


def build_forcing(grid, fc, device, dtype):
    """F = A cos(B x + C t) + D cos(E y + F t); require C=F=0 (steady)."""
    if fc is None:
        return None
    A, B, C, D, E, Ff = (float(fc.get(k, 0.0)) for k in ('A', 'B', 'C', 'D', 'E', 'F'))
    if C != 0.0 or Ff != 0.0:
        raise ValueError("time-dependent forcing (C or F != 0) not supported here")
    x = torch.linspace(0, grid.Lx, grid.Nx, device=device, dtype=dtype)
    y = torch.linspace(0, grid.Ly, grid.Ny, device=device, dtype=dtype)
    X = x[None, :]; Y = y[:, None]
    w = A * torch.cos(B * X) + D * torch.cos(E * Y)
    return w[None]


def _sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


# =========================================================================== #
# Exact local N-derivatives by the chain rule (the "perfect" NN)               #
# =========================================================================== #
def analytic_n_derivs_hat(qh, derivative, L_hat, F_hat, max_order=3):
    """Exact local time-derivatives of N along the QG flow, SPECTRAL, as a list
    [N^(0), N^(1), ..., N^(max_order)].  These are precisely the quantities the
    cheap_deriv NN is trained to predict; feeding them into the R3..R6 brackets
    gives the closure ceiling.  Mirrors build_training_data_mmap.compute_n_derivatives
    (general-Leibniz recursion) but carries everything spectral for the L^k multiplies.

        w^(k)  = L w^(k-1) + N^(k-1),     psi^(k) = inv_lap w^(k)
        N^(m)  = -sum_{j=0}^m C(m,j) J(psi^(m-j), omega^(j))
    """
    from qg.solver.opt.basis import to_spectral, to_physical
    inv_lap = derivative.inv_laplacian

    w_h = [qh]                                   # omega^(k) spectral
    w_phys = [to_physical(qh)]                   # omega^(k) physical (for J)
    psi_phys = [to_physical(inv_lap * qh)]       # psi^(k)   physical (for J)
    N_h = [N_spectral(qh, derivative, F_hat)]    # N^(k) spectral

    for k in range(1, max_order + 1):
        wk_h = L_hat * w_h[k - 1] + N_h[k - 1]   # omega^(k) = L omega^(k-1) + N^(k-1)
        w_h.append(wk_h)
        w_phys.append(to_physical(wk_h))
        psi_phys.append(to_physical(inv_lap * wk_h))
        Nk = None                                # N^(k) = -sum_j C(k,j) J(psi^(k-j), omega^(j))
        for j in range(0, k + 1):
            term = math.comb(k, j) * J_phys(psi_phys[k - j], w_phys[j], derivative)
            Nk = (-term) if Nk is None else (Nk - term)
        N_h.append(to_spectral(Nk))
    return N_h


# =========================================================================== #
# RK4 order-5 elementary differentials (for E_NN1 = match the RK4 trajectory)  #
# =========================================================================== #
def _Bsym(a_phys, b_phys, derivative):
    """Symmetric bilinear B(a,b) = -1/2 [ J(inv_lap a, b) + J(inv_lap b, a) ] (physical).
    Satisfies B(omega,omega) = -J(psi,omega) = N - F, and N'(omega) a = 2 B(omega,a)."""
    from qg.solver.opt.basis import to_spectral, to_physical
    inv = derivative.inv_laplacian
    ia = to_physical(inv * to_spectral(a_phys))
    ib = to_physical(inv * to_spectral(b_phys))
    return -0.5 * (J_phys(ia, b_phys, derivative) + J_phys(ib, a_phys, derivative))


def _Japply(x_phys, omega_phys, derivative, L_hat):
    """J x = G'(omega) x = L x + 2 B(omega, x)  (physical). J = Frechet deriv of G."""
    return L_op(x_phys, L_hat) + 2.0 * _Bsym(omega_phys, x_phys, derivative)


def rk4_T5_hat(qh, derivative, L_hat, F_hat):
    r"""RK4 order-5 elementary-differential combination T5 (SPECTRAL), so that
    tau_RK4 = h^5 T5 + O(h^6) (Prop. 1):

      T5 = 1/120 J^4 f - 1/240 J^2 B(f,f) + 1/120 J B(f,Jf)
           - 1/60 B(f,J^2 f) + 1/120 B(f,B(f,f)) - 1/80 B(Jf,Jf)

    with f = G(omega) = L omega + N, J = L + 2 B(omega, .), B the symmetric bilinear.
    All six trees built directly (no {L,N}-basis reduction); ~20 Jacobians, diagnostic."""
    from qg.solver.opt.basis import to_spectral, to_physical
    omega = to_physical(qh)
    Nh = N_spectral(qh, derivative, F_hat)
    f = to_physical(L_hat * qh + Nh)                       # f = omega_dot
    Jf  = _Japply(f,   omega, derivative, L_hat)
    J2f = _Japply(Jf,  omega, derivative, L_hat)
    J3f = _Japply(J2f, omega, derivative, L_hat)
    J4f = _Japply(J3f, omega, derivative, L_hat)
    Bff   = _Bsym(f,  f,   derivative)
    BfJf  = _Bsym(f,  Jf,  derivative)
    BfJ2f = _Bsym(f,  J2f, derivative)
    BJfJf = _Bsym(Jf, Jf,  derivative)
    BfBff = _Bsym(f,  Bff, derivative)
    J2Bff = _Japply(_Japply(Bff, omega, derivative, L_hat), omega, derivative, L_hat)
    JBfJf = _Japply(BfJf, omega, derivative, L_hat)
    T5 = ((1.0/120.0) * J4f - (1.0/240.0) * J2Bff + (1.0/120.0) * JBfJf
          - (1.0/60.0) * BfJ2f + (1.0/120.0) * BfBff - (1.0/80.0) * BJfJf)
    return to_spectral(T5)


# =========================================================================== #
# Rollouts                                                                     #
# =========================================================================== #
def rollout_rk4_coarse(omega_0, dt, n_steps, checkpoint_steps,
                       derivative, L_hat, F_phys, device):
    """RK4 at the COARSE step dt -- the target trajectory for E_NN1 (match RK4).
    E_NN1's rollout should reproduce THIS to O(h^6), not the fine-RK4 truth."""
    om = omega_0.clone(); out = {}; cps = set(checkpoint_steps)
    if 0 in cps:
        out[0] = om[0].cpu().numpy()
    for s in range(1, n_steps + 1):
        om = rk4_step(om, dt, derivative, L_hat, F_phys)
        if s in cps:
            arr = om[0].cpu().numpy(); out[s] = arr
            if not np.isfinite(np.sqrt(np.mean(arr ** 2))):
                print(f"      [rk4-coarse] non-finite at step {s} -- stopping."); break
    return out
def rollout_fine(omega_0, h_fine, n_fine_steps, checkpoint_steps,
                 derivative, L_hat, F_phys, device):
    om = omega_0.clone()
    out = {}; cps = set(checkpoint_steps)
    w = omega_0.clone()
    for _ in range(3):
        w = rk4_step(w, h_fine, derivative, L_hat, F_phys)
    _sync(device); t0 = time.time()
    if 0 in cps:
        out[0] = om[0].cpu().numpy()
    for s in range(1, n_fine_steps + 1):
        om = rk4_step(om, h_fine, derivative, L_hat, F_phys)
        if s in cps:
            arr = om[0].cpu().numpy(); out[s] = arr
            if s % (max(n_fine_steps // 20, 1)) == 0:
                print(f"      [truth] RK4 {s}/{n_fine_steps}  |omega|_rms="
                      f"{float(np.sqrt(np.mean(arr**2))):.4e}", flush=True)
            if not np.isfinite(np.sqrt(np.mean(arr ** 2))):
                print("      [truth] non-finite -- stopping."); break
    _sync(device); return out, time.time() - t0


def rollout_bare(omega_0, omega_m1, dt, n_steps, checkpoint_steps,
                 derivative, L_hat, F_hat, device):
    from qg.solver.opt.basis import to_spectral, to_physical
    qh_n = to_spectral(omega_0); qh_nm1 = to_spectral(omega_m1)
    denom_hat = 1.0 - 0.5 * dt * L_hat
    out = {}; cps = set(checkpoint_steps)

    def step(qh, Nh, Nh_prev):
        return (qh + dt * (0.5 * L_hat * qh + (1.5 * Nh - 0.5 * Nh_prev))) / denom_hat

    Nh_n = N_spectral(qh_n, derivative, F_hat)
    Nh_nm1 = N_spectral(qh_nm1, derivative, F_hat)
    a, Na, Nam1 = qh_n.clone(), Nh_n.clone(), Nh_nm1.clone()
    for _ in range(3):
        a_new = step(a, Na, Nam1)
        Nam1, Na, a = Na, N_spectral(a_new, derivative, F_hat), a_new
    _sync(device); t0 = time.time()
    if 0 in cps:
        out[0] = to_physical(qh_n)[0].cpu().numpy()
    for s in range(1, n_steps + 1):
        qh_new = step(qh_n, Nh_n, Nh_nm1)
        Nh_nm1, Nh_n, qh_n = Nh_n, N_spectral(qh_new, derivative, F_hat), qh_new
        if s in cps:
            arr = to_physical(qh_n)[0].cpu().numpy(); out[s] = arr
            if not np.isfinite(np.sqrt(np.mean(arr ** 2))):
                print(f"      [bare] non-finite at step {s} -- stopping."); break
    _sync(device); return out, time.time() - t0


def rollout_perfect(omega_0, omega_m1, Delta_T, n_steps, checkpoint_steps,
                    derivative, L_hat, F_hat, device, dealias_nn=False,
                    include_r4=False, r4_n3dot_coef=None, r4_denom=24.0,
                    include_r5=False, r5_denom=240.0,
                    include_r6=False, r6_denom=1440.0, match_rk4=False):
    """Coarse AB2CN2 + e_anal + R3 + (optional) R4/R5/R6, with EXACT chain-rule
    derivatives. coef_p = Delta_T^p (truth = RK4 exact flow -> NO (1-1/K^2) factor).

    match_rk4=False -> E_NN2: remove the full AB2CN2 LTE; closed step -> EXACT flow.
    match_rk4=True  -> E_NN1: additionally SUBTRACT tau_RK4 = Delta_T^5 T5, so the
                       closed step reproduces the coarse-RK4 trajectory instead of the
                       exact flow (delta_1 = delta_2 - h^5 T5, eq. 38). Use with
                       --r4 --r5 so the exact-matching part is consistent through h^5."""
    from qg.solver.opt.basis import to_spectral, to_physical
    coef3 = Delta_T ** 3
    coef4 = Delta_T ** 4
    coef5 = Delta_T ** 5
    coef6 = Delta_T ** 6
    denom_hat = 1.0 - 0.5 * Delta_T * L_hat
    L2 = L_hat ** 2; L3 = L2 * L_hat; L4 = L2 * L2; L5 = L4 * L_hat; L6 = L4 * L2
    out = {}; cps = set(checkpoint_steps)

    max_order = 2
    if include_r4: max_order = max(max_order, 3)
    if include_r5: max_order = max(max_order, 4)
    if include_r6: max_order = max(max_order, 5)

    def one_step(qh_curr, Nh_curr, Nh_minus):
        AB2_Nh = 1.5 * Nh_curr - 0.5 * Nh_minus
        qh_bare = (qh_curr + Delta_T * (0.5 * L_hat * qh_curr + AB2_Nh)) / denom_hat

        Nd = analytic_n_derivs_hat(qh_curr, derivative, L_hat, F_hat, max_order=max_order)
        # Nd[0]=N, [1]=Ndot, [2]=Nddot, [3]=N3dot, [4]=N4dot, [5]=N5dot (all spectral)
        Ndot_h, Nddot_h = Nd[1], Nd[2]

        # --- R3: -(coef3/12)(L^3 w + L^2 N + L Ndot - 5 Nddot) ---
        # analytical part (L^3 w + L^2 N): NOT dealiased; NN part (L Ndot - 5 Nddot): dealiased
        e_anal_hat = -coef3 * (1.0 / 12.0) * (L3 * qh_curr + L2 * Nh_curr)
        f_hat = (1.0 / 12.0) * (L_hat * Ndot_h - 5.0 * Nddot_h)
        if dealias_nn:
            f_hat = _dealias_mul(f_hat, derivative)
        om_new_hat = qh_bare + e_anal_hat - coef3 * f_hat

        # --- R4: -(coef4/r4_denom)(2L^4 w + 2L^3 N + 2L^2 Ndot - 4L Nddot + c N3dot) ---
        if include_r4:
            N3dot_h = Nd[3]
            c4 = 1.0 if r4_n3dot_coef is None else float(r4_n3dot_coef)
            e_r4 = -coef4 * (1.0 / r4_denom) * (2.0 * L4 * qh_curr + 2.0 * L3 * Nh_curr
                                                + 2.0 * L2 * Ndot_h - 4.0 * L_hat * Nddot_h
                                                + c4 * N3dot_h)
            if dealias_nn:
                e_r4 = _dealias_mul(e_r4, derivative)
            om_new_hat = om_new_hat + e_r4

        # --- R5: -(coef5/240)(13L^5 w + 13L^4 N + 13L^3 Ndot - 17L^2 Nddot + 8L N3dot - 7 N4dot) ---
        if include_r5:
            N3dot_h, N4dot_h = Nd[3], Nd[4]
            e_r5 = -coef5 * (1.0 / r5_denom) * (13.0 * L5 * qh_curr + 13.0 * L4 * Nh_curr
                                                + 13.0 * L3 * Ndot_h - 17.0 * L2 * Nddot_h
                                                + 8.0 * L_hat * N3dot_h - 7.0 * N4dot_h)
            if dealias_nn:
                e_r5 = _dealias_mul(e_r5, derivative)
            om_new_hat = om_new_hat + e_r5

        # --- R6: -(coef6/1440)(43L^6 w + 43L^5 N + 43L^4 Ndot - 47L^3 Nddot + 28L^2 N3dot - 17L N4dot + 4 N5dot) ---
        if include_r6:
            N3dot_h, N4dot_h, N5dot_h = Nd[3], Nd[4], Nd[5]
            e_r6 = -coef6 * (1.0 / r6_denom) * (43.0 * L6 * qh_curr + 43.0 * L5 * Nh_curr
                                                + 43.0 * L4 * Ndot_h - 47.0 * L3 * Nddot_h
                                                + 28.0 * L2 * N3dot_h - 17.0 * L_hat * N4dot_h
                                                + 4.0 * N5dot_h)
            if dealias_nn:
                e_r6 = _dealias_mul(e_r6, derivative)
            om_new_hat = om_new_hat + e_r6

        # --- E_NN1: subtract tau_RK4 = h^5 T5 so the closed step matches coarse RK4 ---
        if match_rk4:
            T5h = coef5 * rk4_T5_hat(qh_curr, derivative, L_hat, F_hat)
            if dealias_nn:
                T5h = _dealias_mul(T5h, derivative)
            om_new_hat = om_new_hat - T5h

        Nh_new = N_spectral(om_new_hat, derivative, F_hat)
        return om_new_hat, Nh_new

    qh_curr = to_spectral(omega_0); qh_minus = to_spectral(omega_m1)
    Nh_curr = N_spectral(qh_curr, derivative, F_hat)
    Nh_minus = N_spectral(qh_minus, derivative, F_hat)

    qc, Nc, Nm = qh_curr.clone(), Nh_curr.clone(), Nh_minus.clone()   # warmup (untimed)
    for _ in range(3):
        qn, Nn = one_step(qc, Nc, Nm)
        Nm, Nc, qc = Nc, Nn, qn
    _sync(device); t0 = time.time()
    if 0 in cps:
        out[0] = to_physical(qh_curr)[0].cpu().numpy()
    for s in range(1, n_steps + 1):
        qh_new, Nh_new = one_step(qh_curr, Nh_curr, Nh_minus)
        Nh_minus, Nh_curr, qh_curr = Nh_curr, Nh_new, qh_new
        if s in cps:
            arr = to_physical(qh_curr)[0].cpu().numpy(); out[s] = arr
            if not np.isfinite(np.sqrt(np.mean(arr ** 2))):
                print(f"      [perfect] non-finite at step {s} -- stopping."); break
    _sync(device); return out, time.time() - t0


def rel_l2(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)) / max(np.sqrt(np.mean(b ** 2)), 1e-30))


def _radial_spectrum(field):
    field = np.asarray(field, dtype=np.float64)
    P = np.abs(np.fft.fft2(field)) ** 2
    ny, nx = field.shape
    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    KR = np.rint(np.sqrt(np.add.outer(ky ** 2, kx ** 2))).astype(int)
    kmax = int(KR.max())
    Pr = np.zeros(kmax + 1)
    for k in range(kmax + 1):
        m = (KR == k)
        if m.any():
            Pr[k] = P[m].mean()
    return np.arange(kmax + 1), Pr


# =========================================================================== #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root-dir', type=Path, required=True, help='dataset root (manifest.json)')
    p.add_argument('--load-refs', type=Path, default=None,
                   help='rollout_refs_*.npz from the NN run -- reuse the SAME truth (+bare) for an '
                        'apples-to-apples ceiling. Strongly recommended.')
    p.add_argument('--rerun-bare', action='store_true', help='recompute the bare leg even if refs has it')
    p.add_argument('--n-steps', type=int, default=1000, help='coarse steps (used only without --load-refs)')
    p.add_argument('--n-checkpoints', type=int, default=20)
    p.add_argument('--ic-index', type=int, default=0)
    p.add_argument('--Delta-T-override', type=float, default=None)
    p.add_argument('--K-override', type=int, default=None)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--out-dir', type=Path, default=None)
    p.add_argument('--ic-tag', type=str, default='ft_perfect')
    p.add_argument('--dealias-nn', action='store_true', help='project the correction onto the 2/3 band each step')
    p.add_argument('--r4', action='store_true', help='add the R4 refinement (2L^4 w + 2L^3 N + 2L^2 Ndot - 4L Nddot [+ c N3dot])')
    p.add_argument('--r4-n3dot-coef', type=float, default=None, help='coefficient on the R4 N3dot term (verified value: 1). Only with --r4.')
    p.add_argument('--r4-denom', type=float, default=24.0,
                   help='prefactor denominator of the R4 correction: e_R4 = -(Delta_T^4 / r4_denom) * '
                        '(2L^4 w + 2L^3 N + 2L^2 Ndot - 4L Nddot + c N3dot). Truncation hierarchy is '
                        'h^3/12 (R3), h^4/24 (R4), ... so this defaults to 24.')
    p.add_argument('--r5', action='store_true',
                   help='add the verified R5: -(Delta_T^5/r5_denom)(13L^5 w + 13L^4 N + 13L^3 Ndot '
                        '- 17L^2 Nddot + 8L N3dot - 7 N4dot). Needs N4dot (one more chain step).')
    p.add_argument('--r5-denom', type=float, default=240.0, help='R5 prefactor denominator (verified: 240).')
    p.add_argument('--r6', action='store_true',
                   help='add the verified R6: -(Delta_T^6/r6_denom)(43L^6 w + 43L^5 N + 43L^4 Ndot '
                        '- 47L^3 Nddot + 28L^2 N3dot - 17L N4dot + 4 N5dot). Needs N5dot.')
    p.add_argument('--r6-denom', type=float, default=1440.0, help='R6 prefactor denominator (verified: 1440).')
    p.add_argument('--match-rk4', action='store_true',
                   help='E_NN1: subtract tau_RK4 = Delta_T^5 T5 so the closed step reproduces the '
                        'COARSE-RK4 trajectory instead of the exact flow. Use with --r4 --r5 for an '
                        'O(h^6)-consistent E_NN1. Without it the closure is E_NN2 (match exact flow). '
                        'Also rolls out coarse RK4 and reports the closed error vs BOTH references.')
    p.add_argument('--diag', action='store_true', help='per-term RMS table + rollout_diag_<tag>.png (improvement-vs-time + error spectrum)')
    p.add_argument('--pareto', action='store_true', help='bare dt sweep, with the perfect-closure accuracy drawn as the ceiling line')
    p.add_argument('--pareto-dt-factors', type=str, default='1,2,4,8,16,40,100')
    p.add_argument('--save-refs', action='store_true', help='if computing truth fresh, dump it for reuse')
    # forcing (forced-turb defaults; manifest overrides if present)
    p.add_argument('--fA', type=float, default=-0.1); p.add_argument('--fB', type=float, default=2.0)
    p.add_argument('--fC', type=float, default=0.0);  p.add_argument('--fD', type=float, default=0.1)
    p.add_argument('--fE', type=float, default=2.0);  p.add_argument('--fF', type=float, default=0.0)
    p.add_argument('--no-forcing', action='store_true')
    args = p.parse_args()

    device = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    dtype = torch.float64                       # analytic derivs are cancellation-sensitive
    out_dir = args.out_dir or args.root_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[perfect] device={device} dtype=float64")

    with open(args.root_dir / 'manifest.json') as f:
        manifest = json.load(f)
    Nx, Ny = int(manifest['Nx']), int(manifest['Ny'])
    Lx, Ly = float(manifest['Lx']), float(manifest['Ly'])
    nu = float(manifest['nu']); mu = float(manifest.get('mu', 0.0)); beta = float(manifest.get('beta', 0.0))
    Delta_T = args.Delta_T_override or float(manifest['Delta_T'])
    K = args.K_override or int(manifest.get('K', 100))
    h_fine = Delta_T / K
    print(f"[perfect] grid {Nx}x{Ny} nu={nu} mu={mu} beta={beta}  Delta_T={Delta_T} K={K} h_fine={h_fine}")
    orders = "R3" + ("+R4" if args.r4 else "") + ("+R5" if args.r5 else "") + ("+R6" if args.r6 else "")
    clos_name = "E_NN1 (match RK4)" if args.match_rk4 else "E_NN2 (match exact flow)"
    print(f"[perfect] closure: {clos_name}   orders: {orders}")
    if args.match_rk4 and not (args.r4 and args.r5):
        print("[perfect] WARN: E_NN1 is only O(h^6)-consistent with --r4 --r5 (you are missing one).")

    refs = None
    if args.load_refs is not None:
        rf = np.load(args.load_refs)
        refs_Delta_T = float(rf['Delta_T']); refs_K = int(rf['K'])
        refs_h_fine = refs_Delta_T / refs_K
        if abs(refs_h_fine - h_fine) > 1e-12 * max(refs_h_fine, h_fine):
            raise SystemExit(f"[perfect] --load-refs fine-step mismatch: refs h_fine={refs_h_fine:.3e} "
                             f"vs current {h_fine:.3e}. Set --K-override so Delta_T/K matches.")
        cp_coarse = [int(s) for s in rf['cp_coarse']]
        same_dt = abs(refs_Delta_T - Delta_T) <= 1e-15
        refs = dict(ic_index=int(rf['ic_index']),
                    cp_times=[s * refs_Delta_T for s in cp_coarse],
                    truth_stack=rf['truth_stack'], bare_stack=rf['bare_stack'],
                    t_truth=float(rf['t_truth']), t_bare=float(rf['t_bare']), same_dt=same_dt)
        if 'ic_omega_0' in rf:
            refs['ic_fields'] = (rf['ic_omega_0'], rf['ic_omega_m1'], rf['ic_omega_m2'])
        args.ic_index = refs['ic_index']
        print(f"[perfect] loaded refs: {len(cp_coarse)} checkpoints "
              f"t={refs['cp_times'][0]:.3f}..{refs['cp_times'][-1]:.3f}; reuse truth t={refs['t_truth']:.1f}s")

    from qg.solver.grid.cartesian import CartesianGrid
    from qg.solver.opt.derivative import Derivative
    from qg.solver.opt.basis import to_spectral as _ts, to_physical as _tp
    grid = CartesianGrid(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, device=device, precision='float64')
    derivative = Derivative(grid)
    for attr in ('dx', 'dy', 'laplacian', 'inv_laplacian', 'alias_mask'):
        if hasattr(derivative, attr):
            setattr(derivative, attr, getattr(derivative, attr).to(device))
    L_hat = build_L_hat(derivative, nu, mu, beta).to(device)
    L2 = L_hat ** 2; L3 = L2 * L_hat; L4 = L2 * L2; L5 = L4 * L_hat; L6 = L4 * L2

    if args.no_forcing:
        fc = None
    elif manifest.get('has_forcing') and isinstance(manifest.get('forcing'), dict):
        fc = manifest['forcing']
    else:
        fc = dict(A=args.fA, B=args.fB, C=args.fC, D=args.fD, E=args.fE, F=args.fF)
    F_phys = build_forcing(grid, fc, device, dtype)
    F_hat = _ts(F_phys) if F_phys is not None else None
    print(f"[perfect] forcing: {'none' if F_phys is None else fc}")

    # ---- IC (truth-shared from refs; perfect closure needs only o0 + om1) ---- #
    if refs is not None and 'ic_fields' in refs:
        o0, om1, _ = refs['ic_fields']
        omega_0 = torch.tensor(np.asarray(o0), dtype=dtype, device=device)[None]
        omega_m1 = torch.tensor(np.asarray(om1), dtype=dtype, device=device)[None]
        print(f"[perfect] IC from refs (row {args.ic_index}): "
              f"|omega_0|_rms={float(torch.sqrt((omega_0**2).mean())):.4e}")
    else:
        inp = np.load(args.root_dir / 'packed' / 'inputs.npy', mmap_mode='r')
        cfg_fields = list(manifest.get('input_fields',
                          ['omega_0', 'omega_m1', 'omega_m2', 'psi_0', 'psi_m1', 'psi_m2']))
        ci = {f: cfg_fields.index(f) for f in ('omega_0', 'omega_m1')}
        omega_0 = torch.tensor(np.asarray(inp[args.ic_index, ci['omega_0']]), dtype=dtype, device=device)[None]
        omega_m1 = torch.tensor(np.asarray(inp[args.ic_index, ci['omega_m1']]), dtype=dtype, device=device)[None]
        print(f"[perfect] IC row {args.ic_index}: |omega_0|_rms={float(torch.sqrt((omega_0**2).mean())):.4e}")

    # ---- checkpoints ---- #
    if refs is not None:
        cp_coarse = [int(round(t / Delta_T)) for t in refs['cp_times']]
        M = cp_coarse[-1]
    else:
        M = args.n_steps
        cp_coarse = sorted(set(int(round(f * M)) for f in np.linspace(0, 1, args.n_checkpoints + 1)))
    cp_times = [s * Delta_T for s in cp_coarse]
    print(f"[perfect] horizon T={M*Delta_T:.4f}  {len(cp_coarse)} checkpoints")

    # ---- truth + bare ---- #
    if refs is not None:
        truth_cp = {s * K: refs['truth_stack'][i] for i, s in enumerate(cp_coarse)}
        t_truth = refs['t_truth']
        if refs['same_dt'] and not args.rerun_bare:
            bare_cp = {s: refs['bare_stack'][i] for i, s in enumerate(cp_coarse)}
            t_bare = refs['t_bare']
            print(f"[perfect] truth + bare REUSED from refs")
        else:
            print(f"[perfect] truth REUSED; recomputing bare {M} steps ...")
            bare_cp, t_bare = rollout_bare(omega_0, omega_m1, Delta_T, M, cp_coarse,
                                           derivative, L_hat, F_hat, device)
    else:
        cp_fine = [s * K for s in cp_coarse]
        print(f"[perfect] truth: {M*K} fine RK4 steps ...")
        truth_cp, t_truth = rollout_fine(omega_0, h_fine, M * K, cp_fine, derivative, L_hat, F_phys, device)
        print(f"[perfect] bare: {M} coarse steps ...")
        bare_cp, t_bare = rollout_bare(omega_0, omega_m1, Delta_T, M, cp_coarse, derivative, L_hat, F_hat, device)
        if args.save_refs:
            ref_cp = [s for s in cp_coarse if (s * K in truth_cp and s in bare_cp)]
            np.savez(out_dir / f'rollout_refs_{args.ic_tag}.npz',
                     ic_index=np.int64(args.ic_index), n_steps=np.int64(M),
                     Delta_T=np.float64(Delta_T), K=np.int64(K), h_fine=np.float64(h_fine),
                     cp_coarse=np.asarray(ref_cp, dtype=np.int64),
                     cp_times=np.asarray([s * Delta_T for s in ref_cp], dtype=np.float64),
                     t_truth=np.float64(t_truth), t_bare=np.float64(t_bare),
                     ic_omega_0=omega_0[0].cpu().numpy(), ic_omega_m1=omega_m1[0].cpu().numpy(),
                     ic_omega_m2=omega_m1[0].cpu().numpy(),
                     truth_stack=np.stack([np.asarray(truth_cp[s * K], np.float32) for s in ref_cp]),
                     bare_stack=np.stack([np.asarray(bare_cp[s], np.float32) for s in ref_cp]))

    # ---- perfect closure ---- #
    print(f"[perfect] {clos_name}: {M} coarse steps + analytic derivatives [{orders}]"
          f"{' - h^5 T5' if args.match_rk4 else ''} ...")
    clos_cp, t_clos = rollout_perfect(omega_0, omega_m1, Delta_T, M, cp_coarse,
                                      derivative, L_hat, F_hat, device,
                                      dealias_nn=args.dealias_nn, include_r4=args.r4,
                                      r4_n3dot_coef=args.r4_n3dot_coef, r4_denom=args.r4_denom,
                                      include_r5=args.r5, r5_denom=args.r5_denom,
                                      include_r6=args.r6, r6_denom=args.r6_denom,
                                      match_rk4=args.match_rk4)
    print(f"[perfect]   walltime = {t_clos:.3f}s  ({t_clos/M*1e3:.3f} ms/step; "
          f"diagnostic only -- not a production cost)")

    # ---- coarse-RK4 reference (the E_NN1 target) ---- #
    rk4c_cp = None
    if args.match_rk4:
        print(f"[perfect] coarse RK4 @ Delta_T: {M} steps (E_NN1's target trajectory) ...")
        rk4c_cp = rollout_rk4_coarse(omega_0, Delta_T, M, cp_coarse,
                                     derivative, L_hat, F_phys, device)

    # ---- error table ---- #
    avail = [s for s in cp_coarse if (s in bare_cp and s in clos_cp and s * K in truth_cp)]
    a_times = np.array([s * Delta_T for s in avail])
    rel_bare = np.array([rel_l2(bare_cp[s], truth_cp[s * K]) for s in avail])
    rel_clos = np.array([rel_l2(clos_cp[s], truth_cp[s * K]) for s in avail])
    truth_rms = np.array([float(np.sqrt(np.mean(truth_cp[s * K] ** 2))) for s in avail])
    print(f"\n============= ERROR vs TRUTH (rel-L2): PERFECT closure [{orders}] =============")
    print(f"{'t':>10}{'truth_rms':>14}{'bare':>14}{'perfect':>14}{'improve x':>12}")
    for i, s in enumerate(avail):
        imp = rel_bare[i] / max(rel_clos[i], 1e-30)
        print(f"{a_times[i]:>10.4f}{truth_rms[i]:>14.4e}{rel_bare[i]:>14.4e}{rel_clos[i]:>14.4e}{imp:>12.2f}")

    print("\n==================== CEILING SUMMARY ====================")
    print(f"  closure               = {clos_name}   [{orders}]")
    print(f"  bare    final rel-L2 = {rel_bare[-1]:.4e}   ({t_bare:.3f}s)")
    print(f"  closed  final rel-L2 = {rel_clos[-1]:.4e}   (vs fine-RK4 truth)")
    print(f"  improvement over bare = {rel_bare[-1]/max(rel_clos[-1],1e-30):.1f}x")
    if rk4c_cp is not None:
        avail_r = [s for s in avail if s in rk4c_cp]
        rel_vs_rk4c = np.array([rel_l2(clos_cp[s], rk4c_cp[s]) for s in avail_r])
        rel_rk4c_truth = np.array([rel_l2(rk4c_cp[s], truth_cp[s * K]) for s in avail_r])
        print(f"\n  --- E_NN1 verification (target = coarse RK4, not the fine truth) ---")
        print(f"  closed vs COARSE-RK4 final rel-L2 = {rel_vs_rk4c[-1]:.4e}  "
              f"(-> 0 confirms the closed step reproduces RK4 @ Delta_T)")
        print(f"  coarse-RK4 vs fine-truth final    = {rel_rk4c_truth[-1]:.4e}  "
              f"(RK4's own O(h^5) error; the 'closed vs truth' floor cannot beat this)")
        results['rel_vs_rk4_coarse'] = rel_vs_rk4c.tolist()
        results['rel_rk4coarse_vs_truth'] = rel_rk4c_truth.tolist()
    else:
        print(f"  -> the floor should DROP as you add R5, then R6: confirms AB2CN2+E_NN2")
        print(f"     converges onto the fine-RK4 truth (the exact flow). If the floor stops")
        print(f"     dropping, you have hit the modified-equation radius DT_star at this Delta_T.")

    results = dict(Delta_T=Delta_T, K=K, h_fine=h_fine, n_steps=M, orders=orders,
                   r4=bool(args.r4), r5=bool(args.r5), r6=bool(args.r6),
                   r4_n3dot_coef=(float(args.r4_n3dot_coef) if args.r4_n3dot_coef is not None else None),
                   t_truth=t_truth, t_bare=t_bare, t_clos=t_clos,
                   cp_times=a_times.tolist(), truth_rms=truth_rms.tolist(),
                   rel_bare=rel_bare.tolist(), rel_clos=rel_clos.tolist(),
                   final_bare=float(rel_bare[-1]), final_clos=float(rel_clos[-1]))

    # ---- diagnostics ---- #
    s_end = avail[-1]
    if args.diag:
        print("\n============= CLOSURE TERM RMS (analytic, at IC) =============")
        with torch.no_grad():
            qh0 = _ts(omega_0)
            md = 2 + (1 if args.r4 else 0) + (1 if args.r5 else 0) + (1 if args.r6 else 0)
            md = max(md, 4 if args.r5 else md, 5 if args.r6 else md)
            Nd = analytic_n_derivs_hat(qh0, derivative, L_hat, F_hat, max_order=max(md, 3))
            Nh0 = Nd[0]; Ndd = Nd[2]; Ndo = Nd[1]; N3d = Nd[3] if len(Nd) > 3 else None
            N4d = Nd[4] if len(Nd) > 4 else None
            N5d = Nd[5] if len(Nd) > 5 else None
            cf = 1.0 / 12.0
            def _rms(spec):
                return float(torch.sqrt((_tp(spec) ** 2).mean()))
            terms = {
                'R3 L^3 w':   Delta_T**3 * cf * _rms(L3 * qh0),
                'R3 L^2 N':   Delta_T**3 * cf * _rms(L2 * Nh0),
                'R3 L*Ndot':  Delta_T**3 * cf * _rms(L_hat * Ndo),
                'R3 5*Nddot': Delta_T**3 * cf * 5.0 * _rms(Ndd),
            }
            if args.r4:
                cf4 = 1.0 / args.r4_denom
                terms['R4 2L^4 w']    = Delta_T**4 * cf4 * 2.0 * _rms(L4 * qh0)
                terms['R4 2L^3 N']    = Delta_T**4 * cf4 * 2.0 * _rms(L3 * Nh0)
                terms['R4 2L^2 Ndot'] = Delta_T**4 * cf4 * 2.0 * _rms(L2 * Ndo)
                terms['R4 4L*Nddot']  = Delta_T**4 * cf4 * 4.0 * _rms(L_hat * Ndd)
                if N3d is not None:
                    terms['R4 N3dot'] = Delta_T**4 * cf4 * _rms(N3d)
            if args.r5 and N4d is not None:
                cf5 = 1.0 / args.r5_denom
                terms['R5 13L^5 w']    = Delta_T**5 * cf5 * 13.0 * _rms(L5 * qh0)
                terms['R5 13L^4 N']    = Delta_T**5 * cf5 * 13.0 * _rms(L4 * Nh0)
                terms['R5 13L^3 Ndot'] = Delta_T**5 * cf5 * 13.0 * _rms(L3 * Ndo)
                terms['R5 17L^2 Nddot']= Delta_T**5 * cf5 * 17.0 * _rms(L2 * Ndd)
                terms['R5 8L N3dot']   = Delta_T**5 * cf5 * 8.0 * _rms(L_hat * N3d)
                terms['R5 7 N4dot']    = Delta_T**5 * cf5 * 7.0 * _rms(N4d)
            if args.r6 and N5d is not None:
                cf6 = 1.0 / args.r6_denom
                terms['R6 4 N5dot']    = Delta_T**6 * cf6 * 4.0 * _rms(N5d)
        tot = sum(terms.values()) or 1e-30
        for k, v in terms.items():
            print(f"  {k:<16} rms={v:.4e}  {100*v/tot:5.1f}%")
        print(f"  {'-'*44}\n  total correction rms = {tot:.4e}")
        results['term_rms'] = {k: float(v) for k, v in terms.items()}
        try:
            kk_c, Pc = _radial_spectrum(clos_cp[s_end] - truth_cp[s_end * K])
            kk_b, Pb = _radial_spectrum(bare_cp[s_end] - truth_cp[s_end * K])
            kk_t, Pt = _radial_spectrum(truth_cp[s_end * K])
            figd, axd = plt.subplots(1, 2, figsize=(12, 4.5))
            axd[0].semilogy(a_times, rel_bare / np.maximum(rel_clos, 1e-30), 'o-', ms=3, color='C2')
            axd[0].set_xlabel('physical time'); axd[0].set_ylabel('improvement (bare/perfect)')
            axd[0].set_title(f'CEILING improvement vs time [{orders}]'); axd[0].grid(alpha=0.3)
            axd[1].loglog(kk_t[1:], Pt[1:], 'k-', lw=1, alpha=0.4, label='truth field')
            axd[1].loglog(kk_b[1:], Pb[1:], 'C0-', label='bare error')
            axd[1].loglog(kk_c[1:], Pc[1:], 'C3-', label=f'perfect [{orders}] error')
            axd[1].set_xlabel('radial wavenumber |k|'); axd[1].set_ylabel('power')
            axd[1].set_title(f'error spectrum @ t={s_end*Delta_T:.2f}')
            axd[1].legend(); axd[1].grid(alpha=0.3, which='both')
            figd.tight_layout(); figd.savefig(out_dir / f'rollout_diag_{args.ic_tag}.png', dpi=130)
            print(f"[perfect] wrote rollout_diag_{args.ic_tag}.png")
        except Exception as e:                       # noqa: BLE001
            print(f"[perfect] diag figure skipped ({type(e).__name__}: {e})")

    # ---- Pareto (bare dt sweep, perfect accuracy as ceiling line) ---- #
    pareto = None
    if args.pareto and s_end == M:
        print("\n==================== PARETO (bare dt sweep) ====================")
        pts = []
        for fac in [float(x) for x in args.pareto_dt_factors.split(',')]:
            dt = Delta_T / fac
            if dt < h_fine * 0.999:
                continue
            nst = int(round(M * Delta_T / dt))
            bc, wt = rollout_bare(omega_0, omega_m1, dt, nst, [nst], derivative, L_hat, F_hat, device)
            if nst not in bc:
                print(f"  bare dt={dt:.3e} blew up -- skipped"); continue
            err = rel_l2(bc[nst], truth_cp[M * K]); pts.append((dt, wt, err))
            print(f"  bare dt={dt:.3e} steps={nst} wall={wt:.3f}s rel-L2={err:.4e}")
        pareto = pts
        results['pareto'] = [list(x) for x in pts]

    np.savez(out_dir / f'rollout_perfect_{args.ic_tag}.npz',
             **{k: np.asarray(v) for k, v in results.items() if k not in ('pareto', 'term_rms', 'orders')},
             clos_final=clos_cp[s_end], bare_final=bare_cp[s_end], truth_final=truth_cp[s_end * K])
    (out_dir / f'rollout_perfect_{args.ic_tag}.json').write_text(json.dumps(results, indent=2, default=float))

    # ---- main figure ---- #
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].semilogy(a_times, rel_bare, 'o-', label='bare (coarse)')
    ax[0].semilogy(a_times, rel_clos, 's-', color='C3', label=f'perfect closure [{orders}]')
    ax[0].set_xlabel('physical time'); ax[0].set_ylabel('rel-L2 vs truth')
    ax[0].legend(); ax[0].set_title('error growth: bare vs ceiling'); ax[0].grid(alpha=0.3)
    if pareto:
        dts, wts, errs = zip(*pareto)
        ax[1].loglog(wts, errs, 'o-', color='C0', label='bare @ varying dt')
        ax[1].axhline(rel_clos[-1], color='C3', ls='--', lw=1.5, label=f'perfect [{orders}] (ceiling)')
        ax[1].loglog([t_bare], [rel_bare[-1]], 'C1D', ms=9, label='bare @ Delta_T')
        ax[1].set_xlabel('walltime (s)'); ax[1].set_ylabel('final rel-L2 vs truth')
        ax[1].legend(); ax[1].set_title('cost / accuracy'); ax[1].grid(alpha=0.3, which='both')
    else:
        ax[1].axis('off')
    fig.tight_layout(); fig.savefig(out_dir / f'rollout_perfect_{args.ic_tag}.png', dpi=130)
    print(f"\n[perfect] wrote rollout_perfect_{args.ic_tag}.png / .npz / .json in {out_dir}")


if __name__ == '__main__':
    main()
