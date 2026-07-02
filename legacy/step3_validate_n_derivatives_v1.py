"""
step3_validate_n_derivatives.py  (FLOW PAST CYLINDER, Re ~ 200)

Validate the analytical chain-rule formulas for N_dot and N_ddot against
finite-difference estimates from the decaying-turbulence dt sweep reference run.

Conventions (matching memory edit #7):
    omega_t = L * omega + N(omega)        with N = -J(psi, omega) + F
    psi    = inv_laplacian * omega
    omega_dot = L*omega + N
    psi_dot   = inv_laplacian * omega_dot
    N_dot   = -J(psi_dot, omega) - J(psi, omega_dot)
    omega_ddot = L * omega_dot + N_dot
    psi_ddot   = inv_laplacian * omega_ddot
    N_ddot  = -J(psi_ddot, omega) - 2*J(psi_dot, omega_dot) - J(psi, omega_ddot)

Decaying turbulence has no forcing, no obstacle, no sponge -- a clean QG-with-
diffusion-only setup. This is the cleanest test of the chain-rule machinery,
free from the boundary artifacts that affect the cylinder version.
We use N = -J(psi, omega) consistently on both the analytical and the
finite-difference sides.

This script does TWO tests:

  TEST A -- single-mode sanity check
    Pure code-correctness check on a small periodic grid. Detects sign /
    FFT-normalization bugs in the chain-rule implementation.

  TEST B -- 2D in-the-wild check on decaying turbulence data
    Use snapshots from the dt_1em5 reference run. For each consecutive
    triplet (omega^{n-1}, omega^n, omega^{n+1}) at saved times spaced by
    dt_save = 0.05, compute:
      N_dot_FD  = (N(omega^{n+1}) - N(omega^{n-1})) / (2 * dt_save)
      N_ddot_FD = (N(omega^{n+1}) - 2 N(omega^n) + N(omega^{n-1})) / dt_save^2
      N_dot_AN, N_ddot_AN  via the analytical chain rule
    Both sides use N = -J(psi, omega) (no forcing, no Brinkman, no sponge).
    Should agree to O(dt_save^2) ~ 2.5e-3 relative.

Sweep this script analyzes:
    SWEEP_SUBDIRS = ['dt_1em3','dt_2em3','dt_1p25em4','dt_2p5em4',
                     'dt_5em4','dt_2em5','dt_1em5']
    Reference: dt_1em5 (smallest dt; cleanest snapshots).

Usage:
    python step3_validate_n_derivatives.py \\
        --sweep-root /path/to/decaying_turb_dt_sweep \\
        --out-dir   /path/to/figures \\
        [--n-snapshots 5] [--ref-subdir dt_1em5]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import yaml

# Local QG imports (must be importable in the active venv)
from qg.solver.grid.cartesian import CartesianGrid
from qg.solver.opt.derivative import Derivative
from qg.solver.opt.basis import to_spectral, to_physical


# --------------------------------------------------------------------------- #
# QG operators
#
# These wrap the solver's primitives so we can call them on arbitrary fields
# without instantiating a full QG solver.
# --------------------------------------------------------------------------- #

def J_phys(psi_phys: torch.Tensor, omega_phys: torch.Tensor,
           derivative) -> torch.Tensor:
    """
    Compute J(psi, omega) = u dq/dx + v dq/dy where u = -dpsi/dy, v = +dpsi/dx,
    with psi and omega supplied as INDEPENDENT physical-space fields.

    This is essential for N_dot, where we need J(psi_dot, omega) with
    psi_dot = inv_lap omega_dot != inv_lap omega.

    Sign convention: returns +J(psi, omega). The solver's jacobian_pq returns
    -J(psi, omega) (the contribution to omega_t), so we don't reuse it here
    directly -- we replicate its arithmetic with explicit sign control.
    """
    psih = to_spectral(psi_phys)
    qh = to_spectral(omega_phys)
    uh = -1 * derivative.dy * psih      # u = -dpsi/dy
    vh = +1 * derivative.dx * psih      # v = +dpsi/dx
    u = to_physical(uh)
    v = to_physical(vh)
    q = to_physical(qh)
    # J(psi, omega) = u dq/dx + v dq/dy
    #               = d/dx(u q) + d/dy(v q)   [since div(u)=0 in incompressible]
    uq_h = to_spectral(u * q)
    vq_h = to_spectral(v * q)
    j_hat = derivative.dx * uq_h + derivative.dy * vq_h
    return to_physical(j_hat)


# --------------------------------------------------------------------------- #
# Linear operator
# --------------------------------------------------------------------------- #

def L_op(omega_phys: torch.Tensor, L_hat: torch.Tensor) -> torch.Tensor:
    """
    Apply the linear operator L = nu*laplacian - mu - B*d/dx*inv_laplacian
    in physical space, given its spectral multiplier L_hat.
    """
    qh = to_spectral(omega_phys)
    return to_physical(L_hat * qh)


def build_L_hat(derivative, nu: float, mu: float, B: float) -> torch.Tensor:
    """
    Spectral multiplier for the linear operator. Same definition the solver
    uses internally (nu*laplacian - mu - B*dx*inv_laplacian).
    """
    L_hat = nu * derivative.laplacian
    L_hat = L_hat - mu
    L_hat = L_hat - B * derivative.dx * derivative.inv_laplacian
    return L_hat


# --------------------------------------------------------------------------- #
# Analytical chain-rule N_dot and N_ddot
# --------------------------------------------------------------------------- #

def compute_n_dot_analytical(
    omega: torch.Tensor,
    derivative,
    L_hat: torch.Tensor,
    F: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Analytical N_dot via chain rule + bilinearity of J:
        omega_dot = L*omega + N(omega)
        psi_dot   = inv_lap * omega_dot
        N_dot     = -J(psi_dot, omega) - J(psi, omega_dot)
    All inputs/outputs in physical space.
    """
    # psi from omega
    psi_hat = derivative.inv_laplacian * to_spectral(omega)
    psi = to_physical(psi_hat)

    # N(omega) = -J(psi, omega) + F
    Nval = -1.0 * J_phys(psi, omega, derivative)
    if F is not None:
        Nval = Nval + F

    # omega_dot = L*omega + N
    omega_dot = L_op(omega, L_hat) + Nval

    # psi_dot = inv_lap * omega_dot
    psi_dot_hat = derivative.inv_laplacian * to_spectral(omega_dot)
    psi_dot = to_physical(psi_dot_hat)

    # N_dot
    return -1.0 * J_phys(psi_dot, omega, derivative) \
           - 1.0 * J_phys(psi, omega_dot, derivative)


def compute_n_ddot_analytical(
    omega: torch.Tensor,
    derivative,
    L_hat: torch.Tensor,
    F: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Analytical N_ddot via the chain rule:
        N_ddot = -J(psi_ddot, omega) - 2*J(psi_dot, omega_dot) - J(psi, omega_ddot)
    where omega_ddot = L*omega_dot + N_dot.
    """
    psi_hat = derivative.inv_laplacian * to_spectral(omega)
    psi = to_physical(psi_hat)

    Nval = -1.0 * J_phys(psi, omega, derivative)
    if F is not None:
        Nval = Nval + F
    omega_dot = L_op(omega, L_hat) + Nval

    psi_dot_hat = derivative.inv_laplacian * to_spectral(omega_dot)
    psi_dot = to_physical(psi_dot_hat)

    N_dot = (
        -1.0 * J_phys(psi_dot, omega, derivative)
        - 1.0 * J_phys(psi, omega_dot, derivative)
    )
    omega_ddot = L_op(omega_dot, L_hat) + N_dot

    psi_ddot_hat = derivative.inv_laplacian * to_spectral(omega_ddot)
    psi_ddot = to_physical(psi_ddot_hat)

    return (
        -1.0 * J_phys(psi_ddot, omega, derivative)
        -2.0 * J_phys(psi_dot, omega_dot, derivative)
        -1.0 * J_phys(psi, omega_ddot, derivative)
    )


# --------------------------------------------------------------------------- #
# TEST A -- single-mode sanity check
# --------------------------------------------------------------------------- #

def test_a_single_mode(out_dir: Path) -> None:
    """
    Pick:   omega(x,y) = A * cos(k1 x) * cos(k2 y)
    Then   psi = inv_lap omega  =  -A/(k1^2+k2^2) * cos(k1 x) cos(k2 y)
    And     J(psi, omega) = 0   (psi proportional to omega, J(f,f) = 0)

    With J=0, N = F (if F is on; else 0), constant in time.
    Then trivially N_dot = N_ddot = 0.

    This is a degenerate test that catches sign / FFT-normalization bugs.
    A more interesting non-zero test:
       omega = A * cos(k1 x + k2 y)
       psi   = -A/(k1^2+k2^2) * cos(k1 x + k2 y)
       J(psi, omega) = 0 still (parallel gradients).

    Let's use a sum of two modes which DON'T share a streamfunction:
       omega = A1 cos(k1 x) + A2 cos(k2 y)        (two unrelated modes)
    Then psi has a different spatial structure than omega and J != 0.

    For this test, run the solver-side code on a small grid and verify the
    analytical chain-rule gives results consistent with finite differences
    of N along a short trajectory. Bug-catching, not closed-form check.
    """
    print("\n" + "=" * 72)
    print("TEST A: single-mode sanity check")
    print("=" * 72)

    Nx = Ny = 64
    Lx = Ly = 2 * np.pi
    nu = 1e-3
    mu = 1e-2
    B = 0.0

    grid = CartesianGrid(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, device='cpu',
                         precision='float64')
    derivative = Derivative(grid).to(grid.device)
    L_hat = build_L_hat(derivative, nu=nu, mu=mu, B=B)

    # Build a non-trivial initial omega: two unrelated modes
    x = torch.arange(Nx, device=grid.device) * (Lx / Nx)
    y = torch.arange(Ny, device=grid.device) * (Ly / Ny)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    A1, A2 = 1.0, 0.7
    k1, k2 = 3.0, 5.0
    omega0 = (A1 * torch.cos(k1 * X) + A2 * torch.cos(k2 * Y))[None]   # batch dim

    # No forcing for this test
    F = None

    # Analytical N_dot, N_ddot at omega0
    n_dot_an  = compute_n_dot_analytical(omega0, derivative, L_hat, F)
    n_ddot_an = compute_n_ddot_analytical(omega0, derivative, L_hat, F)
    print(f"  analytical |N_dot|  max  = {n_dot_an.abs().max().item():.4e}")
    print(f"  analytical |N_ddot| max = {n_ddot_an.abs().max().item():.4e}")

    # Finite-difference reference: integrate by tiny dt forward and backward
    # using a high-order one-step (RK4) so the FD baseline isn't AB2-limited.
    dt_fd = 1e-4

    def rhs_phys(omega):
        psi_hat = derivative.inv_laplacian * to_spectral(omega)
        psi = to_physical(psi_hat)
        N = -1.0 * J_phys(psi, omega, derivative)
        return L_op(omega, L_hat) + N

    def N_at(omega):
        psi_hat = derivative.inv_laplacian * to_spectral(omega)
        psi = to_physical(psi_hat)
        return -1.0 * J_phys(psi, omega, derivative)

    def rk4_step(omega, dt):
        k1f = rhs_phys(omega)
        k2f = rhs_phys(omega + 0.5 * dt * k1f)
        k3f = rhs_phys(omega + 0.5 * dt * k2f)
        k4f = rhs_phys(omega + dt * k3f)
        return omega + (dt / 6.0) * (k1f + 2 * k2f + 2 * k3f + k4f)

    # Step backward and forward by dt_fd (uses RK4 accuracy ~ dt^4 -> N derivative
    # FD accuracy ~ dt^2)
    omega_p = rk4_step(omega0, dt_fd)
    omega_m = rk4_step(omega0, -dt_fd)
    N_p = N_at(omega_p)
    N_0 = N_at(omega0)
    N_m = N_at(omega_m)

    n_dot_fd  = (N_p - N_m) / (2 * dt_fd)
    n_ddot_fd = (N_p - 2 * N_0 + N_m) / dt_fd**2

    rel_err_dot  = (n_dot_an  - n_dot_fd ).abs().max() / max(n_dot_fd .abs().max(), 1e-30)
    rel_err_ddot = (n_ddot_an - n_ddot_fd).abs().max() / max(n_ddot_fd.abs().max(), 1e-30)
    print(f"  rel_err N_dot:  max-abs(an - fd)/max-abs(fd) = {rel_err_dot.item():.4e}")
    print(f"  rel_err N_ddot: max-abs(an - fd)/max-abs(fd) = {rel_err_ddot.item():.4e}")
    print(f"  expected ~ O(dt_fd^2) = {dt_fd**2:.1e}; "
          f"acceptable up to ~1e-4 due to RK4 residual + roundoff")

    # ---- visualization: side-by-side fields and per-pixel error ---- #
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fields = [
        (r'$\dot N$ analytical',  n_dot_an[0].cpu().numpy()),
        (r'$\dot N_{FD}$ (RK4)',  n_dot_fd[0].cpu().numpy()),
        (r'$\dot N - \dot N_{FD}$',  (n_dot_an - n_dot_fd)[0].cpu().numpy()),
        (r'$\ddot N$ analytical', n_ddot_an[0].cpu().numpy()),
        (r'$\ddot N_{FD}$ (RK4)', n_ddot_fd[0].cpu().numpy()),
        (r'$\ddot N - \ddot N_{FD}$', (n_ddot_an - n_ddot_fd)[0].cpu().numpy()),
    ]
    for ax, (title, fld) in zip(axes.flat, fields):
        vmax = float(np.max(np.abs(fld)))
        im = ax.imshow(fld, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       origin='lower', interpolation='gaussian', aspect='equal')
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(
        f'TEST A: single-mode sanity check ({Nx}x{Ny}, '
        rf'$dt_{{\rm fd}}={dt_fd}$, RK4 reference)' '\n'
        rf'rel err $\dot N$ = {rel_err_dot.item():.2e}, '
        rf'rel err $\ddot N$ = {rel_err_ddot.item():.2e}',
        fontsize=11,
    )
    fig.tight_layout()
    out_path = out_dir / 'step3_test_a_single_mode.png'
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------- #
# TEST B -- 2D field check using consecutive snapshots from the FR sweep
# --------------------------------------------------------------------------- #

def test_b_2d_in_the_wild(
    sweep_root: Path,
    out_dir: Path,
    n_snapshots: int = 5,
    ref_subdir: str = 'dt_1em5',
    dt_fd: float = 1e-4,
) -> None:
    """
    Use the smallest-dt decaying-turbulence run as the source of central
    snapshots. For each chosen central snapshot omega(t_n), generate the
    plus/minus partners by ultra-fine RK4 integration with step dt_fd
    (forward and backward by dt_fd respectively). Then compute:

      N_dot_FD  = (N(omega^+) - N(omega^-)) / (2 * dt_fd)
      N_ddot_FD = (N(omega^+) - 2 N(omega^0) + N(omega^-)) / dt_fd^2
      N_dot_AN, N_ddot_AN  via the analytical chain rule at omega^0

    Both sides use N = -J(psi, omega): no forcing, no Brinkman, no sponge.
    The actual solver has those terms, but for this code-correctness check
    we just want the chain-rule arithmetic itself to be self-consistent.

    Why this differs from the original Test B: previously we used three
    consecutive saved snapshots separated by dt_save = 0.05, giving an
    O(dt_save^2) ~ 2.5e-3 truncation floor. On chaotic 2D turbulence the
    higher time derivatives of N are huge, so the formal floor is multiplied
    by a giant prefactor and the FD becomes meaningless. Generating the
    plus/minus partners ourselves at dt_fd = 1e-4 reduces the FD truncation
    error to O(dt_fd^2) ~ 1e-8, which is below any reasonable signal level.

    Should agree to leading order O(dt_fd^2). With dt_fd = 1e-4 this is
    ~1e-8 relative, so a healthy match is anywhere below ~1e-4.
    """
    print("\n" + "=" * 72)
    print("TEST B: 2D in-the-wild check on decaying turbulence data")
    print("=" * 72)

    # use the reference run (smallest dt) and load its omega + times
    run_dir = sweep_root / ref_subdir
    omega_path = run_dir / 'DNS_FR_omega.npy'
    times_path = run_dir / 'DNS_FR_times.npy'
    params_path = run_dir / 'DNS_FR_params.yaml'

    if not omega_path.exists():
        raise FileNotFoundError(
            f"{omega_path} not found. Run prepare_npz_for_mmap.py first."
        )

    times = np.load(times_path)
    T_save = len(times)
    omega_loader = _OmegaLoader(omega_path)
    Ny, Nx = omega_loader.spatial_shape
    dt_save = float(times[1] - times[0])
    print(f"  loaded run ({ref_subdir}): shape (T,Ny,Nx)=({T_save},{Ny},{Nx}), "
          f"dt_save={dt_save:.4f}, loader={omega_loader.mode}")

    # read PDE parameters from the run's YAML so we use the right nu, mu, B
    if params_path.exists():
        with open(params_path) as f:
            params = yaml.safe_load(f)
        pde = params.get('pde', {})
        nu = float(pde.get('nu', 1.025e-5))
        mu = float(pde.get('mu', 0.0))
        B  = float(pde.get('B',  0.0))
        Lx = float(params.get('grid', {}).get('Lx', 2 * np.pi))
        Ly = float(params.get('grid', {}).get('Ly', 2 * np.pi))
        print(f"  params: nu={nu}, mu={mu}, B(beta)={B}, "
              f"Lx={Lx:.3f}, Ly={Ly:.3f}")
    else:
        nu = 1.025e-5
        mu = 0.0
        B = 0.0
        Lx = Ly = 2 * np.pi
        print(f"  no params file; falling back to decaying-turbulence YAML defaults: "
              f"nu={nu}, mu={mu}, B={B}, Lx=Ly=8*pi")

    # Build grid + derivative + L_hat at the FULL resolution (512x512)
    grid = CartesianGrid(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, device='cpu',
                         precision='float64')
    derivative = Derivative(grid).to(grid.device)
    L_hat = build_L_hat(derivative, nu=nu, mu=mu, B=B)

    # No forcing in decaying turbulence case
    F = None
    print(f"  no forcing (decaying turb); N = -J(psi, omega)")

    # Helper: N(omega) physical, no forcing
    def N_at(omega):
        psi_hat = derivative.inv_laplacian * to_spectral(omega)
        psi = to_physical(psi_hat)
        return -1.0 * J_phys(psi, omega, derivative)

    # Helper: full PDE RHS = L*omega + N(omega), used by RK4
    def rhs_phys(omega):
        return L_op(omega, L_hat) + N_at(omega)

    # Classical 4th-order Runge-Kutta single-step. Works for both forward
    # (dt > 0) and backward (dt < 0) integration.
    def rk4_step(omega, dt):
        k1 = rhs_phys(omega)
        k2 = rhs_phys(omega + 0.5 * dt * k1)
        k3 = rhs_phys(omega + 0.5 * dt * k2)
        k4 = rhs_phys(omega + dt * k3)
        return omega + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Pick n_snapshots random interior centrals, avoiding the early transient
    rng = np.random.default_rng(42)
    spinup_mask = times >= 5.0    # skip the early high-amplitude transient
    valid_indices = np.where(spinup_mask)[0]
    n_pick = min(n_snapshots, len(valid_indices))
    chosen = rng.choice(valid_indices, size=n_pick, replace=False)
    chosen = sorted(chosen.tolist())
    print(f"  testing on snapshot indices: {chosen}")
    print(f"  FD step:    dt_fd = {dt_fd:.1e}  "
          f"(formal truncation floor O(dt_fd^2) = {dt_fd**2:.1e})")

    # Per-pick stats
    rel_err_dots, rel_err_ddots, t_chosen = [], [], []
    for n in chosen:
        # Read just the central snapshot
        om_0 = torch.tensor(omega_loader.read(n), dtype=torch.float64)[None]

        # Generate plus/minus partners by ultra-fine RK4 from om_0
        # (forward by +dt_fd, backward by -dt_fd). RK4 has truncation error
        # O(dt_fd^5) per step, much smaller than the centered-difference
        # O(dt_fd^2) we're trying to characterize.
        om_p = rk4_step(om_0, +dt_fd)
        om_m = rk4_step(om_0, -dt_fd)

        # finite-difference N_dot, N_ddot at om_0
        N_m = N_at(om_m); N_0 = N_at(om_0); N_p = N_at(om_p)
        ndot_fd  = (N_p - N_m) / (2 * dt_fd)
        nddot_fd = (N_p - 2 * N_0 + N_m) / dt_fd**2

        # analytical at om_0
        ndot_an  = compute_n_dot_analytical (om_0, derivative, L_hat, F)
        nddot_an = compute_n_ddot_analytical(om_0, derivative, L_hat, F)

        # relative L2 error
        def rel_l2(a, b):
            num = torch.sqrt(torch.mean((a - b)**2))
            den = torch.sqrt(torch.mean(b**2))
            return (num / den).item() if den > 0 else float('nan')

        ed = rel_l2(ndot_an, ndot_fd)
        edd = rel_l2(nddot_an, nddot_fd)
        rel_err_dots.append(ed)
        rel_err_ddots.append(edd)
        t_chosen.append(times[n])
        print(f"    t={times[n]:.3f} (idx={n}): rel L2 err  "
              f"N_dot={ed:.3e}  N_ddot={edd:.3e}")

    # --- summary plot --- #
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(t_chosen, rel_err_dots,  'o-', label=r'$\dot N$',  color='C0')
    ax.plot(t_chosen, rel_err_ddots, 's-', label=r'$\ddot N$', color='C3')
    ax.axhline(dt_fd**2, color='k', ls='--', alpha=0.5,
               label=fr'$\mathcal{{O}}(dt_{{\rm fd}}^2) = {dt_fd**2:.1e}$')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'Relative $L^2$ Error: Analytical vs $_{\rm FD}$')
    ax.set_yscale('log')
    ax.set_title(
        rf'Test B: Analytical Chain-Rule vs $_{{\rm FD}}$ via Ultra-Fine RK4, '
        rf'$dt_{{\rm fd}}={dt_fd:.1e}$ (Decaying Turbulence)'
    )
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / 'step3_test_b_2d_in_the_wild_decay.png'
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_path}")

    # one example field comparison -- regenerate plus/minus via RK4 here too
    n_mid = chosen[len(chosen) // 2]
    om_0 = torch.tensor(omega_loader.read(n_mid), dtype=torch.float64)[None]
    om_p = rk4_step(om_0, +dt_fd)
    om_m = rk4_step(om_0, -dt_fd)
    ndot_fd  = (N_at(om_p) - N_at(om_m)) / (2 * dt_fd)
    nddot_fd = (N_at(om_p) - 2 * N_at(om_0) + N_at(om_m)) / dt_fd**2
    ndot_an  = compute_n_dot_analytical (om_0, derivative, L_hat, F)
    nddot_an = compute_n_ddot_analytical(om_0, derivative, L_hat, F)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    rows = [
        (r'\dot N',  ndot_an,  ndot_fd),
        (r'\ddot N', nddot_an, nddot_fd),
    ]
    for irow, (label, an, fd) in enumerate(rows):
        an_np = an[0].cpu().numpy()
        fd_np = fd[0].cpu().numpy()
        diff_np = an_np - fd_np
        vmax_fld = float(np.max(np.abs(np.concatenate([an_np.flat, fd_np.flat]))))
        vmax_diff = float(np.max(np.abs(diff_np)))

        axes[irow, 0].imshow(an_np, cmap='RdBu_r', vmin=-vmax_fld, vmax=vmax_fld,
                             origin='lower', interpolation='gaussian', aspect='equal')
        axes[irow, 0].set_title(rf'${label}$ Analytical', fontsize=10)
        axes[irow, 1].imshow(fd_np, cmap='RdBu_r', vmin=-vmax_fld, vmax=vmax_fld,
                             origin='lower', interpolation='gaussian', aspect='equal')
        axes[irow, 1].set_title(rf'${label}_{{\rm FD}}$ (RK4, $dt_{{\rm fd}}={dt_fd:.0e}$)',
                                fontsize=10)
        axes[irow, 2].imshow(diff_np, cmap='seismic', vmin=-vmax_diff, vmax=vmax_diff,
                             origin='lower', interpolation='gaussian', aspect='equal')
        axes[irow, 2].set_title(rf'${label} - {label}_{{\rm FD}}$', fontsize=10)
        for ax in axes[irow]:
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(rf'Test B: Decaying-Turb Snapshot at $t={times[n_mid]:.3f}$, '
                 rf'Analytical vs RK4-FD',
                 fontsize=12)
    fig.tight_layout()
    out_path = out_dir / 'step3_test_b_field_example_decay.png'
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------- #
# Robust per-snapshot loader (avoids whole-file mmap that can OOM on the      #
# cluster login node)                                                         #
# --------------------------------------------------------------------------- #

class _OmegaLoader:
    """
    Read a single snapshot from a DNS_FR_omega.npy file without holding the
    whole file in memory. Tries np.load(mmap_mode='r') first (fast random
    access), and falls back to a per-snapshot direct read using the .npy
    header layout if mmap fails (e.g. with OSError: cannot allocate memory
    on a memory-constrained login node).

    The fallback reads only one snapshot (Ny * Nx * itemsize bytes) per call.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.mode = None
        self._mmap = None         # used in 'mmap' mode
        self._header = None       # used in 'fallback' mode
        self._header_offset = None
        self._dtype = None
        self._spatial_shape = None
        self._n_snapshots = None
        self._has_batch_dim = False

        # First, parse the .npy header to learn the shape and dtype no matter
        # which mode we end up in.
        from numpy.lib import format as np_format
        with open(self.path, 'rb') as f:
            major, minor = np_format.read_magic(f)
            if (major, minor) == (1, 0):
                shape, fortran_order, dtype = np_format.read_array_header_1_0(f)
            elif (major, minor) == (2, 0):
                shape, fortran_order, dtype = np_format.read_array_header_2_0(f)
            else:
                raise RuntimeError(f"unsupported .npy version {major}.{minor}")
            self._header_offset = f.tell()
        if fortran_order:
            raise RuntimeError("Fortran-order .npy not supported here")
        self._dtype = dtype

        # Layouts we accept:
        #   (T, Ny, Nx)       -- single trajectory, no batch dim
        #   (1, T, Ny, Nx)    -- batch-of-1
        #   (B, T, Ny, Nx)    -- multi-batch (we pick batch 0 by default,
        #                       same fallback as Step 1's prepare_npz)
        self._batch_index = 0  # used only when has_batch_dim
        if len(shape) == 4:
            self._has_batch_dim = True
            B, T, Ny, Nx = shape
            if B != 1:
                print(f"  [warn] {self.path.name}: omega has B={B} batches; "
                      f"using batch {self._batch_index}. Regenerate with "
                      f"prepare_npz_for_mmap.py --batch-index 0 to strip the dim.")
            self._n_snapshots = T
            self._spatial_shape = (Ny, Nx)
        elif len(shape) == 3:
            self._has_batch_dim = False
            self._n_snapshots = shape[0]
            self._spatial_shape = (shape[1], shape[2])
        else:
            raise RuntimeError(f"unexpected omega shape {shape}; expected 3D or 4D")

        # Try mmap; if it fails, use fallback.
        try:
            mm = np.load(self.path, mmap_mode='r')
            if mm.ndim == 4:
                mm = mm[self._batch_index]
            self._mmap = mm
            self.mode = 'mmap'
        except OSError as e:
            print(f"  [info] mmap of {self.path.name} failed ({e}); "
                  f"using per-snapshot fallback loader")
            self.mode = 'fallback'

    @property
    def spatial_shape(self) -> Tuple[int, int]:
        return self._spatial_shape

    @property
    def n_snapshots(self) -> int:
        return self._n_snapshots

    def read(self, idx: int) -> np.ndarray:
        """Read the idx-th snapshot as a fresh numpy array."""
        if idx < 0 or idx >= self._n_snapshots:
            raise IndexError(f"snapshot index {idx} out of range [0, {self._n_snapshots})")
        if self.mode == 'mmap':
            return np.asarray(self._mmap[idx], dtype=self._dtype)
        # Fallback: seek + read just the bytes for this snapshot
        Ny, Nx = self._spatial_shape
        per_snapshot_bytes = Ny * Nx * self._dtype.itemsize
        # Adjust offset for the leading batch dim if present
        if self._has_batch_dim:
            T = self._n_snapshots
            absolute_idx = self._batch_index * T + idx
        else:
            absolute_idx = idx
        offset_to_idx = self._header_offset + absolute_idx * per_snapshot_bytes
        with open(self.path, 'rb') as f:
            f.seek(offset_to_idx)
            buf = f.read(per_snapshot_bytes)
        if len(buf) != per_snapshot_bytes:
            raise RuntimeError(f"short read at idx={idx}: "
                               f"got {len(buf)} bytes, expected {per_snapshot_bytes}")
        return np.frombuffer(buf, dtype=self._dtype).reshape(Ny, Nx).copy()


# --------------------------------------------------------------------------- #
# TEST C -- analytical vs empirical AB2CN2 local truncation error             #
# --------------------------------------------------------------------------- #

def ab2cn2_step_spectral(qh_n: torch.Tensor, qh_nm1: torch.Tensor,
                         dt: float, derivative, L_hat: torch.Tensor,
                         F_phys: Optional[torch.Tensor]) -> torch.Tensor:
    r"""
    One bare AB2CN2 IMEX step in spectral space.

    The solver advances:
        $\bar\omega_t = L \bar\omega + N(\bar\omega)$
    where $L$ is treated implicitly via Crank-Nicolson and $N$ explicitly via
    AB2. Following the solver's convention (see qg/solver/integrator/imex.py
    and qg/qg.py):

        N^n        = N(\bar\omega^n)
        N^{n-1}    = N(\bar\omega^{n-1})
        AB2(N)     = (3/2) N^n - (1/2) N^{n-1}
        rhs        = qh^n + dt * (0.5 * L_hat * qh^n + AB2(N))
        qh^{n+1}   = rhs / (1 - 0.5 * dt * L_hat)

    Args:
        qh_n   : spectral vorticity at level n (current). shape (1, Ny, Nx).
        qh_nm1 : spectral vorticity at level n-1 (previous). shape (1, Ny, Nx).
        dt     : the step size to advance.
        derivative, L_hat: same as elsewhere in this script.
        F_phys : forcing in physical space, or None.

    Returns: qh^{n+1}, spectral.

    Note: the formula uses a single linear-operator denominator. This matches
    qg/solver/integrator/imex.py:CN2.
    """
    # Build N^n and N^{n-1} in physical space (matching the solver's pipeline)
    def N_at_qh(qh):
        psi_hat = derivative.inv_laplacian * qh
        psi = to_physical(psi_hat)
        omega = to_physical(qh)
        N_phys = -1.0 * J_phys(psi, omega, derivative)
        if F_phys is not None:
            N_phys = N_phys + F_phys
        return to_spectral(N_phys)

    Nh_n   = N_at_qh(qh_n)
    Nh_nm1 = N_at_qh(qh_nm1)
    AB2_Nh = 1.5 * Nh_n - 0.5 * Nh_nm1

    # CN2: rhs = qh + dt * (0.5*L*qh + AB2(N));  qh_new = rhs / (1 - 0.5*dt*L)
    rhs_hat = qh_n + dt * (0.5 * L_hat * qh_n + AB2_Nh)
    denom_hat = 1.0 - 0.5 * dt * L_hat
    return rhs_hat / denom_hat


def predicted_local_error_phys(omega: torch.Tensor, derivative,
                               L_hat: torch.Tensor,
                               F: Optional[torch.Tensor],
                               dT: float,
                               K: int) -> torch.Tensor:
    r"""
    Predicted difference between bare AB2CN2 at coarse step $\Delta T$ and
    bare AB2CN2 at fine step $h = \Delta T / K$ ($K$ steps), both starting
    from the same $\bar\omega^n$:

        $\bar\omega^{n+1}_{\rm coarse} - \bar\omega^{n+1}_{\rm fine}$
            $\approx \frac{\Delta T^3}{12}\bigl(1 - \frac{1}{K^2}\bigr)
                      \bigl[L^3 \bar\omega + L^2 N + L\dot N - 5\ddot N\bigr]$

    Both schemes have the same leading-order coefficient $\frac{1}{12}[\dots]$;
    the coarse and fine truncation errors differ only by their step sizes,
    yielding the $(1 - 1/K^2)$ factor.

    All terms are evaluated at the central snapshot $\bar\omega^n$.

    For $K = 1$ this returns zero (degenerate: coarse equals fine).
    For $K \to \infty$, the factor approaches 1 and we recover the standard
    AB2CN2 vs exact-PDE local truncation error.
    """
    # qh = spectral vorticity
    qh = to_spectral(omega)

    # L^3 * omega  (purely spectral)
    L3_omega_phys = to_physical(L_hat**3 * qh)

    # L^2 * N(omega)
    psi_hat = derivative.inv_laplacian * qh
    psi = to_physical(psi_hat)
    N_phys = -1.0 * J_phys(psi, omega, derivative)
    if F is not None:
        N_phys = N_phys + F
    Nh = to_spectral(N_phys)
    L2_N_phys = to_physical(L_hat**2 * Nh)

    # L * Ndot
    Ndot_phys = compute_n_dot_analytical(omega, derivative, L_hat, F)
    Ndot_h    = to_spectral(Ndot_phys)
    L_Ndot_phys = to_physical(L_hat * Ndot_h)

    # Nddot
    Nddot_phys = compute_n_ddot_analytical(omega, derivative, L_hat, F)

    K_factor = 1.0 - 1.0 / (K ** 2)   # zero when K=1, ~1 for large K
    coef = (dT ** 3) / 12.0 * K_factor
    return coef * (L3_omega_phys + L2_N_phys + L_Ndot_phys - 5.0 * Nddot_phys)


def test_c_truncation_error(
    sweep_root: Path,
    out_dir: Path,
    n_starts: int = 3,
    K_values: Tuple[int, ...] = (10, 50, 100, 500, 1000),
    h_fine: float = 1.0e-5,
    h_ultrafine: float = 5.0e-6,
    ref_subdir: str = 'dt_1em5',
) -> None:
    r"""
    End-to-end check that the chain-rule machinery for $\dot N, \ddot N$
    correctly predicts the difference between two bare AB2CN2 schemes
    starting from the SAME two-level stencil $(\bar\omega^{-1}, \bar\omega^0)$:

      - COARSE: one bare AB2CN2 step of size $\Delta T = K \cdot h_{\rm fine}$
      - FINE:   $K$ bare AB2CN2 steps of size $h_{\rm fine}$

    Both starting from $(\bar\omega^{-1}, \bar\omega^0)$ as the AB2 stencil.

    PROCEDURE.
      For each starting time $t_n$ from a saved snapshot (the "seed"):
        1. Generate $\bar\omega^{-1}$ by ultra-fine RK4 integration (step
           $h_{\rm uf} = 5\times 10^{-6}$, twice as fine as $h_{\rm fine}$)
           forward from the seed by exactly $\Delta T$. This is essentially
           exact -- RK4 truncation error at $h_{\rm uf}^4 = 6.25\times 10^{-22}$
           per step is way below anything we are measuring.
        2. Continue ultra-fine integration for another $\Delta T$ to get
           $\bar\omega^0$. So:
              seed at $t_n$
              $\bar\omega^{-1}$ at $t_n + \Delta T$
              $\bar\omega^0$ at $t_n + 2\Delta T$
        3. From $(\bar\omega^{-1}, \bar\omega^0)$ as the AB2 two-level stencil:
              COARSE: one AB2CN2 step of size $\Delta T$ -> $\bar\omega^1_{\rm coarse}$
              FINE:   K AB2CN2 steps of size $h_{\rm fine}$ -> $\bar\omega^K_{\rm fine}$
        4. Empirical: $\bar\omega^1_{\rm coarse} - \bar\omega^K_{\rm fine}$
           Predicted: $\frac{\Delta T^3}{12}(1 - 1/K^2)
                       [L^3\bar\omega^0 + L^2 N^0 + L\dot N^0 - 5\ddot N^0]$.

    EFFICIENCY.
      For each starting point we run the ultra-fine integration just ONCE
      to the largest $\Delta T$, saving the field at every $\Delta T$ value.
      Then we replay the comparison for each (start, K).

    NOTES.
      - The fine integration here uses $(\bar\omega^{-1}, \bar\omega^0)$ as
        its starting two-level stencil even though $\bar\omega^{-1}$ is at
        $t_n + \Delta T$, not $t_n + 2\Delta T - h_{\rm fine}$. This matches
        the modified-equation derivation, which expanded $\bar\omega^{n-1}$
        around $\bar\omega^n$ assuming the *same* $\bar\omega^{n-1}$ for
        coarse and fine schemes.
      - K=1 is excluded by default (degenerate: coarse step equals fine step).
    """
    print("\n" + "=" * 72)
    print("TEST C: analytical vs empirical AB2CN2 coarse-vs-fine difference")
    print("=" * 72)

    run_dir = sweep_root / ref_subdir
    omega_path = run_dir / 'DNS_FR_omega.npy'
    times_path = run_dir / 'DNS_FR_times.npy'
    params_path = run_dir / 'DNS_FR_params.yaml'

    if not omega_path.exists():
        raise FileNotFoundError(
            f"{omega_path} not found. Run prepare_npz_for_mmap.py first."
        )

    # We only need:
    #   * the times array (small)
    #   * a small number of seed snapshots (one per starting point)
    # mmap'ing the entire DNS_FR_omega.npy file may fail with OSError on memory-
    # constrained machines (the whole-file VA reservation is rejected). Use a
    # robust loader that tries mmap first and falls back to a per-snapshot read
    # using the .npy header.
    times = np.load(times_path)
    T_save = len(times)
    h_save = float(times[1] - times[0])

    omega_loader = _OmegaLoader(omega_path)
    Ny, Nx = omega_loader.spatial_shape
    print(f"  loaded ({ref_subdir}): T={T_save}, grid={Ny}x{Nx}, "
          f"h_save={h_save:.4f}, loader={omega_loader.mode}")

    # Read PDE params
    if params_path.exists():
        with open(params_path) as f:
            params = yaml.safe_load(f)
        pde = params.get('pde', {})
        nu = float(pde.get('nu', 1.025e-5))
        mu = float(pde.get('mu', 0.0))
        B  = float(pde.get('B',  0.0))
        Lx = float(params.get('grid', {}).get('Lx', 2 * np.pi))
        Ly = float(params.get('grid', {}).get('Ly', 2 * np.pi))
    else:
        nu, mu, B = 1.025e-5, 0.0, 0.0
        Lx = Ly = 2 * np.pi

    grid = CartesianGrid(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, device='cpu',
                         precision='float64')
    derivative = Derivative(grid).to(grid.device)
    L_hat = build_L_hat(derivative, nu=nu, mu=mu, B=B)

    F = None  # decaying turbulence has no forcing

    # Sort K values ascending; find the largest dT we'll need
    K_values = tuple(sorted(set(K_values)))
    dT_values = np.array([K * h_fine for K in K_values])
    dT_max = float(dT_values.max())
    print(f"  K values:    {K_values}")
    print(f"  dT values:   {[f'{x:.2e}' for x in dT_values]}")
    print(f"  h_fine:      {h_fine:.1e}  (fine AB2CN2 step)")
    print(f"  h_ultrafine: {h_ultrafine:.1e}  (RK4 warmup step)")
    print(f"  dT_max:      {dT_max:.2e}  (max ultra-fine span = 2*dT_max)")

    # Pick starting snapshots. Seed must be far enough into the run to be past
    # spinup; we don't need any forward room in the saved data because we
    # generate everything via ultra-fine integration from the seed.
    rng = np.random.default_rng(13)
    spinup_mask = times >= 10.0
    valid_indices = np.where(spinup_mask)[0]
    n_pick = min(n_starts, len(valid_indices))
    starts = sorted(rng.choice(valid_indices, size=n_pick, replace=False).tolist())
    print(f"  starts (snapshot indices): {starts}")
    print(f"  starts (times):            {[f'{times[s]:.2f}' for s in starts]}")

    # ---- helper: RK4 step in physical space ---- #
    def rhs_phys(omega):
        """R(omega) = L*omega + N(omega), N = -J(psi, omega) (no forcing here)."""
        psi_hat = derivative.inv_laplacian * to_spectral(omega)
        psi = to_physical(psi_hat)
        N = -1.0 * J_phys(psi, omega, derivative)
        return L_op(omega, L_hat) + N

    def rk4_step(omega, dt):
        k1 = rhs_phys(omega)
        k2 = rhs_phys(omega + 0.5 * dt * k1)
        k3 = rhs_phys(omega + 0.5 * dt * k2)
        k4 = rhs_phys(omega + dt * k3)
        return omega + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # ---- Loop over starting points ---- #
    rows_dT, rows_K, rows_emp, rows_pred, rows_diff, rows_relerr, rows_start, rows_om0_l2 = (
        [], [], [], [], [], [], [], []
    )

    # For the second-PNG pointwise-difference figure, save delta_emp and
    # delta_pred at one diagnostic dT per start. We pick the SMALLEST dT,
    # which in panel (b) shows the largest relative residual -- precisely
    # the regime we want to inspect spatially.
    diag_K = min(K_values)
    diag_dT = diag_K * h_fine
    diag_fields = []  # list of dicts, one per start

    for s_idx in starts:
        seed_np = omega_loader.read(s_idx)
        seed = torch.tensor(seed_np, dtype=torch.float64)[None]
        t_seed = times[s_idx]
        print(f"\n  start s_idx={s_idx} t={t_seed:.3f}")

        # Strategy: integrate ultra-fine RK4 from seed forward by 2*dT_max,
        # saving the field at every dT_value AND at every 2*dT_value.
        # That way we have $\bar\omega^{-1}$ at t_seed + dT and $\bar\omega^0$
        # at t_seed + 2*dT for each dT in dT_values, all from a single sweep.
        save_times_offsets = sorted(set(
            [dT for dT in dT_values] + [2 * dT for dT in dT_values]
        ))
        # max time offset
        t_max_offset = save_times_offsets[-1]
        # number of ultra-fine steps to cover that
        n_uf_steps = int(np.ceil(t_max_offset / h_ultrafine))
        print(f"    ultra-fine warmup: {n_uf_steps} RK4 steps of {h_ultrafine:.1e}")

        # Storage: dict from time-offset -> field (snapshot)
        saved_fields = {}
        target_offsets = list(save_times_offsets)
        next_target = target_offsets.pop(0)

        omega_cur = seed.clone()
        t_cur = 0.0
        for step in range(n_uf_steps):
            omega_cur = rk4_step(omega_cur, h_ultrafine)
            t_cur = (step + 1) * h_ultrafine
            # check whether we just crossed a target time (within 1 step)
            while next_target is not None and t_cur >= next_target - 1e-12:
                # we slightly overshoot by at most h_ultrafine; that's fine
                # because RK4 error is so small. But ideally save at exact
                # target by interpolating linearly; here we just take the
                # post-step value.
                saved_fields[next_target] = omega_cur.clone()
                next_target = target_offsets.pop(0) if target_offsets else None
            if next_target is None:
                break
        print(f"    warmup done: {len(saved_fields)} snapshots saved")

        # ---- For each K, run the comparison ---- #
        for K, dT in zip(K_values, dT_values):
            om_nm1 = saved_fields[float(dT)]      # $\bar\omega^{-1}$ at offset dT
            om_n0  = saved_fields[float(2 * dT)]  # $\bar\omega^0$ at offset 2*dT

            # COARSE: one AB2CN2 step of size dT from (om_nm1, om_n0)
            qh_nm1 = to_spectral(om_nm1)
            qh_n0  = to_spectral(om_n0)
            qh_coarse = ab2cn2_step_spectral(qh_n0, qh_nm1, dT, derivative,
                                             L_hat, F)
            om_coarse = to_physical(qh_coarse)

            # FINE: K AB2CN2 steps of size h_fine from same (om_nm1, om_n0)
            # Note: the fine's first step uses om_nm1 (at offset dT) as its
            # "previous level", which is non-standard (the proper previous
            # step would be at offset 2dT - h_fine) -- this matches the
            # modified-equation derivation convention.
            qh_minus = qh_nm1
            qh_curr  = qh_n0
            for _ in range(K):
                qh_next = ab2cn2_step_spectral(qh_curr, qh_minus, h_fine,
                                               derivative, L_hat, F)
                qh_minus = qh_curr
                qh_curr = qh_next
            om_fine = to_physical(qh_curr)

            # Empirical: coarse - fine
            delta_emp = (om_coarse - om_fine)[0]

            # Predicted: formula evaluated at om_n0 with (dT, K)
            delta_pred = predicted_local_error_phys(
                om_n0, derivative, L_hat, F, dT, K
            )[0]

            l2_emp  = float(torch.sqrt(torch.mean(delta_emp**2)).item())
            l2_pred = float(torch.sqrt(torch.mean(delta_pred**2)).item())
            l2_diff = float(torch.sqrt(torch.mean((delta_emp - delta_pred)**2)).item())
            l2_om0  = float(torch.sqrt(torch.mean(om_n0[0]**2)).item())
            rel_err = l2_diff / l2_emp if l2_emp > 0 else float('nan')

            rows_dT.append(dT)
            rows_K.append(K)
            rows_emp.append(l2_emp)
            rows_pred.append(l2_pred)
            rows_diff.append(l2_diff)
            rows_relerr.append(rel_err)
            rows_start.append(s_idx)
            rows_om0_l2.append(l2_om0)
            print(f"    K={K:4d}  dT={dT:.2e}  "
                  f"|emp|={l2_emp:.3e}  |pred|={l2_pred:.3e}  "
                  f"|diff|={l2_diff:.3e}  rel_err={rel_err:.3e}")

            # Save spatial fields at the diagnostic dT for the second PNG
            if K == diag_K:
                diag_fields.append(dict(
                    s_idx=s_idx,
                    t_seed=float(times[s_idx]),
                    dT=dT,
                    K=K,
                    delta_emp=delta_emp.detach().cpu().numpy().copy(),
                    delta_pred=delta_pred.detach().cpu().numpy().copy(),
                ))

    rows_dT      = np.array(rows_dT)
    rows_K       = np.array(rows_K)
    rows_emp     = np.array(rows_emp)
    rows_pred    = np.array(rows_pred)
    rows_diff    = np.array(rows_diff)
    rows_relerr  = np.array(rows_relerr)
    rows_start   = np.array(rows_start)
    rows_om0_l2  = np.array(rows_om0_l2)

    # ============================================================ #
    # PNG 1: aggregate diagnostics (1 x 3)                          #
    #   (a) |emp| and |pred| vs dT, with slope-2 and slope-3 lines #
    #   (b) |emp - pred| absolute, with slope-3 and slope-4 lines  #
    #   (c) |emp - pred| / |omega^0|, with slope-3 and slope-4     #
    # ============================================================ #
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    cmap = mpl.colormaps['viridis']

    # ----- (a) |emp| and |pred| ----- #
    for j, s_idx in enumerate(starts):
        mask = rows_start == s_idx
        col = cmap(j / max(len(starts) - 1, 1))
        axes[0].loglog(rows_dT[mask], rows_emp[mask], 'o-',
                       color=col, ms=8, lw=1.5,
                       label=rf'Start $t={times[s_idx]:.2f}$ (Empirical)')
        axes[0].loglog(rows_dT[mask], rows_pred[mask], 's--',
                       color=col, ms=6, lw=1.0, alpha=0.7,
                       label=rf'Start $t={times[s_idx]:.2f}$ (Predicted)')

    if len(rows_dT) > 0:
        dT_line = np.array([rows_dT.min(), rows_dT.max()])
        ymax_dt = dT_line[-1]
        ymax_emp = rows_emp[rows_dT == ymax_dt].max() if (rows_dT == ymax_dt).any() else rows_emp.max()
        ref3 = ymax_emp * (dT_line / dT_line[-1])**3
        ref2 = ymax_emp * (dT_line / dT_line[-1])**2
        axes[0].loglog(dT_line, ref3, 'k--', alpha=0.6, lw=1.0,
                       label=r'Slope $= 3$')
        axes[0].loglog(dT_line, ref2, 'k:',  alpha=0.6, lw=1.0,
                       label=r'Slope $= 2$')

    axes[0].set_xlabel(r'$\Delta T$')
    axes[0].set_ylabel(r'$\|\bar\omega^1_{\rm coarse} - \bar\omega^K_{\rm fine}\|_2$')
    axes[0].set_title(r'(a) Coarse-vs-Fine Difference vs $\Delta T$')
    axes[0].grid(True, which='both', alpha=0.3)
    axes[0].legend(fontsize=7, loc='lower right')

    # ----- (b) absolute residual |emp - pred| ----- #
    for j, s_idx in enumerate(starts):
        mask = rows_start == s_idx
        col = cmap(j / max(len(starts) - 1, 1))
        axes[1].loglog(rows_dT[mask], rows_diff[mask], 'o-',
                       color=col, ms=8, lw=1.5,
                       label=rf'Start $t={times[s_idx]:.2f}$')
    if len(rows_dT) > 0:
        finite = np.isfinite(rows_diff) & (rows_diff > 0)
        if finite.any():
            ymax_diff = rows_diff[finite].max()
            ref4 = ymax_diff * (dT_line / dT_line[-1])**4
            ref3b = ymax_diff * (dT_line / dT_line[-1])**3
            axes[1].loglog(dT_line, ref4,  'k--', alpha=0.6, lw=1.0,
                           label=r'Slope $= 4$')
            axes[1].loglog(dT_line, ref3b, 'k:',  alpha=0.6, lw=1.0,
                           label=r'Slope $= 3$')
    axes[1].set_xlabel(r'$\Delta T$')
    axes[1].set_ylabel(r'$\|\Delta_{\rm emp} - \Delta_{\rm pred}\|_2$')
    axes[1].set_title(r'(b) Absolute Residual $\|\Delta_{\rm emp} - \Delta_{\rm pred}\|_2$')
    axes[1].grid(True, which='both', alpha=0.3)
    axes[1].legend(fontsize=7, loc='lower right')

    # ----- (c) residual normalized by |omega^0| ----- #
    rows_normed = np.where(rows_om0_l2 > 0, rows_diff / rows_om0_l2, np.nan)
    for j, s_idx in enumerate(starts):
        mask = rows_start == s_idx
        col = cmap(j / max(len(starts) - 1, 1))
        axes[2].loglog(rows_dT[mask], rows_normed[mask], 'o-',
                       color=col, ms=8, lw=1.5,
                       label=rf'Start $t={times[s_idx]:.2f}$')
    if len(rows_dT) > 0:
        finite = np.isfinite(rows_normed) & (rows_normed > 0)
        if finite.any():
            ymax_n = rows_normed[finite].max()
            ref4c = ymax_n * (dT_line / dT_line[-1])**4
            ref3c = ymax_n * (dT_line / dT_line[-1])**3
            axes[2].loglog(dT_line, ref4c, 'k--', alpha=0.6, lw=1.0,
                           label=r'Slope $= 4$')
            axes[2].loglog(dT_line, ref3c, 'k:',  alpha=0.6, lw=1.0,
                           label=r'Slope $= 3$')
    axes[2].set_xlabel(r'$\Delta T$')
    axes[2].set_ylabel(r'$\|\Delta_{\rm emp} - \Delta_{\rm pred}\|_2 / \|\bar\omega^0\|_2$')
    axes[2].set_title(r'(c) Residual Normalized by $\|\bar\omega^0\|_2$')
    axes[2].grid(True, which='both', alpha=0.3)
    axes[2].legend(fontsize=7, loc='lower right')

    fig.suptitle(
        r'Test C: AB2CN2 Coarse-vs-Fine Difference vs Derived Closure '
        r'(Decaying Turbulence, $N=256^2$, $L_x = L_y = 2\pi$, $T = 60$, Batch #0)'
        '\n'
        r'$\bar\omega^1_{\rm coarse} - \bar\omega^K_{\rm fine} \approx '
        r'\frac{\Delta T^3}{12}(1 - 1/K^2)\,[L^3\bar\omega + L^2 N + L\dot N - 5\ddot N]$',
        fontsize=11,
    )
    fig.tight_layout()
    out_path = out_dir / 'step3_test_c_truncation_error_decay.png'
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  wrote {out_path}")

    # ============================================================ #
    # PNG 2: pointwise spatial diagnostics at the diagnostic dT     #
    #   3 rows (one per start) x 3 cols (delta_emp, delta_pred,     #
    #   delta_emp - delta_pred). Reveals whether the discrepancy at #
    #   small dT is structural (different spatial pattern) or just  #
    #   numerical noise (incoherent residual).                      #
    # ============================================================ #
    if len(diag_fields) > 0:
        n_rows = len(diag_fields)
        fig2, ax2 = plt.subplots(n_rows, 3, figsize=(13, 4.0 * n_rows),
                                 gridspec_kw=dict(wspace=0.05, hspace=0.18))
        if n_rows == 1:
            ax2 = ax2[None, :]

        for r, fld in enumerate(diag_fields):
            de = fld['delta_emp']
            dp = fld['delta_pred']
            dr = de - dp

            # Symmetric color scale shared between emp and pred panels;
            # separate scale for the residual panel.
            v_ep = float(np.max(np.abs(np.concatenate([de.flat, dp.flat]))))
            v_r  = float(np.max(np.abs(dr)))
            if v_ep == 0: v_ep = 1e-30
            if v_r == 0:  v_r = 1e-30

            ax2[r, 0].imshow(de, origin='lower', cmap='seismic',
                             vmin=-v_ep, vmax=+v_ep, aspect='equal',
                             interpolation='gaussian')
            ax2[r, 1].imshow(dp, origin='lower', cmap='seismic',
                             vmin=-v_ep, vmax=+v_ep, aspect='equal',
                             interpolation='gaussian')
            ax2[r, 2].imshow(dr, origin='lower', cmap='seismic',
                             vmin=-v_r, vmax=+v_r, aspect='equal',
                             interpolation='gaussian')

            if r == 0:
                ax2[r, 0].set_title(r'$\Delta_{\rm emp} = \bar\omega^1_{\rm coarse} - \bar\omega^K_{\rm fine}$',
                                    fontsize=10)
                ax2[r, 1].set_title(r'$\Delta_{\rm pred} = \frac{\Delta T^3}{12}(1 - 1/K^2)\,E(\bar\omega^0)$',
                                    fontsize=10)
                ax2[r, 2].set_title(r'$\Delta_{\rm emp} - \Delta_{\rm pred}$', fontsize=10)

            ax2[r, 0].set_ylabel(
                rf'Start $t = {fld["t_seed"]:.2f}$' '\n'
                rf'$|\Delta_{{\rm emp}}|_2 = $ {np.sqrt((de**2).mean()):.2e}',
                fontsize=9,
            )
            for ax in ax2[r]:
                ax.set_xticks([]); ax.set_yticks([])

        fig2.suptitle(
            rf'Test C Diagnostic: Pointwise Spatial Comparison at $\Delta T = {diag_dT:.1e}$ '
            rf'($K = {diag_K}$, $h_{{\rm fine}} = {h_fine:.1e}$)'
            '\n'
            r'(Decaying Turbulence, $N = 256^2$, $L_x = L_y = 2\pi$, Batch #0)',
            fontsize=11,
        )
        fig2.tight_layout()
        out_path2 = out_dir / 'step3_test_c_pointwise_decay.png'
        fig2.savefig(out_path2, dpi=160, bbox_inches='tight')
        plt.close(fig2)
        print(f"  wrote {out_path2}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--sweep-root', type=Path, required=True,
                   help='root containing dt_1em5/DNS_FR_omega.npy etc.')
    p.add_argument('--out-dir', type=Path, default=Path.cwd(),
                   help='where to write step3_test_*.png')
    p.add_argument('--n-snapshots', type=int, default=5,
                   help='number of snapshot triplets to test in TEST B')
    p.add_argument('--ref-subdir', type=str, default='dt_1em5',
                   help='which dt subdir to use as the FD reference '
                        '(default dt_1em5; use dt_2em5 etc. if 1em5 not done)')
    p.add_argument('--n-starts', type=int, default=3,
                   help='number of starting snapshots for TEST C')
    p.add_argument('--K-values', type=int, nargs='+',
                   default=[10, 50, 100, 500, 1000],
                   help='ratios K = Delta T / h_fine for TEST C. '
                        'For each K, the test compares 1 coarse AB2CN2 step '
                        'of size Delta T = K * h_fine vs K fine AB2CN2 steps '
                        'of size h_fine, both starting from a (omega^{-1}, '
                        'omega^0) stencil generated by ultra-fine RK4.')
    p.add_argument('--h-fine', type=float, default=1.0e-5,
                   help='fine AB2CN2 step size for TEST C (default 1e-5)')
    p.add_argument('--h-ultrafine', type=float, default=5.0e-6,
                   help='ultra-fine RK4 step size used to generate the '
                        '(omega^{-1}, omega^0) stencil for TEST C '
                        '(default 5e-6, twice as fine as h_fine)')
    p.add_argument('--skip-test-a', action='store_true',
                   help='skip the single-mode sanity check')
    p.add_argument('--skip-test-b', action='store_true',
                   help='skip the 2D in-the-wild check (e.g. when no FR data ready)')
    p.add_argument('--skip-test-c', action='store_true',
                   help='skip the truncation-error check (most expensive test)')
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_test_a:
        test_a_single_mode(args.out_dir)
    if not args.skip_test_b:
        test_b_2d_in_the_wild(args.sweep_root, args.out_dir,
                              n_snapshots=args.n_snapshots,
                              ref_subdir=args.ref_subdir)
    if not args.skip_test_c:
        test_c_truncation_error(args.sweep_root, args.out_dir,
                                n_starts=args.n_starts,
                                K_values=tuple(args.K_values),
                                h_fine=args.h_fine,
                                h_ultrafine=args.h_ultrafine,
                                ref_subdir=args.ref_subdir)


if __name__ == '__main__':
    main()