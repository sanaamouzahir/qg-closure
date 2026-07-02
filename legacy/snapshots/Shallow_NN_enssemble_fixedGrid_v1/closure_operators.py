"""
closure_operators.py

Scheme-specific assembly of the temporal-closure truncation operators R_p from
the *scheme-agnostic* N-derivative channels produced by build_training_data_mmap.py.

Design
------
The training targets are the pure chain-rule N-derivatives N^(1)..N^(5) at omega_0.
They do NOT depend on the time integrator. The dependence on the integrator
(AB2CN2 vs AB4CN2 vs ...) lives ENTIRELY in the coefficients of the truncation
operators R_p. So one dataset feeds every scheme; you pick the scheme here.

Each per-step LTE has the form

    tau = - sum_{p>=3} (h^p / D_p^{scheme}) R_p^{scheme} + O(h^{p_max+1}),

and each operator expands in a single fixed basis,

    R_p = a_{p,0} L^p omega
        + sum_{k=1}^{p} a_{p,k} L^{p-k} N^{(k-1)},      N^{(0)} := N.

The k=0,1 terms (L^p omega, L^{p-1} N) are the ANALYTICAL part: Fourier-diagonal
multiplies of fields already available at inference (omega, and N via one Jacobian).
The k>=2 terms are the LEARNED part: powers of L on the predicted N-derivatives.

Coefficients (verified symbolically; see LTE_RK4_AB4CN2_AB2CN2):

  AB2CN2   D = {3:12, 4:24, 5:240, 6:1440}
    R3 = L^3 w + L^2 N +  L Ndot  - 5 Nddot
    R4 = 2L^4 w + 2L^3 N + 2L^2 Ndot - 4 L Nddot +    Ndddot
    R5 = 13L^5 w + 13L^4 N + 13L^3 Ndot - 17 L^2 Nddot + 8 L Ndddot - 7 N4
    R6 = 43L^6 w + 43L^5 N + 43L^4 Ndot - 47 L^3 Nddot + 28 L^2 Ndddot - 17 L N4 + 4 N5

  AB4CN2   D = {3:12, 4:24, 5:720}     (note: R5 denominator is 720, not 240)
    R3 = L^3 w + L^2 N + L Ndot                       (no Nddot)
    R4 = 2L^4 w + 2L^3 N + 2L^2 Ndot + L Nddot         (no Ndddot)
    R5 = 39L^5 w + 39L^4 N + 39L^3 Ndot + 24 L^2 Nddot + 9 L Ndddot - 251 N4

The single structural difference AB2->AB4 is the explicit-N polynomial fit:
R3^(AB4) drops the -5 Nddot piece and R4^(AB4) trades -4 L Nddot + Ndddot for
+ L Nddot. Everything containing only L on (omega, N) is identical (same CN part).

Channel convention
------------------
`Nderivs` is an ordered sequence with Nderivs[m-1] = N^(m):
    Nderivs[0] = Ndot   (N^(1))
    Nderivs[1] = Nddot  (N^(2))
    Nderivs[2] = Ndddot (N^(3))
    Nderivs[3] = N4     (N^(4))
    Nderivs[4] = N5     (N^(5))
matching TARGET_FIELDS = [N_dot, N_ddot, N_3dot, N_4dot, N_5dot].

All fields are physical-space torch tensors; L is applied spectrally via L_hat
(the same Fourier-diagonal symbol used in the builder/solver).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch

from qg.solver.opt.basis import to_spectral, to_physical


# --------------------------------------------------------------------------- #
# Coefficient tables.  coeffs[p] has length p+1: index k -> a_{p,k}.           #
#   k=0 : L^p omega                                                            #
#   k>=1: L^{p-k} N^{(k-1)}   (N^{(0)} = N)                                     #
# --------------------------------------------------------------------------- #

SCHEMES: Dict[str, Dict] = {
    'ab2cn2': {
        'denom':  {3: 12, 4: 24, 5: 240, 6: 1440},
        'coeffs': {
            3: [1, 1, 1, -5],
            4: [2, 2, 2, -4, 1],
            5: [13, 13, 13, -17, 8, -7],
            6: [43, 43, 43, -47, 28, -17, 4],
        },
    },
    'ab4cn2': {
        'denom':  {3: 12, 4: 24, 5: 720},
        'coeffs': {
            3: [1, 1, 1, 0],
            4: [2, 2, 2, 1, 0],
            5: [39, 39, 39, 24, 9, -251],
        },
    },
}


def available_orders(scheme: str) -> List[int]:
    return sorted(SCHEMES[scheme]['coeffs'].keys())


def n_nn_channels(orders: Sequence[int]) -> int:
    """How many N-derivative channels (N^(1)..N^(p_max-1)) the orders require."""
    return max(orders) - 1


def _Lk(field_phys: torch.Tensor, L_hat: torch.Tensor, k: int) -> torch.Tensor:
    """Apply L^k to a physical field via the Fourier-diagonal symbol."""
    if k == 0:
        return field_phys
    return to_physical((L_hat ** k) * to_spectral(field_phys))


# --------------------------------------------------------------------------- #
# Operator assembly                                                           #
# --------------------------------------------------------------------------- #

def assemble_Rp_split(scheme: str, p: int,
                      omega: torch.Tensor, N: torch.Tensor,
                      Nderivs: Sequence[torch.Tensor],
                      L_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (R_p^anal, R_p^nn) for the given scheme and order.

    R_p^anal = a_{p,0} L^p omega + a_{p,1} L^{p-1} N          (free at inference)
    R_p^nn   = sum_{k>=2} a_{p,k} L^{p-k} N^{(k-1)}            (uses predictions)
    """
    coeffs = SCHEMES[scheme]['coeffs'][p]
    need = p - 1                       # highest N-derivative order used = N^(p-1)
    if len(Nderivs) < need:
        raise ValueError(f"{scheme} R{p} needs N^(1)..N^({need}); "
                         f"got {len(Nderivs)} channels")

    # analytical part: k = 0, 1
    R_anal = coeffs[0] * _Lk(omega, L_hat, p)
    if coeffs[1] != 0:
        R_anal = R_anal + coeffs[1] * _Lk(N, L_hat, p - 1)

    # learned part: k = 2..p  ->  L^{p-k} N^{(k-1)}
    R_nn = torch.zeros_like(omega)
    for k in range(2, p + 1):
        c = coeffs[k]
        if c == 0:
            continue
        Nkm1 = Nderivs[k - 2]          # N^{(k-1)}
        R_nn = R_nn + c * _Lk(Nkm1, L_hat, p - k)
    return R_anal, R_nn


def assemble_Rp(scheme: str, p: int, omega, N, Nderivs, L_hat) -> torch.Tensor:
    Ra, Rn = assemble_Rp_split(scheme, p, omega, N, Nderivs, L_hat)
    return Ra + Rn


def closure_increment(scheme: str,
                      omega: torch.Tensor, N: torch.Tensor,
                      Nderivs: Sequence[torch.Tensor],
                      L_hat: torch.Tensor, dt: float,
                      orders: Sequence[int] = (3,)
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The per-step closure correction delta to ADD to a bare step:

        delta = - sum_{p in orders} (dt^p / D_p) R_p

    Returns (delta_anal, delta_nn, delta_total). The split lets the deployment
    fold delta_anal into the Fourier-diagonal IMEX update for free and add only
    delta_nn from the network's predicted N-derivatives.
    """
    denom = SCHEMES[scheme]['denom']
    delta_anal = torch.zeros_like(omega)
    delta_nn = torch.zeros_like(omega)
    for p in orders:
        if p not in denom:
            raise ValueError(f"{scheme} has no R{p} (orders: {available_orders(scheme)})")
        Ra, Rn = assemble_Rp_split(scheme, p, omega, N, Nderivs, L_hat)
        c = -(dt ** p) / denom[p]
        delta_anal = delta_anal + c * Ra
        delta_nn = delta_nn + c * Rn
    return delta_anal, delta_nn, delta_anal + delta_nn


def closure_increment_batched(scheme: str,
                              omega: torch.Tensor, N: torch.Tensor,
                              Nderivs: Sequence[torch.Tensor],
                              L_hat: torch.Tensor, dt: torch.Tensor,
                              orders: Sequence[int] = (3, 4)) -> torch.Tensor:
    """delta = -sum_{p in orders} (dt^p / D_p) R_p, with a PER-SAMPLE dt vector.

    For the Delta_T sweep, dt varies across the batch, so dt is a (B,) tensor and
    dt^p broadcasts over (B,1,1,1). R_p itself is dt-independent (assembled from the
    batched omega/N/Nderivs), so this just reweights the same operators per sample.
    Returns the full R_p increment (anal + nn) summed over `orders` -- e.g.
    orders=(3,4) is the analytic R3,R4 part; the corrector learns R5 and up.
    """
    denom = SCHEMES[scheme]['denom']
    dtb = dt.reshape(-1, *([1] * (omega.dim() - 1))).to(omega.dtype)   # (B,1,1,1)
    delta = torch.zeros_like(omega)
    for p in orders:
        if p not in denom:
            raise ValueError(f"{scheme} has no R{p} (orders: {available_orders(scheme)})")
        Ra, Rn = assemble_Rp_split(scheme, p, omega, N, Nderivs, L_hat)
        delta = delta + (-(dtb ** p) / denom[p]) * (Ra + Rn)
    return delta


# --------------------------------------------------------------------------- #
# Self-check: verify assembled coefficients reproduce the raw LTE on the       #
# linear test equation (every R_p collapses to powers of lambda).             #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    import sympy as sp

    h, L, Nsym = sp.symbols('h L N')      # scalar surrogates for the operator basis
    w = sp.symbols('w')
    # symbolic N-derivatives N^(1)..N^(5)
    nd = sp.symbols('n1 n2 n3 n4 n5')

    def Lk(x, k):                          # scalar L^k
        return L**k * x

    def Rp_sym(scheme, p):
        c = SCHEMES[scheme]['coeffs'][p]
        R = c[0] * Lk(w, p) + c[1] * Lk(Nsym, p - 1)
        for k in range(2, p + 1):
            if c[k] != 0:
                R += c[k] * Lk(nd[k - 2], p - k)
        return R

    # Reference raw LTE coefficients (from sympy derivation), as -h^p/D_p R_p:
    #   AB2  h^3: -L^3 w/12 - L^2 N/12 - L n1/12 + 5 n2/12
    #   AB4  h^3: -L^3 w/12 - L^2 N/12 - L n1/12
    print("AB2CN2 tau (assembled):")
    tau2 = sum(-(h**p)/SCHEMES['ab2cn2']['denom'][p] * Rp_sym('ab2cn2', p)
               for p in (3, 4, 5, 6))
    for p in (3, 4, 5, 6):
        print(f"  h^{p}:", sp.expand(tau2.coeff(h, p)))

    print("\nAB4CN2 tau (assembled):")
    tau4 = sum(-(h**p)/SCHEMES['ab4cn2']['denom'][p] * Rp_sym('ab4cn2', p)
               for p in (3, 4, 5))
    for p in (3, 4, 5):
        print(f"  h^{p}:", sp.expand(tau4.coeff(h, p)))

    print("\nAB2 - AB4 difference (h^3, h^4):")
    d = sp.expand(tau2 - tau4)
    for p in (3, 4):
        print(f"  h^{p}:", sp.expand(d.coeff(h, p)))
    print("  expected h^3: 5*n2/12 ; h^4: 5*L*n2/24 - n3/24")
