r"""
Absolute stability regions of the explicit Adams-Bashforth methods AB1..AB4 and
of classical RK4, in the complex plane $z=\lambda\Delta t$.

Linear-multistep (AB) methods are encoded by two polynomials
    rho(zeta)   = sum_j alpha_j zeta^j     (y-history)
    sigma(zeta) = sum_j beta_j  zeta^j     (f-history; explicit => leading coeff 0)
A value z is absolutely stable iff every root zeta of the stability polynomial
    pi(zeta;z) = rho(zeta) - z*sigma(zeta)
satisfies |zeta| <= 1.  RK4 is one-step with amplification
    R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24,   stable iff |R(z)| <= 1.

For advection lambda is ~imaginary, so what matters is the imaginary-axis
stability boundary (ISB).  This script measures the real- and imaginary-axis
intercepts and draws the regions (RK4's is vastly larger -- the point of Step 3).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Method definitions.  AB: (rho, sigma) high->low powers; sigma leads with 0.
# --------------------------------------------------------------------------- #
AB = {
    1: ([1, -1.],            [0, 1.]),
    2: ([1, -1, 0.],         [0, 3/2, -1/2]),
    3: ([1, -1, 0, 0.],      [0, 23/12, -4/3, 5/12]),
    4: ([1, -1, 0, 0, 0.],   [0, 55/24, -59/24, 37/24, -3/8]),
}
def R_rk4(z):
    return 1 + z + z**2/2 + z**3/6 + z**4/24      # RK4 amplification factor

# --------------------------------------------------------------------------- #
# Stability predicates (vectorised over a flat complex array Z).
# --------------------------------------------------------------------------- #
def stable_AB(Z, rho, sigma, tol=1e-9):
    rho = np.asarray(rho); sigma = np.asarray(sigma)
    C = rho[None, :] - Z[:, None] * sigma[None, :]   # (N, p+1) coeffs, monic (c0=1)
    p = C.shape[1] - 1
    comp = np.zeros((len(Z), p, p), complex)
    comp[:, 0, :] = -C[:, 1:]                        # top row
    idx = np.arange(p - 1)
    comp[:, idx + 1, idx] = 1.0                      # subdiagonal
    rad = np.max(np.abs(np.linalg.eigvals(comp)), axis=1)
    return rad <= 1 + tol

def stable_RK4(Z, tol=1e-9):
    return np.abs(R_rk4(Z)) <= 1 + tol

def axis_intercept(predicate, direction, hi, N=200000):
    """Largest t along `direction` (from 0) that stays stable."""
    ts = np.linspace(1e-5, hi, N)
    ok = predicate(ts * direction)
    bad = np.where(~ok)[0]
    return ts[bad[0] - 1] if len(bad) else hi

# --------------------------------------------------------------------------- #
# 1) Axis intercepts (negative-real = dissipation, imaginary = advection/ISB).
# --------------------------------------------------------------------------- #
print(f"{'method':6} {'neg-real |z|':>13} {'imag |z| (ISB)':>16}")
for p in (1, 2, 3, 4):
    pred = lambda Z, p=p: stable_AB(Z, *AB[p])
    re = axis_intercept(pred, -1.0, hi=2.5)
    im = axis_intercept(pred,  1j,  hi=2.5)
    print(f" AB{p}   {re:>13.4f} {im:>16.4f}")
re4 = axis_intercept(stable_RK4, -1.0, hi=4.0)
im4 = axis_intercept(stable_RK4,  1j,  hi=4.0)
print(f" RK4   {re4:>13.4f} {im4:>16.4f}   (imag = 2*sqrt(2) = {2*np.sqrt(2):.4f})")

# --------------------------------------------------------------------------- #
# 2) Draw the regions as filled masks on a grid; inset zooms on the AB lobes.
# --------------------------------------------------------------------------- #
def region_mask(predicate, xr, yr, n=700):
    x = np.linspace(*xr, n); y = np.linspace(*yr, n)
    X, Y = np.meshgrid(x, y); Z = (X + 1j*Y).ravel()
    return x, y, predicate(Z).reshape(n, n)

methods = [("AB2", lambda Z: stable_AB(Z, *AB[2]), "C0"),
           ("AB3", lambda Z: stable_AB(Z, *AB[3]), "C3"),
           ("AB4", lambda Z: stable_AB(Z, *AB[4]), "C2"),
           ("RK4", stable_RK4,                      "k")]

XR, YR = (-3.1, 0.9), (-3.2, 3.2)
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 6.2))

# -- left: full view, RK4 dominates ------------------------------------------ #
for name, pred, c in methods:
    x, y, m = region_mask(pred, XR, YR, n=700)
    axL.contour(x, y, m.astype(float), levels=[0.5], colors=c, linewidths=2)
    axL.plot([], [], color=c, lw=2, label=rf"$\mathrm{{{name}}}$")   # legend proxy
axL.axhline(0, color="0.5", lw=.6); axL.axvline(0, color="0.5", lw=.6, ls="--")
axL.plot(0, 2*np.sqrt(2), "k.", ms=7); axL.plot(0, -2*np.sqrt(2), "k.", ms=7)
axL.annotate(r"$\mathrm{RK4}$ ISB $=2\sqrt{2}\approx2.83$", (-1.05, 2.95), fontsize=10)
axL.set_xlim(*XR); axL.set_ylim(*YR); axL.set_aspect("equal")
axL.set_xlabel(r"$\mathrm{Re}\,(\lambda\,\Delta t)$", fontsize=12)
axL.set_ylabel(r"$\mathrm{Im}\,(\lambda\,\Delta t)$", fontsize=12)
axL.set_title(r"$\mathrm{AB}p$ explicit vs $\mathrm{RK4}$ "
              r"(stable region $|\cdot|\leq1$)", fontsize=12)
axL.legend(loc="lower left", fontsize=11); axL.grid(alpha=.25)

# -- right: zoom on the AB lobes --------------------------------------------- #
for name, pred, c in methods[:3]:
    x, y, m = region_mask(pred, (-1.35, 0.6), (-1.0, 1.0), n=520)
    axR.contour(x, y, m.astype(float), levels=[0.5], colors=c, linewidths=2)
    axR.plot([], [], color=c, lw=2, label=rf"$\mathrm{{{name}}}$")
axR.axhline(0, color="0.5", lw=.6); axR.axvline(0, color="0.5", lw=.6, ls="--")
axR.plot([0, 0, 0], [0.7236, 0.4300, 0.0079], "o", ms=5,
         color="none", mec="k")
axR.annotate(r"$\mathrm{AB3}\!:\,0.72\,i$", (0.05, 0.70), fontsize=9, color="C3")
axR.annotate(r"$\mathrm{AB4}\!:\,0.43\,i$", (0.05, 0.40), fontsize=9, color="C2")
axR.annotate(r"$\mathrm{AB2}\!:\,0$ (tangent)", (-0.92, 0.06), fontsize=9, color="C0")
axR.set_xlim(-1.35, 0.6); axR.set_ylim(-1.0, 1.0); axR.set_aspect("equal")
axR.set_xlabel(r"$\mathrm{Re}\,(\lambda\,\Delta t)$", fontsize=12)
axR.set_ylabel(r"$\mathrm{Im}\,(\lambda\,\Delta t)$", fontsize=12)
axR.set_title(r"zoom: $\mathrm{AB}p$ imaginary-axis reach", fontsize=12)
axR.legend(loc="lower left", fontsize=10); axR.grid(alpha=.25)

fig.suptitle(r"Absolute stability regions in $z=\lambda\,\Delta t$", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("ab_stability_regions.png", dpi=140)
print("\nsaved ab_stability_regions.png")
