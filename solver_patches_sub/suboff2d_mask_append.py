

def suboff2d(grid, derivative,  # add state as first argument if time-dependent
             x_nose=0.1, y_center=0.5, hull_height=1.2566,
             tolerance=1e-3, **kwargs):
    """DARPA SUBOFF AFF-1 bare-hull silhouette (DTRC model 5470; Groves,
    Huang & Chang 1989) as a 2D Brinkman mask (Sanaa order 2026-07-23).

    Model coordinates (feet): L = 14.291667, rmax = 0.8333333 (L/D = 8.575).
    Segments: bow ellipse-family 0..3.333333; parallel midbody ..10.645833;
    6th-order polynomial afterbody ..13.979167 (rh = 0.1175, k0 = 10,
    k1 = 44.6244 -- coefficients from the boundary conditions; the polynomial
    sums to exactly 1 at the midbody junction and rh^2 at the tail); closing
    cap ..14.291667. x_nose/y_center are FRACTIONS of Lx/Ly; hull_height is
    the physical max height 2*r_max (default 1.2566 = cylinder D so
    Re_D = 3900 at the production commons U=2, nu=6.4443e-4; hull length
    = 8.575*D = 10.777).
    """
    Lx = grid.Lx
    Ly = grid.Ly
    Nx = grid.Nx
    Ny = grid.Ny
    x = torch.linspace(0, Lx, Nx, device=grid.device)
    y = torch.linspace(0, Ly, Ny, device=grid.device)
    LM, RM = 14.291667, 0.8333333
    rh, k0, k1 = 0.1175, 10.0, 44.6244
    cb1, cb2 = 1.126395101, 0.442874707
    D = hull_height
    L_hull = D / (2.0 * RM) * LM                     # physical hull length
    X = (x - x_nose * Lx) * (LM / L_hull)            # model-feet coordinate
    r = torch.zeros_like(X)
    m = (X >= 0.0) & (X <= 3.333333)                 # bow
    xb = X[m]
    a = 0.3 * xb - 1.0
    b = 1.2 * xb + 1.0
    arg = cb1 * xb * a ** 4 + cb2 * xb ** 2 * a ** 3 + 1.0 - a ** 4 * b
    r[m] = RM * arg.clamp(min=0.0) ** (1.0 / 2.1)
    m = (X > 3.333333) & (X <= 10.645833)            # parallel midbody
    r[m] = RM
    m = (X > 10.645833) & (X <= 13.979167)           # afterbody
    xi = (13.979167 - X[m]) / 3.333333
    poly = (rh ** 2 + rh * k0 * xi ** 2
            + (20.0 - 20.0 * rh ** 2 - 4.0 * rh * k0 - k1 / 3.0) * xi ** 3
            + (-45.0 + 45.0 * rh ** 2 + 6.0 * rh * k0 + k1) * xi ** 4
            + (36.0 - 36.0 * rh ** 2 - 4.0 * rh * k0 - k1) * xi ** 5
            + (-10.0 + 10.0 * rh ** 2 + rh * k0 + k1 / 3.0) * xi ** 6)
    r[m] = RM * poly.clamp(min=0.0) ** 0.5
    m = (X > 13.979167) & (X <= 14.291667)           # closing cap
    argc = 1.0 - (3.2 * X[m] - 44.733333) ** 2
    r[m] = 0.1175 * RM * argc.clamp(min=0.0) ** 0.5
    r_phys = r * ((D / 2.0) / RM)                    # (Nx,) physical half-height
    dy = (y[:, None] - y_center * Ly).abs()          # (Ny,1)
    rrow = r_phys[None, :]                           # (1,Nx)
    mask = torch.zeros(Ny, Nx, device=grid.device)
    mask[dy < rrow] = 1
    mask[(dy - rrow).abs() < tolerance] = 0.5
    return mask[None, :, :]


mask_library["suboff2d"] = suboff2d
