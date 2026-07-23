

def submarine(grid, derivative,  # add state as first argument if time-dependent
              x_center=0.2, y_center=0.5,
              a=2.5133, b=0.6283, tolerance=1e-3,
              **kwargs):
    """Mid-water elongated ellipse hull ('submarine', Sanaa order 2026-07-22).

    x_center / y_center are FRACTIONS of Lx / Ly; a / b are the semi-axes in
    PHYSICAL units. Defaults: hull height 2b = 1.2566 (= the cylinder D, so
    Re_D = U*D/nu = 3900 exactly at the production commons U=2,
    nu=6.4443e-4) and aspect ratio 4:1 (hull length 2a = 5.0266). Mid-water
    (free wake above AND below, unlike the bottom-attached cape).

    Returns mask (1, Ny, Nx) with values in {0, 0.5, 1} (fpc convention).
    """
    Lx = grid.Lx
    Ly = grid.Ly
    Nx = grid.Nx
    Ny = grid.Ny
    x = torch.linspace(0, Lx, Nx, device=grid.device)
    y = torch.linspace(0, Ly, Ny, device=grid.device)
    xc = x_center * Lx
    yc = y_center * Ly
    rho = torch.sqrt(((x[None, :] - xc) / a) ** 2
                     + ((y[:, None] - yc) / b) ** 2)   # yx
    mask = torch.zeros_like(rho)
    mask[rho < 1.0] = 1
    mask[torch.abs(rho - 1.0) < tolerance] = 0.5
    return mask[None, :, :]
