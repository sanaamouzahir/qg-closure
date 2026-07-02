import math   # add at top of mask.py if not already imported


def submarine(grid, derivative,  # add state as first argument if time-dependent
              x_center=0.25, y_center=0.5,
              length=None, radius=None, fineness=5.0,
              nose_exp=0.5, tail_exp=1.0,
              angle=0.0, tolerance=1e-3,
              **kwargs):
    """
    Idealized submarine: a slender streamlined hull suspended in the channel,
    aligned by default with the +x inflow.  Rounded blunt nose upstream, finely
    tapered tail downstream -- the fore/aft asymmetry that makes a streamlined
    body's wake (thin, attached, weak shedding) qualitatively different from a
    bluff cylinder's von Karman street.

    Geometry (body frame, before any rotation):
        axial coordinate   xi = X'/a   in [-1, 1]      (a = half-length)
        half-width         W(xi) = b * p(xi) / p_max
        p(xi) = (1 - xi)**tail_exp * (1 + xi)**nose_exp
      Nose tip at xi = -1 (upstream), tail tip at xi = +1 (downstream).
      Max thickness sits forward of center at
        xi* = (nose_exp - tail_exp) / (nose_exp + tail_exp).
      nose_exp < tail_exp  ->  blunt nose, fine tail (submarine-like).
      nose_exp = tail_exp  ->  fore/aft symmetric (prolate-ellipse-like).

    Parameters
    ----------
    x_center, y_center : float
        Hull center as fractions of (Lx, Ly).  Default places it upstream and
        mid-channel (like `fpc`) so a wake can develop downstream.
    length : float or None
        Full hull length in physical units.  Default Lx / 4.
    radius : float or None
        Max hull half-width b.  If None, set from `fineness`: b = a / fineness.
    fineness : float
        Length-to-diameter ratio L / (2b).  Real submarines are ~6-8; ~5 keeps
        the idealized body visibly thick.  Ignored if `radius` is given.
    nose_exp, tail_exp : float
        Profile exponents controlling nose bluntness and tail fineness.
    angle : float
        Angle of attack in radians (hull rotated CCW from +x).  Default 0.
    tolerance : float
        Soft-boundary (mask = 0.5) thickness in physical units.

    Returns
    -------
    mask : torch.Tensor of shape (1, Ny, Nx) with values in {0, 0.5, 1}.
    """
    Lx = grid.Lx
    Ly = grid.Ly
    Nx = grid.Nx
    Ny = grid.Ny

    x = torch.linspace(0, Lx, Nx, device=grid.device)
    y = torch.linspace(0, Ly, Ny, device=grid.device)

    x_c = x_center * Lx
    y_c = y_center * Ly
    L = (Lx / 4.0) if length is None else length
    a = 0.5 * L                                  # half-length
    b = (a / fineness) if radius is None else radius

    # Coordinates relative to hull center, then rotate into the body frame
    X = x[None, :] - x_c                          # (1, Nx)
    Y = y[:, None] - y_c                          # (Ny, 1)
    if angle != 0.0:
        ca, sa = math.cos(angle), math.sin(angle)
        Xp = ca * X + sa * Y                       # broadcasts to (Ny, Nx)
        Yp = -sa * X + ca * Y
    else:
        Xp = X + 0.0 * Y                          # broadcast to (Ny, Nx)
        Yp = Y + 0.0 * X

    xi = Xp / a                                   # axial coordinate (Ny, Nx)

    # Streamlining profile p(xi) on [-1, 1], normalized so max half-width = b.
    eps = 1e-12
    one_minus = torch.clamp(1.0 - xi, min=0.0)
    one_plus = torch.clamp(1.0 + xi, min=0.0)
    prof = one_minus.pow(tail_exp) * one_plus.pow(nose_exp)
    xi_star = (nose_exp - tail_exp) / (nose_exp + tail_exp + eps)
    p_max = (1.0 - xi_star) ** tail_exp * (1.0 + xi_star) ** nose_exp
    W = b * prof / (p_max + eps)                  # half-width at each xi (Ny, Nx)

    on_body_axially = xi.abs() < 1.0
    gap = W - Yp.abs()                            # >0 inside, =0 on the surface

    mask = torch.zeros((Ny, Nx), device=grid.device, dtype=grid.ftype)
    inside = on_body_axially & (gap > tolerance)
    boundary = on_body_axially & (gap.abs() <= tolerance) & (W > tolerance)
    mask[inside] = 1.0
    mask[boundary] = 0.5

    return mask[None, :, :]


# Add to the registry:
#   mask_library = {
#       'fpc': fpc,
#       'circular': circular,
#       'cape': cape,
#       'submarine': submarine,
#   }
