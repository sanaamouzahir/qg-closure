import torch
from qg.solver.opt.basis import to_physical, to_spectral

def jacobian_pq(op, state):
    """
    Computes the Jacobian of q (vorticity) and p (streampatch) in spectral space (h).
    """
    q = to_physical(state.qh)
    u = to_physical(state.uh)
    v = to_physical(state.vh)

    uq = u * q # calculate products
    vq = v * q

    uqh = to_spectral(uq)
    vqh = to_spectral(vq)
    
    jacobian = -1 * op.derivative.dx * uqh - op.derivative.dy * vqh # - d/dx(u*q) - d/dy(v*q)
    
    return jacobian
    
def advection_uv(op, state):
    ''' - (u . del) u '''
    
    
    # get potential flow contributions:
    u = to_physical(state.uh_p)
    v = to_physical(state.vh_p)
    u_p = to_physical(state.uh_p)
    v_p = to_physical(state.vh_p)
    
    dudy, dudx = torch.gradient(u_p, dim=(-2,-1), spacing=(op.derivative.dy, op.derivative.dx))
    dvdy, dvdx = torch.gradient(v_p, dim=(-2,-1), spacing=(op.derivative.dy, op.derivative.dx))
    # faster than two ffts?
    
    x_adv = -1 * (u * dudx + v * dudy)
    y_adv = -1 * (u * dvdx + v * dvdy)
    
    x_advh = to_spectral(x_adv)
    y_advh = to_spectral(y_adv)
    
    
    
    ### Not correct; mult becomes convolution!    
    # dudx = 1j * op.derivative.kr * state.uh_p
    # dudy = 1j * op.derivative.ky * state.uh_p
    # dvdx = 1j * op.derivative.kr * state.vh_p
    # dvdy = 1j * op.derivative.ky * state.vh_p
    
    # x_adv = - (state.uh * dudx + state.vh * dudy)
    # y_adv = - (state.uh * dvdx + state.vh * dvdy)
    
    return x_advh, y_advh

    