import torch
import torch.nn.functional as F
import inspect
from qg.solver.opt.basis import to_physical, to_spectral

    
def solve_mask(mask, grid, derivative):
    signature = inspect.signature(mask)
    num_params = len(signature.parameters)
    print(f"Mask function {mask.__name__} has {num_params} parameters.")
    
    kernel = 1/16 * torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]]).to(grid.device)[None,None,...]

    if num_params == 3:
        def _mask(op, state):
            basic_mask, mask_vel = mask(state, op.grid, op.derivative)
            chi = F.conv2d(basic_mask[None,:,:,:], kernel, padding='same')[0] # B y x
            vel = F.conv2d(mask_vel[None,:,:,:], kernel, padding='same')[0] # B 2 y x
            return chi, vel
    else:
        mask = mask(grid, derivative)
        chi = F.conv2d(mask[None,:,:,:], kernel, padding='same')[0] # B y x
        vel = torch.zeros([2,]).to(grid.device)
        def _mask(op, state):
            return chi, vel
        
    return _mask    

#####


def compute_normal_vectors(mask):
    # Compute gradients using central differences
    grad_x = torch.gradient(mask, dim=-1) [0]
    grad_y = torch.gradient(mask, dim=-2) [0]
    
    # Compute the magnitude of the gradient
    grad_magnitude = (grad_x**2 + grad_y**2) ** 0.5
    
    # Normalize the gradient to get the unit normal vector
    normal_x = grad_x / (grad_magnitude)  # Add small epsilon to avoid division by zero
    normal_y = grad_y / (grad_magnitude)
    
    # Set normal vectors to zero inside the solid region (where mask == 1)
    normal_x[~torch.isfinite(normal_x)] = 0
    normal_y[~torch.isfinite(normal_y)] = 0
    
    return normal_x, normal_y

#####
    
    
def brinkman_no_slip_penalty(op, state, chi, chi_velocity):
    """
    Computes the brinkman volume penalization for mask (chi) in spectral space (h).
    """
    eta = op.params.penalty * op.dt
    
    u = to_physical(state.uh) # convert to physical space
    v = to_physical(state.vh)

    u_chi = chi * (u - chi_velocity[0]) # products to be damped
    v_chi = chi * (v - chi_velocity[1])

    u_chi_h = to_spectral(u_chi)
    v_chi_h = to_spectral(v_chi)
    
    sponge = (- op.derivative.dx * v_chi_h + op.derivative.dy * u_chi_h) / eta # - d/dx(chi*v) + d/dy(chi*u)
    
    return sponge

def brinkman_friction_slip_penalty(op, state, chi, chi_velocity):
    """
    Computes the brinkman volume penalization for mask (chi) in spectral space (h).
    """
    eta = op.params.penalty * op.dt
    friction = op.params.friction

    u = to_physical(state.uh) # convert to physical space
    v = to_physical(state.vh)
    
    du = (u - chi_velocity[0])
    dv = (v - chi_velocity[1])
    
    normal_x, normal_y = compute_normal_vectors(chi)
    ndot = du * normal_x + dv * normal_y
    dun = ndot * normal_x
    dvn = ndot * normal_y
    dut = (du - dun)
    dvt = (dv - dvn)
    dutr = dut * (1 - friction)
    dvtr = dvt * (1 - friction)

    u_chi = chi * (dun + dut * friction) # products to be damped
    v_chi = chi * (dvn + dvt * friction)

    u_chi_h = to_spectral(u_chi)
    v_chi_h = to_spectral(v_chi)
    
    sponge = (-1 * op.derivative.dx * v_chi_h + op.derivative.dy * u_chi_h) / eta # - d/dx(chi*v) + d/dy(chi*u)
    
    
    # modify flow field inside obstacle
    
    # get closest point along normal
    x = torch.arange(0, chi.shape[-1], device=chi.device)
    y = torch.arange(0, chi.shape[-2], device=chi.device)
    xc = (torch.round(normal_x) + x).to(torch.int32)
    yc = (torch.round(normal_y) + y).to(torch.int32)
    print(xc.shape)
    uc = dutr[...,yc,xc]
    vc = dvtr[...,yc,xc]
    print(yc.shape)
    
    u_corr = u * (1 - chi) + chi * uc
    v_corr = v * (1 - chi) + chi * vc
    
    state.uh = to_spectral(u_corr)
    state.vh = to_spectral(v_corr)
    
    return sponge

# def inlet_outlet(op, state,
#                     inlet_velocity=1.0, mn=0.4, mx=0.6, eta=2.0,
#                     inlet_x = 0.12,                 
#                     **kwargs):
    
#     x = torch.linspace(0, 1, op.grid.Nx,device=op.grid.device)
#     y = torch.linspace(0, 1, op.grid.Ny,device=op.grid.device)
#     X, Y = x[None,:],y[:,None]
    
    
#     ### regions
#     total_outlet_width = inlet_x
#     outlet_width = total_outlet_width / 3
#     outlet_xw = outlet_width / 2
#     outlet_xv = outlet_width * 3 / 2
#     outlet_xu = outlet_width * 5 / 2
    
    
#     _outlet_w = ((torch.abs(X - outlet_xw) < outlet_width) \
#         * (Y >= mn) * (Y <= mx))[None,:,:].to(op.grid.ftype)
#     _outlet_v = ((torch.abs(X - outlet_xv) < outlet_width) \
#         * (Y >= mn) * (Y <= mx))[None,:,:].to(op.grid.ftype)
#     _outlet_ramp_v = torch.nn.functional.relu((outlet_width - torch.abs(X - outlet_xv))/outlet_width)
#     _outlet_u = ((torch.abs(X - outlet_xu) < outlet_width) \
#         * (Y >= mn) * (Y <= mx))[None,:,:].to(op.grid.ftype)
#     _outlet_ramp_u = torch.nn.functional.relu((outlet_width - torch.abs(X - outlet_xu))/outlet_width)

#     _eta = 2 * eta * state.dt
    
    
            
#     ### inlet
#     _inlet = _outlet_u
#     _inlet_ramp = _outlet_ramp_u
#     vx = - inlet_velocity    
    
#     u = to_physical(state.uh)
#     u += (vx-u)  * _inlet
#     state.uh = to_spectral(u)
#     state.update_qp()
    
#     ### vorticity / open bc
#     outlet_vorticity_sponge = to_spectral(_outlet_w * to_physical(1j * op.derivative.kr * state.vh - 1j * op.derivative.ky * state.uh)) / _eta
    
#     ### y-velocity / straighten
#     sv = to_physical(state.vh)
#     dv = _outlet_v * _outlet_ramp_v * (sv)
#     v_chi_h = to_spectral(dv)   
#     outlet_y_velocity_smooth_sponge = (1j * op.derivative.kr * v_chi_h) / _eta

#     # su = to_physical(state.uh)
#     # du = _inlet * (su - vx)
#     # u_chi_h = to_spectral(du)
#     # outlet_x_velocity_sponge = (- 1j * derivative.ky * u_chi_h) / _eta
    

#     outlet_sponge = outlet_vorticity_sponge \
#         + outlet_y_velocity_smooth_sponge \
#         # + outlet_x_velocity_sponge   
#     return dealias(outlet_sponge, op.derivative, 1/3)