from qg.solver.util import _Math

from qg.solver.opt.operator.jacobian import jacobian_pq
from qg.solver.opt.operator.obstacle import solve_mask, brinkman_no_slip_penalty, brinkman_friction_slip_penalty
# from qg.solver.opt.operator.vortex import vortex_stretching

def define_explicit_operator(param, grid, derivative, logger, args, sources, **kwargs):
    patches = []
    
    if param.bc is not None:
        patches.append(lambda op, state: param.bc(state, grid, derivative))
    
    if param.forcing is not None:
        logger.info(f"Forced turbulence")
        patches.append(lambda op, state: param.forcing(state, grid, derivative))
        
    if param.pde.penalty > 0:
        
        # if param.pde.friction is not None:
        #     logger.info("Using Brinkman penalty (friction-slip) operator")
        #     brinkman_penalty = brinkman_friction_slip_penalty
        # else:
        logger.info("Using Brinkman penalty (no-slip) operator")
        brinkman_penalty = brinkman_no_slip_penalty
        
        mask = solve_mask(param.mask, grid, derivative)
        patches.append(lambda op, state: brinkman_penalty(op, state, *mask(op, state)))
        
    # if param.pde.rossby_radius is not None:
    #     logger.info("Using vortex stretching operator")
    #     patches.append(vortex_stretching)

    patches.extend(sources)
            
    return Operator(*args, patch_list=patches)
        
class Operator:
    def __init__(self, dt, grid, derivative, params, patch_list = []):        
        self.dt = dt
        self.grid = grid
        self.derivative = derivative
        self.params = params
        self.device = grid.device

        self.patch_list = [*patch_list,jacobian_pq]
        
    def source(self, state):
        return self.derivative.dealias(sum([f(self, state) for f in self.patch_list]))
      
    def __repr__(self):
        return (f"Operator: device={self.device}")  

## special implicit linear operator (since this is the only one for now)

class ImplicitLinearOperator(_Math):
    def __init__(self, grid, derivative, params):
        self.derivative = derivative
        self.params = params
        self.device = grid.device
        super().__init__(value = self._linear_term()) # precompute

    def _linear_term(self):
        nu = self.params.nu
        mu = self.params.mu
        B = self.params.B
        
        # first term is diffusion: nu del^2 omega
        # then bottom drag: - mu omega
        # then Coriolis with beta term: - beta d psi/ dx (where omega = del^2 psi)
        
        return nu * self.derivative.laplacian - mu - B * self.derivative.dx * self.derivative.inv_laplacian

    def __repr__(self):
        return (f"ImplictLinearOperator(nu={self.params.nu}, mu={self.params.mu},"
                f"B={self.params.B}, device={self.device})")
