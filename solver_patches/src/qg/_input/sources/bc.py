import torch
import torch.nn.functional as F
from qg.solver.opt.basis import to_physical, to_spectral

from functools import lru_cache

class Sponge:
    @staticmethod
    def vorticity_sponge(state, derivative, sponge, mask, uh, vh):
        if sponge <= 0:
            print("Warning: sponge must be positive for sponge layer to work effectively.")
            return 0
        
        _ramp = mask
        _eta = sponge * state.dt
        
        ### vorticity / open bc
        outlet_vorticity_sponge = -1 * to_spectral(_ramp * to_physical(state.qh)) / _eta
        return outlet_vorticity_sponge
    
    @staticmethod
    def diffuse_sponge(state, derivative, sponge, masks, uh, vh):
        if sponge <= 0:
            print("Warning: sponge must be positive for sponge layer to work effectively.")
            return 0
        
        _outlet_v1, _outlet_v2, _outlet_v1_ramp = masks
        _eta = sponge * state.dt
        
        ### velocity / closed bc
        masked_vh_delta = to_spectral(_outlet_v2 * to_physical(state.vh - vh)) 
        masked_uh_delta = to_spectral(_outlet_v2 * to_physical(state.uh - uh))
        outlet_velocity_sponge = (derivative.dx * masked_vh_delta - derivative.dy * masked_uh_delta) / _eta
        
        ### diffusion
        outlet_diffusion = 0.1 * derivative.laplacian * to_spectral(_outlet_v1_ramp * to_physical(state.qh))
            
        return outlet_diffusion + outlet_velocity_sponge # outlet_vorticity_sponge + 
    

class Flow:
    @staticmethod
    def const_x_flow(state, inlet_velocity):
        
        @lru_cache(maxsize=1)
        def base_flow():
            flow_uh = torch.zeros_like(state.uh)
            flow_uh[...,0,0] = inlet_velocity
            flow_vh = flow_uh * 0.0
            return flow_uh, flow_vh
        
        flow_uh, flow_vh = base_flow()
        state.uh[...,0,0] = inlet_velocity
        state.vh[...,0,0] = 0.0
        
        return flow_uh, flow_vh

class Region:
    
    @staticmethod
    def vertical_outlet_single_mask(grid, X, Y, _min, _max, x, width):
        _mask_1 = ((X > x) * (X < x + width) \
            * (Y >= _min) * (Y <= _max))[None,:,:].to(grid.ftype)
        
        _mask_ramp = _mask_1 \
            * (X - x) / width
            
        return _mask_ramp


    @staticmethod
    def vertical_outlet_double_mask(grid, X, Y, _min, _max, x, width):
        _mask_1 = ((X > x) * (X < x + width) \
            * (Y >= _min) * (Y <= _max))[None,:,:].to(grid.ftype)
        _mask_2 = ((X > x + width) * (X < x + 2 * width) \
            * (Y >= _min) * (Y <= _max))[None,:,:].to(grid.ftype)
        
        _mask_ramp = _mask_1 \
            * (X - x) / width
            
        return _mask_1, _mask_2, _mask_ramp
    
    @staticmethod
    def vertical_bidirectional_triple_mask(grid, X, Y, _min, _max, x, width):
        _mask_1 = ((X > x) * (X < x + width) \
            * (Y >= _min) * (Y <= _max))[None,:,:].to(grid.ftype)
        _mask_2 = ((X > x + width) * (X < x + 2 * width) \
            * (Y >= _min) * (Y <= _max))[None,:,:].to(grid.ftype)
        _mask_3 = ((X > x + 2 * width) * (X < x + 3 * width) \
            * (Y >= _min) * (Y <= _max))[None,:,:].to(grid.ftype) 
        _mask_ramp = _mask_1 * (X - x) / width \
            + _mask_3 * ((x + 3 * width) - X) / width
            
        return _mask_1, _mask_2, _mask_ramp
    
    
    @staticmethod
    def horizontal_bidirectional_double_mask(grid, X, Y, _min, _max, y, width):
        _mask_1 = ((Y > y) * (Y < y + width) \
            * (X >= _min) * (X <= _max))[None,:,:].to(grid.ftype)
        _mask_2 = ((Y > y + width) * (Y < y + 2 * width) \
            * (X >= _min) * (X <= _max))[None,:,:].to(grid.ftype)
        
        _mask_ramp = \
            _mask_1  * (Y - y) / width + \
            _mask_2 * ((y + 2 * width) - Y) / width
            
        return _mask_ramp
    
    @staticmethod
    def horizontal_bidirectional_triple_mask(grid, X, Y, _min, _max, y, width):
        _mask_1 = ((Y > y) * (Y < y + width) \
            * (X >= _min) * (X <= _max))[None,:,:].to(grid.ftype)
        _mask_2 = ((Y > y + width) * (Y < y + 2 * width) \
            * (X >= _min) * (X <= _max))[None,:,:].to(grid.ftype)
        _mask_3 = ((Y > y + 2 * width) * (Y < y + 3 * width) \
            * (X >= _min) * (X <= _max))[None,:,:].to(grid.ftype)
        
        
        _mask_ramp = \
            _mask_1  * (Y - y) / width + \
            _mask_3 * ((y + 3 * width) - Y) / width
            
        return _mask_1 + _mask_3, _mask_2, _mask_ramp
    
    @staticmethod
    @lru_cache(maxsize=1)
    def outlet_mask_r(grid, _min, _max, width, type='single'): # right outlet mask
        x = torch.linspace(0, 1, grid.Nx,device=grid.device)
        y = torch.linspace(0, 1, grid.Ny,device=grid.device)
        X, Y = x[None,:],y[:,None]
        
        match type:
            case 'single':
                return Region.vertical_outlet_single_mask(grid, X,  Y, _min, _max, 1 - width, width)
            case 'double':
                return Region.vertical_outlet_double_mask(grid, X,  Y, _min, _max, 1 - 2 * width, width)
    
    @staticmethod
    @lru_cache(maxsize=1)
    def outlet_mask_rtd(grid, _min, _max, width, type='single'):
        
        x = torch.linspace(0, 1, grid.Nx,device=grid.device)
        y = torch.linspace(0, 1, grid.Ny,device=grid.device)
        X, Y = x[None,:],y[:,None]
            
        match type:
            case 'single':
                _outlet_ramp = Region.vertical_outlet_single_mask(grid, X,  Y, _min, _max, 1 - width, width)
                _birect_ramp = Region.horizontal_bidirectional_double_mask(grid, X, Y, _min, _max, 1 - 2 * width, width) # just above lower image boundary
                return (_outlet_ramp + _birect_ramp).clamp(0.0, 1.0)
            
            case 'double': 
                _outlet_1, _outlet_2, _outlet_ramp = Region.vertical_outlet_double_mask(grid, X,  Y, _min, _max, 1 - 2 * width, width)
                _bidirect_1, _bidirect_2, _bidirect_ramp = Region.horizontal_bidirectional_triple_mask(grid, X, Y, _min, _max, 1 - 3*width, width) # just above lower image boundary
                return (_outlet_1 + _bidirect_1, _outlet_2 + _bidirect_2, (_outlet_ramp + _bidirect_ramp).clamp(0.0, 1.0))
            
            case 'triple': 
                _outlet_1, _outlet_2, _outlet_ramp = Region.vertical_bidirectional_triple_mask(grid, X,  Y, _min, _max, 1 - 3 * width, width)
                _bidirect_1, _bidirect_2, _bidirect_ramp = Region.horizontal_bidirectional_triple_mask(grid, X, Y, _min, _max, 1 - 3*width, width) # just above lower image boundary
                return (_outlet_1 + _bidirect_1, _outlet_2 + _bidirect_2, (_outlet_ramp + _bidirect_ramp).clamp(0.0, 1.0))

### 
class BC:
    @staticmethod
    def const_outlet_vorticity_r(state, grid, derivative, 
                        inlet_velocity=1.0, _min=0.0, _max=1.0, sponge=4.0,
                        width = 0.05,                 
                        **kwargs):
        return Sponge.vorticity_sponge(state, derivative, sponge,
                                    Region.outlet_mask_r(grid, _min, _max, width, type='single'),
                                    *Flow.const_x_flow(state, inlet_velocity))
        

    @staticmethod
    def const_outlet_vorticity_rtd(state, grid, derivative, 
                        inlet_velocity=1.0, _min=0.0, _max=1.0, sponge=4.0,
                        width = 0.05,                 
                        **kwargs):
        return Sponge.vorticity_sponge(state, derivative, sponge,
                                       Region.outlet_mask_rtd(grid, _min, _max, width, type='single'),
                                       *Flow.const_x_flow(state, inlet_velocity))
    @staticmethod
    def const_outlet_diffuse_r(state, grid, derivative, 
                        inlet_velocity=1.0, _min=0.0, _max=1.0, sponge=4.0,
                        width = 0.05,                 
                        **kwargs):
        return Sponge.diffuse_sponge(state, derivative, sponge,
                                    Region.outlet_mask_r(grid, _min, _max, width, type='double'),
                                    *Flow.const_x_flow(state, inlet_velocity))
        

    @staticmethod
    def const_outlet_diffuse_rtd(state, grid, derivative, 
                        inlet_velocity=1.0, _min=0.0, _max=1.0, sponge=4.0,
                        width = 0.05,                 
                        **kwargs):
        return Sponge.diffuse_sponge(state, derivative, sponge,
                                       Region.outlet_mask_rtd(grid, _min, _max, width, type='triple'),
                                       *Flow.const_x_flow(state, inlet_velocity))

    @staticmethod
    def none(state, grid, derivative,                  
                        **kwargs):
        return 0

####################################################################################################

valid_bc = lambda _bc: hasattr(_bc, 'function') and _bc.function in bc_library

bc_library = {
    'periodic': BC.none,
    'const-outlet-vorticity-r': BC.const_outlet_vorticity_r,
    'const-outlet-vorticity-rtd': BC.const_outlet_vorticity_rtd,
    'const-outlet-diffuse-r': BC.const_outlet_diffuse_r,
    'const-outlet-diffuse-rtd': BC.const_outlet_diffuse_rtd,
}

def solve_bc(_bc):
    if _bc is None or not isinstance(_bc, dict) or 'function' not in _bc or _bc['function'] not in bc_library:
        return _bc
    return lambda *args: bc_library[_bc['function']](*args, **_bc)