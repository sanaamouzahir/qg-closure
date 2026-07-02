import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from qg.solver.opt.basis import _state, to_spectral, to_physical
from qg.solver.integrator.imex import CN2, AB2

import qg.config as vc

from qg.solver.grid.cartesian import CartesianGrid
from qg.solver.opt.derivative import Derivative
from qg.solver.opt.operator import ImplicitLinearOperator, define_explicit_operator

from qg.solver.opt.operator.jacobian import advection_uv

import os
import logging
import jpcm.draw as draw

from omegaconf import DictConfig, OmegaConf

class QG():
    def __init__(self, param,
                 grid = CartesianGrid,
                 derivative = Derivative,
                 implicit_linear_operator = ImplicitLinearOperator,
                 explicit_sources = [],
                 logger = logging.getLogger(__name__)):
        
        # DictConfig already supports attribute access, so just use it directly
        # No need to convert - validate() expects object with attributes
        
        param = vc.validate(param).solve()
        self.param = param
        self.logger = logger
        self.logger.addHandler(logging.StreamHandler())
        
        self.grid = grid(**param.grid)
        self.derivative = derivative(self.grid).to(self.grid.device)
        self.implicit_linear_operator = implicit_linear_operator(self.grid, self.derivative, param.pde)
        self.operator = define_explicit_operator(param, self.grid, self.derivative, self.logger,
                                        args=(param.time.dt, self.grid, self.derivative, param.pde),
                                        sources=explicit_sources) 
        
        self.dt = param.time.dt
        self.logger.info(f"Initialized QG model with {self.grid.Nx}x{self.grid.Ny} grid on {self.grid.device}")

    def step(self, state):
        # state.dt = self.dt # Not sure if this is necessary, need to think about adaptive time stepping TODO

        # vorticity step
        explicit_source = AB2(self.operator.source(state)) # source term
        state.qh = CN2(state.qh, explicit_source, state.dt, self.implicit_linear_operator) # Crank-Nicolson        
        
        # potential flow velocity step
        # state.x_adv, state.y_adv = advection_uv(self.operator, state)
        
        # print(torch.max(to_physical(explicit_source)), torch.min(to_physical(explicit_source)))
        # print(torch.max(to_physical(state.qh)), torch.min(to_physical(state.qh)))
        
        # update fields
        state.update_uv()
        # state.update_potential_flow() # also potential_flow
        state.update_t()

    def init(self):  
        return _state(self.param.ic(self.grid, self.derivative), self.dt, self.derivative) # In spectral space
          
    def _run(self, prof=None):
        save_rate = self.param.time.save_rate
        steps = int(self.param.time.T / self.dt)  # Number of time steps
        
        state = self.init()
        
        # print(torch.max(to_physical(state.qh)), torch.min(to_physical(state.qh)))
        
        B = state.qh.shape[0]  # Number of batches
        solution = torch.zeros([B, int(steps/save_rate)+1, 4, self.grid.Ny, self.grid.Nx])
        
        for it in tqdm(range(steps - 1)):
            self.step(state)            
            
            if (it+1) % save_rate == 0:
                save_index = (it + 1) // save_rate
                solution[:, save_index, ...] = state.out() # B T C H W
            
            if prof is not None:
                prof.step()  # Step the profiler
            
                
        return solution
    
    def solve(self, save_path, name='DNS', clamp=0.3): # for direct user call
        if hasattr(self.param, 'profile') and self.param.profile:
            self.logger.info(f"Profiling enabled.")
            from torch.profiler import profile, ProfilerActivity, record_function
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],record_shapes=True, with_stack=True) as prof:
                with record_function("_run"):
                    solution_torch = self._run(prof)            
            print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
            prof.export_chrome_trace(os.path.join(save_path, f'{name}_trace.json'))
            self.logger.info(f"Profile trace saved at {os.path.join(save_path, f'{name}_trace.json')}")
        else:
            solution_torch = self._run()
        solution = solution_torch.cpu().numpy()
        self.logger.info(f"Simulation complete.")
            
        
        np.save(os.path.join(save_path,f'{name}.npy'), solution)
        self.logger.info(f"Simulation saved at {save_path}")
        
        # select a couple batches for visualization (permute 0,1 axes)
        solution_b = np.transpose(solution[0:4,:,:1,...],(1,0,2,3,4))  # T (selected_B) C H W

        draw.mp4(os.path.join(save_path,f'{name}.mp4'), solution_b,
                   fps=self.param.fps, triplet=True)
        draw.mp4(os.path.join(save_path,f'{name}_clamped.mp4'), solution_b,
                   fps=self.param.fps, triplet=True, clamp=clamp)
        draw.mp4(os.path.join(save_path,f'{name}_seismic.mp4'), solution_b,
                   fps=self.param.fps, triplet=True, cmap='seismic', clamp=clamp)  
        
        # draw.mp4(os.path.join(save_path,'DNS.mp4'), solution_b,
        #            fps=20, triplet=False, mn = [4,1])
        # draw.mp4(os.path.join(save_path,'DNS_clamped.mp4'), solution_b,
        #            fps=20, triplet=False, mn = [4,1], clamp=0.3)
        # draw.mp4(os.path.join(save_path,'DNS_seismic.mp4'), solution_b,
        #            fps=20, triplet=False, mn = [4,1], cmap='seismic', clamp=0.3)    
        
        # # make streamlines from the vorticity field
        # draw.streamlines(os.path.join(save_path,f'{name}_streamlines.mp4'), solution_b[:,:,1,...], -solution_b[:,:,2,...], 
        #                  fps=self.param.fps) # u, -v
        
        
        
        
           
        self.logger.info(f"Videos saved.")
        
        return solution_torch

    def nn_step(self, u, dt=None):
        if dt is None:
            dt = self.dt
        qh = to_spectral(u) # assumes B H W, vorticity only
        state = _state(qh, dt, self.derivative) # In spectral space
        self.step(state)
        return state._out()[:,None,None,...]  # B H W