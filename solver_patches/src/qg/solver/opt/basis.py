import torch
import numpy as np

@staticmethod
def puv(qh, derivative):
    ph = derivative.inv_laplacian * qh
    uh = -1 * derivative.dy * ph
    vh = derivative.dx * ph
    return ph, uh, vh

class _state:
    def __init__(self, qh, dt, derivative):
        self.qh = qh
        self.t = 0.0
        self.dt = dt
        self.derivative = derivative
        
        if self.qh is not None:
            self.update_uv()
        
    def update_uv(self):
        self.ph, self.uh, self.vh = puv(self.qh, self.derivative)
        
    # def update_potential_flow(self):    
    #     self.uh = self.uh + self.uh_p + self.x_adv * self.dt
    #     self.vh = self.vh + self.vh_p + self.y_adv * self.dt
        
    # def update_qp(self):
    #     self.qh = - 1j * self.derivative.kr * self.vh + 1j * self.derivative.ky * self.uh
    #     self.ph = self.qh * self.derivative.inv_laplacian
        
    #     uh_w = 1j * self.derivative.ky * self.ph
    #     vh_w = -1j * self.derivative.kr * self.ph
        
    #     self.uh_p = self.uh - uh_w
    #     self.vh_p = self.vh - vh_w
        
    def update_t(self):
        self.t += self.dt
        
    def update_uvt(self):
        self.update_uv()
        self.update_t()
        
    def _out(self):
        return to_physical(self.qh)  
        
    def out(self, cdim=1):
        return torch.stack(
            [to_physical(self.qh),
            to_physical(self.ph),
            to_physical(self.uh),
            # to_physical(self.uh_p),
            to_physical(self.vh)],
            dim=cdim, # assume batched
        )

def to_physical(spectral_field):
    """
    Convert a spectral field to physical space (inverse FFT).
    """
    return torch.fft.irfftn(spectral_field,dim=(-2, -1),norm='forward')

def to_spectral(physical_field):
    """
    Convert a physical field to spectral space (FFT).
    """
    return torch.fft.rfftn(physical_field,dim=(-2, -1),norm='forward')