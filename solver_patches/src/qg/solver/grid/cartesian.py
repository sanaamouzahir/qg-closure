import torch
import math


class CartesianGrid:
    def __init__(self, Lx=2*math.pi, Ly=2*math.pi, Nx=512, Ny=512, device=None, precision='float32', **kwargs):
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.precision = precision
        self.ftype = {
            'float32': torch.float32,
            'float64': torch.float64
        }[precision]
        self.ctype = {
            'float32': torch.complex64,
            'float64': torch.complex128
        }[precision]
        torch.set_default_dtype(self.ftype)


        self.size=self.Nx*self.Ny
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.x = torch.arange(-self.Lx/2, self.Lx/2, self.dx, device=self.device)
        self.y = torch.arange(-self.Ly/2, self.Ly/2, self.dy, device=self.device)
        
        # number of wavenumber components (half of real grid in x-direction)
        self.dk = int(self.Nx / 2 + 1)

        # pure wavenumbers
        self.ky = torch.reshape((torch.fft.fftfreq(self.Ny, self.Ly / (self.Ny * 2 * torch.pi))), 
            (self.Ny, 1)
        )[None,:,:] 
        
        self.kx = torch.reshape((torch.fft.rfftfreq(self.Nx, self.Lx / (self.Nx * 2 * torch.pi))), 
            (1, self.dk)
        )[None,:,:]
        
        self.ksq = self.kx**2 + self.ky**2
        
    def to(self, device):
        """ Move grid tensors to another device. """
        self.device = device
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        
        # wavenumbers
        self.kx = self.kx.to(device)
        self.ky = self.ky.to(device)
        self.ksq = self.ksq.to(device)

    def __repr__(self):
        # print(grid)
        return (f"Grid(Lx={self.Lx}, Ly={self.Ly}, Nx={self.Nx}, Ny={self.Ny}, "
                f"dx={self.dx:.4f}, dy={self.dy:.4f}, device={self.device})")