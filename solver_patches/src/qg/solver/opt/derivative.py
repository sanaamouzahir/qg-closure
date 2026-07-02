import torch
import math

### Set up spectral derivatives (first and second derivatives)
class Derivative:
    def __init__(self, grid):
        self.grid = grid
        self.device = grid.device
        
        # normal model
        self.mkx = grid.kx
        self.mky = grid.ky
        
        # Zeitlin sine?
        # _dx = torch.pi / grid.Nx
        # _dy = torch.pi / grid.Ny
        # self.mkx = torch.sin(grid.kx * _dx) / _dx
        # self.mky = torch.sin(grid.ky * _dy) / _dy
        
        self.dx = 1j * self.mkx
        self.dy = 1j * self.mky

        # laplacian
        self.mksq = self.mkx**2 + self.mky**2  
        self.laplacian = - self.mksq
        self.inv_laplacian = 1.0/self.laplacian
        self.inv_laplacian[:,0,0] = 0.0 # 0/0 = 0 
        
        # dealiasing for stability and mask
        dealias_factor=1/3
        self.k_cut = math.sqrt(2) * (1 - dealias_factor) * min(self.mky.max(), self.mkx.max())
        self.alias_mask = (torch.sqrt(self.mksq) > self.k_cut)
        
    def dealias(self, y):
        """
        Apply dealiasing to the field based on the ratio (usually 1/3 rule).
        The field's high-frequency components are truncated.
        """
        # Apply dealiasing: set high-frequency components to zero
        y[self.alias_mask.expand_as(y)] = 0
        return y

        
    def to(self, device):
        self.device = device
        self.dx = self.dx.to(device)
        self.dy = self.dy.to(device)
        self.laplacian = self.laplacian.to(device)
        self.inv_laplacian = self.inv_laplacian.to(device)
        self.alias_mask = self.alias_mask.to(device)
        return self

    def __repr__(self):
        return (f"Derivative: Nx={self.grid.Nx}, Ny={self.grid.Ny}, dk={self.dk}, "
                f"Lx={self.grid.Lx:.4f}, Ly={self.grid.Ly:.4f}, device={self.device}")

if __name__ == "__main__":
    from qg.solver.grid.cartesian import CartesianGrid
    grid = CartesianGrid(Nx=64, Ny=64)
    derivative = Derivative(grid)
    print(derivative.mkx)