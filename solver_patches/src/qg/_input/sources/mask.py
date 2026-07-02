import torch
import numpy as np
from PIL import Image
import os


def add_margin(pil_img, width, height, top, left, color):
    result = Image.new(pil_img.mode, (width, height), color)
    result.paste(pil_img, (left, top))
    return result

def circular(grid, derivative, # add state as first argument if time-dependent
             r, tolerance=1e-3,
             **kwargs):
    # Use grid object for domain size and number of grid points
    Lx = grid.Lx
    Ly = grid.Ly
    Nx = grid.Nx
    Ny = grid.Ny

    # Create a grid of coordinates (x, y)
    x = torch.linspace(0, Lx, Nx, device = grid.device)
    y = torch.linspace(0, Ly, Ny, device = grid.device)

    # Find the center of the domain
    x_center = Lx / 2
    y_center = Ly / 2

    # Compute the distance of each point from the center
    distance = torch.sqrt((x[None,:] - x_center)**2 + (y[:,None] - y_center)**2) # yx

    # Create the mask: inside the circle (distance < r) is 1
    mask = torch.zeros_like(distance)
    mask[distance < r] = 1  # Inside the circle
    mask[torch.abs(distance - r) < tolerance] = 0.5  # Boundary (within tolerance)
    
    return mask[None,:,:]  # Add batch dimension


def fpc(grid, derivative, # add state as first argument if time-dependent
             tolerance=1e-3,
             **kwargs):
    # Use grid object for domain size and number of grid points
    Lx = grid.Lx
    Ly = grid.Ly
    Nx = grid.Nx
    Ny = grid.Ny

    # Create a grid of coordinates (x, y)
    x = torch.linspace(0, Lx, Nx, device = grid.device)
    y = torch.linspace(0, Ly, Ny, device = grid.device)

    # Find the center of the domain
    x_center = Lx / 8
    y_center = Ly / 2
    r = Lx / 16  # Radius of the circle

    # Compute the distance of each point from the center
    distance = torch.sqrt((x[None,:] - x_center)**2 + (y[:,None] - y_center)**2) # yx

    # Create the mask: inside the circle (distance < r) is 1
    mask = torch.zeros_like(distance)
    mask[distance < r] = 1  # Inside the circle
    mask[torch.abs(distance - r) < tolerance] = 0.5  # Boundary (within tolerance)
    
    return mask[None,:,:]  # Add batch dimension

####################################################################################################

valid_mask = lambda _mask: hasattr(_mask, 'function') and _mask.function in mask_library

mask_library = {
    'fpc': fpc,
    'circular': circular,
}        

def solve_mask(_mask):
    if _mask is None or not isinstance(_mask, dict) or 'function' not in _mask or _mask['function'] not in mask_library:
        return _mask
    return lambda *args: mask_library[_mask['function']](*args, **_mask)

