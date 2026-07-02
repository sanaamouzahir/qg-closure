import torch
from qg.solver.opt.basis import puv

# CU compatibility workaround
def abs(x):
    if x.is_cuda:
        return torch.sqrt(x.real**2 + x.imag**2)
    else:
        return torch.abs(x)


def int_sq(y):
    Y = torch.sum(abs(y[:, 0])**2) + 2*torch.sum(abs(y[:, 1:])**2)
    return Y

# Generates initial conditions based on specified energy and wavenumber limits
def _init_randn(grid, derivative,
                energy=0.0, wavenumbers=[3.0, 5.0], n_batch = 1,
                seed=86, persistent=True,
                **kwargs):
    
    if persistent and 'ic_' in globals():
        return globals()['ic_'].detach().clone()
    
    torch.manual_seed(seed)
    
    # use grid wavenumbers
    K = torch.sqrt(grid.ksq).repeat(n_batch, 1, 1)  # Wavenumber of each point in frequency space
    k = grid.kx.repeat(n_batch, grid.Ny, 1)         # Ensure proper shape for k

    # random vorticity in spectral space
    qh = torch.randn(k.size(), dtype=torch.complex128).to(grid.device)
    
    # filter wavenumber range with zero mean
    qh[K < wavenumbers[0]] = 0.0
    qh[K > wavenumbers[1]] = 0.0
    qh[k == 0.0] = 0.0
    
    # normalize to specified energy
    ph, uh, vh = puv(qh, derivative)
    E = 0.5 * (int_sq(uh) + int_sq(vh))
    qh *= torch.sqrt(energy / E)
    
    # store the initial condition for persistent use
    if persistent:
        globals()['ic_'] = qh.detach().clone()
    
    # print("Initial condition energy:", E0)
    # print(torch.max((qh).abs()), torch.min((qh).abs()))
    
    return qh

####################################################################################################

valid_ic = lambda _ic: hasattr(_ic, 'function') and _ic.function in ic_library

ic_library = {
    'randn': _init_randn,
}

def solve_ic(_ic):
    if _ic is None or not isinstance(_ic, dict) or 'function' not in _ic or _ic['function'] not in ic_library:
        return _ic
    return lambda *args: ic_library[_ic['function']](*args, **_ic)