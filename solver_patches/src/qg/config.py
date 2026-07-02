"""Hydra-native configuration for QG solver using dataclasses."""

import math
from dataclasses import dataclass, field
from typing import Optional, List, Union
from hydra.core.config_store import ConfigStore


@dataclass
class GridConfig:
    """Grid configuration."""
    Nx: int
    Ny: int
    Lx: float
    Ly: float
    precision: str
    device: str


@dataclass
class TimeConfig:
    """Time stepping configuration."""
    dt: float
    T: int
    save_rate: int


@dataclass
class PDEConfig:
    """PDE physics parameters."""
    mu: float
    nu: float
    B: float
    nv: float
    penalty: float
    friction: Optional[float]
    rossby_radius: Optional[float]
    closure_function: Optional[str]
    closure: float
    width: Optional[float]


@dataclass
class ICConfig:
    """Initial conditions configuration."""
    function: str
    energy: float
    wavenumbers: List[float]
    seed: int
    n_batch: int


@dataclass
class BCConfig:
    """Boundary conditions configuration."""
    function: str
    inlet_velocity: float
    width: float
    _min: float
    _max: float
    sponge: float


@dataclass
class ForcingConfig:
    """Forcing configuration."""
    function: str
    A: float
    B: float
    C: float
    D: float
    E: float
    F: float


@dataclass
class MaskConfig:
    """Mask configuration for obstacles."""
    function: str
    r: float
    x_center: float
    y_center: float
    tol: float


@dataclass
class QGConfig:
    """Main QG solver configuration."""
    seed: int
    grid: GridConfig
    time: TimeConfig
    pde: PDEConfig
    ic: ICConfig
    bc: BCConfig
    forcing: Optional[ForcingConfig]
    mask: Optional[MaskConfig]
    fps: int
    profile: bool


def register_configs():
    """Register configs with Hydra ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(name="qg_base", node=QGConfig)
    cs.store(group="scenario", name="base", node=QGConfig)


# Legacy validation system for compatibility
class validate():
    """Validates and resolves config to functions."""
    def __init__(self, param):
        from omegaconf import OmegaConf
        
        self.runner = 'run.py'
        self.fps = param.fps
        self.profile = param.profile
        
        # Set IC seed from global seed before converting to dict
        ic_config = param.ic
        if hasattr(param, 'seed'):
            OmegaConf.set_struct(ic_config, False)
            ic_config.seed = param.seed
            OmegaConf.set_struct(ic_config, True)
        
        self.time = param.time
        self.pde = param.pde
        self.grid = OmegaConf.to_container(param.grid, resolve=True)
        self.ic = OmegaConf.to_container(ic_config, resolve=True)
        self.bc = OmegaConf.to_container(param.bc, resolve=True) if param.bc is not None else None
        self.forcing = OmegaConf.to_container(param.forcing, resolve=True) if param.forcing is not None else None
        self.mask = OmegaConf.to_container(param.mask, resolve=True) if param.mask is not None else None

    def solve(self):
        # imports here to avoid import on load
        from qg._input.sources.ic import solve_ic
        from qg._input.sources.bc import solve_bc
        from qg._input.sources.forcing import solve_forcing
        from qg._input.mask.mask import solve_mask
        
        self.ic = solve_ic(self.ic)
        self.bc = solve_bc(self.bc)
        self.mask = solve_mask(self.mask)
        self.forcing = solve_forcing(self.forcing)
        
        return self
