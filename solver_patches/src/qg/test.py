"""
QG Dataset Generation Tests using Hydra Multirun

Examples:
    # Generate single scenario
    python -m qg.test scenario=decaying_turbulence
    
    # Sweep over multiple scenarios
    python -m qg.test --multirun scenario=decaying_turbulence,flow_past_cylinder,cape_high_re
    
    # Sweep over grid sizes
    python -m qg.test --multirun qg.grid.Nx=128,256,512 qg.grid.Ny=128,256,512
    
    # Sweep over viscosity values
    python -m qg.test --multirun qg.pde.nu=1e-5,5e-5,1e-4
    
    # Combine sweeps
    python -m qg.test --multirun scenario=decaying_turbulence qg.grid.Nx=128,256 qg.ic.n_batch=10,20
"""

import hydra
from omegaconf import DictConfig
from qg.train import main as train_main

# Reuse the main training function for tests
# Hydra handles all the sweep logic

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Test entrypoint that reuses train.main with test-specific settings."""
    # You can add test-specific overrides here if needed
    train_main(cfg)


if __name__ == "__main__":
    main()
