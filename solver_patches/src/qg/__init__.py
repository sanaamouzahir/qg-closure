from qg.config import QGConfig
from qg.solver.qg import QG

# from mura import data_run

# def autorun(_config, **kwargs):
#     """Legacy autorun function."""
#     with data_run(_config) as save_path:
#         qg = QG(_config)
#         qg.solve(save_path=save_path, **kwargs)

# expose basic config from hydra


def direct_solver(config_overrides: dict = None) -> QG:
    """Initialize QG solver with basic config.
    
    Args:
        config_overrides: Optional dictionary to override default config values.
                         Uses dot notation for nested values, e.g. {'qg.grid.Nx': 256}
    
    Returns:
        QG solver instance ready to run.
    
    Example:
        >>> qg = direct_solver({'qg.grid.Nx': 256, 'qg.grid.Ny': 256})
        >>> data = qg.solve(save_path='./output')
    """
    from hydra import initialize, compose, GlobalHydra
    from omegaconf import OmegaConf
    from qg.config import register_configs
    import os
    from pathlib import Path
    
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()
    
    # Register configs
    register_configs()
    
    # Get absolute path to config directory
    config_dir = Path(__file__).parent / "conf"
    
    # Initialize Hydra with config path
    with initialize(version_base="1.3", config_path=str(config_dir)):
        # Compose config with optional overrides
        overrides = []
        if config_overrides:
            for key, value in config_overrides.items():
                overrides.append(f"{key}={value}")
        
        cfg = compose(config_name="config", overrides=overrides)
        
        # Debug: print resolved config
        print(f"Grid config: Nx={cfg.qg.grid.Nx}, Ny={cfg.qg.grid.Ny}, device={cfg.qg.grid.device}")
        
        # Initialize and return QG solver
        return QG(cfg.qg)

