"""
Minimal driver script to run a QG simulation from a scenario YAML,
without the wandb / mura / lightning / git overhead in train.py.

Usage:
    python run_qg.py +scenario=flow_past_cape \
        qg.grid.Nx=128 qg.grid.Ny=128 \
        qg.time.T=0.5 qg.time.save_rate=50

Hydra will manage the output directory under outputs/<date>/<time>/
unless you override it with `hydra.run.dir=...`.
"""

import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from qg.config import register_configs

# Register configs
register_configs()


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run a QG simulation and save outputs."""
    from hydra.core.hydra_config import HydraConfig

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    print(f"Output directory: {output_dir}")

    # Save the resolved config alongside outputs for reference
    with open(output_dir / 'config.yaml', 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Build the solver and run
    from qg.solver.qg import QG
    solver = QG(cfg.qg)

    print(f"Running QG simulation (T={cfg.qg.time.T}, dt={cfg.qg.time.dt})...")
    data = solver.solve(
        save_path=str(output_dir),
        name='DNS',
        clamp=cfg.get('clamp', 0.3),
    )
    print(f"Simulation complete: {data.shape}")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()