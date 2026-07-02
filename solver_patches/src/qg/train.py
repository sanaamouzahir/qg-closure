"""Hydra-driven QG dataset generation with WandB tracking.

This script generates QG turbulence datasets with proper experiment tracking:
- WandB logging with full config
- Git tracking with automatic diff saving
- Version management
- Artifact uploading
"""

import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

# QG configuration
from qg.config import register_configs, QGConfig

# Use Mura's Hydra utilities
from mura.hydra import (
    register_resolvers,
    compute_git_info,
    EnhancedGitCallback,
)

# Register configs and resolvers
register_configs()
register_resolvers()


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Generate QG dataset with tracking."""
    
    # Get output directory from Hydra
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    
    # Initialize WandB if enabled
    if cfg.wandb.mode != "disabled":
        # Compute git info
        git_info = compute_git_info()
        
        # Create run name
        run_name = f"qg-{cfg.qg.grid.Nx}x{cfg.qg.grid.Ny}-nu{cfg.qg.pde.nu:.0e}"
        
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.get('entity'),
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.mode,
            tags=cfg.wandb.get('tags', []) + ['qg', 'dataset-generation'],
        )
        
        # Log git info
        wandb.config.update({
            'git_sha': git_info['sha'],
            'git_dirty': git_info['dirty'],
        })
        
        # Save git diff if dirty
        if git_info['dirty'] and git_info['diff_text']:
            diff_path = output_dir / 'git_diff.patch'
            diff_path.write_text(git_info['diff_text'])
            wandb.save(str(diff_path))
        
        print(f"✅ WandB initialized: {wandb_run.url}")
    else:
        wandb_run = None
    
    # Save full config
    config_path = output_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    
    # Initialize solver with Hydra config directly
    from qg.solver.qg import QG
    solver = QG(cfg.qg)
    
    # Run simulation
    print(f"Running QG simulation (T={cfg.qg.time.T}, dt={cfg.qg.time.dt})...")
    data = solver.solve(
        save_path=str(output_dir),
        name='qg_data',
        clamp=cfg.get('clamp', 0.3)
    )
    
    print(f"✅ QG simulation complete: {data.shape}")
    print(f"   Output saved to: {output_dir}")
    
    # Log to WandB
    if wandb_run:
        # Log summary statistics
        import torch
        wandb.log({
            'dataset/num_samples': data.shape[0],
            'dataset/num_timesteps': data.shape[1],
            'dataset/channels': data.shape[2],
            'dataset/height': data.shape[3],
            'dataset/width': data.shape[4],
            'dataset/mean': float(data.mean()),
            'dataset/std': float(data.std()),
            'dataset/min': float(data.min()),
            'dataset/max': float(data.max()),
        })
        
        # Upload artifacts
        artifact = wandb.Artifact(
            name=f"qg-dataset-{cfg.qg.grid.Nx}x{cfg.qg.grid.Ny}",
            type="dataset",
            description=f"QG turbulence dataset: {cfg.qg.grid.Nx}x{cfg.qg.grid.Ny}, nu={cfg.qg.pde.nu}",
        )
        # artifact.add_file(str(output_dir / 'qg_data.npy'))
        artifact.add_file(str(config_path))
        
        # Add videos if they exist
        for video in output_dir.glob('*.mp4'):
            artifact.add_file(str(video))
        
        wandb.log_artifact(artifact)
        print(f"✅ Artifacts uploaded to WandB")
        
        wandb.finish()


if __name__ == "__main__":
    main()
