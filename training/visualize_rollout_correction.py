"""
visualize_rollout_correction.py - show the closure's effect at the field level.

Per sample, two rows of 5 panels:

  Row 1 (NO closure):
    [omega_truth] [omega_coarse] [truth - coarse]   [E(k)]   [Z(k)]
  Row 2 (WITH closure):
    [omega_truth] [omega_coarse + e_anal + e_NN_pred] [truth - corrected]
                                                      [E(k)]   [Z(k)]

Where the E(k) panel overlays:  E_truth, E_field-of-row, E_error
And the Z(k) panel overlays:    Z_truth, Z_field-of-row, Z_error
With:
    E(k) = (1/2) * k^-2 * |omega_hat(k)|^2     (kinetic energy spectrum)
    Z(k) = (1/2) * |omega_hat(k)|^2            (enstrophy spectrum)

Inputs are read directly from the saved npz (no need to know which subset of
channels the trained model used; we always need the 6 v2 fields for the
network input + omega_K_fine, omega_1_coarse, e_anal_incr for the figure).

Usage:
    python visualize_rollout_correction.py \
        --run-dir /path/to/training_runs/<run_name> \
        --root-dir /path/to/decaying_turbulence_dT_1em3_fixD_v2_OLD_f32bug \
        --n-samples 3 \
        --split val \
        --device cpu

Outputs:  <run-dir>/rollout_correction_visualization.png   (or --out-path)
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Importable from $QG_DIR/training/
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path('.').resolve()))


# --------------------------------------------------------------------------- #
# Operators                                                                   #
# --------------------------------------------------------------------------- #

def build_L_hat(Nx: int, Ny: int, Lx: float, Ly: float,
                nu: float, mu: float, beta: float) -> np.ndarray:
    """Spectral linear operator L_hat = nu*lap - mu - B*dx*inv_lap."""
    nx_freq = np.fft.fftfreq(Nx, d=Lx / Nx) * 2.0 * np.pi
    ny_freq = np.fft.fftfreq(Ny, d=Ly / Ny) * 2.0 * np.pi
    KX, KY = np.meshgrid(nx_freq, ny_freq, indexing='xy')
    K2 = KX ** 2 + KY ** 2
    K2_safe = np.where(K2 == 0, 1.0, K2)
    L_hat = -nu * K2 - mu
    L_hat = L_hat + 1j * beta * KX / K2_safe
    L_hat[0, 0] = -mu
    return L_hat


def apply_closure_correction(f_NN_pred_phys: np.ndarray, Delta_T: float,
                             K: int) -> np.ndarray:
    """Convert NN's bracket prediction into the additive coarse-step increment.

    Under Fix D's target convention,
        f_NN_target = (1/12) [L*Ndot - 5*Nddot]   (coarse-fine sign)
    so the increment to ADD to the coarse step is
        e_NN_incr = -Delta_T^3 * (1 - 1/K^2) * f_NN_target.
    """
    return -1.0 * (Delta_T ** 3) * (1.0 - 1.0 / (K ** 2)) * f_NN_pred_phys


def radial_average(field2d: np.ndarray, weights2d: np.ndarray = None):
    """Azimuthal-average a (Ny, Nx) array of |hat|^2 over radial bins of k.

    Returns (k_centers, radial_psd).
    """
    Ny, Nx = field2d.shape
    fhat = np.fft.fft2(field2d) / (Nx * Ny)
    psd = np.abs(fhat) ** 2  # shape (Ny, Nx)
    if weights2d is not None:
        psd = psd * weights2d
    kx = np.fft.fftfreq(Nx, d=1.0 / Nx)
    ky = np.fft.fftfreq(Ny, d=1.0 / Ny)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    Kmag = np.sqrt(KX ** 2 + KY ** 2)
    kmax = int(np.floor(Kmag.max()))
    bins = np.arange(0, kmax + 2)
    sp, _ = np.histogram(Kmag.ravel(), bins=bins, weights=psd.ravel())
    cnt, _ = np.histogram(Kmag.ravel(), bins=bins)
    sp_avg = np.where(cnt > 0, sp / np.maximum(cnt, 1), 0.0)
    kc = 0.5 * (bins[:-1] + bins[1:])
    return kc, sp_avg


def energy_spectrum(omega_phys: np.ndarray, Lx: float, Ly: float):
    """Kinetic energy spectrum:  E(k) = 0.5 * |hat psi|^2 * k^2  =  0.5 * k^-2 * |hat omega|^2

    With omega = lap psi, |hat omega|^2 = k^4 |hat psi|^2, so
        E(k) = 0.5 k^2 |hat psi|^2 = 0.5 k^{-2} |hat omega|^2
    Done by computing |hat omega|^2 azimuthally then dividing by k^2.
    """
    Ny, Nx = omega_phys.shape
    kc, sp = radial_average(omega_phys)  # = <|hat omega|^2>(k)
    # protect k=0
    k_safe = np.where(kc <= 0, np.inf, kc)
    return kc, 0.5 * sp / (k_safe ** 2)


def enstrophy_spectrum(omega_phys: np.ndarray):
    """Enstrophy spectrum:  Z(k) = 0.5 * |hat omega|^2."""
    kc, sp = radial_average(omega_phys)
    return kc, 0.5 * sp


# --------------------------------------------------------------------------- #
# Model loading                                                               #
# --------------------------------------------------------------------------- #

def load_model(run_dir: Path, n_in: int, hidden: int, kernel: int,
               base_channels: int, depth: int, model_name: str,
               device: str = 'cpu'):
    """Load a model from run_dir/model_best.pt with the right factory."""
    if model_name in ('bilinear_closure', 'bilin', 'fixd_v2'):
        from model_fixD import build_model
        model = build_model(model_name, in_channels=n_in,
                            hidden=hidden, kernel=kernel)
    else:
        from model import build_model
        if model_name == 'unet':
            model = build_model('unet', in_channels=n_in,
                                base_channels=base_channels,
                                kernel=kernel)
        else:
            model = build_model('cnn', in_channels=n_in,
                                hidden_channels=hidden,
                                depth=depth, kernel=kernel)

    ckpt_path = run_dir / 'model_best.pt'
    if not ckpt_path.exists():
        ckpt_path = run_dir / 'model_last.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(f"no model_best.pt or model_last.pt in {run_dir}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'model_state' in state:
        model.load_state_dict(state['model_state'])
    elif 'state_dict' in state:
        model.load_state_dict(state['state_dict'])
    else:
        model.load_state_dict(state)
    model.to(device).eval()
    print(f"  loaded {ckpt_path}")
    return model


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir',  type=Path, required=True,
                   help='training run dir (config.json, model_best.pt)')
    p.add_argument('--root-dir', type=Path, required=True,
                   help='dataset root (manifest.json + samples/)')
    p.add_argument('--n-samples', type=int, default=3)
    p.add_argument('--split', type=str, default='val',
                   choices=['train', 'val', 'test'])
    p.add_argument('--out-path', type=Path, default=None)
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    out_path = args.out_path or (args.run_dir / 'rollout_correction_visualization.png')

    # --- Manifest --- #
    with open(args.root_dir / 'manifest.json') as f:
        manifest = json.load(f)
    Nx, Ny = int(manifest['Nx']), int(manifest['Ny'])
    Lx, Ly = float(manifest['Lx']), float(manifest['Ly'])
    nu     = float(manifest['nu'])
    mu     = float(manifest.get('mu', 0.0))
    beta   = float(manifest.get('beta', 0.0))
    Delta_T = float(manifest['Delta_T'])
    K       = int(manifest.get('K', 100))
    print(f"  manifest: Nx={Nx} Ny={Ny} Lx={Lx:.3f} Ly={Ly:.3f}")
    print(f"            nu={nu} mu={mu} beta={beta} Delta_T={Delta_T} K={K}")

    # --- Config --- #
    cfg_path = args.run_dir / 'config.json'
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    input_fields = tuple(cfg.get('input_fields',
                                 ['omega_0', 'omega_m1', 'omega_m2',
                                  'psi_0',   'psi_m1',   'psi_m2']))
    target_field = cfg.get('target_field', 'f_NN_target')
    model_name = cfg.get('model', 'bilinear_closure')
    print(f"  inputs : {input_fields}")
    print(f"  target : {target_field}")
    print(f"  model  : {model_name}")

    # --- Dataset (for normalization stats + split indices) --- #
    from dataset import ClosureDataset
    ds = ClosureDataset(
        root_dir=args.root_dir, split=args.split,
        input_fields=input_fields, target_field=target_field,
        normalize=cfg.get('normalize', True),
    )

    model = load_model(args.run_dir,
                       n_in=len(input_fields),
                       hidden=cfg.get('hidden_channels', 64),
                       kernel=cfg.get('kernel', 3),
                       base_channels=cfg.get('base_channels', 32),
                       depth=cfg.get('depth', 6),
                       model_name=model_name,
                       device=args.device)

    # --- Pick samples --- #
    rng = np.random.default_rng(args.seed)
    sample_idx = sorted(rng.choice(len(ds), size=args.n_samples,
                                    replace=False).tolist())
    print(f"  picked sample indices: {sample_idx}")

    samples_dir = args.root_dir / 'samples'
    split_indices = ds.indices

    # --- Build figure: 2 rows per sample, 5 cols --- #
    n_cols = 5
    fig_h_per_sample = 6.5
    fig = plt.figure(figsize=(n_cols * 3.0, fig_h_per_sample * args.n_samples))
    outer = gridspec.GridSpec(2 * args.n_samples, n_cols, figure=fig,
                              hspace=0.55, wspace=0.30)

    for i, ds_idx in enumerate(sample_idx):
        manifest_idx = int(split_indices[ds_idx])
        rec = np.load(samples_dir / f'sample_{manifest_idx:06d}.npz')

        omega_K_fine   = rec['omega_K_fine'].astype(np.float64)
        omega_1_coarse = rec['omega_1_coarse'].astype(np.float64)
        e_anal_incr    = rec['e_anal_incr'].astype(np.float64)

        if omega_K_fine.ndim == 3:
            omega_K_fine   = omega_K_fine[0]
            omega_1_coarse = omega_1_coarse[0]
            e_anal_incr    = e_anal_incr[0]

        # NN prediction
        x, _ = ds[ds_idx]
        with torch.no_grad():
            yhat = model(x.unsqueeze(0).to(args.device))[0]
        f_NN_pred = (yhat * ds.target_std + ds.target_mean)[0].cpu().numpy().astype(np.float64)

        e_NN_incr_pred = apply_closure_correction(f_NN_pred, Delta_T, K)
        omega_corrected = omega_1_coarse + e_anal_incr + e_NN_incr_pred

        diff_no_closure   = omega_K_fine - omega_1_coarse
        diff_with_closure = omega_K_fine - omega_corrected

        rms_truth   = float(np.sqrt(np.mean(omega_K_fine ** 2)))
        rms_no_cl   = float(np.sqrt(np.mean(diff_no_closure ** 2)))
        rms_with_cl = float(np.sqrt(np.mean(diff_with_closure ** 2)))
        rel_no_cl   = rms_no_cl / rms_truth if rms_truth > 0 else float('nan')
        rel_with_cl = rms_with_cl / rms_truth if rms_truth > 0 else float('nan')
        improvement = rms_no_cl / rms_with_cl if rms_with_cl > 0 else float('inf')

        print(f"  sample {ds_idx} (manifest #{manifest_idx}):")
        print(f"    |truth|        = {rms_truth:.4e}")
        print(f"    |truth-coarse| = {rms_no_cl:.4e}   (rel {rel_no_cl:.4f})")
        print(f"    |truth-corr|   = {rms_with_cl:.4e}   (rel {rel_with_cl:.4f})")
        print(f"    improvement factor = {improvement:.2f}x")

        # Pre-compute spectra
        kc, E_truth = energy_spectrum(omega_K_fine,    Lx, Ly)
        _,  E_co    = energy_spectrum(omega_1_coarse,  Lx, Ly)
        _,  E_corr  = energy_spectrum(omega_corrected, Lx, Ly)
        _,  E_dno   = energy_spectrum(diff_no_closure, Lx, Ly)
        _,  E_dwith = energy_spectrum(diff_with_closure, Lx, Ly)

        _,  Z_truth = enstrophy_spectrum(omega_K_fine)
        _,  Z_co    = enstrophy_spectrum(omega_1_coarse)
        _,  Z_corr  = enstrophy_spectrum(omega_corrected)
        _,  Z_dno   = enstrophy_spectrum(diff_no_closure)
        _,  Z_dwith = enstrophy_spectrum(diff_with_closure)

        # ===== Row 1: NO closure ===== #
        v_omega = float(np.abs(np.concatenate([
            omega_K_fine.flat, omega_1_coarse.flat, omega_corrected.flat])).max())
        if v_omega == 0:
            v_omega = 1e-30
        v_diff_no = float(np.abs(diff_no_closure).max())
        if v_diff_no == 0:
            v_diff_no = 1e-30

        ax = fig.add_subplot(outer[2 * i, 0])
        ax.imshow(omega_K_fine, cmap='RdBu_r', vmin=-v_omega, vmax=v_omega,
                  origin='lower', aspect='equal', interpolation='gaussian')
        ax.set_title(r'truth: $\omega^{K}_{\rm fine}$', fontsize=11)
        ax.set_ylabel(rf'sample {ds_idx}'
                      '\n'
                      r'$\bf{No\ closure}$'
                      '\n'
                      rf'rel $L^2 = {rel_no_cl:.3f}$', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

        ax = fig.add_subplot(outer[2 * i, 1])
        ax.imshow(omega_1_coarse, cmap='RdBu_r', vmin=-v_omega, vmax=v_omega,
                  origin='lower', aspect='equal', interpolation='gaussian')
        ax.set_title(r'coarse: $\omega^{1}_{\rm coarse}$', fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

        ax = fig.add_subplot(outer[2 * i, 2])
        ax.imshow(diff_no_closure, cmap='RdBu_r',
                  vmin=-v_diff_no, vmax=v_diff_no,
                  origin='lower', aspect='equal', interpolation='gaussian')
        ax.set_title(r'truth $-$ coarse'
                     '\n'
                     rf'$\|\cdot\|_2 = {rms_no_cl:.3e}$', fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

        ax = fig.add_subplot(outer[2 * i, 3])
        ax.loglog(kc[1:], E_truth[1:] + 1e-30, 'k-',  lw=1.6, label='truth')
        ax.loglog(kc[1:], E_co[1:]    + 1e-30, 'C0-', lw=1.2, label='coarse')
        ax.loglog(kc[1:], E_dno[1:]   + 1e-30, 'r:',  lw=1.4, label='error')
        ax.set_xlabel(r'$k$', fontsize=10)
        ax.set_ylabel(r'$E(k) = \frac{1}{2}\,k^{-2}\,|\hat{\omega}(k)|^2$', fontsize=10)
        ax.set_title('Energy spectrum', fontsize=10)
        ax.grid(True, which='both', alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, loc='best')

        ax = fig.add_subplot(outer[2 * i, 4])
        ax.loglog(kc[1:], Z_truth[1:] + 1e-30, 'k-',  lw=1.6, label='truth')
        ax.loglog(kc[1:], Z_co[1:]    + 1e-30, 'C0-', lw=1.2, label='coarse')
        ax.loglog(kc[1:], Z_dno[1:]   + 1e-30, 'r:',  lw=1.4, label='error')
        ax.set_xlabel(r'$k$', fontsize=10)
        ax.set_ylabel(r'$Z(k) = \frac{1}{2}\,|\hat{\omega}(k)|^2$', fontsize=10)
        ax.set_title('Enstrophy spectrum', fontsize=10)
        ax.grid(True, which='both', alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, loc='best')

        # ===== Row 2: WITH closure ===== #
        v_diff_with = float(np.abs(diff_with_closure).max())
        if v_diff_with == 0:
            v_diff_with = 1e-30

        ax = fig.add_subplot(outer[2 * i + 1, 0])
        ax.imshow(omega_K_fine, cmap='RdBu_r', vmin=-v_omega, vmax=v_omega,
                  origin='lower', aspect='equal', interpolation='gaussian')
        ax.set_title(r'truth: $\omega^{K}_{\rm fine}$', fontsize=11)
        ax.set_ylabel(r'$\bf{With\ closure}$'
                      '\n'
                      rf'rel $L^2 = {rel_with_cl:.3f}$'
                      '\n'
                      rf'improvement: ${improvement:.2f}\times$', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

        ax = fig.add_subplot(outer[2 * i + 1, 1])
        ax.imshow(omega_corrected, cmap='RdBu_r', vmin=-v_omega, vmax=v_omega,
                  origin='lower', aspect='equal', interpolation='gaussian')
        ax.set_title(r'coarse $+$ closure'
                     '\n'
                     r'$\omega^{1}_{\rm coarse} + e_{\rm anal} + e^{\rm pred}_{NN}$',
                     fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

        ax = fig.add_subplot(outer[2 * i + 1, 2])
        ax.imshow(diff_with_closure, cmap='RdBu_r',
                  vmin=-v_diff_with, vmax=v_diff_with,
                  origin='lower', aspect='equal', interpolation='gaussian')
        ax.set_title(r'truth $-$ corrected'
                     '\n'
                     rf'$\|\cdot\|_2 = {rms_with_cl:.3e}$', fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

        ax = fig.add_subplot(outer[2 * i + 1, 3])
        ax.loglog(kc[1:], E_truth[1:] + 1e-30, 'k-',  lw=1.6, label='truth')
        ax.loglog(kc[1:], E_corr[1:]  + 1e-30, 'C2-', lw=1.2, label='corrected')
        ax.loglog(kc[1:], E_dwith[1:] + 1e-30, 'r-',  lw=1.4, label='error (with)')
        ax.loglog(kc[1:], E_dno[1:]   + 1e-30, 'C0:', lw=1.0, alpha=0.7,
                  label='error (no closure)')
        ax.set_xlabel(r'$k$', fontsize=10)
        ax.set_ylabel(r'$E(k) = \frac{1}{2}\,k^{-2}\,|\hat{\omega}(k)|^2$', fontsize=10)
        ax.set_title('Energy spectrum', fontsize=10)
        ax.grid(True, which='both', alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, loc='best')

        ax = fig.add_subplot(outer[2 * i + 1, 4])
        ax.loglog(kc[1:], Z_truth[1:] + 1e-30, 'k-',  lw=1.6, label='truth')
        ax.loglog(kc[1:], Z_corr[1:]  + 1e-30, 'C2-', lw=1.2, label='corrected')
        ax.loglog(kc[1:], Z_dwith[1:] + 1e-30, 'r-',  lw=1.4, label='error (with)')
        ax.loglog(kc[1:], Z_dno[1:]   + 1e-30, 'C0:', lw=1.0, alpha=0.7,
                  label='error (no closure)')
        ax.set_xlabel(r'$k$', fontsize=10)
        ax.set_ylabel(r'$Z(k) = \frac{1}{2}\,|\hat{\omega}(k)|^2$', fontsize=10)
        ax.set_title('Enstrophy spectrum', fontsize=10)
        ax.grid(True, which='both', alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, loc='best')

    fig.suptitle(rf'Rollout Correction: $\omega^{{K}}_{{\rm fine}}$ vs '
                 rf'$\omega^{{1}}_{{\rm coarse}}$ '
                 rf'$\pm$ closure  ({args.split} set, {model_name})',
                 fontsize=13, y=0.998)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"\nwrote {out_path}")


if __name__ == '__main__':
    main()
