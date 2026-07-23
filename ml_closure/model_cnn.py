"""
CNN-only Pi_FF closure: FiLM-CNN + 1x1 conv head, NO GP (Sanaa order
2026-07-22). Question under test: can the conditioned CNN alone train well?
Lock this, then re-add the GP head.

Prediction contract:
    yhat_std(x)        = head(FiLMCNN(x; zeta, zeta_dot))          [standardized]
    Pi_hat(x)          = yhat_std(x) * sigma_loc(sdf(x))           [physical Pi*]
sigma_loc is a FROZEN per-pixel scale: the train-split rms of the normalized
target Pi* binned by sdf/D (piecewise-LINEAR interpolation between bin centers
— a step profile would print bin seams into the physical field), recorded as
buffers in every checkpoint, inverted exactly at prediction. Loss = plain MSE
in the standardized space = MSE weighted by 1/sigma_loc^2 in physical space.
This carries the ~40x near/far amplitude gap with ZERO learnable weighting —
the D/E ELBO-buy-out family and the mtgp task machinery cannot exist here.
"""

import numpy as np
import torch
import torch.nn as nn

from model_piff import FiLMCNN


class PiffCNN(nn.Module):
    """FiLM-CNN + 1x1 conv head. Inputs identical to PiffModel's CNN level:
    x (B,4,H,W) = [omega*, u*, v*, sdf*], conditioner (zeta, zeta_dot/zdot_sd)."""

    def __init__(self, conf):
        super().__init__()
        mc = conf['model']
        dc = conf.get('data', {}) or {}
        self.use_zeta_dot = bool(mc.get('use_zeta_dot', False))
        # v2 (Sanaa order 2026-07-22 night): |lap(omega_bar)|* as a 5th CNN
        # INPUT channel, log1p-compressed by the recorded lap_scale (the
        # 2026-07-16 heavy-tail lesson; raw range 0..332 -> ~[0, 5.9]).
        # Default OFF -> v1 checkpoints byte-identical (buffer non-persistent).
        self.use_lap_input = bool(mc.get('use_lap_input', False))
        cond_dim = 1 + int(self.use_zeta_dot)
        self.cnn = FiLMCNN(in_channels=int(mc['in_channels']),
                           channels=mc['channels'], kernel=int(mc['kernel']),
                           film=bool(mc['film']),
                           film_hidden=int(mc['film_hidden']),
                           cond_dim=cond_dim)
        self.head = nn.Conv2d(self.cnn.out_dim, 1, kernel_size=1)
        self.sdf_clip_D = float(dc.get('sdf_clip_D', 2.0))
        nb = int(mc.get('sigma_loc_bins', 24))
        # sigma_loc profile buffers — recorded in every ckpt (same contract as
        # PiffModel.y_mu/y_sd). Centers over s = clip(sdf/D, 0, clipD);
        # sig_rms init 1.0 = identity until set_sigma_profile is called.
        edges = torch.linspace(0.0, self.sdf_clip_D, nb + 1)
        self.register_buffer('sig_edges', edges)
        self.register_buffer('sig_centers', 0.5 * (edges[:-1] + edges[1:]))
        self.register_buffer('sig_rms', torch.ones(nb))
        self.register_buffer('sig_set', torch.zeros(()))
        self.register_buffer('zdot_sd', torch.ones(()),
                             persistent=self.use_zeta_dot)
        self.register_buffer('lap_scale', torch.ones(()),
                             persistent=self.use_lap_input)

    # ---- recorded constants ---------------------------------------------- #
    def set_sigma_profile(self, rms):
        """Install the train-split rms(Pi*) per sdf/D bin (train_cnn.sigma_profile
        output). Refuses non-positive values — empty bins must be filled by the
        caller (nearest non-empty), a zero rms would divide the loss by 0."""
        rms = torch.as_tensor(np.asarray(rms, dtype=np.float64),
                              dtype=self.sig_rms.dtype)
        if rms.shape != self.sig_rms.shape:
            raise ValueError(f"sigma profile shape {tuple(rms.shape)} != "
                             f"{tuple(self.sig_rms.shape)}")
        if not bool((rms > 0).all()):
            raise ValueError(f"sigma profile has non-positive bins: {rms.tolist()}")
        self.sig_rms.copy_(rms)
        self.sig_set.fill_(1.0)
        return {'sigma_loc_rms': [float(v) for v in rms],
                'sigma_loc_edges': [float(v) for v in self.sig_edges]}

    def set_zdot_sd(self, zdot_sd):
        zdot_sd = float(zdot_sd)
        if not zdot_sd > 0.0:
            raise ValueError(f"zdot_sd={zdot_sd}: no zeta_dot variance in the "
                             f"pool — use_zeta_dot is meaningless, refuse loudly")
        self.zdot_sd.fill_(zdot_sd)
        return {'zdot_sd': zdot_sd}

    def set_lap_scale(self, lap_scale):
        lap_scale = float(lap_scale)
        if not lap_scale > 0.0:
            raise ValueError(f"bad lap_scale={lap_scale}")
        self.lap_scale.fill_(lap_scale)
        return {'lap_scale': lap_scale}

    # ---- forward ----------------------------------------------------------- #
    def _cond(self, zeta, zeta_dot):
        if self.use_zeta_dot:
            if zeta_dot is None:
                raise ValueError("use_zeta_dot=true but zeta_dot not supplied")
            return torch.stack([zeta, zeta_dot / self.zdot_sd], dim=-1)  # (B,2)
        return zeta

    def features_in(self, x, lap=None):
        """CNN input tensor: the 4 canonical channels, +log1p(|lap|/scale)
        as channel 4 when use_lap_input (sdf stays channel 3 — sigma_loc and
        the task/region logic keep reading x[:, 3] unchanged)."""
        if not self.use_lap_input:
            return x
        if lap is None:
            raise ValueError("use_lap_input=true but lap not supplied")
        return torch.cat(
            [x, torch.log1p(lap / self.lap_scale).unsqueeze(1)], dim=1)

    def forward(self, x, zeta, zeta_dot=None, lap=None):
        """(B,4,H,W),(B,)[,(B,)][,(B,H,W)] -> standardized prediction."""
        f = self.cnn(self.features_in(x, lap), self._cond(zeta, zeta_dot))
        return self.head(f).squeeze(1)

    def sigma_loc(self, x):
        """Per-pixel frozen scale (B,H,W) from input channel 3: s = sdf_star *
        sdf_clip_D clamped to [0, clipD], piecewise-linear in the bin-center
        rms profile (constant beyond the first/last center)."""
        if not bool(self.sig_set):
            raise RuntimeError("sigma_loc profile not set — call "
                               "set_sigma_profile before any prediction")
        c = self.sig_centers
        s = (x[:, 3] * self.sdf_clip_D).clamp(float(c[0]), float(c[-1]))
        hi = torch.bucketize(s, c).clamp(1, c.numel() - 1)   # right center idx
        lo = hi - 1
        w = (s - c[lo]) / (c[hi] - c[lo])
        return (1.0 - w) * self.sig_rms[lo] + w * self.sig_rms[hi]

    def predict_physical(self, x, zeta, zeta_dot=None, lap=None):
        """Physical-units prediction Pi_hat* (B,H,W): standardization inverted
        exactly by the recorded profile."""
        return self.forward(x, zeta, zeta_dot, lap) * self.sigma_loc(x)

    # ---- logging ----------------------------------------------------------- #
    def film_norms(self):
        """||dgamma||, ||beta|| at the conditioner-range midpoint 0 (same probe
        as PiffModel.film_norms — the FiLM-activity signal for the monitor)."""
        with torch.no_grad():
            gb = self.cnn.film_mlp(torch.zeros(
                1, self.cnn.cond_dim, device=self.sig_rms.device))
            dgamma, beta = gb.chunk(2, dim=-1)
            return float(dgamma.norm()), float(beta.norm())
