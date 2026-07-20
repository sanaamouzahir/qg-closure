"""
TWO-BAND blended Pi_FF inference (Sanaa GO 2026-07-20).

WHY TWO MODELS. Measured on the same frames/truth/scale, one model cannot
serve both regions: near-wall targets have RMS ~3-6.6, far-field ~0.08-0.42
(a 40x gap), while a single SVGP has ONE y-standardization, ONE inducing set
and ONE ARD geometry -- so capacity goes to whichever regime dominates the
pooled loss. The measured trade:

    wallv2 (grad+lap+wall gate) NEAR r2 .954 err 21%  |  FAR r2 .874 err 35%
    lap    (grad+lap, no gate)  NEAR r2 .907 err 30%  |  FAR r2 .934 err 25%

SPEC REVISION 2026-07-20: the delivered pair does NOT mix recipes. BOTH
specialists use the plain `lap` recipe (grad + lap GP-ARD channels, no wall
gate, no tail bias) and differ ONLY by data.band, so band-vs-pooled is the one
variable under test. Tail-bias and the wall gate are redundant under band
restriction (tail-bias fought a dilution the band removes by construction; the
gate fought far-field contamination of the feature scales, which per-band
conditioning_stats/target_stats handle -- and the gate is what caused the ep-0
NaN). Both code paths stay guarded and default-off for a later ablation, and
this class still supports a wall-gated near model if one is ever blended in
(see FEATURE CONVENTIONS).

Each model is trained as a BAND SPECIALIST (dataset_piff data.band restricts
the valid mask, hence each band's own y-standardization / feature scales /
inducing set), and this module blends them at inference. Pre-registered bar:
NEAR r2 >= 0.954 AND FAR r2 >= 0.934 simultaneously.

BLEND. Partition of unity over the signed distance s = sdf / D:

    w(s) = 1                          s <= overlap_lo   (pure NEAR)
    w(s) = 1 - (3 t^2 - 2 t^3),  t = (s - lo)/(hi - lo)   in the overlap
    w(s) = 0                          s >= overlap_hi   (pure FAR)

    Pi   = w * Pi_near + (1 - w) * Pi_far
    var  = w^2 * var_near + (1 - w)^2 * var_far

VARIANCE CHOICE -- DOCUMENTED ASSUMPTION: the w^2/(1-w)^2 form is the variance
of a weighted sum of INDEPENDENT experts. The two specialists are trained on
overlapping data (the 1.0-1.5 D overlap band) and share the same inputs, so
their errors are certainly POSITIVELY correlated there; the independent-experts
variance is therefore a LOWER bound on the true predictive variance inside the
overlap (it omits the 2 w (1-w) cov term). It is exact outside the overlap,
where one weight is 0 and the other 1 -- i.e. everywhere except a 0.5 D collar.
The alternative (w*var_near + (1-w)*var_far, the fully-correlated / mixture-of-
experts moment) would be an UPPER bound. We take the independent form because
it is the standard product-of-experts convention and because sigma is already
recalibrated downstream by the conformal/structural sidecars; anyone using
blended sigma for a coverage claim inside the overlap must re-verify it there.

WHERE THE PIXEL SDF COMES FROM. Input channel 3 is sdf_star =
clip(sdf, +/-clipD)/clipD with clipD = data.sdf_clip_D * D, so

    s = sdf / D = sdf_star * sdf_clip_D

exactly, for every pixel with |sdf| < clipD. The configs use sdf_clip_D = 2.0
and overlap_hi = 1.5, so the recovery is EXACT throughout the blend region and
throughout the near model's support; beyond 2 D the channel saturates and the
recovered s reads 2.0 -- which still gives w = 0 (correct), so saturation is
harmless. __init__ asserts sdf_clip_D >= overlap_hi. This is why the blended
path needs no extra plumbing to reach RunData.sdf.

FEATURE CONVENTIONS -- THE CRITICAL PART. The two sub-models have DIFFERENT
feature normalizations and y-standardizations, so each computes its OWN
features and GP inputs from the raw batch; nothing is shared but the raw
tensors. Even with the SAME recipe on both sides, the two carry different
g_scale / g2_scale / lap_scale / y_mu / y_sd (each calibrated on its own band)
and different inducing sets -- which is exactly the capacity split we are
buying, and exactly why gpin must never be shared between them.

Callers must build the dataset with NO data.band (the blend is scored on the
whole field) and model.use_wall_gate false -- i.e. the
conf_piff_<geometry>_gjs_lap.yaml config, which is the recipe both specialists
were trained with. With the delivered pair that is the whole story and the
gate path below is dormant.

RESIDUAL SUPPORT for a wall-gated near model (the deferred ablation): if the
near specialist WAS trained gated, its g/lap channels were multiplied by
exp(-max(sdf,0)/D) in the dataset and its scales calibrated on gated data. The
blended model then consumes the UNGATED convention and reconstructs that gate
itself, per pixel, from the recovered s. The reconstruction is exact wherever
w > 0 (s <= 1.5 D < clipD). Beyond the clip the reconstructed gate is too
large, but g/lap are POINTWISE GP-ARD columns (they never enter the CNN, so
there is no receptive-field leakage) and every such pixel has w = 0 exactly,
so the error is multiplied by zero.

The class exposes the interface the diagnostics use: masked_gp_inputs,
features, predict_physical, the use_* flags, noise_prior, het_noise-adjacent
helpers, ard_lengthscales, y_mu/y_sd, eval()/to()/double() (free from
nn.Module). `.gp` and `.likelihood` are deliberately NOT provided: paths that
reach into the raw GP (e.g. replot_eval_fields' --recal sidecar, or the
structural sigma recalibration) are not defined for a blend and must fail
loudly rather than silently use one band's GP for the whole field.
"""

import numpy as np
import torch
import torch.nn as nn

from model_piff import PiffModel


def blend_weight(s, lo, hi):
    """Smoothstep partition of unity: 1 at s <= lo (pure near), 0 at s >= hi
    (pure far), C^1 in between. `s` = sdf / D (torch tensor or ndarray)."""
    if not hi > lo:
        raise ValueError(f"overlap must satisfy hi > lo (got lo={lo}, hi={hi})")
    if isinstance(s, torch.Tensor):
        t = ((s - lo) / (hi - lo)).clamp(0.0, 1.0)
        return 1.0 - (3.0 * t * t - 2.0 * t * t * t)
    t = np.clip((np.asarray(s) - lo) / (hi - lo), 0.0, 1.0)
    return 1.0 - (3.0 * t * t - 2.0 * t * t * t)


class BlendedGPInputs:
    """Carrier for the per-pixel inputs of BOTH specialists plus the blend
    weight, standing in for the plain (P, D) gpin tensor the callers chunk.

    Supports exactly what the diagnostics do with a gpin: `.shape[0]` and
    slicing `gpin[i0:i0+chunk]`. Every field is sliced together, so a chunk of
    the carrier is a consistent chunk of near-inputs, far-inputs, blend weights
    and the two structural-noise g columns."""

    __slots__ = ('gp_near', 'gp_far', 'w', 'g_near', 'g_far', 'masked')

    def __init__(self, gp_near, gp_far, w, g_near=None, g_far=None,
                 masked=True):
        self.gp_near, self.gp_far, self.w = gp_near, gp_far, w
        self.g_near, self.g_far = g_near, g_far
        # masked=False => built by features() over ALL pixels: the GP inputs
        # are (B,H,W,D) and the g planes (B,H,W), whereas predict_physical
        # needs per-pixel rows (P,D) and (P,). The flag lets that trap fail
        # loudly instead of silently mis-shaping het_noise (G4 MINOR 2026-07-20).
        self.masked = bool(masked)

    @property
    def shape(self):
        return self.gp_near.shape

    def __len__(self):
        return int(self.gp_near.shape[0])

    def __getitem__(self, sl):
        cut = lambda t: (None if t is None else t[sl])          # noqa: E731
        return BlendedGPInputs(self.gp_near[sl], self.gp_far[sl], self.w[sl],
                               cut(self.g_near), cut(self.g_far),
                               masked=self.masked)


class BlendedPiffModel(nn.Module):
    """Two band specialists under one PiffModel-shaped interface. See the
    module docstring for the blend, the variance assumption, the sdf recovery
    and the feature-convention requirement (ungated, unbanded eval config)."""

    def __init__(self, near, far, overlap_lo, overlap_hi, sdf_clip_D,
                 geometry=None):
        # overlap_lo/overlap_hi/sdf_clip_D are REQUIRED, never defaulted
        # (G4 MINOR 2026-07-20): a silently-defaulted blend geometry would hand
        # over at a distance nobody chose, and the manifest is the only place
        # that decision is recorded.
        super().__init__()
        self.near, self.far = near, far
        self.overlap_lo, self.overlap_hi = float(overlap_lo), float(overlap_hi)
        self.sdf_clip_D = float(sdf_clip_D)
        self.geometry = geometry
        if not self.overlap_hi > self.overlap_lo:
            raise ValueError(f"overlap_hi {self.overlap_hi} must exceed "
                             f"overlap_lo {self.overlap_lo}")
        if self.sdf_clip_D < self.overlap_hi:
            raise ValueError(
                f"sdf_clip_D {self.sdf_clip_D} < overlap_hi {self.overlap_hi}: "
                f"the sdf recovered from input channel 3 saturates INSIDE the "
                f"blend region, so the partition of unity would be wrong there")
        # the two must agree on everything that shapes the RAW BATCH; they are
        # expected to DISAGREE on use_wall_gate (that is the whole point)
        for fl in ('use_zeta_dot', 'use_grad_feature', 'use_lap_feature',
                   'noise_prior', 'likelihood_type'):
            a, b = getattr(near, fl), getattr(far, fl)
            if a != b:
                raise ValueError(f"blend: near {fl}={a!r} != far {fl}={b!r} — "
                                 f"the two specialists must consume the same "
                                 f"batch tensors")
        self.use_zeta_dot = near.use_zeta_dot
        self.use_grad_feature = near.use_grad_feature
        self.use_lap_feature = near.use_lap_feature
        self.noise_prior = near.noise_prior
        self.likelihood_type = near.likelihood_type
        # DATA convention this model consumes: UNGATED g/lap (the far model's).
        # The near model's gate is applied internally, per pixel. Callers must
        # therefore build the dataset with use_wall_gate false and NO data.band.
        self.use_wall_gate = False
        self.near_wall_gate = bool(getattr(near, 'use_wall_gate', False))

    # ---- interface the diagnostics touch ---------------------------------- #
    @property
    def y_mu(self):
        """FAR band's standardization (the bulk of the domain). Reported only;
        the blend inverts each sub-model's own standardization internally."""
        return self.far.y_mu

    @property
    def y_sd(self):
        return self.far.y_sd

    def is_student_t(self):
        return self.near.is_student_t()

    def student_nu(self):
        return self.near.student_nu()

    def zeta_ard_lengthscale(self):
        return self.far.zeta_ard_lengthscale()

    def ard_lengthscales(self):
        """Per-band lengthscales, flattened with near_/far_ prefixes plus the
        far band's bare keys so summary writers expecting 'zeta' still work."""
        out = {f'far_{k}': v for k, v in self.far.ard_lengthscales().items()}
        out.update({f'near_{k}': v for k, v in self.near.ard_lengthscales().items()})
        out.update(self.far.ard_lengthscales())
        return out

    def load_state_dict(self, state_dict, strict=True):
        """The sub-models are already loaded from their own checkpoints by
        piff_model_loader. The blended handle carries an EMPTY 'model' dict so
        the `model.load_state_dict(ck['model'])` line in every diagnostic stays
        untouched; anything non-empty is a caller error."""
        if state_dict:
            raise RuntimeError(
                "BlendedPiffModel.load_state_dict: the specialists load from "
                "their own ckpts (blended_manifest.yaml); got a non-empty "
                f"state dict with {len(state_dict)} keys")
        return torch.nn.modules.module._IncompatibleKeys([], [])

    # ---- the blend --------------------------------------------------------- #
    def _sdf_over_D(self, x):
        """Per-pixel s = sdf/D recovered EXACTLY from input channel 3
        (sdf_star = clip(sdf,+/-clipD)/clipD). See the module docstring."""
        return x[:, 3] * self.sdf_clip_D                    # (B,H,W)

    def _near_gate(self, s):
        """The near specialist's dataset-side wall gate exp(-max(sdf,0)/D),
        reconstructed per pixel. Identity when the near model is ungated."""
        if not self.near_wall_gate:
            return None
        return torch.exp(-s.clamp_min(0.0))

    def features(self, x, zeta, zeta_dot=None, g=None, lap=None):
        """Per-pixel inputs for BOTH specialists + blend weight, over ALL
        pixels. Each sub-model computes its OWN features from the raw batch —
        their normalizations and y-standardizations differ and are never
        shared. g/lap must be the UNGATED planes (see module docstring).

        NOT a prediction path: the returned carrier is UNMASKED, so its GP
        inputs are (B,H,W,D) and its g planes (B,H,W). predict_physical needs
        per-pixel rows and REFUSES an unmasked carrier — use
        masked_gp_inputs(x, zeta, mask, ...) to predict."""
        s = self._sdf_over_D(x)
        gate = self._near_gate(s)
        g_n = g if (g is None or gate is None) else g * gate
        lap_n = lap if (lap is None or gate is None) else lap * gate
        return BlendedGPInputs(
            self.near.features(x, zeta, zeta_dot=zeta_dot, g=g_n, lap=lap_n),
            self.far.features(x, zeta, zeta_dot=zeta_dot, g=g, lap=lap),
            blend_weight(s, self.overlap_lo, self.overlap_hi),
            g_near=g_n, g_far=g, masked=False)

    def masked_gp_inputs(self, x, zeta, mask, zeta_dot=None, g=None, lap=None):
        """Flatten to masked pixels, same contract as PiffModel: returns a
        BlendedGPInputs whose every field is indexed by the SAME mask, so the
        pixel order matches across near/far/weights."""
        s = self._sdf_over_D(x)
        gate = self._near_gate(s)
        g_n = g if (g is None or gate is None) else g * gate
        lap_n = lap if (lap is None or gate is None) else lap * gate
        return BlendedGPInputs(
            self.near.features(x, zeta, zeta_dot=zeta_dot, g=g_n, lap=lap_n)[mask],
            self.far.features(x, zeta, zeta_dot=zeta_dot, g=g, lap=lap)[mask],
            blend_weight(s, self.overlap_lo, self.overlap_hi)[mask],
            g_near=(None if g_n is None else g_n[mask]),
            g_far=(None if g is None else g[mask]))

    def predict_physical(self, gpin, g_masked=None):
        """Blended predictive (mean, var) in PHYSICAL target units. Each
        specialist inverts its OWN y-standardization first, then

            mu  = w mu_near + (1-w) mu_far
            var = w^2 var_near + (1-w)^2 var_far   (independent experts — see
                                                    the module docstring)

        `g_masked` is IGNORED: the carrier already holds each band's correctly
        gated / ungated g column (they differ), and using the caller's single
        version would feed the near model far-field-scaled noise features."""
        if not isinstance(gpin, BlendedGPInputs):
            raise TypeError(
                "BlendedPiffModel.predict_physical needs the BlendedGPInputs "
                "carrier from this model's masked_gp_inputs — a plain gpin "
                "tensor cannot carry the two bands' distinct features")
        if not gpin.masked:
            raise ValueError(
                "BlendedPiffModel.predict_physical was handed an UNMASKED "
                "carrier (from features()): its GP inputs are (B,H,W,D) and "
                "its g planes (B,H,W), but prediction needs per-pixel rows "
                "(P,D) and (P,) — het_noise would silently receive the wrong "
                "shape. Correct usage:\n"
                "    gpin = model.masked_gp_inputs(x, zeta, mask, "
                "zeta_dot=..., g=..., lap=...)\n"
                "    mu, var = model.predict_physical(gpin)\n"
                "features() is for inspecting the two bands' inputs, not for "
                "predicting.")
        mu_n, var_n = self.near.predict_physical(gpin.gp_near, g_masked=gpin.g_near)
        mu_f, var_f = self.far.predict_physical(gpin.gp_far, g_masked=gpin.g_far)
        w = gpin.w.to(mu_n.dtype)
        one_m = 1.0 - w
        return (w * mu_n + one_m * mu_f,
                w * w * var_n + one_m * one_m * var_f)

    def band_fractions(self, x, mask):
        """Diagnostic: (pure-near, overlap, pure-far) pixel shares of a masked
        batch — the sanity check that a frame actually exercises both experts."""
        s = self._sdf_over_D(x)[mask]
        n = max(int(s.numel()), 1)
        return {'pure_near': float((s <= self.overlap_lo).sum()) / n,
                'overlap': float(((s > self.overlap_lo)
                                  & (s < self.overlap_hi)).sum()) / n,
                'pure_far': float((s >= self.overlap_hi).sum()) / n}

    def describe(self):
        return {'kind': 'blended_two_band', 'geometry': self.geometry,
                'overlap_lo_D': self.overlap_lo, 'overlap_hi_D': self.overlap_hi,
                'sdf_clip_D': self.sdf_clip_D,
                'near_wall_gate': self.near_wall_gate,
                'use_grad_feature': self.use_grad_feature,
                'use_lap_feature': self.use_lap_feature,
                'use_zeta_dot': self.use_zeta_dot,
                'noise_prior': self.noise_prior,
                'variance_blend': 'independent experts: w^2 var_near + '
                                  '(1-w)^2 var_far (lower bound in the overlap)'}


def build_blended(near_ck, far_ck, overlap_lo, overlap_hi, sdf_clip_D,
                  geometry=None, device='cpu'):
    """Construct both specialists from their loaded checkpoint dicts (each
    carries the conf it was TRAINED with) and wrap them. The blend geometry
    (overlap_lo, overlap_hi, sdf_clip_D) is REQUIRED — never defaulted
    (G4 MINOR 2026-07-20)."""
    near = PiffModel(near_ck['conf']).to(device)
    near.load_state_dict(near_ck['model'])
    far = PiffModel(far_ck['conf']).to(device)
    far.load_state_dict(far_ck['model'])
    m = BlendedPiffModel(near, far, overlap_lo=overlap_lo,
                         overlap_hi=overlap_hi, sdf_clip_D=sdf_clip_D,
                         geometry=geometry).to(device)
    m.eval()
    return m
