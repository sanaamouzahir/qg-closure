"""
FiLM-CNN feature extractor + pointwise SVGP head (ML SPEC 01 S2).

CNN: 3 conv layers k5, periodic padding both directions, 4 -> 32 -> 32 -> F=16,
swish; RF = 13 coarse points (measured Pi–omega locality ~7, Srinivasan 2024 —
do NOT deepen without a B-item). FiLM after every conv: h <- gamma(zeta)*h +
beta(zeta), one shared MLP zeta -> 16 (swish) -> 2*(32+32+16), output layer
ZERO-initialized so gamma = 1 + dgamma is exactly identity at init (T3).
`film=False` freezes gamma=1, beta=0 (Re-blind ablation).

SVGP (Phase-1 pointwise design): each output pixel = one GP evaluation on the
F-dim feature vector + zeta appended (F+1 kernel inputs). RBF-ARD over F+1,
M=512 inducing in feature space (k-means init on ~10k pixels of the untrained-
FiLM net), whitened variational parameterization, Gaussian homoscedastic
likelihood. GPyTorch implementation — hand-rolling SVGP is FORBIDDEN by spec;
if gpytorch is missing this module raises a FLAG-worthy ImportError at GP
construction (CNN-only paths still import fine for T1–T4).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_t

try:
    import gpytorch
    _GPYTORCH_ERR = None
except ImportError as e:          # CNN tests must not require gpytorch
    gpytorch = None
    _GPYTORCH_ERR = e


def periodic_pad(x, p):
    """Circular padding in both spatial dims (domain is periodic; the SDF
    channel carries the obstacle)."""
    return F_t.pad(x, (p, p, p, p), mode='circular')


class FiLMCNN(nn.Module):
    def __init__(self, in_channels=4, channels=(32, 32, 16), kernel=5,
                 film=True, film_hidden=16):
        super().__init__()
        self.channels = tuple(int(c) for c in channels)
        self.kernel = int(kernel)
        self.pad = self.kernel // 2
        self.film = bool(film)
        convs, cin = [], int(in_channels)
        for c in self.channels:
            convs.append(nn.Conv2d(cin, c, self.kernel, padding=0))
            cin = c
        self.convs = nn.ModuleList(convs)
        self.n_film = sum(self.channels)
        self.film_mlp = nn.Sequential(
            nn.Linear(1, int(film_hidden)), nn.SiLU(),
            nn.Linear(int(film_hidden), 2 * self.n_film))
        # identity init: zero output layer => dgamma = beta = 0 exactly (T3)
        nn.init.zeros_(self.film_mlp[-1].weight)
        nn.init.zeros_(self.film_mlp[-1].bias)

    @property
    def out_dim(self):
        return self.channels[-1]

    def forward(self, x, zeta):
        """x (B,4,H,W); zeta (B,) -> features (B,F,H,W)."""
        if self.film:
            gb = self.film_mlp(zeta.reshape(-1, 1))          # (B, 2*n_film)
            dgamma, beta = gb.chunk(2, dim=-1)
        off = 0
        h = x
        for conv in self.convs:
            h = conv(periodic_pad(h, self.pad))
            if self.film:
                c = conv.out_channels
                g = 1.0 + dgamma[:, off:off + c, None, None]
                b = beta[:, off:off + c, None, None]
                h = g * h + b
                off += c
            h = F_t.silu(h)
        return h


def _kmeans(x, k, iters=50, seed=0):
    """Plain seeded Lloyd k-means in torch (no sklearn dependency risk).
    x (N,D) -> centers (k,D)."""
    g = torch.Generator(device='cpu').manual_seed(int(seed))
    idx = torch.randperm(x.shape[0], generator=g)[:k]
    centers = x[idx].clone()
    for _ in range(int(iters)):
        d = torch.cdist(x, centers)                    # (N,k)
        assign = d.argmin(dim=1)
        for j in range(k):
            m = assign == j
            if m.any():
                centers[j] = x[m].mean(dim=0)
    return centers


if gpytorch is not None:

    class PiffSVGP(gpytorch.models.ApproximateGP):
        """Pointwise SVGP: RBF-ARD over F+1 dims (features + zeta), whitened
        variational strategy, M inducing points in feature space."""

        def __init__(self, inducing_points):
            m = inducing_points.shape[0]
            var_dist = gpytorch.variational.CholeskyVariationalDistribution(m)
            strategy = gpytorch.variational.VariationalStrategy(
                self, inducing_points, var_dist, learn_inducing_locations=True)
            # VariationalStrategy default whitening = 'cholesky' (whitened parameterization)
            super().__init__(strategy)
            d = inducing_points.shape[1]
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=d))

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x))

else:
    class PiffSVGP:  # placeholder that fails loudly (spec S2.2: no hand-rolled SVGP)
        def __init__(self, *a, **k):
            raise ImportError(
                f"gpytorch unavailable ({_GPYTORCH_ERR}); install gpytorch==1.13 into "
                f"qg-env-piff — do NOT hand-roll an SVGP (ML SPEC 01 S2.2). FLAG if "
                f"the cluster install fails.")


class PiffModel(nn.Module):
    """FiLM-CNN + pointwise SVGP + Gaussian likelihood, end-to-end trainable."""

    def __init__(self, conf):
        super().__init__()
        mc = conf['model']
        self.cnn = FiLMCNN(in_channels=int(mc['in_channels']),
                           channels=mc['channels'], kernel=int(mc['kernel']),
                           film=bool(mc['film']), film_hidden=int(mc['film_hidden']))
        self.gp_input_dim = self.cnn.out_dim + 1     # F + zeta
        init_z = torch.randn(int(mc['n_inducing']), self.gp_input_dim)
        self.gp = PiffSVGP(init_z)
        if gpytorch is None:
            raise ImportError(str(_GPYTORCH_ERR))
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        # recorded, invertible y-standardization constants (2026-07-12 ruling
        # fallback): buffers -> saved in every checkpoint state_dict. Defaults
        # (0,1) = identity. The GP is trained on y_t = (y - y_mu)/y_sd;
        # predict_physical() inverts exactly. Spec-S1.2 target DEFINITION
        # untouched — this is an affine reparameterization for GP conditioning
        # (a large raw outputscale ~var(y) makes the float32 K_zz Cholesky
        # numerically singular: absolute jitter 1e-6 is ~1e-10 relative).
        self.register_buffer('y_mu', torch.zeros(()))
        self.register_buffer('y_sd', torch.ones(()))

    def set_y_standardization(self, y_mean, y_var):
        """Set the recorded standardization constants from exact train-target
        stats. Returns them for the run/checkpoint manifest."""
        y_mean, y_var = float(y_mean), float(y_var)
        if not y_var > 0.0:
            raise ValueError(f"bad target stats: var={y_var}")
        self.y_mu.fill_(y_mean)
        self.y_sd.fill_(y_var ** 0.5)
        return {'y_mean': y_mean, 'y_std': y_var ** 0.5}

    def standardize_y(self, y):
        return (y - self.y_mu) / self.y_sd

    def predict_physical(self, gpin):
        """Likelihood-included predictive (mean, var) in PHYSICAL target units
        (standardization inverted exactly). Non-Gaussian likelihoods (e.g.
        StudentT, arm-C B-item experiment): gpytorch's marginal is Monte-Carlo
        sampled -> moments carry a leading sample dim; reduce by the law of
        total variance (E[var] + Var[mean]) so callers always get 1-D."""
        pred = self.likelihood(self.gp(gpin))
        mu, var = pred.mean, pred.variance
        if mu.dim() > 1:
            var = var.mean(dim=0) + mu.var(dim=0)
            mu = mu.mean(dim=0)
        return mu * self.y_sd + self.y_mu, var * self.y_sd * self.y_sd

    def features(self, x, zeta):
        """(B,4,H,W),(B,) -> per-pixel GP inputs (B,H,W,F+1)."""
        f = self.cnn(x, zeta)                        # (B,F,H,W)
        f = f.permute(0, 2, 3, 1)                    # (B,H,W,F)
        z = zeta.reshape(-1, 1, 1, 1).expand(*f.shape[:3], 1)
        return torch.cat([f, z], dim=-1)

    def masked_gp_inputs(self, x, zeta, mask):
        """Flatten to masked pixels: returns (P, F+1). Loss/eval masking lives
        HERE — inputs are never zeroed (spec S1.5)."""
        return self.features(x, zeta)[mask]

    def zeta_ard_lengthscale(self):
        """Learned ARD lengthscale of the zeta input dim (last) — reported in
        every eval summary (spec S2.2)."""
        return float(self.gp.covar_module.base_kernel.lengthscale[0, -1])

    @torch.no_grad()
    def init_inducing_kmeans(self, dataset, n_pixels, iters, seed, device='cpu'):
        """k-means init of inducing points on features of the initial
        (untrained-FiLM) network (spec S2.2)."""
        rng = np.random.default_rng(int(seed))
        feats = []
        need = int(n_pixels)
        order = rng.permutation(len(dataset))
        for i in order:
            s = dataset[int(i)]
            f = self.masked_gp_inputs(s['x'][None].to(device),
                                      s['zeta'][None].to(device),
                                      s['mask'][None].to(device))
            if f.shape[0] == 0:
                continue
            take = min(f.shape[0], max(1, need // 4))
            sel = torch.from_numpy(rng.choice(f.shape[0], size=take, replace=False))
            feats.append(f[sel].cpu())
            if sum(x.shape[0] for x in feats) >= need:
                break
        pts = torch.cat(feats)[:need].to(torch.float32)
        if pts.shape[0] < self.gp.variational_strategy.inducing_points.shape[0]:
            raise ValueError(f"only {pts.shape[0]} pixels for k-means < M inducing")
        centers = _kmeans(pts, self.gp.variational_strategy.inducing_points.shape[0],
                          iters=iters, seed=seed)
        self.gp.variational_strategy.inducing_points.data.copy_(centers.to(device))
        return pts.shape[0]

    @torch.no_grad()
    def init_hyperparams_from_stats(self, y_mean, y_var, noise_frac=0.1):
        """Data-informed hyperparameter INIT (Sanaa autonomy ruling 2026-07-12):
        constant mean = target mean, kernel outputscale = (1-noise_frac)*var,
        likelihood noise = noise_frac*var — in the space the GP is trained in.
        HISTORY: applied in RAW target space (var ~7.6e3) this NaN'd instantly
        (float32 K_zz Cholesky, jitter 1e-6 absolute ~1e-10 relative — job
        1830733); with set_y_standardization the GP space has var(y_t)=1, so
        call this with (0.0, 1.0) — same information, conditioned kernels.
        INIT ONLY; all three remain trainable."""
        y_mean, y_var = float(y_mean), float(y_var)
        nf = float(noise_frac)
        if not (y_var > 0.0 and 0.0 < nf < 1.0):
            raise ValueError(f"bad init stats: var={y_var}, noise_frac={nf}")
        self.gp.mean_module.constant.data.fill_(y_mean)
        self.gp.covar_module.outputscale = (1.0 - nf) * y_var
        self.likelihood.noise = nf * y_var
        return {'gp_mean_init': y_mean, 'gp_space_var': y_var, 'noise_frac': nf,
                'outputscale_init': (1.0 - nf) * y_var, 'noise_init': nf * y_var}

    def film_norms(self):
        """||dgamma||, ||beta|| summaries for logging (spec S3.5). Probes the
        MLP at the training zeta range midpoint 0."""
        with torch.no_grad():
            w = self.cnn.film_mlp[-1]
            gb = self.cnn.film_mlp(torch.zeros(1, 1, device=w.weight.device))
            dg, b = gb.chunk(2, dim=-1)
            return float(dg.norm()), float(b.norm())
