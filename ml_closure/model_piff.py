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
                 film=True, film_hidden=16, cond_dim=1):
        super().__init__()
        self.channels = tuple(int(c) for c in channels)
        self.kernel = int(kernel)
        self.pad = self.kernel // 2
        self.film = bool(film)
        self.cond_dim = int(cond_dim)
        convs, cin = [], int(in_channels)
        for c in self.channels:
            convs.append(nn.Conv2d(cin, c, self.kernel, padding=0))
            cin = c
        self.convs = nn.ModuleList(convs)
        self.n_film = sum(self.channels)
        self.film_mlp = nn.Sequential(
            nn.Linear(self.cond_dim, int(film_hidden)), nn.SiLU(),
            nn.Linear(int(film_hidden), 2 * self.n_film))
        # identity init: zero output layer => dgamma = beta = 0 exactly (T3)
        nn.init.zeros_(self.film_mlp[-1].weight)
        nn.init.zeros_(self.film_mlp[-1].bias)

    @property
    def out_dim(self):
        return self.channels[-1]

    def forward(self, x, zeta):
        """x (B,4,H,W); zeta (B,) or (B,cond_dim) -> features (B,F,H,W)."""
        if self.film:
            gb = self.film_mlp(zeta.reshape(-1, self.cond_dim))  # (B, 2*n_film)
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

    class StructuralStudentTLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
        """Heteroscedastic Student-t observation model for the arm-F structural
        noise head (B-item, kurtosis ~1e4 verdict). scale^2(x) = het_noise(g)
        is passed in as the `noise` kwarg — EXACTLY the FixedNoiseGaussian-
        Likelihood contract, so the structural head (noise_a/noise_b/g2_scale)
        is reused verbatim, never duplicated. ONE extra learned dof parameter
        nu = softplus(raw_nu) + 2 (>2 so the variance exists). Training uses the
        _OneDimensionalLikelihood Gauss-Hermite quadrature for expected_log_prob
        (deterministic, low variance); marginal() draws likelihood samples (the
        leading sample dim is reduced by law-of-total-variance in
        PiffModel.predict_physical, the arm-C path). float64-safe: scale/nu are
        cast to the incoming function-sample dtype."""

        def __init__(self, nu_init=5.0):
            super().__init__()
            import math as _math
            nu0 = max(float(nu_init), 2.0 + 1.0e-3)
            self.register_parameter(                       # softplus(raw_nu)+2 == nu0
                'raw_nu', torch.nn.Parameter(
                    torch.tensor(_math.log(_math.expm1(nu0 - 2.0)))))

        @property
        def nu(self):
            return torch.nn.functional.softplus(self.raw_nu) + 2.0

        def forward(self, function_samples, *args, noise=None, **kwargs):
            if noise is None:
                raise ValueError("StructuralStudentTLikelihood needs the "
                                 "structural noise (scale^2) via the `noise` kwarg")
            scale = noise.clamp_min(1.0e-12).sqrt().to(function_samples.dtype)
            nu = self.nu.to(function_samples.dtype)
            return torch.distributions.StudentT(
                df=nu, loc=function_samples, scale=scale)

    class PiffMultitaskSVGP(gpytorch.models.ApproximateGP):
        """Coregionalized 2-task SVGP -- ICM / Hadamard multitask head (Sanaa GO
        2026-07-21). Kernel

            K((x,t),(x',t')) = k_RBF-ARD(x, x') * B[t, t']

        with B the learned coregionalization matrix supplied by gpytorch
        IndexKernel (rank r): B = W W^T + diag(v), W of shape (n_tasks, r). Read
        B two ways -- its DIAGONAL is each task's own OUTPUT SCALE (on top of the
        per-task y-standardization), its OFF-DIAGONAL is the LEARNED cross-task
        correlation, so a far-field prediction borrows strength from near-wall
        data through B[0,1] instead of being band-isolated (the two-band
        specialists' regression that motivated this model).

        WHY IndexKernel AND NOT LMCVariationalStrategy. Every pixel is a SINGLE
        task (near OR far), i.e. one scalar observation per input -- the textbook
        Hadamard-multitask case. IndexKernel keeps the target SCALAR and the
        predictive per-point scalar, so the whole ELBO / predict_physical /
        every diagnostic stays SHAPE-IDENTICAL to the single-task PiffSVGP.
        LMCVariationalStrategy instead emits ALL task outputs per point and needs
        a full (N, n_tasks) target matrix + MultitaskGaussianLikelihood, which
        would fork evaluate(), predict_physical() and every consumer. The task
        index is the LAST input column (integer 0/1); the first `feat_dim`
        columns are the shared GP features. Inducing points live in the shared
        FEATURE space (feature k-means); each carries a task label so K_zz keeps
        the same product form (VariationalStrategy calls forward on them)."""

        def __init__(self, inducing_points, feat_dim, n_tasks=2, coreg_rank=1):
            m = inducing_points.shape[0]
            var_dist = gpytorch.variational.CholeskyVariationalDistribution(m)
            strategy = gpytorch.variational.VariationalStrategy(
                self, inducing_points, var_dist, learn_inducing_locations=True)
            super().__init__(strategy)
            self.feat_dim = int(feat_dim)
            self.n_tasks = int(n_tasks)
            self.mean_module = gpytorch.means.ConstantMean()
            self.data_covar = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=self.feat_dim))
            self.task_covar = gpytorch.kernels.IndexKernel(
                num_tasks=self.n_tasks, rank=int(coreg_rank))

        def forward(self, x):
            xf = x[..., :self.feat_dim]
            # task index column. round()+clamp keeps the IndexKernel indices in
            # [0, n_tasks-1] and INTEGER even though the inducing points' task
            # column is a learnable location (learn_inducing_locations=True): a
            # no-op on the exact-integer DATA indices, and round() has zero
            # gradient so the inducing task labels stay frozen at their half/half
            # init (no drift out of range, no spurious learning of the label).
            ti = x[..., self.feat_dim:].round().clamp(0.0, float(self.n_tasks - 1))
            mean_x = self.mean_module(xf)
            # canonical gpytorch Hadamard product-kernel: evaluate the data and
            # task kernels and multiply (linear_operator 0.6.1 .mul)
            covar = self.data_covar(xf).mul(self.task_covar(ti))
            return gpytorch.distributions.MultivariateNormal(mean_x, covar)

        def coreg_matrix(self):
            """Learned n_tasks x n_tasks coregionalization matrix B (detached)."""
            tk = self.task_covar
            cf = tk.covar_factor
            return (cf @ cf.transpose(-1, -2) + torch.diag_embed(tk.var)).detach()

else:
    class PiffSVGP:  # placeholder that fails loudly (spec S2.2: no hand-rolled SVGP)
        def __init__(self, *a, **k):
            raise ImportError(
                f"gpytorch unavailable ({_GPYTORCH_ERR}); install gpytorch==1.13 into "
                f"qg-env-piff — do NOT hand-roll an SVGP (ML SPEC 01 S2.2). FLAG if "
                f"the cluster install fails.")

    class PiffMultitaskSVGP:  # same loud-failure placeholder as PiffSVGP
        def __init__(self, *a, **k):
            raise ImportError(str(_GPYTORCH_ERR))


class PiffModel(nn.Module):
    """FiLM-CNN + pointwise SVGP + Gaussian likelihood, end-to-end trainable."""

    def __init__(self, conf):
        super().__init__()
        mc = conf['model']
        # Ensemble conditioning (Sanaa ORDER 3, 2026-07-13). Both flags default
        # OFF -> exact pre-order model (all existing ckpts/configs load as-is).
        # use_zeta_dot: zeta_dot = d(zeta)/dt (table-Re, T_shed-smoothed,
        #   ensemble-std normalized via the recorded zdot_sd buffer) enters BOTH
        #   the FiLM conditioner (cond_dim 2) and the GP as an ARD dim — the
        #   wake-state-lag coordinate (ramp-up vs sine-down at equal zeta).
        # use_grad_feature: local |grad omega_bar|* (crop-computed by the
        #   dataset, train-mean normalized via g_scale) as one more ARD dim —
        #   arm F's structural sigma prior moved INTO the kernel.
        # use_lap_feature (Sanaa GO 2026-07-16): local |lap omega_bar|*
        #   (crop-computed by the dataset, train-mean normalized via
        #   lap_scale) as one more ARD dim — GP-side only, exactly like
        #   use_grad_feature (FiLM-CNN inputs untouched). Default OFF ->
        #   bit-identical to the pre-lap model.
        self.use_zeta_dot = bool(mc.get('use_zeta_dot', False))
        self.use_grad_feature = bool(mc.get('use_grad_feature', False))
        self.use_lap_feature = bool(mc.get('use_lap_feature', False))
        # use_wall_gate (wallv2, Sanaa GO 2026-07-18): the g and lap channels
        # are multiplied by exp(-max(sdf,0)/D) IN THE DATASET
        # (dataset_piff.RunData — upstream of every scale calibration, so
        # g_scale/g2_scale/lap_scale are stats of the GATED features). The
        # flag lives on the model so checkpoints round-trip it via the saved
        # conf, exactly like use_lap_feature; the model math itself is
        # unchanged. Default OFF -> bit-identical to the pre-wallv2 model.
        self.use_wall_gate = bool(mc.get('use_wall_gate', False))
        if self.use_wall_gate and not (self.use_grad_feature
                                       and self.use_lap_feature):
            raise ValueError("model.use_wall_gate=true requires "
                             "use_grad_feature and use_lap_feature both true")
        # arm-F structural noise prior, PRODUCTION port (Sanaa 2026-07-13
        # night): sigma^2(x) = softplus(a) + softplus(b) * s_feat(x), s_feat =
        # train-mean-normalized g^2 (g = the grad feature; requires
        # use_grad_feature). Only scalars a, b learn (the C/D/E ladder: any
        # FREE per-pixel sigma lets the ELBO buy out the signal). Init
        # softplus(a)=0.1, softplus(b)=0.01, b MOBILE (10x lr in the trainer);
        # sigma capped to [1e-3, 10] in std space (standardized units).
        self.noise_prior = str(mc.get('noise_prior', 'none'))
        if self.noise_prior not in ('none', 'structural'):
            raise ValueError(f"noise_prior {self.noise_prior!r}")
        if self.noise_prior == 'structural' and not self.use_grad_feature:
            raise ValueError("structural noise prior needs use_grad_feature")
        # observation model (B-item, Sanaa 2026-07-14): gaussian (default, zero
        # behavior change) or student_t (heavy-tailed, scale^2 = the SAME
        # structural het_noise head, one extra learned dof nu). student_t is
        # only meaningful with the structural scale it reuses.
        self.likelihood_type = str(mc.get('likelihood', 'gaussian'))
        if self.likelihood_type not in ('gaussian', 'student_t'):
            raise ValueError(f"model.likelihood {self.likelihood_type!r} "
                             f"(gaussian | student_t)")
        if self.likelihood_type == 'student_t' and self.noise_prior != 'structural':
            raise ValueError("model.likelihood=student_t requires "
                             "noise_prior=structural (scale^2 = het_noise)")
        # MULTITASK / coregionalized 2-task head (Sanaa GO 2026-07-21). Default
        # OFF -> the single-task PiffSVGP path below is BYTE-IDENTICAL (this
        # block does not execute, the extra buffers keep their scalar shape, no
        # task column is appended). ON -> task 0 = near-wall closure, task 1 =
        # far-field closure; the per-pixel task is DERIVED from input channel 3
        # (sdf_star) at forward time exactly like BlendedPiffModel, so nothing
        # threads a task tensor through the dataset / crop / diagnostics. The
        # single-assignment split is the MIDPOINT of the two specialist bands
        # (near sdf<=1.5D, far sdf>=1.0D -> split 1.25D): cleaner than
        # duplicating overlap pixels into both tasks (keeps the ELBO num_data a
        # clean pixel count) and the correlation is learned from the JOINT
        # objective + shared inducing/data-kernel, not from shared pixels.
        self.multitask = bool(mc.get('multitask', False))
        if self.multitask:
            if self.likelihood_type == 'student_t':
                raise ValueError("model.multitask + student_t not supported "
                                 "(per-task scale not defined); use gaussian")
            self.n_tasks = 2
            dc = conf.get('data', {}) or {}
            self.sdf_clip_D = float(dc.get('sdf_clip_D', 2.0))
            mtb = dc.get('multitask_bands', {}) or {}
            self.near_sdf_hi = float(mtb.get('near_sdf_hi', 1.5))
            self.far_sdf_lo = float(mtb.get('far_sdf_lo', 1.0))
            self.task_split_D = 0.5 * (self.near_sdf_hi + self.far_sdf_lo)
            if self.sdf_clip_D < self.task_split_D:
                raise ValueError(
                    f"multitask: data.sdf_clip_D={self.sdf_clip_D} < task split "
                    f"{self.task_split_D} D -- channel 3 saturates before the "
                    f"near/far boundary, so the per-pixel task would be wrong")
            self.coreg_rank = int(mc.get('coreg_rank', 1))
        cond_dim = 1 + int(self.use_zeta_dot)
        self.cnn = FiLMCNN(in_channels=int(mc['in_channels']),
                           channels=mc['channels'], kernel=int(mc['kernel']),
                           film=bool(mc['film']), film_hidden=int(mc['film_hidden']),
                           cond_dim=cond_dim)
        self.zeta_idx = self.cnn.out_dim                  # position of zeta in GP inputs
        self.gp_input_dim = (self.cnn.out_dim + 1 + int(self.use_zeta_dot)
                             + int(self.use_grad_feature)
                             + int(self.use_lap_feature))
        if self.multitask:
            # inducing points: shared FEATURE space + a task label column. Half
            # the M points are labelled task 0, half task 1, so both tasks are
            # represented in K_zz at init; init_inducing_kmeans overwrites the
            # feature columns with a feature k-means and keeps these labels.
            M = int(mc['n_inducing'])
            init_feat = torch.randn(M, self.gp_input_dim)
            tcol = torch.zeros(M, 1)
            tcol[M // 2:] = 1.0
            init_z = torch.cat([init_feat, tcol], dim=-1)
            self.gp = PiffMultitaskSVGP(init_z, feat_dim=self.gp_input_dim,
                                        n_tasks=self.n_tasks,
                                        coreg_rank=self.coreg_rank)
        else:
            init_z = torch.randn(int(mc['n_inducing']), self.gp_input_dim)
            self.gp = PiffSVGP(init_z)
        if gpytorch is None:
            raise ImportError(str(_GPYTORCH_ERR))
        if self.noise_prior == 'structural':
            import math as _math
            self.noise_a = torch.nn.Parameter(
                torch.tensor(_math.log(_math.expm1(0.1))))   # softplus -> 0.1
            self.noise_b = torch.nn.Parameter(
                torch.tensor(_math.log(_math.expm1(0.01))))  # softplus -> 0.01
            if self.likelihood_type == 'student_t':
                self.likelihood = StructuralStudentTLikelihood()
            else:
                self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                    noise=torch.full((1,), 0.1), learn_additional_noise=False)
        else:
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        # recorded, invertible y-standardization constants (2026-07-12 ruling
        # fallback): buffers -> saved in every checkpoint state_dict. Defaults
        # (0,1) = identity. The GP is trained on y_t = (y - y_mu)/y_sd;
        # predict_physical() inverts exactly. Spec-S1.2 target DEFINITION
        # untouched — this is an affine reparameterization for GP conditioning
        # (a large raw outputscale ~var(y) makes the float32 K_zz Cholesky
        # numerically singular: absolute jitter 1e-6 is ~1e-10 relative).
        # multitask: PER-TASK (mean, var) buffers (near stats from near pixels,
        # far from far) so the ~40x near/far amplitude gap is carried by the
        # task structure, not one shared scale. Single-task keeps the scalar
        # buffers verbatim -> pre-multitask ckpts round-trip byte-identically.
        if self.multitask:
            self.register_buffer('y_mu', torch.zeros(self.n_tasks))
            self.register_buffer('y_sd', torch.ones(self.n_tasks))
        else:
            self.register_buffer('y_mu', torch.zeros(()))
            self.register_buffer('y_sd', torch.ones(()))
        # recorded conditioning normalizations (same contract as y_mu/y_sd:
        # buffers -> in every ckpt; defaults = identity). persistent only when
        # the flag is on, so pre-ORDER-3 ckpts still load strict=True.
        self.register_buffer('zdot_sd', torch.ones(()), persistent=self.use_zeta_dot)
        self.register_buffer('g_scale', torch.ones(()), persistent=self.use_grad_feature)
        self.register_buffer('lap_scale', torch.ones(()), persistent=self.use_lap_feature)
        self.register_buffer('g2_scale', torch.ones(()),
                             persistent=self.noise_prior == 'structural')

    def het_noise(self, g_masked):
        """Structural per-pixel noise in STANDARDIZED variance units:
        sigma^2 = softplus(a) + softplus(b) * s_feat, s_feat = g^2/g2_scale
        (unit train mean); sigma clamped to [1e-3, 10] std space (arm F)."""
        s_feat = (g_masked ** 2) / self.g2_scale.to(g_masked.dtype)
        s2 = (torch.nn.functional.softplus(self.noise_a).to(g_masked.dtype)
              + torch.nn.functional.softplus(self.noise_b).to(g_masked.dtype)
              * s_feat)
        # the VALIDATED arm-F band, VARIANCE units [1e-3, 10] (G4 2026-07-13:
        # my first port squared it to [1e-6,100] — a 3x looser sigma cap that
        # reopens the buy-out channel under b@10x-lr)
        return s2.clamp(1.0e-3, 10.0)

    def set_conditioning_stats(self, zdot_sd=None, g_scale=None, lap_scale=None):
        """Recorded normalization constants for the conditioning inputs, from
        exact train-split stats. Returns the manifest dict."""
        out = {}
        if self.use_zeta_dot:
            zdot_sd = float(zdot_sd)
            if not zdot_sd > 0.0:
                raise ValueError(f"zdot_sd={zdot_sd}: ensemble has no zeta_dot "
                                 f"variance (const-only pool?) — use_zeta_dot "
                                 f"is meaningless here, refuse loudly")
            self.zdot_sd.fill_(zdot_sd)
            out['zdot_sd'] = zdot_sd
        if self.use_grad_feature:
            g_scale = float(g_scale)
            if not g_scale > 0.0:
                raise ValueError(f"bad g_scale={g_scale}")
            self.g_scale.fill_(g_scale)
            out['g_scale'] = g_scale
        if self.use_lap_feature:
            lap_scale = float(lap_scale)
            if not lap_scale > 0.0:
                raise ValueError(f"bad lap_scale={lap_scale}")
            self.lap_scale.fill_(lap_scale)
            out['lap_scale'] = lap_scale
        return out

    def is_student_t(self):
        return getattr(self, 'likelihood_type', 'gaussian') == 'student_t'

    def student_nu(self):
        """Learned degrees-of-freedom nu = softplus(raw_nu)+2 (float), or None
        for the Gaussian likelihood."""
        return float(self.likelihood.nu) if self.is_student_t() else None

    @torch.no_grad()
    def set_student_nu(self, nu_value):
        """Set the Student-t dof from a value (INIT only; raw_nu stays
        learnable). Inverts softplus: raw_nu = log(expm1(nu-2))."""
        if not self.is_student_t():
            raise ValueError("set_student_nu on a non-student_t likelihood")
        import math as _math
        nu0 = max(float(nu_value), 2.0 + 1.0e-3)
        self.likelihood.raw_nu.data.fill_(_math.log(_math.expm1(nu0 - 2.0)))
        return float(self.likelihood.nu)

    def student_scale(self, g_masked):
        """Physical Student-t SCALE field = sqrt(het_noise(g)) * y_sd (the
        structural scale mapped out of standardized space). NLL/coverage under
        the t use this + student_nu(); the predictive VARIANCE is
        scale^2 * nu/(nu-2) (finite for nu>2), returned by predict_physical."""
        s = self.het_noise(g_masked).clamp_min(1.0e-12).sqrt()
        return s * self.y_sd.to(g_masked.dtype)

    def set_noise_feature_scale(self, g2_scale):
        """Recorded train-mean of g^2 (s_feat normalization, structural prior)."""
        g2_scale = float(g2_scale)
        if not g2_scale > 0.0:
            raise ValueError(f"bad g2_scale={g2_scale}")
        self.g2_scale.fill_(g2_scale)
        return {'g2_scale': g2_scale}

    def set_y_standardization(self, y_mean, y_var):
        """Set the recorded standardization constants from exact train-target
        stats. Multitask: y_mean / y_var are length-n_tasks (per-task) and fill
        the per-task buffers; single-task takes scalars -> byte-identical.
        Returns them for the run/checkpoint manifest."""
        if getattr(self, 'multitask', False):
            ym = np.asarray(y_mean, dtype=np.float64).reshape(-1)
            yv = np.asarray(y_var, dtype=np.float64).reshape(-1)
            if ym.shape != (self.n_tasks,) or yv.shape != (self.n_tasks,):
                raise ValueError(f"multitask y-standardization needs "
                                 f"{self.n_tasks} per-task stats, got "
                                 f"means {ym.shape} vars {yv.shape}")
            if not np.all(yv > 0.0):
                raise ValueError(f"bad per-task target var: {yv.tolist()}")
            self.y_mu.copy_(torch.as_tensor(ym, dtype=self.y_mu.dtype))
            self.y_sd.copy_(torch.as_tensor(np.sqrt(yv), dtype=self.y_sd.dtype))
            return {'y_mean': ym.tolist(), 'y_std': np.sqrt(yv).tolist()}
        y_mean, y_var = float(y_mean), float(y_var)
        if not y_var > 0.0:
            raise ValueError(f"bad target stats: var={y_var}")
        self.y_mu.fill_(y_mean)
        self.y_sd.fill_(y_var ** 0.5)
        return {'y_mean': y_mean, 'y_std': y_var ** 0.5}

    def standardize_y(self, y, task=None):
        """Standardize targets. Multitask: PER-PIXEL by the pixel's own task
        (task = the last column of the pixel's GP inputs, an integer 0/1);
        single-task ignores `task` -> byte-identical."""
        if getattr(self, 'multitask', False):
            if task is None:
                raise ValueError("multitask standardize_y needs the per-pixel task")
            t = task.long()
            return (y - self.y_mu[t]) / self.y_sd[t]
        return (y - self.y_mu) / self.y_sd

    def predict_physical(self, gpin, g_masked=None):
        """Likelihood-included predictive (mean, var) in PHYSICAL target units
        (standardization inverted exactly). Structural noise prior: pass the
        masked grad feature (same pixel order as gpin) so sigma^2(x) enters
        the predictive. Non-Gaussian likelihoods (e.g. StudentT, arm-C B-item
        experiment): gpytorch's marginal is Monte-Carlo sampled -> moments
        carry a leading sample dim; reduce by the law of total variance
        (E[var] + Var[mean]) so callers always get 1-D."""
        if self.noise_prior == 'structural':
            if g_masked is None:
                raise ValueError("structural noise prior: predict_physical "
                                 "needs g_masked")
            pred = self.likelihood(self.gp(gpin), noise=self.het_noise(g_masked))
        else:
            pred = self.likelihood(self.gp(gpin))
        mu, var = pred.mean, pred.variance
        if mu.dim() > 1:
            var = var.mean(dim=0) + mu.var(dim=0)
            mu = mu.mean(dim=0)
        if getattr(self, 'multitask', False):
            # invert each pixel's OWN task standardization (task = last gpin col)
            t = gpin[..., -1].long()
            sd = self.y_sd[t]
            return mu * sd + self.y_mu[t], var * sd * sd
        return mu * self.y_sd + self.y_mu, var * self.y_sd * self.y_sd

    def features(self, x, zeta, zeta_dot=None, g=None, lap=None):
        """(B,4,H,W),(B,)[,(B,)][,(B,H,W)][,(B,H,W)] -> per-pixel GP inputs
        (B,H,W,F+1[+zdot][+grad][+lap]). Column order: F CNN features, zeta,
        then zeta_dot (normalized), then |grad omega_bar|* (normalized),
        then |lap omega_bar|* (normalized)."""
        if self.use_zeta_dot:
            if zeta_dot is None:
                raise ValueError("use_zeta_dot=true but zeta_dot not supplied")
            zd = zeta_dot / self.zdot_sd
            cond = torch.stack([zeta, zd], dim=-1)   # (B,2) FiLM conditioner
        else:
            cond = zeta
        f = self.cnn(x, cond)                        # (B,F,H,W)
        f = f.permute(0, 2, 3, 1)                    # (B,H,W,F)
        cols = [f, zeta.reshape(-1, 1, 1, 1).expand(*f.shape[:3], 1)]
        if self.use_zeta_dot:
            cols.append(zd.reshape(-1, 1, 1, 1).expand(*f.shape[:3], 1))
        if self.use_grad_feature:
            if g is None:
                raise ValueError("use_grad_feature=true but g not supplied")
            gn = g / self.g_scale
            if self.use_wall_gate:
                # wallv2 (probe2 2026-07-19): the GATED g distribution is
                # extreme (most pixels ~0, near-wall tail to ~184x scale) --
                # linear input made K_zz numerically NotPSD (eig -1.9e-4 >
                # gpytorch's 1e-6 jitter ceiling) = the ep-0 NaN. Same
                # log1p tail compression as lap (Sanaa ruling 2026-07-16).
                # Applied ONLY under use_wall_gate: every pre-wallv2 ckpt
                # keeps the linear transform byte-identically.
                gn = torch.log1p(gn)
            cols.append(gn.unsqueeze(-1))                   # (B,H,W,1)
        if self.use_lap_feature:
            if lap is None:
                raise ValueError("use_lap_feature=true but lap not supplied")
            # log1p tail compression (Sanaa ruling 2026-07-16, run 1836219
            # postmortem): standardized |lap| is heavy-tailed (observed range
            # 0..332); raw it makes quantile inducing columns kernel-orthogonal
            # to typical data and defeats the near-inert warm init. log1p maps
            # the range to ~[0, 5.9] so lengthscale-20 is inert for ALL pixels.
            cols.append(torch.log1p(lap / self.lap_scale).unsqueeze(-1))
        if getattr(self, 'multitask', False):
            # append the per-pixel task index as the LAST GP-input column,
            # derived from input channel 3 (sdf_star) -- see _pixel_task
            cols.append(self._pixel_task(x).unsqueeze(-1))
        return torch.cat(cols, dim=-1)

    def _pixel_task(self, x):
        """Per-pixel task index (0 near / 1 far) recovered EXACTLY from input
        channel 3 (sdf_star = clip(sdf, +/-clipD)/clipD): s = sdf/D = sdf_star *
        sdf_clip_D. near = s <= task_split_D (task 0), far = s > task_split_D
        (task 1). Exact at the split because task_split_D < sdf_clip_D
        (asserted in __init__). Mirrors BlendedPiffModel._sdf_over_D."""
        s = x[:, 3] * self.sdf_clip_D                    # (B,H,W)
        return (s > self.task_split_D).to(x.dtype)       # 0.0 near / 1.0 far

    def _base_kernel(self):
        """RBF-ARD base kernel of the GP -- data_covar in multitask, covar_module
        otherwise (the two heads differ)."""
        return (self.gp.data_covar.base_kernel if getattr(self, 'multitask', False)
                else self.gp.covar_module.base_kernel)

    def masked_gp_inputs(self, x, zeta, mask, zeta_dot=None, g=None, lap=None):
        """Flatten to masked pixels: returns (P, gp_input_dim). Loss/eval
        masking lives HERE — inputs are never zeroed (spec S1.5)."""
        return self.features(x, zeta, zeta_dot=zeta_dot, g=g, lap=lap)[mask]

    def zeta_ard_lengthscale(self):
        """Learned ARD lengthscale of the zeta input dim — reported in every
        eval summary (spec S2.2). Indexed by position (zeta is no longer the
        last dim when the ORDER-3 conditioning flags are on)."""
        return float(self._base_kernel().lengthscale[0, self.zeta_idx])

    def ard_lengthscales(self):
        """Named ARD lengthscales of the conditioning dims (acceptance
        predictions, ORDER 3d): zeta always; zeta_dot / grad / lap when
        enabled."""
        ls = self._base_kernel().lengthscale[0]
        out = {'zeta': float(ls[self.zeta_idx])}
        i = self.zeta_idx + 1
        if self.use_zeta_dot:
            out['zeta_dot'] = float(ls[i]); i += 1
        if self.use_grad_feature:
            out['grad'] = float(ls[i]); i += 1
        if self.use_lap_feature:
            out['lap'] = float(ls[i])
        return out

    def coreg_report(self):
        """Learned coregionalization summary (multitask only, for logging):
        per-task output variance B[t,t] and the cross-task correlation
        B[0,1]/sqrt(B[0,0]B[1,1]). None single-task."""
        if not getattr(self, 'multitask', False):
            return None
        B = self.gp.coreg_matrix().cpu()
        d = torch.diagonal(B).clamp_min(1e-30)
        corr = float(B[0, 1] / (d[0] * d[1]).sqrt())
        return {'task_var': [float(d[0]), float(d[1])], 'cross_corr': corr}

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
            f = self.masked_gp_inputs(
                s['x'][None].to(device), s['zeta'][None].to(device),
                s['mask'][None].to(device),
                zeta_dot=(s['zeta_dot'][None].to(device) if self.use_zeta_dot else None),
                g=(s['g'][None].to(device) if self.use_grad_feature else None),
                lap=(s['lap'][None].to(device) if self.use_lap_feature else None))
            if f.shape[0] == 0:
                continue
            take = min(f.shape[0], max(1, need // 4))
            sel = torch.from_numpy(rng.choice(f.shape[0], size=take, replace=False))
            feats.append(f[sel].cpu())
            if sum(x.shape[0] for x in feats) >= need:
                break
        pts = torch.cat(feats)[:need].to(torch.float32)
        M = self.gp.variational_strategy.inducing_points.shape[0]
        if getattr(self, 'multitask', False):
            # k-means over the SHARED FEATURE columns only (drop the appended
            # task-index column), then re-attach half/half task labels so both
            # tasks are covered in the inducing set (the ARD kernel sees the
            # coordinate values; the task column is handled by IndexKernel).
            pts = pts[:, :self.gp.feat_dim]
        if pts.shape[0] < M:
            raise ValueError(f"only {pts.shape[0]} pixels for k-means < M inducing")
        centers = _kmeans(pts, M, iters=iters, seed=seed)
        if getattr(self, 'multitask', False):
            tcol = torch.zeros(centers.shape[0], 1, dtype=centers.dtype)
            tcol[centers.shape[0] // 2:] = 1.0
            centers = torch.cat([centers, tcol], dim=-1)
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
        if getattr(self, 'multitask', False):
            # data kernel outputscale; the IndexKernel coregionalization matrix
            # carries the residual per-task scale + cross-task correlation
            self.gp.data_covar.outputscale = (1.0 - nf) * y_var
        else:
            self.gp.covar_module.outputscale = (1.0 - nf) * y_var
        if not self.is_student_t():          # StudentT carries no scalar .noise
            self.likelihood.noise = nf * y_var
        return {'gp_mean_init': y_mean, 'gp_space_var': y_var, 'noise_frac': nf,
                'outputscale_init': (1.0 - nf) * y_var, 'noise_init': nf * y_var}

    def film_norms(self):
        """||dgamma||, ||beta|| summaries for logging (spec S3.5). Probes the
        MLP at the training zeta range midpoint 0."""
        with torch.no_grad():
            w = self.cnn.film_mlp[-1]
            gb = self.cnn.film_mlp(torch.zeros(1, self.cnn.cond_dim,
                                               device=w.weight.device))
            dg, b = gb.chunk(2, dim=-1)
            return float(dg.norm()), float(b.norm())
