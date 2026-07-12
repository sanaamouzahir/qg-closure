"""
T1–T6 gate tests (ML SPEC 01 S5). T7 is a jobscript (piff_smoke_job.sh), not a
test function. Run from ml_closure/ (flat sibling imports):

    CPU arm (all.q):    pytest tests_piff.py -k "t1 or t2 or t3 or t4" -v
    GPU arm (ibgpu.q):  pytest tests_piff.py -k "t5 or t6" -v

T1/T6 need the FPC-const Step-0 canonical artifacts (make_dataset_manifest.py);
they skip with a loud message if absent. T5/T6 need gpytorch (qg-env-piff).
"""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

import dataset_piff as dp
from model_piff import FiLMCNN, gpytorch

HERE = Path(__file__).resolve().parent
CONF = dp.load_conf(HERE / 'conf_piff.yaml')
RUN0 = Path(CONF['data']['runs'][0])
HAVE_DATA = (RUN0 / 'DATASET_MANIFEST.md').exists() and \
            (RUN0 / f"DNS_LES_s{int(CONF['data']['scale'])}.npz").exists()

needs_data = pytest.mark.skipif(
    not HAVE_DATA, reason=f"Step-0 canonical artifacts missing in {RUN0} — "
                          f"run make_dataset_manifest.py first")
needs_gpytorch = pytest.mark.skipif(
    gpytorch is None, reason="gpytorch missing — install into qg-env-piff (never hand-roll)")


def small_conf(**data_over):
    conf = copy.deepcopy(CONF)
    small = {'crops_per_epoch_train': 32, 'crops_per_epoch_val': 16}
    small.update(data_over)  # explicit overrides win; no duplicate-kwarg collision
    conf['data'].update(small)
    return conf


# --------------------------------------------------------------------------- #
@needs_data
def test_t1_loader_regression_hash():
    """Fixed seed -> identical batch tensors across two independent builds."""
    conf = small_conf()
    h = []
    for _ in range(2):
        runs = dp.build_runs(conf)
        ds = dp.PiffCropDataset(runs, 'train', conf, seed=1234)
        h.append(ds.epoch_hash())
    assert h[0] == h[1], f"loader not deterministic: {h[0][:16]} != {h[1][:16]}"
    # and a different epoch/seed must NOT collide
    runs = dp.build_runs(conf)
    ds2 = dp.PiffCropDataset(runs, 'train', conf, seed=1234)
    ds2.set_epoch(1)
    assert ds2.epoch_hash() != h[0]


def test_t2_normalization_identity():
    """The same nondimensional state realized at two different U(t) gives
    identical normalized channels up to float tolerance (spec S1.2: U from the
    table, no per-sample statistics)."""
    rng = np.random.default_rng(0)
    D, U1, U2 = 1.2566, 2.0, 3.1
    om1 = rng.standard_normal((64, 64)); u1 = rng.standard_normal((64, 64))
    v1 = rng.standard_normal((64, 64)); pi1 = rng.standard_normal((64, 64))
    s = U2 / U1
    a = dp.normalize_fields(om1, u1, v1, pi1, U1, D)
    b = dp.normalize_fields(om1 * s, u1 * s, v1 * s, pi1 * s * s, U2, D)
    for x, yv, name in zip(a, b, ('omega*', 'u*', 'v*', 'Pi*')):
        assert np.allclose(x, yv, rtol=1e-12, atol=1e-12), f"{name} not U-invariant"


def test_t3_film_identity_init_exact_zero():
    """At init, FiLM-on vs FiLM-frozen outputs differ by exactly 0."""
    torch.manual_seed(0)
    net_on = FiLMCNN(film=True)
    net_off = FiLMCNN(film=False)
    net_off.load_state_dict(net_on.state_dict())
    x = torch.randn(2, 4, 64, 64)
    zeta = torch.tensor([0.7, -1.3])
    with torch.no_grad():
        d = (net_on(x, zeta) - net_off(x, zeta)).abs().max()
    assert float(d) == 0.0, f"FiLM not identity at init: max|diff| = {float(d)}"


def test_t4_periodic_pad_equivariance():
    """Cyclic translation of the input cyclically translates the CNN output
    (tolerance 1e-5)."""
    torch.manual_seed(1)
    net = FiLMCNN(film=True)
    x = torch.randn(1, 4, 64, 64)
    zeta = torch.tensor([0.3])
    sy, sx = 7, 13
    with torch.no_grad():
        ref = torch.roll(net(x, zeta), shifts=(sy, sx), dims=(2, 3))
        out = net(torch.roll(x, shifts=(sy, sx), dims=(2, 3)), zeta)
    err = float((ref - out).abs().max())
    assert err <= 1e-5, f"equivariance violated: max|diff| = {err:.3e}"


# --------------------------------------------------------------------------- #
@needs_gpytorch
def test_t5_svgp_noise_recovery_and_coverage():
    """Synthetic 1D-feature GP regression with known noise: recovered noise
    within 20%, +/-2 sigma coverage in [90, 99]% (spec S5)."""
    from model_piff import PiffSVGP
    torch.manual_seed(0)
    np.random.seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n, sigma_true = 4000, 0.1
    x = torch.rand(n, 1) * 6 - 3
    y = torch.sin(2.0 * x[:, 0]) + sigma_true * torch.randn(n)
    x, y = x.to(device), y.to(device)

    z0 = x[torch.randperm(n)[:64]].clone()
    gp = PiffSVGP(z0).to(device)
    lik = gpytorch.likelihoods.GaussianLikelihood().to(device)
    mll = gpytorch.mlls.VariationalELBO(lik, gp, num_data=n)
    opt = torch.optim.Adam(list(gp.parameters()) + list(lik.parameters()), lr=1e-2)
    gp.train(); lik.train()
    for _ in range(600):
        opt.zero_grad()
        loss = -mll(gp(x), y)
        loss.backward()
        opt.step()

    gp.eval(); lik.eval()
    with torch.no_grad():
        pred = lik(gp(x))
        mu, sd = pred.mean, pred.variance.sqrt()
        cov2 = float(((y - mu).abs() <= 2.0 * sd).float().mean())
    noise = float(lik.noise.sqrt())
    assert abs(noise - sigma_true) / sigma_true <= 0.20, \
        f"noise {noise:.4f} vs true {sigma_true} (>20% off)"
    assert 0.90 <= cov2 <= 0.99, f"2-sigma coverage {cov2:.3f} outside [0.90, 0.99]"


@needs_data
@needs_gpytorch
def test_t6_overfit_500_crops():
    """Capacity check: 500 crops from FPC-const, train R2 > 0.95 within 50
    epochs (spec S5)."""
    from model_piff import PiffModel
    from train_piff import batches, evaluate
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    conf = small_conf(crops_per_epoch_train=500)
    torch.manual_seed(0)
    np.random.seed(0)
    runs = dp.build_runs(conf)
    ds = dp.PiffCropDataset(runs, 'train', conf, seed=0)   # fixed table: pure overfit
    model = PiffModel(conf).to(device)
    model.init_inducing_kmeans(ds, 10000, int(conf['model']['kmeans_iters']),
                               seed=0, device=device)
    # same path as train_piff.py (2026-07-12 ruling, standardization fallback):
    # recorded y-standardization + data-informed init in standardized space
    ystats = dp.target_stats(runs, 'train', conf)
    std_const = model.set_y_standardization(ystats['mean'], ystats['var'])
    hyper0 = model.init_hyperparams_from_stats(
        0.0, 1.0, noise_frac=dp._f(conf['train']['init_noise_frac']))
    print(f"[t6] y-standardization {std_const} + GP init: {hyper0}")
    n_pix = int(sum(ds[i]['mask'].sum() for i in range(len(ds))))
    mll = gpytorch.mlls.VariationalELBO(model.likelihood, model.gp, num_data=n_pix)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    r2 = -np.inf
    for ep in range(50):
        model.train()
        for b in batches(ds, int(conf['data']['batch_crops'])):
            gpin = model.masked_gp_inputs(b['x'].to(device), b['zeta'].to(device),
                                          b['mask'].to(device))
            yt = b['y'].to(device)[b['mask'].to(device)]
            opt.zero_grad()
            loss = -mll(model.gp(gpin), model.standardize_y(yt))
            loss.backward()
            opt.step()
        m = evaluate(model, ds, device, int(conf['train']['gp_chunk']))
        r2 = m['r2']
        print(f"[t6] ep {ep:02d} train R2 {r2:.4f} RMSE {m['rmse']:.3e}")
        if r2 > 0.95:
            break
    assert r2 > 0.95, f"overfit failed: train R2 {r2:.4f} <= 0.95 after 50 epochs"
