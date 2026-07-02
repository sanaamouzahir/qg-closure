"""
rollout_timed_pareto.py
=======================

On-the-fly, *timed* multi-step rollout for the temporal (step-size) closure,
forced-turbulence + cheap_deriv edition. Everything is generated from a single
shared IC -- nothing is loaded -- because the point is to TIME each scheme:

    truth   : M*K RK4 steps of h_fine = Delta_T/K      (fine reference)
    bare    : M     AB2CN2 steps of Delta_T            (coarse, no closure)
    closure : M     AB2CN2 steps of Delta_T, each + e_anal + e_NN  (cheap_deriv)

All three start from the same omega_0 (loaded from the dataset's packed IC).
We store only SPARSE checkpoints at shared physical times (the fine run is
M*K steps; we never keep every frame), and compare bare/closure to truth at
those times. Robust wall-clock timing (CUDA-synced, warmup excluded) gives the
per-scheme cost, and --pareto adds a bare-at-many-dt sweep to trace the
cost-vs-accuracy front with the closure point overlaid.

Closure assembly
----------------
e_anal = -(Delta_T^3)(1-1/K^2) * (1/12)(L^3 omega + L^2 N)         [analytic]
e_NN   = -(Delta_T^3)(1-1/K^2) * f_NN
  cheap_deriv: model outputs [Ndot, Nddot, ...]; f_NN = (1/12)(L*Ndot - 5*Nddot)
  bilinear:    model outputs f_NN directly (denormalized)

Usage
-----
  python rollout_timed_pareto.py --run-dir <run> --root-dir <dataset_root> \
      --n-steps 200 --n-checkpoints 10 --ic-index 100 --device cuda \
      [--Delta-T-override 5e-3] [--pareto] [--out-dir .]
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class _NoProf:
    """No-op profiler -- the default threaded through the hot path so the clean
    timed loop and the bare leg pay ZERO instrumentation cost (mark() is a single
    Python attr-return)."""
    enabled = False
    def mark(self, label):        # noqa: D401  (intentional no-op)
        return
    def step_begin(self):
        return
    def step_end(self):
        return
_NOPROF = _NoProf()
def _find_training_dir():
    here = Path(__file__).resolve().parent
    for ancestor in [here, *here.parents]:
        if (ancestor / 'dataset.py').exists():
            return ancestor
    return here
_training_dir = _find_training_dir()
sys.path.insert(0, str(_training_dir))
sys.path.insert(0, str(Path(__file__).resolve().parent))


# =========================================================================== #
# Solver primitives -- verbatim from build_training_data_fixD_v2.py /          #
# rollout_multistep_comparison.py (dealiased pseudo-spectral Jacobian).        #
# =========================================================================== #
def J_phys(psi_phys, omega_phys, derivative):
    from qg.solver.opt.basis import to_spectral, to_physical
    psih = to_spectral(psi_phys); qh = to_spectral(omega_phys)
    u = to_physical(-1 * derivative.dy * psih)
    v = to_physical(+1 * derivative.dx * psih)
    q = to_physical(qh)
    uq_h = to_spectral(u * q).clone()
    vq_h = to_spectral(v * q).clone()
    derivative.dealias(uq_h); derivative.dealias(vq_h)
    j_hat = derivative.dx * uq_h + derivative.dy * vq_h
    return to_physical(j_hat)


def L_op(omega_phys, L_hat):
    from qg.solver.opt.basis import to_spectral, to_physical
    return to_physical(L_hat * to_spectral(omega_phys))


def _dealias_mul(yh, derivative):
    """Dealias by multiplying by a cached 0/1 keep-mask -- numerically identical to
    derivative.dealias (zeros the same |k|>k_cut modes) but elementwise, avoiding the
    boolean-scatter assignment which forces a GPU sync and is much slower per step."""
    keep = getattr(derivative, '_keep_mask', None)
    if keep is None or keep.device != yh.device:
        # build on yh's device (alias_mask may live on CPU); rebuild if a cached
        # mask is on the wrong device.
        keep = (~derivative.alias_mask).to(device=yh.device, dtype=torch.float64)
        derivative._keep_mask = keep
    return yh * keep


def _N_core(qh, derivative, F_hat, prof=_NOPROF):
    """Minimal one-Jacobian N-eval -- 5 FFTs, N kept SPECTRAL (the AB2CN2 step uses
    it spectral, so no iFFT back). psi/u/v stay spectral until the physical product;
    F is added in spectral (F_hat precomputed once). No physical<->spectral round
    trips (the old path round-tripped psi, omega, and j -> 11 FFTs). Returns
    (N_hat, omega_phys, psih_spectral)."""
    from qg.solver.opt.basis import to_spectral, to_physical
    prof.mark('N_ifft')                                  # 3 iFFTs (psih multiply is free)
    psih = derivative.inv_laplacian * qh                 # spectral, free
    u = to_physical(-1 * derivative.dy * psih)           # 1 iFFT
    v = to_physical(+1 * derivative.dx * psih)           # 1 iFFT
    omega = to_physical(qh)                              # 1 iFFT
    prof.mark('N_prodfft')                               # 2 physical products + 2 FFTs
    uq_h = to_spectral(u * omega).clone()                # 1 FFT
    vq_h = to_spectral(v * omega).clone()                # 1 FFT
    prof.mark('N_spectral')                              # 2 dealias-muls + spectral combine
    uq_h = _dealias_mul(uq_h, derivative)
    vq_h = _dealias_mul(vq_h, derivative)
    j_hat = derivative.dx * uq_h + derivative.dy * vq_h  # spectral, free
    N_hat = (F_hat - j_hat) if F_hat is not None else (-1.0 * j_hat)
    return N_hat, omega, psih


def N_spectral(qh, derivative, F_hat):
    """Bare path: N only, MINIMAL 5 FFTs (psi never iFFT'd)."""
    return _N_core(qh, derivative, F_hat)[0]


def N_spectral_fields(qh, derivative, F_hat, prof=_NOPROF):
    """Closure path: also returns omega, psi PHYSICAL for the NN stencil. The psi
    iFFT (+1) is the closure's one genuine extra over bare -> 6 FFTs vs bare's 5."""
    from qg.solver.opt.basis import to_physical
    N_hat, omega, psih = _N_core(qh, derivative, F_hat, prof=prof)
    prof.mark('N_psi')
    psi = to_physical(psih)                              # +1 iFFT (NN stencil)
    return N_hat, omega, psi


def build_L_hat(derivative, nu, mu, beta):
    L_hat = nu * derivative.laplacian - mu
    if beta != 0.0:
        L_hat = L_hat - beta * derivative.dx * derivative.inv_laplacian
    return L_hat


def ab2cn2_step_spectral(qh_n, qh_nm1, dt, derivative, L_hat, F_phys):
    from qg.solver.opt.basis import to_spectral, to_physical
    def N_at_qh(qh):
        psi = to_physical(derivative.inv_laplacian * qh)
        omega = to_physical(qh)
        N_phys = -1.0 * J_phys(psi, omega, derivative)
        if F_phys is not None:
            N_phys = N_phys + F_phys
        return to_spectral(N_phys)
    Nh_n = N_at_qh(qh_n); Nh_nm1 = N_at_qh(qh_nm1)
    AB2_Nh = 1.5 * Nh_n - 0.5 * Nh_nm1
    rhs_hat = qh_n + dt * (0.5 * L_hat * qh_n + AB2_Nh)
    denom_hat = 1.0 - 0.5 * dt * L_hat
    return rhs_hat / denom_hat


def rk4_step(omega, dt, derivative, L_hat, F_phys):
    def rhs(om):
        from qg.solver.opt.basis import to_spectral, to_physical
        psi = to_physical(derivative.inv_laplacian * to_spectral(om))
        N = -1.0 * J_phys(psi, om, derivative)
        if F_phys is not None:
            N = N + F_phys
        return L_op(om, L_hat) + N
    k1 = rhs(omega); k2 = rhs(omega + 0.5 * dt * k1)
    k3 = rhs(omega + 0.5 * dt * k2); k4 = rhs(omega + dt * k3)
    return omega + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def E_analytical_phys(omega_phys, derivative, L_hat, F_phys):
    from qg.solver.opt.basis import to_spectral, to_physical
    qh = to_spectral(omega_phys)
    L3_omega = to_physical(L_hat ** 3 * qh)
    psi = to_physical(derivative.inv_laplacian * qh)
    N = -1.0 * J_phys(psi, omega_phys, derivative)
    if F_phys is not None:
        N = N + F_phys
    L2_N = to_physical(L_hat ** 2 * to_spectral(N))
    return (1.0 / 12.0) * (L3_omega + L2_N)


def psi_from_omega(omega_phys, derivative):
    from qg.solver.opt.basis import to_spectral, to_physical
    return to_physical(derivative.inv_laplacian * to_spectral(omega_phys))


# =========================================================================== #
# Forcing (forced turb: time-independent when C=F=0)                           #
# =========================================================================== #
def build_forcing(grid, fc, device, dtype):
    """F = A cos(B x + C t) + D cos(E y + F t); we require C=F=0 (steady)."""
    if fc is None:
        return None
    A, B, C, D, E, Ff = (float(fc.get(k, 0.0)) for k in ('A', 'B', 'C', 'D', 'E', 'F'))
    if C != 0.0 or Ff != 0.0:
        raise ValueError("time-dependent forcing (C or F != 0) not supported here")
    x = torch.linspace(0, grid.Lx, grid.Nx, device=device, dtype=dtype)
    y = torch.linspace(0, grid.Ly, grid.Ny, device=device, dtype=dtype)
    X = x[None, :]; Y = y[:, None]
    w = A * torch.cos(B * X) + D * torch.cos(E * Y)   # (Ny, Nx)
    return w[None]                                     # (1, Ny, Nx)


# =========================================================================== #
# Model load + closure assembly                                                #
# =========================================================================== #
def load_model(run_dir, cfg, n_in, manifest, device, dt_rollout, dtype=torch.float64, nn_float64=False):
    model_name = cfg.get('model', 'bilinear_closure')
    ckpt = run_dir / 'model_best.pt'
    if not ckpt.exists():
        ckpt = run_dir / 'model_last.pt'
    state = torch.load(ckpt, map_location=device, weights_only=False)
    sd = state.get('model_state', state.get('state_dict', state))

    if model_name in ('cheap_deriv', 'deriv_closure'):
        from model_deriv_closure import build_model
        Nx = int(manifest['Nx']); Ny = int(manifest['Ny'])
        dx = float(manifest['Lx']) / Nx; dy = float(manifest['Ly']) / Ny
        # TimeFD divides snapshot differences by dt; snapshots are at the
        # ROLLOUT Delta_T spacing, so the operator's dt must match it.
        dt = dt_rollout
        # infer grad_kernel from the trained stencil shape if present
        gk = cfg.get('grad_kernel', None)
        for k, v in sd.items():
            if k.endswith('grad.wx'):        # fused stencil parameter carries the width
                gk = int(v.shape[-1])
        model = build_model('cheap_deriv', in_channels=n_in,
                            out_orders=cfg.get('out_orders', 3),
                            n_time=n_in // 2,
                            grad_kernel=gk if gk else 3,
                            dt=dt, dx=dx, dy=dy,
                            physics_init=cfg.get('physics_init', True))
    elif model_name in ('bilinear_closure', 'bilin', 'fixd_v2'):
        from model_fixD import build_model
        model = build_model(model_name, in_channels=n_in,
                            hidden=cfg.get('hidden_channels', 64),
                            kernel=cfg.get('kernel', 3))
    else:
        from model import build_model
        model = build_model('cnn', in_channels=n_in,
                            hidden_channels=cfg.get('hidden_channels', 64),
                            depth=cfg.get('depth', 6), kernel=cfg.get('kernel', 3))
    # strict=False + report: surfaces any key/shape mismatch instead of failing
    # hard. The model is the fused parametrization (time_fd.weight, grad.wx/wy);
    # cross-dt/dx deployment is handled by rescaling the loaded stencils, not by
    # the parametrization.
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[load] state_dict mismatch -- missing={list(missing)} "
              f"unexpected={list(unexpected)}")

    # ---- dt portability (cheap_deriv): rescale the loaded time stencil from the
    # TRAINING dt to the DEPLOYMENT dt. The fused time_fd.weight was trained at the
    # dataset's dt (manifest Delta_T) and carries its dt^-k scaling; deploying at a
    # different dt_rollout needs, per row k (the order-k derivative, ~dt^-k):
    #     W_deploy[k] = W_train[k] * (dt_train / dt_deploy)^k .
    # This resets only the dt scaling and preserves the learned dimensionless
    # refinement. The spatial kernel is left alone -- same grid here, so dx is
    # unchanged. (No-op when dt_deploy == dt_train.) ----
    if model_name in ('cheap_deriv', 'deriv_closure'):
        dt_train = float(manifest['Delta_T'])
        dt_deploy = float(dt_rollout)
        if abs(dt_train - dt_deploy) > 1e-15:
            with torch.no_grad():
                W = model.time_fd.weight                 # (nt, nt), row k = order-k stencil
                ratio = dt_train / dt_deploy
                factors = torch.tensor([ratio ** k for k in range(W.shape[0])],
                                       dtype=W.dtype, device=W.device)
                W.mul_(factors[:, None])
            print(f"[load] dt-portability rescale: dt_train={dt_train:.3e} -> "
                  f"dt_deploy={dt_deploy:.3e}; time-stencil row factors "
                  f"{[round(ratio ** k, 6) for k in range(W.shape[0])]}")
    # The NN is a learned approximation (error ~1e-2..1e-4), so its float32 round-off
    # (~1e-7) is noise -- but a float64 conv is ~30x slower on the A6000 (FP64 = 1/32
    # FP32) and was the dominant closure cost. Run inference in float32 by default
    # (this also homogenizes the mixed physics_init/conv params); --nn-float64 forces
    # float64. The output is cast to the float64 rollout dtype before the FFT, so the
    # physics stays float64.
    mdtype = torch.float64 if nn_float64 else torch.float32
    model.to(device=device, dtype=mdtype).eval()
    print(f"[load] NN compute dtype = {mdtype}")
    if model_name in ('cheap_deriv', 'deriv_closure') and not nn_float64:
        print(f"[load] cheap_deriv MIXED precision: TimeFD differencing in float64 "
              f"(cancellation-clean omega_3dot), spatial convs in float32 (~30x faster). "
              f"Pass --nn-float64 to force full float64 (slow; for validation).")
    print(f"[load] model={model_name}  ckpt={ckpt.name}  "
          f"params={sum(p.numel() for p in model.parameters())}")
    return model, model_name


def maybe_compile_model(model, input_fields, omega_stack, psi_stack, dtype, device, enabled):
    """Optionally torch.compile the NN (the `nn_conv` block) and VERIFY it before
    trusting it. Inductor fuses the model's pointwise/bilinear chain, cutting the
    intermediate full-grid memory traffic that makes the small (627-param) net
    memory-bound. Returns the model to actually run: the compiled one only if it
    matches eager, else eager (so the rollout is always correct).

    The check is on a REAL assembled input (exact rollout shape/dtype), comparing
    eager vs compiled. Inductor may reorder float ops, so a ~1e-6 (float32) diff is
    expected and fine; a large diff or non-finite output means a graph problem and
    we fall back. The compile cost is paid HERE (the probe forward), before timing."""
    if not enabled:
        return model
    if not str(device).startswith('cuda'):
        print("[compile] --compile ignored on non-cuda device.")
        return model
    if not hasattr(torch, 'compile'):
        print("[compile] torch.compile unavailable (needs PyTorch>=2.0); using eager.")
        return model
    model_dtype = next(model.parameters()).dtype
    xprobe = assemble_inputs(input_fields, omega_stack, psi_stack, dtype, device).to(dtype=model_dtype)
    with torch.no_grad():
        y_eager = model(xprobe).clone()                       # genuinely eager (pre-compile)
    try:
        compiled = torch.compile(model)
        with torch.no_grad():
            y_comp = compiled(xprobe)                          # triggers compilation
    except Exception as e:                                     # noqa: BLE001
        print(f"[compile] torch.compile FAILED ({type(e).__name__}: {e}); using eager.")
        return model
    d = (y_comp.to(torch.float64) - y_eager.to(torch.float64)).abs()
    max_abs = float(d.max())
    denom = float(y_eager.abs().to(torch.float64).max())
    max_rel = max_abs / (denom if denom > 1e-30 else 1e-30)
    finite = bool(torch.isfinite(y_comp).all())
    print(f"[compile] correctness probe {tuple(xprobe.shape)} {model_dtype}: "
          f"max|delta|={max_abs:.3e}  max_rel={max_rel:.3e}  finite={finite}")
    tol = 1e-3
    if (not finite) or (max_rel > tol):
        print(f"[compile] WARNING: compiled vs eager rel {max_rel:.3e} exceeds {tol:.0e} "
              f"(or non-finite) -- NOT using torch.compile; falling back to eager so the "
              f"rollout stays correct. (A real graph break, not fp reordering, which is ~1e-6.)")
        return model
    print(f"[compile] OK (rel {max_rel:.3e} <= {tol:.0e}); using compiled NN. Compile cost "
          f"paid on this probe, so the closure timing (warmup-excluded) is clean.")
    return compiled


def assemble_inputs(input_fields, omega_stack, psi_stack, dtype, device):
    # n_time-agnostic: maps omega_0 / omega_m1 / ... / omega_m{k} (and psi_*) to the
    # matching stack depth, so the same path serves the 3- and 4-snapshot (and deeper)
    # cheap_deriv models. omega_stack/psi_stack must be at least as deep as the largest
    # m{k} requested.
    chans = []
    for field in input_fields:
        if field.startswith('omega_'):
            tag = field.split('_', 1)[1]            # '0','m1','m2','m3',...
            k = 0 if tag == '0' else int(tag[1:])
            chans.append(omega_stack[k])
        elif field.startswith('psi_'):
            tag = field.split('_', 1)[1]
            k = 0 if tag == '0' else int(tag[1:])
            chans.append(psi_stack[k])
        else:
            raise ValueError(f"rollout supports only omega/psi snapshot inputs, got {field}")
    x = torch.stack([c[0] for c in chans], dim=0)[None]
    return x.to(device=device, dtype=dtype)


def nn_derivs_hat(model, model_name, x, dtype, target_mean, target_std, normalize,
                  prof=_NOPROF, want_n3dot=False):
    """Return the NN's spectral time-derivatives (Ndot_hat, Nddot_hat, N3dot_hat) -- 2
    FFTs, or 3 when want_n3dot (the full-R4 N3dot term). The L-weighting and the R3 / R4
    spectral assembly happen in the closure (free multiplies), so the corrections share
    these FFTs. N3dot_hat is None unless want_n3dot."""
    from qg.solver.opt.basis import to_spectral
    model_dtype = next(model.parameters()).dtype
    # cheap_deriv runs MIXED precision: feed the snapshot stack at the rollout dtype
    # (float64) so the model's TimeFD differencing is cancellation-clean; the model
    # casts internally to its param dtype (float32) for the spatial convs. Other models
    # consume the param dtype directly.
    in_dtype = dtype if model_name in ('cheap_deriv', 'deriv_closure') else model_dtype
    prof.mark('nn_conv')                                 # the conv forward pass
    with torch.no_grad():
        yhat = model(x.to(dtype=in_dtype)).to(dtype=dtype)
    if model_name in ('cheap_deriv', 'deriv_closure'):
        prof.mark('nn_fft')                              # 2 (+1) FFTs (Ndot, Nddot[, N3dot])
        Ndot_hat = to_spectral(yhat[:, 0:1, :, :][0])    # 1 FFT
        Nddot_hat = to_spectral(yhat[:, 1:2, :, :][0])   # 1 FFT
        N3dot_hat = None
        if want_n3dot:
            if yhat.shape[1] < 3:
                raise ValueError(f"--r4-n3dot-coef needs a model with >=3 output orders "
                                 f"(Ndot, Nddot, N3dot); this model outputs {yhat.shape[1]}.")
            N3dot_hat = to_spectral(yhat[:, 2:3, :, :][0])   # 1 FFT
        return Ndot_hat, Nddot_hat, N3dot_hat
    raise ValueError(f"NN derivative extraction not defined for model '{model_name}'")
    # bilinear / cnn: single-channel f_NN_target (optionally normalized)
    f = yhat[:, 0:1, :, :]
    if normalize and target_std is not None:
        f = f * target_std + target_mean
    return f[0][None]
    # bilinear / cnn: single-channel f_NN_target (optionally normalized)
    f = yhat[:, 0:1, :, :]
    if normalize and target_std is not None:
        f = f * target_std + target_mean
    return f[0][None]


# =========================================================================== #
# Timing helpers                                                               #
# =========================================================================== #
def _sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()





class _StepProfiler:
    """Per-component GPU timer for one closure step. Brackets each labelled block
    with a cuda.Event; `mark(L)` closes the previous block and opens block L; the
    block's device time is event(L) -> event(next). step_end() drops a sentinel,
    SYNCS once, and reads every interval (events must complete before elapsed_time
    is valid). Accumulates device-ms per label over the profiled window; a host
    perf_counter around the step (taken by the caller, no sync) gives CPU issue
    time, so host>>device flags a launch/Python-bound step.

    Only the short --profile-step window uses this; the headline walltime comes
    from the un-instrumented loop and is untouched."""
    enabled = True
    def __init__(self, device):
        self.dev = device
        self.dev_ms = {}      # label -> accumulated device ms
        self.order = []       # label first-seen order, for stable printing
        self.host_ms = 0.0    # accumulated CPU issue ms (set by caller)
        self.nsteps = 0
        self._marks = []      # [(label, event)] for the in-flight step
    def step_begin(self):
        self._marks = []
    def mark(self, label):
        e = torch.cuda.Event(enable_timing=True); e.record()
        self._marks.append((label, e))
        if label not in self.dev_ms:
            self.dev_ms[label] = 0.0; self.order.append(label)
    def step_end(self):
        if not self._marks:
            return
        sentinel = torch.cuda.Event(enable_timing=True); sentinel.record()
        torch.cuda.synchronize()
        for i, (lbl, ev0) in enumerate(self._marks):
            ev1 = self._marks[i + 1][1] if i + 1 < len(self._marks) else sentinel
            self.dev_ms[lbl] += ev0.elapsed_time(ev1)
        self.nsteps += 1
        self._marks = []
    # ---- reporting ---------------------------------------------------------- #
    def report(self, clean_ms_per_step, n_fft):
        n = max(self.nsteps, 1)
        dev = {k: v / n for k, v in self.dev_ms.items()}
        dev_sum = sum(dev.values())
        host = self.host_ms / n
        def g(*keys):
            return sum(dev.get(k, 0.0) for k in keys)
        transforms = g('nn_fft', 'N_ifft', 'N_prodfft', 'N_psi')
        nn_conv    = g('nn_conv')
        bracket    = g('e_anal', 'f_NN', 'r4', 'combine')
        dealias    = g('dealias')
        other      = g('bare', 'inputs', 'N_spectral')
        print("\n==================== STEP TIMING BREAKDOWN (closure) "
              "====================")
        print(f"  profiled window  = {n} steps")
        print(f"  clean wall/step  = {clean_ms_per_step:.3f} ms   "
              f"(un-instrumented loop -- the headline number)")
        print(f"  host issue/step  = {host:.3f} ms   (CPU enqueue time, no sync)")
        print(f"  device sum/step  = {dev_sum:.3f} ms   (sum of GPU blocks below)")
        if dev_sum > 0:
            if host > dev_sum:
                print(f"  -> HOST/LAUNCH-BOUND: CPU issue ({host:.3f}) > GPU work "
                      f"({dev_sum:.3f}); ~{100*(host-dev_sum)/host:.0f}% of the "
                      f"step is enqueue/Python overhead. Fusing (torch.compile / "
                      f"CUDA graph) targets this.")
            else:
                print(f"  -> DEVICE-BOUND: GPU work ({dev_sum:.3f}) >= CPU issue "
                      f"({host:.3f}); the kernels themselves dominate.")
        print(f"\n  {'block':<12}{'dev ms/step':>14}{'% dev':>10}")
        for lbl in self.order:
            v = dev[lbl]
            print(f"  {lbl:<12}{v:>14.4f}{100*v/dev_sum if dev_sum else 0:>9.1f}%")
        print("  " + "-" * 36)
        print(f"  {'TRANSFORMS':<12}{transforms:>14.4f}{100*transforms/dev_sum if dev_sum else 0:>9.1f}%"
              f"   ({n_fft} FFT/iFFT)")
        print(f"  {'NN_CONV':<12}{nn_conv:>14.4f}{100*nn_conv/dev_sum if dev_sum else 0:>9.1f}%")
        print(f"  {'BRACKET':<12}{bracket:>14.4f}{100*bracket/dev_sum if dev_sum else 0:>9.1f}%"
              f"   (e_anal+f_NN+r4+combine -- the L^k assembly)")
        print(f"  {'DEALIAS':<12}{dealias:>14.4f}{100*dealias/dev_sum if dev_sum else 0:>9.1f}%")
        print(f"  {'OTHER':<12}{other:>14.4f}{100*other/dev_sum if dev_sum else 0:>9.1f}%"
              f"   (bare AB2 + assemble_inputs + N spectral combine)")
        print("=" * 72)


# =========================================================================== #
# Rollouts                                                                     #
# =========================================================================== #
def rollout_fine(omega_0, h_fine, n_fine_steps, checkpoint_steps,
                 derivative, L_hat, F_phys, device):
    """Truth: RK4 at h_fine (single-step; no stencil bootstrap)."""
    om = omega_0.clone()
    out = {}; cps = set(checkpoint_steps)
    w = omega_0.clone()
    for _ in range(3):                       # warmup (untimed)
        w = rk4_step(w, h_fine, derivative, L_hat, F_phys)
    _sync(device); t0 = time.time()
    if 0 in cps:
        out[0] = om[0].cpu().numpy()
    for s in range(1, n_fine_steps + 1):
        om = rk4_step(om, h_fine, derivative, L_hat, F_phys)
        if s in cps:
            arr = om[0].cpu().numpy()
            out[s] = arr
            rms = float(np.sqrt(np.mean(arr ** 2)))
            el = time.time() - t0
            eta = el * (n_fine_steps - s) / max(s, 1)
            bad = not np.isfinite(rms)
            print(f"      [truth] RK4 step {s:>9d}/{n_fine_steps}  "
                  f"|omega|_rms={rms:.4e}  elapsed={el/60:.1f}m eta={eta/60:.1f}m"
                  f"{'   *** NON-FINITE -- stopping ***' if bad else ''}", flush=True)
            if bad:
                break
    _sync(device); wall = time.time() - t0
    return out, wall


def rollout_bare(omega_0, omega_m1, dt, n_steps, checkpoint_steps,
                 derivative, L_hat, F_hat, device):
    from qg.solver.opt.basis import to_spectral, to_physical
    qh_n = to_spectral(omega_0); qh_nm1 = to_spectral(omega_m1)
    denom_hat = 1.0 - 0.5 * dt * L_hat
    out = {}; cps = set(checkpoint_steps)

    def step(qh, Nh, Nh_prev):
        AB2_Nh = 1.5 * Nh - 0.5 * Nh_prev
        return (qh + dt * (0.5 * L_hat * qh + AB2_Nh)) / denom_hat

    # proper multistep: ONE new N per step; the previous step's N is reused.
    Nh_n = N_spectral(qh_n, derivative, F_hat)
    Nh_nm1 = N_spectral(qh_nm1, derivative, F_hat)

    # warmup (untimed) -- exercises the same code path
    a, Na, Nam1 = qh_n.clone(), Nh_n.clone(), Nh_nm1.clone()
    for _ in range(3):
        a_new = step(a, Na, Nam1)
        Nam1, Na, a = Na, N_spectral(a_new, derivative, F_hat), a_new

    _sync(device); t0 = time.time()
    if 0 in cps:
        out[0] = to_physical(qh_n)[0].cpu().numpy()
    for s in range(1, n_steps + 1):
        qh_new = step(qh_n, Nh_n, Nh_nm1)
        Nh_nm1, Nh_n, qh_n = Nh_n, N_spectral(qh_new, derivative, F_hat), qh_new
        if s in cps:
            arr = to_physical(qh_n)[0].cpu().numpy()
            out[s] = arr
            if not np.isfinite(np.sqrt(np.mean(arr ** 2))):
                print(f"      [bare] non-finite at step {s} -- stopping.", flush=True)
                break
    _sync(device); wall = time.time() - t0
    return out, wall


def rollout_closure(omega_stack, psi_stack, Delta_T, K, n_steps, checkpoint_steps,
                    model, model_name, input_fields, derivative, L_hat, F_hat,
                    dtype, device, target_mean, target_std, normalize, dealias_nn=False,
                    include_r4=False, profile_step=0, r4_n3dot_coef=None, r4_denom=24.0):
    from qg.solver.opt.basis import to_spectral, to_physical
    coef = Delta_T ** 3      # truth = RK4 (exact flow) -> NO (1-1/K^n) factor, just Taylor
    coef4 = Delta_T ** 4
    denom_hat = 1.0 - 0.5 * Delta_T * L_hat
    # precompute L_hat powers ONCE (they are step-invariant). The per-step e_anal /
    # r4 assembly previously rebuilt L_hat**2/**3/**4 every step -- ~22% of the step
    # at 512^2. Numerically identical; pure hoist.
    L_hat2 = L_hat ** 2
    L_hat3 = L_hat2 * L_hat
    L_hat4 = L_hat2 * L_hat2          # only used by --r4
    om = list(omega_stack); ps = list(psi_stack)
    out = {}; cps = set(checkpoint_steps)
    _want_n3dot = bool(include_r4 and r4_n3dot_coef is not None)

    def one_step(om, ps, qh_curr, Nh_curr, Nh_minus, prof=_NOPROF):
        # everything stays SPECTRAL until a single iFFT for the new state
        prof.mark('bare')
        AB2_Nh = 1.5 * Nh_curr - 0.5 * Nh_minus
        qh_bare = (qh_curr + Delta_T * (0.5 * L_hat * qh_curr + AB2_Nh)) / denom_hat
        # e_anal: free spectral multiplies on the cached qh_curr and Nh_curr
        prof.mark('e_anal')
        e_anal_hat = -coef * (1.0 / 12.0) * (L_hat3 * qh_curr + L_hat2 * Nh_curr)
        prof.mark('inputs')
        x = assemble_inputs(input_fields, om, ps, dtype, device)
        Ndot_hat, Nddot_hat, N3dot_hat = nn_derivs_hat(model, model_name, x, dtype,
                                            target_mean, target_std, normalize,
                                            prof=prof, want_n3dot=_want_n3dot)        # 2 (+1) FFTs
        prof.mark('f_NN')
        f_NN_hat = (1.0 / 12.0) * (L_hat * Ndot_hat - 5.0 * Nddot_hat)           # R3 (learned)
        # partial R4 (free spectral): analytic L^4 w, L^3 N + the NN's L^2 Ndot, L Nddot.
        # N3dot term intentionally dropped. Raw Taylor coefs (2,2,2,-4); the 1/12 and
        # sign are carried over from R3. NO (1-1/K^n) factor -- truth is RK4 (exact).
        prof.mark('r4')
        e_r4_hat = None
        if include_r4:
            e_r4_hat = -coef4 * (1.0 / r4_denom) * (2.0 * L_hat4 * qh_curr
                                                + 2.0 * L_hat3 * Nh_curr
                                                + 2.0 * L_hat2 * Ndot_hat
                                                - 4.0 * L_hat * Nddot_hat)
            if r4_n3dot_coef is not None:
                # full-R4 N3dot (third N-derivative) term. The coefficient is supplied
                # via --r4-n3dot-coef from the R4 modified-equation derivation -- NOT
                # guessed here. bracket += c * N3dot ; same -coef4*(1/r4_denom) prefactor.
                e_r4_hat = e_r4_hat - coef4 * (1.0 / r4_denom) * (float(r4_n3dot_coef) * N3dot_hat)
        prof.mark('dealias')
        if dealias_nn:
            f_NN_hat = _dealias_mul(f_NN_hat, derivative)   # 1/3 rule, spectral -- free
            if e_r4_hat is not None:
                e_r4_hat = _dealias_mul(e_r4_hat, derivative)
        prof.mark('combine')
        om_new_hat = qh_bare + e_anal_hat - coef * f_NN_hat
        if e_r4_hat is not None:
            om_new_hat = om_new_hat + e_r4_hat
        # the new N eval hands back omega/psi physical -> reused for the stencil,
        # so neither is iFFT'd a second time
        Nh_new, om_new, psi_new = N_spectral_fields(om_new_hat, derivative, F_hat, prof=prof)
        return om_new_hat, om_new, psi_new, Nh_new

    qh_curr = to_spectral(om[0]); qh_minus = to_spectral(om[1])
    Nh_curr = N_spectral(qh_curr, derivative, F_hat)
    Nh_minus = N_spectral(qh_minus, derivative, F_hat)

    # warmup (untimed)
    om_w, ps_w = list(om), list(ps)
    qc, Nc, Nm = qh_curr.clone(), Nh_curr.clone(), Nh_minus.clone()
    for _ in range(3):
        qnew, onew, pnew, Nnew = one_step(om_w, ps_w, qc, Nc, Nm)
        om_w = [onew] + om_w[:-1]
        ps_w = [pnew] + ps_w[:-1]
        qc, Nm, Nc = qnew, Nc, Nnew
    _sync(device); t0 = time.time()
    if 0 in cps:
        out[0] = om[0][0].cpu().numpy()
    for s in range(1, n_steps + 1):
        qh_new, om_new, psi_new, Nh_new = one_step(om, ps, qh_curr, Nh_curr, Nh_minus)
        om = [om_new] + om[:-1]
        ps = [psi_new] + ps[:-1]
        qh_minus, qh_curr = qh_curr, qh_new
        Nh_minus, Nh_curr = Nh_curr, Nh_new
        if s in cps:
            arr = om_new[0].cpu().numpy()
            out[s] = arr
            if not np.isfinite(np.sqrt(np.mean(arr ** 2))):
                print(f"      [closure] non-finite at step {s} -- stopping.", flush=True)
                break
    _sync(device); wall = time.time() - t0

    # ---- optional per-component breakdown (separate short window) ----------- #
    # The headline `wall` above is from the un-instrumented loop. Here we step a
    # few more times WITH cuda-event marks to attribute the per-step cost. We
    # continue from the developed end-state (forced turb is stationary, so it's
    # representative) and discard the outputs. A per-step sync makes this window
    # slower in absolute terms, but the per-block split and the host-vs-device
    # comparison are what we read -- not this window's walltime.
    if profile_step and str(device).startswith('cuda'):
        nprof = min(int(profile_step), n_steps if n_steps > 0 else int(profile_step))
        prof = _StepProfiler(device)
        # small warmup so the first profiled step isn't paying lazy-init costs
        for _ in range(3):
            qh_new, om_new, psi_new, Nh_new = one_step(om, ps, qh_curr, Nh_curr, Nh_minus)
            om = [om_new] + om[:-1]; ps = [psi_new] + ps[:-1]
            qh_minus, qh_curr = qh_curr, qh_new
            Nh_minus, Nh_curr = Nh_curr, Nh_new
        _sync(device)
        for _ in range(nprof):
            prof.step_begin()
            h0 = time.perf_counter()
            qh_new, om_new, psi_new, Nh_new = one_step(om, ps, qh_curr, Nh_curr,
                                                       Nh_minus, prof=prof)
            prof.host_ms += (time.perf_counter() - h0) * 1e3   # CPU enqueue, no sync
            prof.step_end()                                    # syncs + reads events
            om = [om_new] + om[:-1]; ps = [psi_new] + ps[:-1]
            qh_minus, qh_curr = qh_curr, qh_new
            Nh_minus, Nh_curr = Nh_curr, Nh_new
        n_fft = 8 + (1 if _want_n3dot else 0)   # +1 NN FFT for the N3dot term (full R4)
        prof.report(clean_ms_per_step=wall / max(n_steps, 1) * 1e3, n_fft=n_fft)

    return out, wall


def rel_l2(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)) / max(np.sqrt(np.mean(b ** 2)), 1e-30))


# --------------------------------------------------------------------------- #
# Plot labels (LaTeX, mathtext-safe) -- shared by the timed and diag figures.  #
# --------------------------------------------------------------------------- #
LBL_TRUTH = r'$u_{\mathrm{RK4},\,\delta t=\Delta T/K}$'
LBL_BARE  = r'$u_{\mathrm{AB2CN2},\,\delta t=\Delta T}$'
LBL_ML    = r'$u_{\mathrm{AB2CN2}+\epsilon_{\mathrm{NN}},\,\delta t=\Delta T}$'
# relative L2 error against the fine RK4 truth (the quantity plotted vs time)
ERR_LABEL = (r'rel. $L_2$ error  '
             r'$\frac{\|u-u_{\mathrm{RK4}}\|_2}{\|u_{\mathrm{RK4}}\|_2}$')
# improvement = ratio of the two error norms (bare over ML)
IMPROV_LABEL = (r'$\frac{\|u_{\mathrm{AB2CN2}}-u_{\mathrm{RK4}}\|_2}'
                r'{\|u_{\mathrm{AB2CN2}+\epsilon_{\mathrm{NN}}}-u_{\mathrm{RK4}}\|_2}$')
# scalar L2 norm definition, shown as a footnote so the axis math is unambiguous
NORM_DEF = r'$\|v\|_2=\sqrt{\sum_i v_i^2}$'


def _radial_spectrum(field):
    """Isotropic (radially-binned) power spectrum of a 2D field -- for locating where
    in wavenumber the closure/bare error lives (broadband vs high-k pile-up)."""
    field = np.asarray(field, dtype=np.float64)
    P = np.abs(np.fft.fft2(field)) ** 2
    ny, nx = field.shape
    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    KR = np.rint(np.sqrt(np.add.outer(ky ** 2, kx ** 2))).astype(int)
    kmax = int(KR.max())
    Pr = np.zeros(kmax + 1)
    for k in range(kmax + 1):
        m = (KR == k)
        if m.any():
            Pr[k] = P[m].mean()
    return np.arange(kmax + 1), Pr


# =========================================================================== #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--root-dir', type=Path, required=True)
    p.add_argument('--n-steps', type=int, default=200, help='coarse steps (horizon = n_steps*Delta_T)')
    p.add_argument('--n-checkpoints', type=int, default=10)
    p.add_argument('--fine-save-every', type=int, default=0,
                   help='checkpoint every N fine RK4 steps (snapped to a multiple of K so '
                        'it lands on coarse steps too). Overrides --n-checkpoints. Use for '
                        'long stability runs, e.g. 100000.')
    p.add_argument('--ic-index', type=int, default=0, help='row of packed/inputs.npy to use as IC')
    p.add_argument('--Delta-T-override', type=float, default=None)
    p.add_argument('--K-override', type=int, default=None)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--out-dir', type=Path, default=None)
    p.add_argument('--ic-tag', type=str, default='ft')
    p.add_argument('--pareto', action='store_true',
                   help='also run bare at a sweep of dt to trace the cost/accuracy front')
    p.add_argument('--pareto-dt-factors', type=str, default='1,2,4,8,16,40,100',
                   help='bare dt = Delta_T / factor for each factor (capped at h_fine)')
    # forcing (forced-turb defaults; C=F=0 -> steady)
    p.add_argument('--fA', type=float, default=-0.1); p.add_argument('--fB', type=float, default=2.0)
    p.add_argument('--fC', type=float, default=0.0);  p.add_argument('--fD', type=float, default=0.1)
    p.add_argument('--fE', type=float, default=2.0);  p.add_argument('--fF', type=float, default=0.0)
    p.add_argument('--no-forcing', action='store_true', help='decaying turb (F=None)')
    p.add_argument('--dealias-nn', action='store_true',
                   help='project the NN correction onto the 1/3-rule resolved band each step '
                        '(matches the solver source, which dealiases). Recommended; A/B it.')
    p.add_argument('--r4', action='store_true',
                   help='add the partial R4 refinement (2L^4 w + 2L^3 N + 2L^2 Ndot - 4L Nddot; '
                        'N3dot term dropped). Free spectral multiplies, no extra FFT. A/B vs R3.')
    p.add_argument('--r4-n3dot-coef', type=float, default=None,
                   help='full R4: coefficient of the N3dot (third N-derivative) term in the R4 '
                        'bracket, taken from YOUR R4 modified-equation derivation (the table left '
                        'it as "..."; not guessed here). Only active with --r4. Omit to keep the '
                        'partial R4 (N3dot dropped, as before). Adds 1 NN FFT/step and needs a '
                        'model with >=3 output orders; pair with --nn-float64 for a clean N3dot.')
    p.add_argument('--r4-denom', type=float, default=24.0,
                   help='prefactor denominator of the R4 correction term (R3 stays /12). Your '
                        'truncation hierarchy is h^3/12, h^4/24, so this defaults to 24 -- matching '
                        'rollout_perfect_closure.py. The earlier code used 12 (factor-2 too large, '
                        'R4 over-corrects -> caps at the R3/R4 ratio ~240x). Pass 12 to reproduce.')
    p.add_argument('--nn-float64', action='store_true',
                   help='run the NN in float64 (default float32). float64 conv is ~30x slower '
                        'on the A6000 for ~no accuracy gain (NN error >> float32 eps).')
    p.add_argument('--profile-step', type=int, default=0, metavar='N',
                   help='after the clean closure timing, step N more times WITH per-component '
                        'cuda-event timers and print a breakdown (FFT vs NN-conv vs L^k bracket '
                        'vs dealias) plus a host-issue-vs-device-time comparison to flag whether '
                        'the step is launch/Python-bound. cuda only; 0 = off. Try 500.')
    p.add_argument('--compile', action='store_true',
                   help='torch.compile the NN to fuse its pointwise/bilinear chain (cuts the '
                        'nn_conv memory traffic). Verified against eager on a real probe input '
                        'before use; falls back to eager automatically if they disagree. cuda only.')
    p.add_argument('--diag', action='store_true',
                   help='after the rollout, print a per-term RMS breakdown of the closure '
                        'correction (analytic L^k vs NN-predicted Ndot/Nddot/N3dot terms -> the '
                        'NN-predicted fraction x the ~3-4%% per-derivative error is the NN-limited '
                        'error floor) and write rollout_diag_<tag>.png (improvement-vs-time + error '
                        'power spectrum vs |k|).')
    p.add_argument('--save-refs', action='store_true',
                   help='dump the full truth+bare checkpoint stacks to rollout_refs_<tag>.npz '
                        'so a later run can reuse them via --load-refs (skips recomputing the '
                        'expensive RK4 truth and the bare AB2CN2 leg).')
    p.add_argument('--load-refs', type=Path, default=None,
                   help='load truth+bare from a rollout_refs_*.npz and run ONLY the closure leg. '
                        'Takes ic-index / n-steps / checkpoints from the file; Delta_T, K, h_fine '
                        'must match the file or it aborts.')
    p.add_argument('--rerun-bare', action='store_true',
                   help='with --load-refs: reuse the (expensive) RK4 truth from the file but '
                        'RECOMPUTE the bare AB2CN2 leg (cheap) -- e.g. to get a fresh bare '
                        'walltime under the cached stepper. Closure runs as usual.')
    args = p.parse_args()

    device = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    dtype = torch.float64
    out_dir = args.out_dir or args.run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[rollout] device={device} dtype=float64")

    with open(args.root_dir / 'manifest.json') as f:
        manifest = json.load(f)
    Nx, Ny = int(manifest['Nx']), int(manifest['Ny'])
    Lx, Ly = float(manifest['Lx']), float(manifest['Ly'])
    nu = float(manifest['nu']); mu = float(manifest.get('mu', 0.0)); beta = float(manifest.get('beta', 0.0))
    Delta_T = args.Delta_T_override or float(manifest['Delta_T'])
    K = args.K_override or int(manifest.get('K', 100))
    h_fine = Delta_T / K
    print(f"[rollout] grid {Nx}x{Ny}  nu={nu} mu={mu} beta={beta}")
    print(f"[rollout] Delta_T={Delta_T} K={K} h_fine={h_fine}")

    refs = None
    if args.load_refs is not None:
        rf = np.load(args.load_refs)
        refs_Delta_T = float(rf['Delta_T']); refs_K = int(rf['K'])
        refs_h_fine = refs_Delta_T / refs_K
        # The truth is the fine RK4 flow: it depends only on the FINE step h_fine
        # and the physical checkpoint times, NOT on the coarse Delta_T. So a truth
        # integrated at one Delta_T is reusable at another whenever the fine step
        # matches -- e.g. (Delta_T=1e-3, K=100) and (Delta_T=1e-2, K=1000) both give
        # h_fine=1e-5, the same reference. We map truth snapshots by physical time;
        # the bare leg is Delta_T-specific and is recomputed when Delta_T changes.
        if abs(refs_h_fine - h_fine) > 1e-12 * max(refs_h_fine, h_fine):
            raise SystemExit(
                f"[rollout] --load-refs fine-step mismatch: refs h_fine={refs_h_fine:.3e} "
                f"(Delta_T={refs_Delta_T}, K={refs_K}) vs current h_fine={h_fine:.3e} "
                f"(Delta_T={Delta_T}, K={K}). The truth must be the SAME fine flow -- "
                f"set --K-override so that Delta_T/K equals the refs' fine step "
                f"({refs_h_fine:.3e}); e.g. at Delta_T={Delta_T} use K={int(round(Delta_T/refs_h_fine))}.")
        refs_cp_coarse = [int(s) for s in rf['cp_coarse']]
        refs_cp_times = [s * refs_Delta_T for s in refs_cp_coarse]   # physical times of truth snapshots
        same_dt = abs(refs_Delta_T - Delta_T) <= 1e-15
        refs = dict(ic_index=int(rf['ic_index']), cp_times=refs_cp_times,
                    truth_stack=rf['truth_stack'], bare_stack=rf['bare_stack'],
                    t_truth=float(rf['t_truth']), t_bare=float(rf['t_bare']),
                    same_dt=same_dt)
        if 'ic_omega_0' in rf:
            refs['ic_fields'] = (rf['ic_omega_0'], rf['ic_omega_m1'], rf['ic_omega_m2'])
        if args.ic_index != refs['ic_index']:
            print(f"[rollout] --load-refs: overriding --ic-index {args.ic_index} "
                  f"-> {refs['ic_index']} (must match the saved truth)")
        args.ic_index = refs['ic_index']
        bare_status = 'reused' if (same_dt and not args.rerun_bare) else f'recomputed at Delta_T={Delta_T}'
        print(f"[rollout] loaded refs from {args.load_refs}: {len(refs_cp_times)} truth checkpoints "
              f"t={refs_cp_times[0]:.3f}..{refs_cp_times[-1]:.3f}, h_fine={refs_h_fine:.3e}; "
              f"reusing t_truth={refs['t_truth']:.1f}s, bare {bare_status}")

    cfg = json.loads((args.run_dir / 'config.json').read_text())
    input_fields = tuple(cfg.get('input_fields',
                                 ['omega_0', 'omega_m1', 'omega_m2', 'psi_0', 'psi_m1', 'psi_m2']))
    normalize = cfg.get('normalize', False)
    print(f"[rollout] inputs={input_fields}  normalize={normalize}")

    from qg.solver.grid.cartesian import CartesianGrid
    from qg.solver.opt.derivative import Derivative
    grid = CartesianGrid(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, device=device, precision='float64')
    derivative = Derivative(grid)
    for attr in ('dx', 'dy', 'laplacian', 'inv_laplacian', 'alias_mask'):
        if hasattr(derivative, attr):
            setattr(derivative, attr, getattr(derivative, attr).to(device))
    L_hat = build_L_hat(derivative, nu, mu, beta).to(device)

    if args.no_forcing:
        fc = None
    elif manifest.get('has_forcing') and isinstance(manifest.get('forcing'), dict):
        fc = manifest['forcing']                     # authoritative
    else:
        fc = dict(function='unscaled_cosine', A=args.fA, B=args.fB, C=args.fC,
                  D=args.fD, E=args.fE, F=args.fF)
    F_phys = build_forcing(grid, fc, device, dtype)
    print(f"[rollout] forcing: {'none' if F_phys is None else fc}")
    # spectral forcing for the bare/closure minimal N-eval (added in spectral so N
    # never needs an iFFT back); the truth's RK4 keeps the physical F_phys.
    from qg.solver.opt.basis import to_spectral as _to_spectral
    F_hat = _to_spectral(F_phys) if F_phys is not None else None

    # ---- IC stencil: build an n_time-deep backward stack (3- or 4-snapshot model) ----
    # The cheap_deriv TimeFD needs n_time backward snapshots [omega_0, omega_m1, ...].
    n_time = sum(1 for f in input_fields if f.startswith('omega_'))
    if refs is not None and 'ic_fields' in refs:
        # Truth-shared IC: the saved truth was integrated from the refs' EXACT o0, so
        # the closure must start from those same o0/m1/m2 (bit-for-bit, no chaos-amplified
        # IC drift). Older refs store only 3; a 4-snapshot model also needs omega_m3..,
        # which we pull from this run's packed inputs -- after verifying the packed IC at
        # this ic-index is the SAME physical t0 as the refs (so the extra snapshots are
        # the true earlier states, not a different window).
        omega_stack = [torch.tensor(np.asarray(a), dtype=dtype, device=device)[None]
                       for a in refs['ic_fields']]                 # [o0, om1, om2]
        if n_time > len(omega_stack):
            inputs = np.load(args.root_dir / 'packed' / 'inputs.npy', mmap_mode='r')
            ic = args.ic_index
            tags = ['omega_0'] + [f'omega_m{k}' for k in range(1, n_time)]
            ci = {f: input_fields.index(f) for f in tags}
            p_o0 = torch.tensor(np.asarray(inputs[ic, ci['omega_0']]), dtype=dtype, device=device)[None]
            rms = float(torch.sqrt((omega_stack[0] ** 2).mean()))
            rel = float(torch.sqrt(((p_o0 - omega_stack[0]) ** 2).mean())) / max(rms, 1e-30)
            if rel > 1e-1:
                raise SystemExit(
                    f"[rollout] IC mismatch: packed omega_0 at ic-index {ic} (dataset "
                    f"'{args.root_dir.name}') is rel {rel:.2e} from refs o0 -- too large to be the "
                    f"same trajectory. This ic-index maps to a DIFFERENT physical state than the "
                    f"saved refs; the omega_m3 stencil would be meaningless. Find the index whose "
                    f"|omega_0| matches the refs, or regenerate refs from this dataset.")
            elif rel > 1e-5:
                print(f"[rollout] WARNING: packed omega_0 at ic-index {ic} differs from refs o0 by "
                      f"rel {rel:.2e}: SAME trajectory, but '{args.root_dir.name}' is not bit-identical "
                      f"to the build that produced the refs. Keeping refs o0/m1/m2 (truth-shared, exact) "
                      f"as the IC and using packed only for omega_m3.. The mismatch enters ONLY the "
                      f"order-3 time-FD (omega_m3 weight ~1/dt^3), and only for the first ~{n_time-1} "
                      f"steps before it flushes; with the R3/R4 dt^3/dt^4 prefactors the injected error "
                      f"is ~1e-6, negligible vs the closure error. For a pristine comparison, regenerate "
                      f"refs from this dataset (drop --load-refs, add --save-refs).", flush=True)
            for k in range(len(omega_stack), n_time):             # append omega_m3.. from packed
                omega_stack.append(torch.tensor(np.asarray(inputs[ic, ci[f'omega_m{k}']]),
                                                 dtype=dtype, device=device)[None])
            print(f"[rollout] IC: refs o0/m1/m2 (truth-shared) + omega_m{len(refs['ic_fields'])}.."
                  f"m{n_time-1} from packed (o0 match rel {rel:.2e}); n_time={n_time}")
        print(f"[rollout] IC from refs (row {args.ic_index}): "
              f"|omega_0|_rms={float(torch.sqrt((omega_stack[0]**2).mean())):.4e}")
    else:
        inputs = np.load(args.root_dir / 'packed' / 'inputs.npy', mmap_mode='r')
        ic = args.ic_index
        tags = ['omega_0'] + [f'omega_m{k}' for k in range(1, n_time)]
        ci = {f: input_fields.index(f) for f in tags}
        omega_stack = [torch.tensor(np.asarray(inputs[ic, ci[f]]), dtype=dtype, device=device)[None]
                       for f in tags]
        print(f"[rollout] IC row {ic} (n_time={n_time}): "
              f"|omega_0|_rms={float(torch.sqrt((omega_stack[0]**2).mean())):.4e}")
    omega_0, omega_m1, omega_m2 = omega_stack[0], omega_stack[1], omega_stack[2]
    psi_stack = [psi_from_omega(o, derivative) for o in omega_stack]

    # ---- checkpoints (shared physical times) ---- #
    if refs is not None:
        # place the closure's checkpoints at the SAME physical times as the saved
        # truth, expressed in the current coarse Delta_T (so the truth_stack maps by
        # index even though Delta_T may differ from the run that produced the truth).
        cp_coarse = [int(round(t / Delta_T)) for t in refs['cp_times']]
        for s, t in zip(cp_coarse, refs['cp_times']):
            if abs(s * Delta_T - t) > 1e-9 * max(Delta_T, abs(t), 1.0):
                raise SystemExit(
                    f"[rollout] --load-refs: truth checkpoint t={t:.6f} is not an integer "
                    f"multiple of Delta_T={Delta_T} (nearest coarse step {s} -> {s*Delta_T:.6f}). "
                    f"Pick a Delta_T that divides the saved checkpoint times.")
        M = cp_coarse[-1]
        cp_fine = [s * K for s in cp_coarse]
    else:
        M = args.n_steps
        if args.fine_save_every and args.fine_save_every > 0:
            fse = int(args.fine_save_every)
            if fse % K != 0:
                fse = max(K, int(round(fse / K)) * K)   # snap onto a coarse step
                print(f"[rollout] fine-save-every snapped to {fse} (multiple of K={K})")
            cp_fine = list(range(0, M * K + 1, fse))
            if cp_fine[-1] != M * K:
                cp_fine.append(M * K)                    # always checkpoint the end
            cp_coarse = [s // K for s in cp_fine]
        else:
            cp_coarse = sorted(set(int(round(f * M)) for f in
                                   np.linspace(0, 1, args.n_checkpoints + 1)))
            cp_fine = [s * K for s in cp_coarse]
    cp_times = [s * Delta_T for s in cp_coarse]
    save_every = (cp_fine[1] - cp_fine[0]) if len(cp_fine) > 1 else 0
    print(f"[rollout] horizon T={M*Delta_T:.4f}  {len(cp_coarse)} checkpoints  "
          f"(every {save_every} fine RK4 steps)")

    # ---- truth + bare: computed, or reused from refs ---- #
    if refs is not None:
        truth_cp = {s * K: refs['truth_stack'][i] for i, s in enumerate(cp_coarse)}
        t_truth = refs['t_truth']
        if refs['same_dt'] and not args.rerun_bare:
            bare_cp = {s: refs['bare_stack'][i] for i, s in enumerate(cp_coarse)}
            t_bare = refs['t_bare']
            print(f"[rollout] truth + bare REUSED from refs "
                  f"(skipped {M*K} RK4 + {M} AB2CN2 steps)")
        else:
            why = '--rerun-bare' if refs['same_dt'] else 'Delta_T differs from refs'
            print(f"[rollout] truth REUSED from refs (skipped {M*K} RK4 steps)")
            print(f"[rollout] bare: recomputing {M} coarse steps ({why}) ...")
            bare_cp, t_bare = rollout_bare(omega_0, omega_m1, Delta_T, M, cp_coarse,
                                           derivative, L_hat, F_hat, device)
            print(f"[rollout]   bare walltime  = {t_bare:.3f}s  ({t_bare/M*1e3:.3f} ms/coarse-step)")
    else:
        print(f"\n[rollout] truth: {M*K} fine RK4 steps ...")
        truth_cp, t_truth = rollout_fine(omega_0, h_fine, M * K, cp_fine,
                                         derivative, L_hat, F_phys, device)
        print(f"[rollout]   truth walltime = {t_truth:.2f}s  ({t_truth/(M*K)*1e3:.3f} ms/RK4-step)")

        print(f"[rollout] bare: {M} coarse steps ...")
        bare_cp, t_bare = rollout_bare(omega_0, omega_m1, Delta_T, M, cp_coarse,
                                       derivative, L_hat, F_hat, device)
        print(f"[rollout]   bare walltime  = {t_bare:.3f}s  ({t_bare/M*1e3:.3f} ms/coarse-step)")

    # ---- save truth+bare stacks for reuse (skip if we just loaded them) ---- #
    if args.save_refs and refs is None:
        ref_cp = [s for s in cp_coarse if (s * K in truth_cp and s in bare_cp)]
        truth_stack = np.stack([np.asarray(truth_cp[s * K], dtype=np.float32) for s in ref_cp])
        bare_stack = np.stack([np.asarray(bare_cp[s], dtype=np.float32) for s in ref_cp])
        refs_path = out_dir / f'rollout_refs_{args.ic_tag}.npz'
        np.savez(refs_path,
                 ic_index=np.int64(args.ic_index), n_steps=np.int64(M),
                 Delta_T=np.float64(Delta_T), K=np.int64(K), h_fine=np.float64(h_fine),
                 cp_coarse=np.asarray(ref_cp, dtype=np.int64),
                 cp_times=np.asarray([s * Delta_T for s in ref_cp], dtype=np.float64),
                 t_truth=np.float64(t_truth), t_bare=np.float64(t_bare),
                 ic_omega_0=omega_0[0].detach().cpu().numpy(),
                 ic_omega_m1=omega_m1[0].detach().cpu().numpy(),
                 ic_omega_m2=omega_m2[0].detach().cpu().numpy(),
                 truth_stack=truth_stack, bare_stack=bare_stack)
        print(f"[rollout] saved refs -> {refs_path}  "
              f"({truth_stack.nbytes/1e6:.0f} MB truth + {bare_stack.nbytes/1e6:.0f} MB bare, "
              f"{len(ref_cp)} checkpoints) -- reuse with --load-refs {refs_path}")

    model, model_name = load_model(args.run_dir, cfg, len(input_fields), manifest, device, Delta_T, dtype,
                                   nn_float64=args.nn_float64)
    model = maybe_compile_model(model, input_fields, omega_stack, psi_stack,
                                dtype, device, args.compile)
    target_mean = target_std = None
    if normalize:
        from dataset import ClosureDataset
        ds = ClosureDataset(root_dir=args.root_dir, split='val', input_fields=input_fields,
                            target_field=cfg.get('target_field', 'f_NN_target'), normalize=True)
        target_mean = torch.as_tensor(ds.target_mean, dtype=dtype, device=device)
        target_std = torch.as_tensor(ds.target_std, dtype=dtype, device=device)

    print(f"[rollout] closure: {M} coarse steps + NN ...")
    clos_cp, t_clos = rollout_closure(omega_stack, psi_stack, Delta_T, K, M, cp_coarse,
                                      model, model_name, input_fields, derivative, L_hat,
                                      F_hat, dtype, device, target_mean, target_std, normalize,
                                      dealias_nn=args.dealias_nn, include_r4=args.r4,
                                      profile_step=args.profile_step, r4_n3dot_coef=args.r4_n3dot_coef,
                                      r4_denom=args.r4_denom)
    print(f"[rollout]   closure walltime = {t_clos:.3f}s  ({t_clos/M*1e3:.3f} ms/closure-step)")

    # ---- errors at checkpoints (only those all three trajectories reached) ---- #
    avail = [s for s in cp_coarse if (s in bare_cp and s in clos_cp and s * K in truth_cp)]
    if len(avail) != len(cp_coarse):
        print(f"[rollout] NOTE: comparing on {len(avail)}/{len(cp_coarse)} checkpoints "
              f"(a trajectory stopped early).")
    a_times = np.array([s * Delta_T for s in avail])
    rel_bare = np.array([rel_l2(bare_cp[s], truth_cp[s * K]) for s in avail])
    rel_clos = np.array([rel_l2(clos_cp[s], truth_cp[s * K]) for s in avail])
    truth_rms = np.array([float(np.sqrt(np.mean(truth_cp[s * K] ** 2))) for s in avail])
    print("\n==================== ERROR vs TRUTH (rel-L2) ====================")
    print(f"{'t':>10}{'truth_rms':>14}{'bare':>14}{'closure':>14}{'improve x':>12}")
    for i, s in enumerate(avail):
        imp = rel_bare[i] / max(rel_clos[i], 1e-30)
        print(f"{a_times[i]:>10.4f}{truth_rms[i]:>14.4e}{rel_bare[i]:>14.4e}"
              f"{rel_clos[i]:>14.4e}{imp:>12.2f}")

    print("\n==================== COST ====================")
    print(f"  bare    : {t_bare:.3f}s   final rel-L2 = {rel_bare[-1]:.4e}")
    print(f"  closure : {t_clos:.3f}s   final rel-L2 = {rel_clos[-1]:.4e}")
    print(f"  truth   : {t_truth:.2f}s  (RK4 reference, {K}x the coarse step count)")
    print(f"  closure/bare walltime ratio = {t_clos/max(t_bare,1e-9):.2f}x")
    print(f"  truth/closure walltime ratio = {t_truth/max(t_clos,1e-9):.1f}x  "
          f"(closure buys ~truth accuracy at this fraction of the cost)")

    results = dict(Delta_T=Delta_T, K=K, h_fine=h_fine, n_steps=M,
                   t_truth=t_truth, t_bare=t_bare, t_clos=t_clos,
                   cp_times=a_times.tolist(), truth_rms=truth_rms.tolist(),
                   rel_bare=rel_bare.tolist(), rel_clos=rel_clos.tolist(),
                   final_bare=float(rel_bare[-1]), final_clos=float(rel_clos[-1]))

    # ---- Pareto: bare at a sweep of dt (compared to truth at the last common time) ---- #
    s_end = avail[-1]

    # ---- diagnostics: per-term RMS + error spectrum ---- #
    if args.diag and model_name in ('cheap_deriv', 'deriv_closure'):
        from qg.solver.opt.basis import to_spectral as _ts, to_physical as _tp
        L2 = L_hat ** 2; L3 = L2 * L_hat; L4 = L2 * L2
        print("\n==================== CLOSURE TERM RMS (at IC) ====================")
        with torch.no_grad():
            qh0 = _ts(omega_stack[0]); Nh0 = N_spectral(qh0, derivative, F_hat)
            x0 = assemble_inputs(input_fields, omega_stack, psi_stack, dtype, device)
            Nd, Ndd, N3d = nn_derivs_hat(model, model_name, x0, dtype, target_mean,
                                         target_std, normalize, want_n3dot=True)
            cf = 1.0 / 12.0
            def _rms(spec):
                return float(torch.sqrt((_tp(spec) ** 2).mean()))
            terms = {
                'R3 L^3 w   (anal)': Delta_T**3 * cf * _rms(L3 * qh0),
                'R3 L^2 N   (anal)': Delta_T**3 * cf * _rms(L2 * Nh0),
                'R3 L*Ndot  (NN)':   Delta_T**3 * cf * _rms(L_hat * Nd),
                'R3 5*Nddot (NN)':   Delta_T**3 * cf * 5.0 * _rms(Ndd),
            }
            if args.r4:
                terms['R4 2L^4 w   (anal)'] = Delta_T**4 * cf * 2.0 * _rms(L4 * qh0)
                terms['R4 2L^3 N   (anal)'] = Delta_T**4 * cf * 2.0 * _rms(L3 * Nh0)
                terms['R4 2L^2 Ndot(NN)']   = Delta_T**4 * cf * 2.0 * _rms(L2 * Nd)
                terms['R4 4L*Nddot (NN)']   = Delta_T**4 * cf * 4.0 * _rms(L_hat * Ndd)
                if args.r4_n3dot_coef is not None and N3d is not None:
                    terms['R4 c*N3dot (NN)'] = Delta_T**4 * cf * abs(float(args.r4_n3dot_coef)) * _rms(N3d)
        tot = sum(terms.values()) or 1e-30
        nn_mass = sum(v for k, v in terms.items() if '(NN)' in k)
        for k, v in terms.items():
            print(f"  {k:<20} rms={v:.4e}  {100*v/tot:5.1f}%")
        print(f"  {'-'*46}")
        print(f"  total correction rms = {tot:.4e}")
        print(f"  NN-predicted fraction = {100*nn_mass/tot:.1f}%  -> x the ~3-4% per-derivative "
              f"NN error sets the closure's NN-limited error floor (cf. the ~{rel_bare[-1]/max(rel_clos[-1],1e-30):.0f}x "
              f"improvement).")
    # ---- diagnostic figure (improvement + spectra + error spectra): always saved ----
    try:
        te = s_end * K              # truth index at the last common step
        tf = truth_cp[te]; bf = bare_cp[s_end]; cf = clos_cp[s_end]
        kk, Pt_f = _radial_spectrum(tf)        # spectra of the fields themselves
        _,  Pb_f = _radial_spectrum(bf)
        _,  Pc_f = _radial_spectrum(cf)
        _,  Pb_e = _radial_spectrum(bf - tf)   # spectra of the errors
        _,  Pc_e = _radial_spectrum(cf - tf)
        t_end = s_end * Delta_T
        # resolved band: drop k=0 and the dealiased high-|k| tail (the cliff
        # down to ~1e-9 where modes are zeroed by the 2/3 rule)
        sl = (Pt_f > Pt_f.max() * 1e-8); sl[0] = False

        figd, axd = plt.subplots(1, 3, figsize=(16.5, 4.6))
        # (1) improvement = ||bare err|| / ||ML err|| vs time.  At t=0 both
        # trajectories share the IC (0/0), and the earliest steps sit at the
        # roundoff floor -- either way the ratio blows up to ~1e14. Mask the
        # first point and anything where the closure error is not yet
        # physically resolved, so only the meaningful ratio is plotted.
        imp = rel_bare / np.maximum(rel_clos, 1e-30)
        m = np.isfinite(imp) & (rel_clos > 1e-8); m[0] = False
        axd[0].plot(a_times[m], imp[m], 'o-', ms=3, color='C2')
        axd[0].set_xlabel('physical time')
        axd[0].set_ylabel(IMPROV_LABEL, fontsize=11)
        axd[0].set_title('closure improvement vs time'); axd[0].grid(alpha=0.3)
        # (2) the spectra themselves
        axd[1].loglog(kk[sl], Pt_f[sl], 'k-',  lw=1.2, alpha=0.7, label=LBL_TRUTH)
        axd[1].loglog(kk[sl], Pb_f[sl], 'C0-', lw=1.2, label=LBL_BARE)
        axd[1].loglog(kk[sl], Pc_f[sl], 'C3-', lw=1.2, label=LBL_ML)
        axd[1].set_xlabel(r'radial wavenumber $|k|$')
        axd[1].set_ylabel(r'power spectrum $E(|k|)$')
        axd[1].set_title(f'spectra @ t={t_end:.2f}')
        axd[1].legend(fontsize=8); axd[1].grid(alpha=0.3, which='both')
        # (3) error spectra (field minus truth)
        axd[2].loglog(kk[sl], Pb_e[sl], 'C0-', lw=1.2, label=f'{LBL_BARE} error')
        axd[2].loglog(kk[sl], Pc_e[sl], 'C3-', lw=1.2, label=f'{LBL_ML} error')
        axd[2].set_xlabel(r'radial wavenumber $|k|$')
        axd[2].set_ylabel(r'error power $E_{\mathrm{err}}(|k|)$')
        axd[2].set_title(f'error spectrum @ t={t_end:.2f}')
        axd[2].legend(fontsize=8); axd[2].grid(alpha=0.3, which='both')

        figd.suptitle(f'Online inference results, $K={K}$', fontsize=13)
        figd.text(0.5, 0.005, NORM_DEF, ha='center', fontsize=9)
        figd.tight_layout(rect=[0, 0.03, 1, 0.94])
        figd.savefig(out_dir / f'rollout_diag_{args.ic_tag}.png', dpi=130)
        print(f"[rollout] wrote rollout_diag_{args.ic_tag}.png")
    except Exception as e:                       # noqa: BLE001  (diagnostics must never kill the run)
        print(f"[rollout] diag figure skipped ({type(e).__name__}: {e})")

    pareto = None
    if args.pareto:
        if s_end != M:
            print(f"\n[rollout] PARETO skipped: truth/closure did not reach the full "
                  f"horizon (last common step {s_end} < {M}).")
        else:
            print("\n==================== PARETO (bare dt sweep) ====================")
            factors = [float(x) for x in args.pareto_dt_factors.split(',')]
            pts = []
            for fac in factors:
                dt = Delta_T / fac
                if dt < h_fine * 0.999:
                    continue
                nst = int(round(M * Delta_T / dt))
                bc, wt = rollout_bare(omega_0, omega_m1, dt, nst, [nst],
                                      derivative, L_hat, F_hat, device)
                if nst not in bc:
                    print(f"  bare dt={dt:.3e}  blew up -- skipped"); continue
                err = rel_l2(bc[nst], truth_cp[M * K])
                pts.append((dt, wt, err))
                print(f"  bare dt={dt:.3e}  steps={nst}  wall={wt:.3f}s  rel-L2={err:.4e}")
            pareto = pts
            results['pareto'] = [list(x) for x in pts]

    np.savez(out_dir / f'rollout_timed_{args.ic_tag}.npz',
             **{k: np.asarray(v) for k, v in results.items() if k != 'pareto'},
             clos_final=clos_cp[s_end], bare_final=bare_cp[s_end],
             truth_final=truth_cp[s_end * K])
    (out_dir / f'rollout_timed_{args.ic_tag}.json').write_text(json.dumps(results, indent=2, default=float))

    # ---- figures ---- #
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].semilogy(a_times, rel_bare, 'o-', color='C0', label=LBL_BARE)
    ax[0].semilogy(a_times, rel_clos, 's-', color='C3', label=LBL_ML)
    ax[0].set_xlabel('physical time'); ax[0].set_ylabel(ERR_LABEL, fontsize=10)
    ax[0].legend(fontsize=8); ax[0].set_title('error growth vs truth'); ax[0].grid(alpha=0.3)

    if pareto:
        dts, wts, errs = zip(*pareto)
        ax[1].loglog(wts, errs, 'o-', color='C0', label=r'bare @ varying $\delta t$')
        ax[1].loglog([t_clos], [rel_clos[-1]], 'r*', ms=16, label=LBL_ML)
        ax[1].loglog([t_bare], [rel_bare[-1]], 'C1D', ms=9, label=LBL_BARE)
        # dashed line at the closure's accuracy: where bare must reach to match it
        ax[1].axhline(rel_clos[-1], color='r', ls='--', lw=1, alpha=0.6,
                      label='closure accuracy')
        ax[1].set_xlabel('walltime (s)')
        ax[1].set_ylabel('final  ' + ERR_LABEL, fontsize=10)
        ax[1].legend(fontsize=8); ax[1].set_title('cost / accuracy Pareto'); ax[1].grid(alpha=0.3, which='both')
    else:
        ax[1].axis('off')
    fig.suptitle(f'Online inference results, $K={K}$', fontsize=13)
    fig.text(0.5, 0.005, NORM_DEF, ha='center', fontsize=9)
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.savefig(out_dir / f'rollout_timed_{args.ic_tag}.png', dpi=130)
    print(f"\n[rollout] wrote rollout_timed_{args.ic_tag}.png / .npz / .json in {out_dir}")


if __name__ == '__main__':
    main()
