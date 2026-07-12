"""
train_deriv_rollout.py -- rollout-in-the-loss trainer (Sanaa ruling 2026-07-09).

Objective: unroll M coarse AB2CN2+closure steps from a developed-flow deep
window and penalize the TRAJECTORY, not the derivatives:

    loss = mean_{m=1..M}  relL2( omega_rollout_m , omega_truth_m )        (per sample)

Truth = the deep 28-mark builds (data/ensemble_*/<MEMBER>/forced_turbulence_dT_*/,
n_snapshots_per_sample=28, packed memmap: inputs.npy (Nwin, 56, Ny, Nx) float32,
channel k = omega k marks BEFORE the newest, channels 28..55 = psi likewise --
so chronological mark j in 0..27 lives at channel 27-j). Marks are RK4-ultrafine
references spaced Delta_T_deep = 5e-3; rollout dT = stride * Delta_T_deep with
stride s in {1,2,3} <-> dT in {5e-3, 1e-2, 1.5e-2}.

Per window at stride s:
    input stencil : 7 marks at chrono {0, s, ..., 6s}  (newest first for the model)
    truth marks   : chrono 6s + m*s,  m = 1..M
    M_max(s)      : (27 - 6s)//s  ->  21 / 7 / 3  at s = 1 / 2 / 3.

THE STEPPER IS NOT REIMPLEMENTED: `rollout_aposteriori.run_arm(...,
return_stepper=True, nn_grad=True)` hands back the exact validated closure
one_step (minimal-FFT AB2CN2: bare 5 / closure 8 FFTs; coef = dT^3, NO
(1-1/K^n) -- session 7b ruling; implicit L^3 fold, explicit analytic L^2 N,
NN f = (1/12)(L Ndot - 5 Nddot); NN end-projection via the SAME
Derivative.alias_mask, --dealias-nn default ON; psi history carried by the
stepper; cond_local sigma-hat context from the stepper's own spectral states,
zero extra FFTs). Everything in that path is autograd-safe: no in-place ops on
graph tensors, rfft/irfft differentiable, sigma_hat_spec is scatter_add_ on a
fresh zero tensor (has backward). The only edit to rollout_aposteriori was the
conditional no_grad (nn_grad flag) + the stepper export.

Gradient modes:
    --grad-mode full       full BPTT through all M steps.
    --grad-mode trunc:<k>  detach the carried state every k steps: the graph is
                           segmented, each per-step loss term backprops at most
                           k steps -- one backward at the end. NB this bounds
                           GRADIENT DEPTH only, not activation memory (all M
                           segment graphs are alive until backward); use
                           --checkpoint-steps to bound memory.
    --checkpoint-steps N   (full mode) wrap each one_step in
                           torch.utils.checkpoint (use_reentrant=False) when
                           the unroll length M >= N (0 = never). Trades one
                           extra forward per step for O(1) activation memory.

OPTION 2 (Sanaa ruling 2026-07-11, truth-free annulus stability term):
after the M supervised steps, keep rolling K TRUTH-FREE steps and penalize
GROWTH of the aliased-annulus enstrophy Z_ann = sum_{|k| > (2/3) kmax_axis}
|omega_hat|^2 (512^2: mode radius > 170.67; the solver ball caps content at
241.4 -- exactly the band where every observed NN blow-up seeds):

    loss = mean_{m=1..M} relL2(omega_m, truth_m)
         + free_weight * mean_{f=1..K} relu( log Z_ann(f) / Z_ann(f-1) )

The hinge penalizes growth only -- draining the annulus is NOT rewarded
(the p170 projection already showed that collapses the closure's value).
--free-horizon H sets K = max(0, H - M) per (root, stride): every window
is rolled ~H total steps regardless of how much truth it has, which is
how 16-step behaviour at dT=1.5e-2 (M_max=3) becomes trainable at all.
The supervised->free boundary always DETACHES (the approved 'truncated
gradient'); inside both segments --grad-mode trunc:<k> bounds gradient
depth, and in trunc mode each closed segment backward()s immediately, so
activation memory is bounded by k steps, not M+K.

Batching: --batch-size = windows per OPTIMIZER step, executed as B=1 unrolls
with gradient accumulation. Deliberate: the reused stepper is B=1 (it is the
inference stepper), every member keeps its own L_hat/F_hat/derivative with
zero per-sample plumbing, and per-sample dx,dy is trivially exact (hard rule).
Batches never mix members.

Splits: each deep dir's own split.npz (train/val/test window indices) if
present, else a by-window random split is created (never within-window --
windows are whole samples here, so no anchor leakage by construction).
A stack-roughness screen (||omega_c25 - 2 omega_c26 + omega_c27|| / ||omega||
>= --roughness-min on the three oldest marks) drops residual quiescent/zonal
windows (rule 15 safety net; omega itself is O(1) so the relL2 loss cannot
blow up the way the derivative loss did, but a frozen window teaches nothing).

Val metric per epoch: the same unrolled loss at fixed --val-unroll-M, PLUS the
step-1 residual fraction  relL2(closure step 1) / relL2(bare step 1)  per
stride -- the offline-consistency number (Sanaa's 4-arm check, automated; the
bare step-1 errors are model-independent and cached once).

Gates (--gate, exits after; runner: scripts/sge/rollout_gates_job.sh):
    r1 : (a) trainer unroll with the BARE stepper (closure fully off), M=4,
         vs rollout_aposteriori.run_arm('bare') -- must match bit-exactly;
         (b) closure stepper with the NN forced to ZERO vs run_arm('r3only')
         (zero NN != bare: the analytic L^3-implicit + L^2 N terms remain, so
         the correct zero-NN reference is r3only). Prints max|d omega|.
    r2 : M=1 residual fractions on the FIRST deep root at strides 1/2/3 over
         --gate-windows val windows; with the cond ep63 warm start on kf4 the
         means must reproduce ~0.0575 / 0.0586 / 0.0646 (4-arm table).
    r3 : tiny-overfit -- 2 train windows, stride 1, M=2, 50 Adam steps at
         --gate-lr: loss must drop >10x (gradient-flow sanity). gate-lr
         default 1e-3 intentionally exceeds the 5e-5 production cap: this is
         a 2-sample overfit probe, not training; the cap protects pooled
         relative losses from quiescent leverage, which cannot occur here.

Usage (smoke, DO NOT submit without GO):
    python train_deriv_rollout.py \
        --deep-roots data/ensemble_N5/FRC-kf4/forced_turbulence_dT_5em3 \
                     data/ensemble_N5_7lag/FRC-256/forced_turbulence_dT_5em3 \
        --init-ckpt data/ensemble_N5_7lag/training_runs/deriv7_cond_local_v2/best.pt \
        --model auto --unroll-schedule 1:10,2:10,4:10 --lr 5e-5 \
        --batch-size 1 --grad-clip 1.0 --run-name rollout_ft_cond
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint as _torch_checkpoint

import rollout_aposteriori as ra
from rollout_aposteriori import run_arm
from rollout_timed_pareto import (N_spectral, build_L_hat, build_forcing,
                                  psi_from_omega)
from model_deriv_closure import build_model

DT_TAGS = {1: '5e-3', 2: '1e-2', 3: '1.5e-2'}
GATE_R2_EXPECT = {1: 0.0575, 2: 0.0586, 3: 0.0646}   # kf4 / cond ep63 (4-arm)


# --------------------------------------------------------------------------- #
# per-member context                                                           #
# --------------------------------------------------------------------------- #

class RootCtx:
    """One deep root: manifest physics, solver operators, split, mmap."""

    def __init__(self, root: Path, device: str, roughness_min: float,
                 seed: int):
        from qg.solver.grid.cartesian import CartesianGrid
        from qg.solver.opt.derivative import Derivative
        from qg.solver.opt.basis import to_spectral
        self.root = Path(root)
        self.member = self.root.parent.name
        man = json.loads((self.root / 'manifest.json').read_text())
        self.man = man
        self.Nx, self.Ny = int(man['Nx']), int(man['Ny'])
        self.Lx, self.Ly = float(man['Lx']), float(man['Ly'])
        if abs(self.Lx - self.Ly) > 1e-9 or self.Nx != self.Ny:
            raise SystemExit(f"[{self.member}] ANISOTROPIC domain "
                             f"(Lx!=Ly or Nx!=Ny) -- refused (hard rule).")
        self.dx, self.dy = self.Lx / self.Nx, self.Ly / self.Ny
        self.dt_deep = float(man['Delta_T'])
        self.n_marks = int(man['n_snapshots_per_sample'])
        nu = float(man['nu']); mu = float(man.get('mu', 0.0))
        beta = float(man.get('beta', 0.0))
        grid = CartesianGrid(Nx=self.Nx, Ny=self.Ny, Lx=self.Lx, Ly=self.Ly,
                             device=device, precision='float64')
        self.derivative = Derivative(grid)
        for attr in ('dx', 'dy', 'laplacian', 'inv_laplacian', 'alias_mask'):
            if hasattr(self.derivative, attr):
                setattr(self.derivative, attr,
                        getattr(self.derivative, attr).to(device))
        self.L_hat = build_L_hat(self.derivative, nu, mu, beta).to(device)
        fc = man.get('forcing') if man.get('has_forcing') else None
        self.F_phys = build_forcing(grid, fc, device, torch.float64)
        self.F_hat = (to_spectral(self.F_phys)
                      if self.F_phys is not None else None)
        p = self.root / 'packed' / 'inputs.npy'
        if not p.exists():
            p = self.root / 'inputs.npy'
        self.inputs = np.load(p, mmap_mode='r')      # (Nwin, 2*n_marks, Ny, Nx)
        Nwin = self.inputs.shape[0]
        assert self.inputs.shape[1] == 2 * self.n_marks, \
            f"{self.member}: channel count {self.inputs.shape[1]} != 2*{self.n_marks}"

        # ---- split: use the deep dir's own split.npz if present ----
        sp = self.root / 'split.npz'
        if not sp.exists():
            sp = self.root / 'packed' / 'split.npz'
        if sp.exists():
            s = np.load(sp)
            self.train_idx = [int(i) for i in s['train_idx']]
            self.val_idx = [int(i) for i in s['val_idx']]
            self.split_src = str(sp)
        else:                                        # by-window random split
            rng = np.random.default_rng(seed)
            perm = rng.permutation(Nwin)
            n_tr = int(0.70 * Nwin); n_va = int(0.15 * Nwin)
            self.train_idx = [int(i) for i in perm[:n_tr]]
            self.val_idx = [int(i) for i in perm[n_tr:n_tr + n_va]]
            self.split_src = 'by-window random (no split.npz found)'

        # ---- roughness screen (rule-15 safety net) on the 3 OLDEST marks ----
        self.n_dropped = 0
        if roughness_min > 0:
            keep_tr, keep_va = [], []
            c = self.n_marks - 1                     # oldest chrono channel
            for lst, keep in ((self.train_idx, keep_tr),
                              (self.val_idx, keep_va)):
                for w in lst:
                    o0 = np.asarray(self.inputs[w, c], np.float64)
                    o1 = np.asarray(self.inputs[w, c - 1], np.float64)
                    o2 = np.asarray(self.inputs[w, c - 2], np.float64)
                    rough = (np.sqrt(np.mean((o0 - 2 * o1 + o2) ** 2))
                             / max(np.sqrt(np.mean(o0 ** 2)), 1e-30))
                    if rough >= roughness_min:
                        keep.append(w)
                    else:
                        self.n_dropped += 1
            self.train_idx, self.val_idx = keep_tr, keep_va
        print(f"[root] {self.member:10s} {self.Ny}x{self.Nx} L={self.Lx:.4f} "
              f"beta={beta} nu={nu} mu={mu} marks={self.n_marks} "
              f"dt_deep={self.dt_deep}  train={len(self.train_idx)} "
              f"val={len(self.val_idx)} dropped_quiescent={self.n_dropped}  "
              f"forced={self.F_phys is not None}  split={self.split_src}")

    def m_max(self, stride: int) -> int:
        return (self.n_marks - 1 - 6 * stride) // stride

    def window_tensors(self, win: int, stride: int, M: int, device: str):
        """(omega_stack newest-first [7 x (1,H,W)], truth [M x (H,W)]) f64."""
        assert 1 <= M <= self.m_max(stride), \
            f"M={M} outside 1..{self.m_max(stride)} at stride {stride} " \
            f"(negative channels would wrap into the psi block)"
        nm = self.n_marks - 1                                     # = 27
        hist_ch = [nm - 6 * stride + i * stride for i in range(7)]
        truth_ch = [nm - (6 + m) * stride for m in range(1, M + 1)]
        arr = np.asarray(self.inputs[win, hist_ch + truth_ch], np.float64)
        t = torch.as_tensor(arr, dtype=torch.float64, device=device)
        omega_stack = [t[i][None] for i in range(7)]
        truth = [t[7 + m] for m in range(M)]
        return omega_stack, truth

    def annulus(self, device):
        """Boolean rfft-grid mask of the aliased annulus: mode radius
        r > (2/3)*(min(N)/2). 512^2: r > 170.67 (the solver's sqrt2 ball
        already zeroes r > 241.36, so no upper bound is needed -- those
        modes carry exactly zero energy). Half-plane note: the kx=0 column
        is single-counted vs the doubled interior, but Z_ann only ever
        enters as a RATIO under one fixed mask, so the growth measure is
        weighting-invariant."""
        dev = torch.device(device)
        if getattr(self, '_ann', None) is None or self._ann.device != dev:
            ky = torch.fft.fftfreq(self.Ny, d=1.0 / self.Ny)
            kx = torch.fft.rfftfreq(self.Nx, d=1.0 / self.Nx)
            r = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
            self._ann = (r > (2.0 / 3.0) * (min(self.Ny, self.Nx) / 2.0)
                         ).to(dev)
        return self._ann


def annulus_enstrophy(qh, mask):
    """Z_ann = sum_mask |omega_hat|^2 (graph-carrying torch scalar)."""
    return (qh.real ** 2 + qh.imag ** 2)[..., mask].sum()


def set_globals(rc: RootCtx):
    """rollout_aposteriori's arm constants read module globals; pin them to
    THIS member before stepper creation and before every unroll (the cond
    sigma-hat path reads _LX/_LY at call time)."""
    ra._DX, ra._DY = rc.dx, rc.dy
    ra._LX, ra._LY = rc.Lx, rc.Ly


def make_stepper(rc: RootCtx, stride: int, model, device, arm='closure',
                 dealias_nn=True, nn_grad=True):
    """The EXACT validated stepper from rollout_aposteriori (return_stepper)."""
    set_globals(rc)
    dummy = [torch.zeros(1, rc.Ny, rc.Nx, dtype=torch.float64, device=device)]
    input_fields = None
    if arm.startswith('closure'):
        input_fields = (['omega_0'] + [f'omega_m{k}' for k in range(1, 7)]
                        + ['psi_0'] + [f'psi_m{k}' for k in range(1, 7)])
    return run_arm(arm, dummy, dummy, stride * rc.dt_deep, 0, [],
                   rc.derivative, rc.L_hat, rc.F_hat, device,
                   model=model, input_fields=input_fields,
                   dealias_nn=dealias_nn, nn_grad=nn_grad,
                   return_stepper=True)


class _ZeroNN(torch.nn.Module):
    """Gate R1(b): closure arm with the NN forced to exact zero."""

    def forward(self, x, dt=None, dx=None, dy=None, **kw):
        return torch.zeros(x.shape[0], 3, x.shape[-2], x.shape[-1],
                           dtype=x.dtype, device=x.device)


# --------------------------------------------------------------------------- #
# unroll                                                                       #
# --------------------------------------------------------------------------- #

def init_state(rc: RootCtx, omega_stack, device):
    """Same init sequence as run_arm's own loop (bit-compat for gate R1)."""
    from qg.solver.opt.basis import to_spectral
    psi_stack = [psi_from_omega(o, rc.derivative) for o in omega_stack]
    qh_curr = to_spectral(omega_stack[0])
    qh_minus = to_spectral(omega_stack[1])
    Nh_curr = N_spectral(qh_curr, rc.derivative, rc.F_hat)
    Nh_minus = N_spectral(qh_minus, rc.derivative, rc.F_hat)
    om = [s.clone() for s in omega_stack]
    ps = [s.clone() for s in psi_stack]
    return qh_curr, qh_minus, Nh_curr, Nh_minus, om, ps


def unroll_losses(rc: RootCtx, one_step, omega_stack, truth, M,
                  is_closure=True, trunc_k=0, use_checkpoint=False,
                  free_K=0, free_weight=1.0, ann_mask=None,
                  backward_scale=None, free_cap=10.0):
    """Drive one_step M supervised steps (per-step relL2 vs truth), then
    free_K TRUTH-FREE steps collecting the annulus stability terms
    relu(log Z_ann(f)/Z_ann(f-1)) (OPTION 2 -- see module docstring).

    Returns (losses, stab_terms, blown). Graph-carrying torch scalars,
    UNLESS backward_scale is set (train + trunc mode): then each segment
    backward()s as it closes -- segment loss = (sum_seg sup)/M
    + free_weight*(sum_seg stab)/free_K, scaled by backward_scale -- the
    carried state detaches at the boundary, and PLAIN FLOATS are returned.
    Gradient-identical to one final backward (a segment's loss terms reach
    parameters only through that segment's steps); activation memory is
    bounded by trunc_k steps instead of M+free_K.

    Stops early on non-finite state (returns what it has + blown=True);
    in the free segment the already-collected finite growth terms keep
    their gradient -- a blow-up still teaches."""
    from qg.solver.opt.basis import to_physical
    assert free_K == 0 or (is_closure and ann_mask is not None), \
        "free rolling needs the closure arm and an annulus mask"
    assert backward_scale is None or trunc_k, \
        "segment backward is only defined at trunc boundaries"
    set_globals(rc)
    qc, qm, Nc, Nm, om, ps = init_state(rc, omega_stack,
                                        omega_stack[0].device)
    n_snap = len(om)

    def step_flat(qc, qm, Nc, Nm, *hist):
        return one_step(qc, qm, Nc, Nm,
                        list(hist[:n_snap]), list(hist[n_snap:]))

    losses, stab, blown = [], [], False
    seg_sup, seg_stab = [], []              # open-segment terms (graph mode:
                                            # aliases of losses/stab entries)

    def flush_segment():
        if backward_scale is None:
            seg_sup.clear(); seg_stab.clear()
            return
        terms = []
        if seg_sup:
            terms.append(torch.stack(seg_sup).sum() / max(M, 1))
        if seg_stab:
            terms.append(free_weight * torch.stack(seg_stab).sum()
                         / max(free_K, 1))
        if terms:
            (backward_scale * sum(terms)).backward()
        seg_sup.clear(); seg_stab.clear()

    def detach_state():
        nonlocal qc, qm, Nc, Nm, om, ps
        qc, qm, Nc, Nm = (v.detach() for v in (qc, qm, Nc, Nm))
        om = [v.detach() for v in om]
        ps = [v.detach() for v in ps]

    # ---- supervised segment ----
    for m in range(1, M + 1):
        if use_checkpoint:
            out = _torch_checkpoint(
                step_flat, qc, qm, Nc, Nm, *om, *ps, use_reentrant=False)
        else:
            out = step_flat(qc, qm, Nc, Nm, *om, *ps)
        qh_new, Nh_new, om_new, ps_new = out
        if is_closure:
            om = [om_new] + om[:-1]
            ps = [ps_new] + ps[:-1]
            om_m = om_new[0]                          # physical, free (8-FFT path)
        else:
            om_m = to_physical(qh_new)[0]
        qm, qc = qc, qh_new
        Nm, Nc = Nc, Nh_new
        if not torch.isfinite(om_m).all():
            blown = True
            break
        t = truth[m - 1]
        rel = (torch.linalg.vector_norm(om_m - t)
               / torch.linalg.vector_norm(t).clamp_min(1e-30))
        if backward_scale is not None:
            losses.append(float(rel))
        else:
            losses.append(rel)
        seg_sup.append(rel)
        if trunc_k and m % trunc_k == 0 and (m < M or free_K):
            flush_segment()                           # truncated BPTT boundary
            detach_state()

    # ---- truth-free segment (annulus stability) ----
    if free_K and not blown:
        flush_segment()
        detach_state()      # the approved truncated gradient: stability
        z_prev = None       # gradient lives in the free segment only
        for f in range(1, free_K + 1):
            qh_new, Nh_new, om_new, ps_new = step_flat(qc, qm, Nc, Nm,
                                                       *om, *ps)
            om = [om_new] + om[:-1]
            ps = [ps_new] + ps[:-1]
            qm, qc = qc, qh_new
            Nm, Nc = Nc, Nh_new
            if not torch.isfinite(om_new[0]).all():
                blown = True
                break
            if z_prev is None:                        # Z at segment entry
                z_prev = annulus_enstrophy(qm, ann_mask).clamp_min(1e-300)
            z = annulus_enstrophy(qc, ann_mask).clamp_min(1e-300)
            if not torch.isfinite(z):
                # |omega_hat|^2 overflows LONG before the field itself goes
                # non-finite (e^4.6/step growth at 1.5e-2 overflows the
                # squared sum by step ~10 while omega is still ~1e150) --
                # a finite-field check alone lets log(inf) poison backward
                # (epoch-0 incident, run 1830425). Treat as blown.
                blown = True
                break
            # cap the hinge: log-growth beyond free_cap/step is already
            # unambiguous blow-up; an uncapped term makes the free segment's
            # gradient scale be set by the most-exploded window
            g = torch.relu(torch.log(z / z_prev)).clamp_max(free_cap)
            if backward_scale is not None:
                stab.append(float(g))
            else:
                stab.append(g)
            seg_stab.append(g)
            z_prev = z
            if trunc_k and f % trunc_k == 0 and f < free_K:
                flush_segment()
                detach_state()
                z_prev = z_prev.detach()
    flush_segment()
    return losses, stab, blown


# --------------------------------------------------------------------------- #
# model                                                                        #
# --------------------------------------------------------------------------- #

def build_or_load_model(args, ref_rc: RootCtx, device):
    """--model auto|cheap_deriv|cond_local (+ optional --init-ckpt warm start).

    Warm-start load is strict on PARAMETERS: missing keys are tolerated ONLY
    if they are registered buffers of the freshly built model (older ckpts
    predate buffers like dt_ref_cond -- the 'pre-buffer' compat); any
    unexpected key or missing parameter is a hard error."""
    name, cfg = args.model, {}
    if args.init_ckpt is not None:
        cfg_path = args.init_ckpt.parent / 'config.json'
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
        if name == 'auto':
            name = cfg.get('model', 'cheap_deriv')
            print(f"[model] --model auto -> {name} "
                  f"({'from ' + str(cfg_path) if cfg_path.exists() else 'default'})")
    elif name == 'auto':
        name = 'cheap_deriv'
    n_snap = int(cfg.get('n_snapshots', args.n_snapshots))
    if n_snap != args.n_snapshots:
        raise SystemExit(f"[model] ckpt n_snapshots={n_snap} != "
                         f"--n-snapshots {args.n_snapshots}")
    gk = int(cfg.get('grad_kernel', args.grad_kernel))
    sd = None
    if args.init_ckpt is not None:
        state = torch.load(args.init_ckpt, map_location=device,
                           weights_only=False)
        sd = state.get('model', state.get('model_state',
                                          state.get('state_dict', state)))
        for k, v in sd.items():
            if k.endswith('grad.wx'):
                gk = int(v.shape[-1])
    model = build_model(name, in_channels=2 * n_snap,
                        out_orders=args.out_orders, n_time=n_snap,
                        grad_kernel=gk, dt=args.strides[0] * ref_rc.dt_deep,
                        dx=ref_rc.dx, dy=ref_rc.dy, physics_init=True,
                        learnable_stencils=True)
    if sd is not None:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        buffers = {k for k, _ in model.named_buffers()}
        bad_missing = [k for k in missing if k not in buffers]
        if unexpected:
            raise SystemExit(f"[model] UNEXPECTED ckpt keys: {unexpected}")
        if bad_missing:
            raise SystemExit(f"[model] MISSING parameter keys: {bad_missing}")
        if missing:
            print(f"[model] pre-buffer ckpt: keeping fresh buffers {missing}")
        print(f"[model] warm start from {args.init_ckpt} "
              f"(epoch={state.get('epoch', '?')})")
    model.to(device=device, dtype=torch.float64)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {name}  n_snapshots={n_snap}  grad_kernel={gk}  "
          f"trainable params={n_par:,}  dtype=float64")
    return model, name, n_snap


# --------------------------------------------------------------------------- #
# curriculum                                                                   #
# --------------------------------------------------------------------------- #

def parse_schedule(sched: str | None, unroll_M: int, epochs: int):
    """'1:10,2:10,4:10' = M:duration segments; epoch -> M (last M persists)."""
    if not sched:
        return [unroll_M] * epochs
    seg = []
    for part in sched.split(','):
        m, dur = part.split(':')
        seg += [int(m)] * int(dur)
    if len(seg) < epochs:
        seg += [seg[-1]] * (epochs - len(seg))
    return seg[:epochs]


# --------------------------------------------------------------------------- #
# gates                                                                        #
# --------------------------------------------------------------------------- #

def gate_r1(rc: RootCtx, device):
    print("\n===== GATE R1: trainer unroll vs validated run_arm (M=4) =====")
    win = (rc.val_idx or rc.train_idx)[0]
    omega_stack, truth = rc.window_tensors(win, 1, 4, device)
    ok = True
    for arm_ref, arm_step, mdl, label in (
            ('bare', 'bare', None, 'bare      (closure fully OFF)'),
            ('r3only', 'closure', _ZeroNN().to(device).double(),
             'r3only vs zero-NN closure')):
        one_step = make_stepper(rc, 1, mdl, device, arm=arm_step,
                                nn_grad=False)
        with torch.no_grad():
            # trainer-side unroll, capturing the fields for the comparison
            from qg.solver.opt.basis import to_physical
            set_globals(rc)
            qc, qm, Nc, Nm, om, ps = init_state(rc, omega_stack, device)
            mine = {}
            for m in range(1, 5):
                qh_new, Nh_new, om_new, ps_new = one_step(qc, qm, Nc, Nm,
                                                          om, ps)
                if arm_step == 'closure':
                    om = [om_new] + om[:-1]; ps = [ps_new] + ps[:-1]
                    mine[m] = om_new[0].cpu().numpy()
                else:
                    mine[m] = to_physical(qh_new)[0].cpu().numpy()
                qm, qc = qc, qh_new; Nm, Nc = Nc, Nh_new
            set_globals(rc)
            psi_stack = [psi_from_omega(o, rc.derivative)
                         for o in omega_stack]
            ref = run_arm(arm_ref, omega_stack, psi_stack, rc.dt_deep, 4,
                          [0, 1, 2, 3, 4], rc.derivative, rc.L_hat, rc.F_hat,
                          device, model=mdl,
                          input_fields=(['omega_0']
                                        + [f'omega_m{k}' for k in range(1, 7)]
                                        + ['psi_0']
                                        + [f'psi_m{k}' for k in range(1, 7)]))
        dmax = max(float(np.abs(mine[m] - ref['fields'][m]).max())
                   for m in range(1, 5))
        scale = float(np.abs(ref['fields'][4]).max())
        passed = dmax <= 1e-13 * max(scale, 1.0)
        ok &= passed
        print(f"  {label:32s} max|d omega| over steps 1..4 = {dmax:.3e} "
              f"(field scale {scale:.3e})  -> {'PASS' if passed else 'FAIL'}")
    print(f"===== GATE R1 {'PASS' if ok else 'FAIL'} =====")
    return ok


def residual_fractions(rc: RootCtx, model, device, strides, windows,
                       bare_cache=None, dealias_nn=True):
    """mean/median over windows of relL2(closure step1)/relL2(bare step1)."""
    out = {}
    for s in strides:
        clos = make_stepper(rc, s, model, device, arm='closure',
                            dealias_nn=dealias_nn, nn_grad=False)
        bare = make_stepper(rc, s, None, device, arm='bare', nn_grad=False)
        fr = []
        with torch.no_grad():
            for w in windows:
                omega_stack, truth = rc.window_tensors(w, s, 1, device)
                lc, _, bc = unroll_losses(rc, clos, omega_stack, truth, 1,
                                          is_closure=True)
                key = (rc.member, s, w)
                if bare_cache is not None and key in bare_cache:
                    lb = bare_cache[key]
                else:
                    lbl, _, _ = unroll_losses(rc, bare, omega_stack, truth, 1,
                                              is_closure=False)
                    lb = float(lbl[0]) if lbl else float('nan')
                    if bare_cache is not None:
                        bare_cache[key] = lb
                if bc or not lc or not np.isfinite(lb) or lb < 1e-30:
                    continue
                fr.append(float(lc[0]) / lb)
        out[s] = (float(np.mean(fr)) if fr else float('nan'),
                  float(np.median(fr)) if fr else float('nan'), len(fr))
    return out


def gate_r2(rc: RootCtx, model, device, n_windows):
    print(f"\n===== GATE R2: step-1 residual fraction, member={rc.member}, "
          f"{n_windows} val windows =====")
    wins = (rc.val_idx or rc.train_idx)[:n_windows]
    out = residual_fractions(rc, model, device, [1, 2, 3], wins)
    is_kf4 = 'kf4' in rc.member
    ok = True
    print(f"  {'dT':>8} {'mean':>10} {'median':>10} {'n':>4}"
          + ("   expected(4-arm kf4/IC837)" if is_kf4 else ""))
    for s in (1, 2, 3):
        mean, med, n = out[s]
        line = f"  {DT_TAGS[s]:>8} {mean:>10.4f} {med:>10.4f} {n:>4}"
        if is_kf4:
            exp = GATE_R2_EXPECT[s]
            line += f"   {exp:.4f}"
            # single-IC table vs multi-window mean: allow 30% relative slack
            if not (np.isfinite(mean) and abs(mean - exp) <= 0.30 * exp):
                ok = False
        print(line)
    verdict = ('PASS' if ok else 'FAIL') if is_kf4 else 'INFO (non-kf4 root)'
    print(f"===== GATE R2 {verdict} =====")
    return ok


def gate_r3(rc: RootCtx, model, device, lr, grad_clip):
    print(f"\n===== GATE R3: perturbed-init recovery (2 windows, M=2, "
          f"50 Adam steps, lr={lr}) =====")
    # Physics init already sits at the loss floor (~5e-7) on the easy stride-1
    # M=2 probe, so "drop 10x from init" is unpassable-by-construction (first
    # run 2026-07-09: loss rose then recovered — gradients flow, gate failed).
    # Instead: PERTURB the trainable weights, then demand recovery to within
    # 10x of the unperturbed floor. Tests forward+backward+step end-to-end
    # with a target that is actually reachable.
    wins = rc.train_idx[:2]
    data = [rc.window_tensors(w, 1, 2, device) for w in wins]
    one_step = make_stepper(rc, 1, model, device, arm='closure', nn_grad=True)

    def probe_loss():
        with torch.no_grad():
            tot = 0.0
            for omega_stack, truth in data:
                losses, _, blown = unroll_losses(rc, one_step, omega_stack,
                                                 truth, 2, is_closure=True)
                if blown or not losses:
                    raise SystemExit("[gate r3] rollout blew up -- FAIL")
                tot += float(torch.stack(losses).mean())
        return tot / len(data)

    floor = probe_loss()
    gen = torch.Generator(device='cpu').manual_seed(20260709)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.mul_(1.0 + 0.25 * torch.randn(p.shape, generator=gen,
                                                dtype=p.dtype).to(p.device))
    perturbed = probe_loss()
    print(f"  unperturbed floor={floor:.4e}   after perturbation="
          f"{perturbed:.4e}  ({perturbed / max(floor, 1e-30):.1f}x floor)")
    if perturbed < 10.0 * floor:
        raise SystemExit("[gate r3] perturbation did not move the loss -- "
                         "probe invalid, FAIL")
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(trainable, lr=lr)
    model.train(True)
    first = last = None
    for it in range(50):
        optim.zero_grad()
        tot = 0.0
        for omega_stack, truth in data:
            losses, _, blown = unroll_losses(rc, one_step, omega_stack,
                                             truth, 2, is_closure=True)
            if blown or not losses:
                raise SystemExit("[gate r3] rollout blew up -- FAIL")
            loss = torch.stack(losses).mean() / len(data)
            loss.backward()
            tot += float(loss) * len(data)
        torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        optim.step()
        tot /= len(data)
        if it == 0:
            first = tot
        last = tot
        if it % 10 == 0 or it == 49:
            print(f"  iter {it:3d}  loss={tot:.6e}")
    ok = (last < first / 10.0) or (last < 10.0 * floor)
    print(f"  loss {first:.4e} -> {last:.4e}  ({first / max(last, 1e-30):.1f}x"
          f" drop; floor {floor:.4e})  -> "
          f"{'PASS' if ok else 'FAIL (neither >10x drop nor near floor)'}")
    print(f"===== GATE R3 {'PASS' if ok else 'FAIL'} =====")
    return ok


def gate_r4(rc: RootCtx, model, device, grad_clip):
    """OPTION-2 wiring probe: 2 windows, M=2 supervised + K=4 free steps,
    trunc:2 segment backward. PASS = all terms finite, gradients populated,
    and 10 Adam steps stay finite. A zero stability term is HEALTHY here
    (relu hinge: the warm model is stable on a 6-step horizon) -- the gate
    checks plumbing, not that the penalty is active."""
    print("\n===== GATE R4: option-2 free-segment wiring (M=2 + K=4, "
          "trunc:2, segment backward) =====")
    wins = rc.train_idx[:2]
    mask = rc.annulus(device)
    one_step = make_stepper(rc, 1, model, device, arm='closure', nn_grad=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(trainable, lr=1e-6)
    model.train(True)
    ok = True
    for it in range(10):
        optim.zero_grad()
        sup_v, stab_v = [], []
        for w in wins:
            omega_stack, truth = rc.window_tensors(w, 1, 2, device)
            losses, stab, blown = unroll_losses(
                rc, one_step, omega_stack, truth, 2, is_closure=True,
                trunc_k=2, free_K=4, free_weight=1.0, ann_mask=mask,
                backward_scale=1.0 / len(wins))
            if blown:
                print("  rollout blew up -- FAIL"); ok = False
            sup_v += losses; stab_v += stab
        gnorm = float(torch.nn.utils.clip_grad_norm_(trainable, grad_clip))
        optim.step()
        finite = (all(np.isfinite(v) for v in sup_v + stab_v)
                  and np.isfinite(gnorm))
        ok &= finite and gnorm > 0.0
        if it in (0, 9):
            print(f"  iter {it}  sup={np.mean(sup_v):.4e}  "
                  f"stab={np.mean(stab_v) if stab_v else 0.0:.4e}  "
                  f"|grad|={gnorm:.4e}  finite={finite}")
    print(f"===== GATE R4 {'PASS' if ok else 'FAIL'} =====")
    return ok


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--deep-roots', type=Path, nargs='+', required=True,
                   help='deep 28-mark dirs (forced_turbulence_dT_*), one per '
                        'member; NOT the sliced sweep_dT_* dirs')
    p.add_argument('--strides', type=str, default='1,2,3',
                   help='mark strides = dT/5e-3 (1,2,3 -> 5e-3,1e-2,1.5e-2)')
    p.add_argument('--unroll-M', type=int, default=1,
                   help='fixed unroll length (per-stride clamp to M_max '
                        '= (27-6s)/s -> 21/7/3); overridden by --unroll-schedule')
    p.add_argument('--unroll-schedule', type=str, default=None,
                   help="curriculum 'M:dur,M:dur,...' e.g. '1:10,2:10,4:10' "
                        "(M=1 for 10 epochs, then 2, then 4; last M persists)")
    p.add_argument('--grad-mode', type=str, default='full',
                   help="'full' (BPTT through all M steps) or 'trunc:<k>' "
                        "(detach carried state every k steps)")
    p.add_argument('--checkpoint-steps', type=int, default=0,
                   help='full mode: torch.utils.checkpoint each step when '
                        'M >= this value (0 = never)')
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--init-ckpt', type=Path, default=None,
                   help='warm start (best.pt from train_deriv.py); buffers '
                        'missing from older ckpts are tolerated')
    p.add_argument('--model', default='auto',
                   choices=['auto', 'cheap_deriv', 'cond_local'],
                   help="'auto' reads config.json next to --init-ckpt "
                        "(fallback cheap_deriv)")
    p.add_argument('--n-snapshots', type=int, default=7)
    p.add_argument('--out-orders', type=int, default=3)
    p.add_argument('--grad-kernel', type=int, default=15)
    p.add_argument('--dealias-nn', action=argparse.BooleanOptionalAction,
                   default=True,
                   help='end-project the NN correction (Derivative.alias_mask)')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=5e-5,
                   help='production cap 5e-5 (quiescent postmortem rule)')
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--batch-size', type=int, default=1,
                   help='windows per optimizer step (B=1 unrolls, grad accum)')
    p.add_argument('--windows-per-epoch', type=int, default=0,
                   help='random train windows per (root,stride) per epoch '
                        '(0 = all)')
    p.add_argument('--val-windows', type=int, default=24,
                   help='val windows per (root,stride) (0 = all)')
    p.add_argument('--val-unroll-M', type=int, default=4,
                   help='fixed val unroll length (per-stride clamp)')
    p.add_argument('--free-horizon', type=int, default=0,
                   help='OPTION 2: roll every window to ~this many total '
                        'steps, K = max(0, H - M) truth-free (0 = off)')
    p.add_argument('--free-steps', type=int, default=0,
                   help='OPTION 2: fixed truth-free step count K '
                        '(ignored when --free-horizon is set)')
    p.add_argument('--free-weight', type=str, default='1.0',
                   help='weight of the annulus stability hinge: one value '
                        'for all strides, or a comma list matched to '
                        '--strides in order (per-stride lever, e.g. '
                        '1.0e-3,1.0e-3,2.5e-4 -- the s1 hinge is inactive '
                        'on a stable model anyway; the list lets s3 trade '
                        'less accuracy for damping than s2)')
    p.add_argument('--free-cap', type=float, default=10.0,
                   help='clamp on the per-step log-growth hinge (gradient '
                        'is zero above the cap; pre-blowup sub-cap steps '
                        'carry the signal)')
    p.add_argument('--val-free-steps', type=int, default=16,
                   help='val probe: truth-free steps rolled from the '
                        'stencil (blow-up fraction + annulus growth)')
    p.add_argument('--val-free-windows', type=int, default=8,
                   help='val windows per (root,stride) for the free probe')
    p.add_argument('--roughness-min', type=float, default=1e-4,
                   help='drop windows with stack roughness below this '
                        '(rule-15 safety net; 0 disables)')
    p.add_argument('--compute-dtype', choices=['float64'], default='float64',
                   help='float64 mandatory (closure signal ~ dT^3)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--run-name', type=str, default=None)
    p.add_argument('--out-root', type=Path, default=None)
    p.add_argument('--print-every', type=int, default=1)
    p.add_argument('--gate', type=str, default=None,
                   choices=['r1', 'r2', 'r3', 'r4', 'all'],
                   help='run acceptance gate(s) and exit (no training)')
    p.add_argument('--gate-windows', type=int, default=16)
    p.add_argument('--gate-lr', type=float, default=1e-3,
                   help='gate r3 overfit lr (probe only, not training)')
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = (args.device if (args.device == 'cpu'
                              or torch.cuda.is_available()) else 'cpu')
    args.strides = strides = [int(s) for s in args.strides.split(',')]
    # --free-weight: scalar broadcast or per-stride list (order = --strides)
    fw = [float(x) for x in str(args.free_weight).split(',')]
    if len(fw) == 1:
        fw = fw * len(strides)
    if len(fw) != len(strides):
        raise SystemExit(f"--free-weight has {len(fw)} values for "
                         f"{len(strides)} strides; give 1 or one per stride")
    args.free_weight_map = dict(zip(strides, fw))
    trunc_k = 0
    if args.grad_mode != 'full':
        if not args.grad_mode.startswith('trunc:'):
            raise SystemExit(f"--grad-mode {args.grad_mode!r}: use 'full' or "
                             f"'trunc:<k>'")
        trunc_k = int(args.grad_mode.split(':')[1])

    # ---- roots ----
    rcs = [RootCtx(r, device, args.roughness_min, args.seed)
           for r in args.deep_roots]
    members = [rc.member for rc in rcs]
    if len(set(members)) != len(members):
        raise SystemExit(f"duplicate member names across --deep-roots "
                         f"({members}): steppers/bare_cache are keyed by "
                         f"member -- pass one deep dir per member")
    for rc in rcs:
        for s in strides:
            if rc.m_max(s) < 1:
                raise SystemExit(f"[{rc.member}] stride {s}: no truth marks "
                                 f"beyond the stencil (M_max<1)")
    # reference grid for model construction = most common full grid
    from collections import Counter
    cnt = Counter((rc.Ny, rc.Nx, rc.Lx, rc.Ly) for rc in rcs)
    ref_key = cnt.most_common(1)[0][0]
    ref_rc = next(rc for rc in rcs
                  if (rc.Ny, rc.Nx, rc.Lx, rc.Ly) == ref_key)

    # ---- model ----
    model, model_name, n_snap = build_or_load_model(args, ref_rc, device)
    if n_snap != 7:
        raise SystemExit("stencil depth != 7 unsupported by the deep-window "
                         "indexing (chrono 0..6s)")

    # ---- gates (exit before training) ----
    if args.gate:
        ok = True
        if args.gate in ('r1', 'all'):
            ok &= gate_r1(rcs[0], device)
        if args.gate in ('r2', 'all'):
            ok &= gate_r2(rcs[0], model, device, args.gate_windows)
        if args.gate in ('r3', 'all'):
            ok &= gate_r3(rcs[0], model, device, args.gate_lr, args.grad_clip)
        if args.gate in ('r4', 'all'):
            ok &= gate_r4(rcs[0], model, device, args.grad_clip)
        raise SystemExit(0 if ok else 1)

    # ---- steppers: one per (root, stride), capture the trained model ----
    steppers = {(rc.member, s): make_stepper(rc, s, model, device,
                                             arm='closure',
                                             dealias_nn=args.dealias_nn,
                                             nn_grad=True)
                for rc in rcs for s in strides}

    schedule = parse_schedule(args.unroll_schedule, args.unroll_M, args.epochs)
    trainable = [q for q in model.parameters() if q.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr,
                              weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim,
                                                       T_max=args.epochs)

    out_root = args.out_root or rcs[0].root.parent.parent
    run = args.run_name or f"rollout_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = out_root / 'training_runs' / run
    run_dir.mkdir(parents=True, exist_ok=True)
    free_on = bool(args.free_horizon or args.free_steps)
    (run_dir / 'config.json').write_text(json.dumps(
        vars(args) | {'model': model_name, 'n_snapshots': n_snap,
                      'objective': ('rollout_relL2+annulus_stab'
                                    if free_on else 'rollout_relL2'),
                      'schedule_M': schedule,
                      'deep_roots': [str(r.root) for r in rcs]},
        indent=2, default=str))
    log = run_dir / 'log.csv'
    # column names follow diagnostics/monitor_training.py conventions:
    # train_relL2/val_relL2/best_val are its headline columns; per-stride
    # val_s{1,2,3} land in its generic per-order ('val_*') track. Option-2
    # columns: train_stab (mean hinge), fb_s* (val free-roll blow-up
    # fraction over --val-free-steps), ag_s* (median max annulus
    # log-growth per step on surviving free rolls).
    log.write_text('epoch,lr,M_train,train_relL2,val_relL2,best_val,'
                   + ','.join(f'val_s{s}' for s in strides) + ','
                   + ','.join(f'rf_mean_s{s}' for s in strides) + ','
                   + ','.join(f'rf_med_s{s}' for s in strides)
                   + ',n_blown,n_blown_val,train_stab,n_skip,'
                   + ','.join(f'fb_s{s}' for s in strides) + ','
                   + ','.join(f'ag_s{s}' for s in strides)
                   + ',elapsed_s\n')
    print(f"[rollout-train] run={run}  strides={strides}  "
          f"grad_mode={args.grad_mode}  checkpoint_steps={args.checkpoint_steps}  "
          f"free_weight={args.free_weight_map}  "
          f"schedule={schedule[:12]}{'...' if len(schedule) > 12 else ''}")

    bare_cache = {}                       # (member, stride, win) -> bare step-1

    def train_epoch(ep, M_epoch, rng):
        model.train(True)
        samples = []
        for rc in rcs:
            for s in strides:
                wins = list(rc.train_idx)
                if args.windows_per_epoch and len(wins) > args.windows_per_epoch:
                    wins = list(rng.choice(wins, args.windows_per_epoch,
                                           replace=False))
                samples += [(rc, s, w) for w in wins]
        # Generator.shuffle rejects lists of tuples -- permute indices instead
        samples = [samples[i] for i in rng.permutation(len(samples))]
        tot, nb, n_blown = 0.0, 0, 0
        # chunked grad accumulation: each chunk is one optimizer step, and the
        # per-sample loss is scaled by the ACTUAL chunk size (the ragged last
        # chunk would otherwise be under-weighted).
        # INVARIANT: backward() runs inside the same chunk iteration as the
        # forward, BEFORE the next sample's set_globals(rc) repins
        # rollout_aposteriori._LX/_LY -- do not defer backward across samples
        # (checkpoint recompute / the cond sigma-hat context would read
        # another member's globals).
        tot_stab, n_skip = 0.0, 0
        for c0 in range(0, len(samples), args.batch_size):
            chunk = samples[c0:c0 + args.batch_size]
            optim.zero_grad()
            n_ok = 0
            for rc, s, w in chunk:
                M = min(M_epoch, rc.m_max(s))
                fw_s = args.free_weight_map[s]
                K = (max(0, args.free_horizon - M) if args.free_horizon
                     else args.free_steps)
                use_cp = (args.checkpoint_steps > 0
                          and M >= args.checkpoint_steps and trunc_k == 0)
                omega_stack, truth = rc.window_tensors(w, s, M, device)
                # trunc mode: unroll_losses backward()s per segment (memory
                # bounded by trunc_k) and returns floats; full mode: graph
                # tensors, one backward here.
                bscale = (1.0 / len(chunk)) if trunc_k else None
                losses, stab, blown = unroll_losses(
                    rc, steppers[(rc.member, s)], omega_stack, truth, M,
                    is_closure=True, trunc_k=trunc_k, use_checkpoint=use_cp,
                    free_K=K, free_weight=fw_s,
                    ann_mask=rc.annulus(device) if K else None,
                    backward_scale=bscale, free_cap=args.free_cap)
                if blown:
                    n_blown += 1
                if not losses and not stab:
                    continue
                if bscale is None:
                    # full mode: graph tensors -- one backward here
                    sup_t = torch.stack(losses).mean() if losses else None
                    stab_t = torch.stack(stab).mean() if stab else None
                    terms = ([sup_t] if sup_t is not None else []) \
                        + ([fw_s * stab_t]
                           if stab_t is not None else [])
                    (sum(terms) / len(chunk)).backward()
                    sup = float(sup_t) if sup_t is not None else 0.0
                    stv = float(stab_t) if stab_t is not None else 0.0
                else:
                    # trunc mode: unroll_losses already backward()ed per
                    # segment and returned plain floats
                    sup = (float(np.mean(losses)) if losses else 0.0)
                    stv = (float(np.mean(stab)) if stab else 0.0)
                if losses:
                    tot += sup; nb += 1
                tot_stab += stv
                n_ok += 1
            if n_ok:
                gn = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                if torch.isfinite(gn):
                    optim.step()
                else:
                    # non-finite gradient: one poisoned step would NaN the
                    # weights and kill the whole run (epoch-0 incident,
                    # 1830425) -- drop this chunk's update instead
                    n_skip += 1
                    optim.zero_grad()
        optim.zero_grad()
        return tot / max(nb, 1), tot_stab / max(nb, 1), n_blown, n_skip

    def val_epoch():
        model.train(False)
        per_stride, rf = {}, {}
        n_blown_val = 0
        with torch.no_grad():
            for s in strides:
                vals = []
                for rc in rcs:
                    M = min(args.val_unroll_M, rc.m_max(s))
                    wins = rc.val_idx[:args.val_windows or None]
                    for w in wins:
                        omega_stack, truth = rc.window_tensors(w, s, M, device)
                        losses, _, blown = unroll_losses(
                            rc, steppers[(rc.member, s)], omega_stack, truth,
                            M, is_closure=True)
                        # blown windows are COUNTED, not averaged: one inf
                        # would otherwise poison the whole stride and best-
                        # ckpt selection. n_blown_val is logged; a nonzero
                        # count IS the instability signal.
                        if blown or not losses:
                            n_blown_val += 1
                            continue
                        vals.append(float(torch.stack(losses).mean()))
                per_stride[s] = float(np.mean(vals)) if vals else float('nan')
        # free-roll probe: --val-free-steps truth-free steps from the raw
        # stencil -- blow-up fraction + median max annulus log-growth. THE
        # stability number to watch (the supervised val cannot see past
        # M_max: 3 steps at stride 3).
        fb, ag = {}, {}
        with torch.no_grad():
            for s in strides:
                blown_n, tot_n, growth = 0, 0, []
                for rc in rcs:
                    mask = rc.annulus(device)
                    for w in rc.val_idx[:args.val_free_windows or None]:
                        omega_stack, _ = rc.window_tensors(w, s, 1, device)
                        _, stab_t, blown = unroll_losses(
                            rc, steppers[(rc.member, s)], omega_stack, [], 0,
                            is_closure=True, free_K=args.val_free_steps,
                            ann_mask=mask)
                        tot_n += 1
                        if blown:
                            blown_n += 1
                        elif stab_t:
                            growth.append(max(float(g) for g in stab_t))
                fb[s] = blown_n / max(tot_n, 1)
                ag[s] = float(np.median(growth)) if growth else float('nan')
        for s in strides:
            means, meds, ns = [], [], 0
            for rc in rcs:
                wins = rc.val_idx[:args.val_windows or None]
                out = residual_fractions(rc, model, device, [s], wins,
                                         bare_cache=bare_cache,
                                         dealias_nn=args.dealias_nn)
                m, md, n = out[s]
                if n:
                    means.append(m * n); meds.append(md); ns += n
            rf[s] = ((sum(means) / ns, float(np.mean(meds)))
                     if ns else (float('nan'), float('nan')))
        pooled = float(np.mean([v for v in per_stride.values()
                                if np.isfinite(v)]))
        return pooled, per_stride, rf, n_blown_val, fb, ag

    best = float('inf'); t0 = time.time()
    rng = np.random.default_rng(args.seed)
    for ep in range(args.epochs):
        te0 = time.time()
        M_epoch = schedule[ep]
        tr, tr_stab, n_blown, n_skip = train_epoch(ep, M_epoch, rng)
        va, va_s, rf, n_blown_val, fb, ag = val_epoch()
        sched.step()
        improved = va < best
        payload = {'model': model.state_dict(), 'epoch': ep, 'val': va,
                   'config': vars(args) | {'model': model_name,
                                           'n_snapshots': n_snap,
                                           'grad_kernel': args.grad_kernel,
                                           'out_orders': args.out_orders}}
        if improved:
            best = va
            torch.save(payload, run_dir / 'best.pt')
        torch.save(payload, run_dir / 'last.pt')
        with open(log, 'a') as f:
            f.write(f'{ep},{optim.param_groups[0]["lr"]:.3e},{M_epoch},'
                    f'{tr:.6e},{va:.6e},{best:.6e},'
                    + ','.join(f'{va_s[s]:.6e}' for s in strides) + ','
                    + ','.join(f'{rf[s][0]:.6e}' for s in strides) + ','
                    + ','.join(f'{rf[s][1]:.6e}' for s in strides)
                    + f',{n_blown},{n_blown_val},{tr_stab:.6e},{n_skip},'
                    + ','.join(f'{fb[s]:.4f}' for s in strides) + ','
                    + ','.join(f'{ag[s]:.6e}' for s in strides)
                    + f',{time.time() - t0:.1f}\n')
        if ep % args.print_every == 0 or improved:
            vs = ' '.join(f's{s}={va_s[s]:.3e}' for s in strides)
            rs = ' '.join(f's{s}={rf[s][0]:.4f}/{rf[s][1]:.4f}'
                          for s in strides)
            fs = ' '.join(f's{s}={fb[s]:.2f}/{ag[s]:.2e}' for s in strides)
            print(f"  ep {ep:3d} {'*' if improved else ' '} M={M_epoch}  "
                  f"train={tr:.4e}(+stab {tr_stab:.2e})  val={va:.4e}  "
                  f"best={best:.4e}  [{vs}]  rf(mean/med)=[{rs}]  "
                  f"free(blown/grow)=[{fs}]  blown={n_blown}/val:{n_blown_val}"
                  f"  skip={n_skip}  ({time.time() - te0:.1f}s)")

    print(f"\n[rollout-train] done in {(time.time() - t0) / 60:.1f} min, "
          f"best val={best:.4e}  -> {run_dir}")


if __name__ == '__main__':
    main()
