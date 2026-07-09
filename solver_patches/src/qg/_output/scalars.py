r"""scalars.py -- opt-in per-step scalar recorder (SGS-closure branch, AMENDMENT_01 SC).

Activated ONLY when the run config has a `qg.diag.scalar_rate` key; without it
`ScalarRecorder.build` returns None and every hook in qg.py / bc.py /
obstacle.py is a no-op (`if rec is not None` / `getattr(state,'_rec',None)`),
leaving the solver bit-identical to the unhooked code path (Gate 1 check 1).

Config block (all keys under qg.diag; only scalar_rate is required):
    diag:
      scalar_rate: 10          # record every N steps (10 = 2.5e-3 t.u. at dt 2.5e-4)
      out: /abs/path/scalars.npz   # default: ./scalars.npz in the job cwd
      length: 1.256637         # normalizing length (default 2*mask.r for circular)
      u_mid: 2.0               # fixed-U normalization for Cd_mid/Cl_mid
      flush_every: 2000        # samples per atomic rewrite (kill-safety window)
      probes: [[x1,y1], ...]   # optional explicit physical probe coords
                               # (REQUIRED for non-circular masks, e.g. cape)

FORCE DERIVATION (from the discrete update as implemented -- obstacle.py:53-70):
The solver's explicit vorticity RHS contains the Brinkman patch
    sponge_h = ( -dx(chi*(v - v_o)) + dy(chi*(u - u_o)) ) / eta,
    eta = qg.pde.penalty * dt   (the YAML's factor*dt convention).
This is exactly curl_z of the momentum sink
    F_fluid = -chi * (u - u_o, v - v_o) / eta        [per unit area, rho = 1],
since curl_z F = dx F_y - dy F_x reproduces sponge_h term by term. The force ON
THE OBSTACLE is the reaction, integrated over the domain:
    Fx = (1/eta) * sum(chi * (u - u_o)) * dx * dy,
    Fy = (1/eta) * sum(chi * (v - v_o)) * dx * dy,
with u_o = v_o = 0 for the static masks used on this branch. chi is the
3x3-Gaussian-SMOOTHED mask exactly as the solver applies it (obstacle.solve_mask),
and u, v are the SOURCE-TIME physical velocities INCLUDING the mean inlet flow:
the bc patch is first in the operator patch list, so it has already pinned
uh[...,0,0] = U_inlet when the penalty evaluates. The recorder therefore stashes
the penalty's own chi*(u-u_o), chi*(v-v_o) tensors (obstacle.py hook) -- the
discrete momentum sink actually applied that step -- and reduces them only on
sampled steps. eta and the resulting PHYSICAL eta value are stored in meta.

SAMPLING CONVENTION: a row is written for every step index n with n % rate == 0:
fields at t_n = n*dt (source time, mean flow included), forces from the step
t_n -> t_n + dt, U_inlet = the value the bc actually assigned during that step
(read back from the state path, not from the table -- Gate 1 check 3 overlays
this against the prescribed table). E = 0.5*<u^2+v^2> and Z = 0.5*<q^2> are
DOMAIN MEANS (matching DNS_FR_diagnostics.npz), but E here INCLUDES the
mean-flow energy: E_deviation = E - 0.5*U_inlet^2. Z is computed from the t_n
spectral state by Parseval (norm='forward': mean(q^2) = sum w_k |qh_k|^2,
w = 2 on interior rfft columns). No extra FFTs are performed at any step.

COST (SC.2 of the amendment, reproduced): at scalar_rate=10 and dt=2.5e-4 the
scalar series has ~830 samples per FASTEST shedding period (T_shed = 2.08 at
Re 5600) -- sampling-noise-free spectra/instantaneous-frequency estimates; the
storage is ~48k samples x ~25 scalars x 8 B < 10 MB/run. Field snapshots stay
at save_rate=1000 (0.25 t.u., 8.3 per fastest period): fine for Pi_FF and
mean/rms statistics, NOT used for St. Recorder work is reductions-only on every
rate-th step (no FFTs), so walltime overhead is well under the 2% budget; if
Gate 1 measures more, rate 20 is proposed before anything changes.

Output scalars.npz keys: t, step (n,), then (n,B) arrays U_inlet, Re_inlet,
Fx, Fy, Cd_inst, Cl_inst, Cd_mid, Cl_mid, U_cyl, Re_cyl, E, Z,
probe_u, probe_v (n, B, n_probes), and meta (json string: every definition,
probe coordinates and indices, eta, conventions). Flushed atomically
(tmp + os.replace) every flush_every samples -- a killed job loses at most one
flush interval.
"""
from __future__ import annotations
import json
import os

import numpy as np
import torch


def _get_diag(raw_param):
    raw = raw_param.get('qg', raw_param) if isinstance(raw_param, dict) else {}
    d = raw.get('diag', None)
    if not d or 'scalar_rate' not in d:
        return None, raw
    return d, raw


class ScalarRecorder:
    """Built by QG._run; stashed on the state as `state._rec` so the bc and
    penalty hooks can hand over per-step quantities without recomputation."""

    @staticmethod
    def build(raw_param, grid, steps):
        diag, raw = _get_diag(raw_param)
        if diag is None:
            return None
        return ScalarRecorder(diag, raw, grid, steps)

    def __init__(self, diag, raw, grid, steps):
        self.rate = int(diag['scalar_rate'])
        self.out = str(diag.get('out', 'scalars.npz'))
        self.flush_every = int(diag.get('flush_every', 2000))
        self.u_mid = float(diag.get('u_mid', 2.0))
        self.grid = grid
        self.dxdy = float(grid.dx) * float(grid.dy)
        self.nu = float(raw['pde']['nu'])
        self.dt = float(raw['time']['dt'])
        self.eta = float(raw['pde']['penalty']) * self.dt   # physical eta, as applied

        mask_conf = raw.get('mask', None) or {}
        self.mask_function = mask_conf.get('function', 'none')
        if 'length' in diag:
            self.length = float(diag['length'])
        elif self.mask_function == 'circular':
            self.length = 2.0 * float(mask_conf['r'])
        else:
            raise ValueError('qg.diag.length is required for non-circular masks')

        # probe bookkeeping is completed lazily on the first sampled step,
        # when chi is available (stashed by the penalty hook) to locate the
        # obstacle centroid and mask out solid points.
        self.explicit_probes = [tuple(map(float, p)) for p in diag.get('probes', [])] or None
        self._geom_ready = False

        n_rows = (steps - 1) // self.rate + 1
        self.n_rows = n_rows
        self.i = 0                     # rows written
        self._since_flush = 0
        self._want = False
        self._stash = {}
        self._alloc_done = False

        self.meta = dict(
            scalar_rate=self.rate, dt=self.dt, eta=self.eta,
            penalty_factor=float(raw['pde']['penalty']), nu=self.nu,
            length=self.length, u_mid=self.u_mid,
            mask_function=self.mask_function,
            force='Fx = sum(chi*u)*dx*dy/eta on the SMOOTHED chi as applied; '
                  'source-time u,v incl. mean inlet; see module docstring',
            sampling='row at every step n % rate == 0; fields at t_n, force over '
                     '[t_n, t_n+dt], U_inlet as assigned by the bc that step',
            E='0.5*<u^2+v^2> domain mean, INCLUDES mean-flow energy',
            Z='0.5*<q^2> domain mean via Parseval on qh(t_n)',
        )

    # ---- hooks (called from qg.py loop / bc.py / obstacle.py) -------------- #

    def begin_step(self, it, state):
        self._want = (it % self.rate == 0)
        if self._want:
            self._stash = {'it': it, 'qh': state.qh}   # t_n spectral vorticity

    def stash_inlet(self, inlet_velocity):
        if self._want:
            self._stash['U'] = float(inlet_velocity)

    def stash_penalty(self, u, v, u_chi, v_chi, chi):
        if self._want:
            self._stash.update(u=u, v=v, u_chi=u_chi, v_chi=v_chi, chi=chi)

    def after_step(self, it, state):
        if not self._want:
            return
        s, B = self._stash, state.qh.shape[0]
        if not self._alloc_done:
            self._alloc(B, s)
        row = self.i

        self.t_arr[row] = it * self.dt
        self.step_arr[row] = it
        U = s.get('U', np.nan)
        self.d['U_inlet'][row] = U
        self.d['Re_inlet'][row] = U * self.length / self.nu

        if 'u_chi' in s:
            fx = (s['u_chi'].sum(dim=(-2, -1)) * (self.dxdy / self.eta)).cpu().numpy()
            fy = (s['v_chi'].sum(dim=(-2, -1)) * (self.dxdy / self.eta)).cpu().numpy()
            self.d['Fx'][row], self.d['Fy'][row] = fx, fy
            qdyn_inst = U * U * self.length          # 0.5*rho*U^2*D, x2 in coeff
            qdyn_mid = self.u_mid * self.u_mid * self.length
            self.d['Cd_inst'][row] = 2.0 * fx / qdyn_inst if U else np.nan
            self.d['Cl_inst'][row] = 2.0 * fy / qdyn_inst if U else np.nan
            self.d['Cd_mid'][row] = 2.0 * fx / qdyn_mid
            self.d['Cl_mid'][row] = 2.0 * fy / qdyn_mid

        if 'u' in s:
            u, v = s['u'], s['v']                     # t_n physical, mean included
            self.d['E'][row] = (0.5 * (u * u + v * v)).mean(dim=(-2, -1)).cpu().numpy()
            if not self._geom_ready:
                self._setup_geometry(s['chi'])
            iy, ix = self._upwin
            band_u = u[..., iy[0]:iy[1], ix]
            if self._upwin_keep is not None:
                band_u = band_u[..., self._upwin_keep]
            self.d['U_cyl'][row] = band_u.mean(dim=-1).cpu().numpy()
            self.d['Re_cyl'][row] = self.d['U_cyl'][row] * self.length / self.nu
            for k, (py, px) in enumerate(self._probe_idx):
                self.d['probe_u'][row, :, k] = u[..., py, px].cpu().numpy()
                self.d['probe_v'][row, :, k] = v[..., py, px].cpu().numpy()

        # Z by Parseval on qh(t_n): mean(q^2) = sum w |qh|^2 (norm='forward')
        qh = s['qh']
        w = self._parseval_w
        self.d['Z'][row] = (0.5 * (w * (qh.real ** 2 + qh.imag ** 2))
                            .sum(dim=(-2, -1))).cpu().numpy()

        self._stash = {}
        self.i += 1
        self._since_flush += 1
        if self._since_flush >= self.flush_every:
            self.flush()

    def close(self):
        self.flush()

    # ---- internals ---------------------------------------------------------- #

    def _alloc(self, B, s):
        n = self.n_rows
        self.t_arr = np.full(n, np.nan)
        self.step_arr = np.zeros(n, dtype=np.int64)
        names = ['U_inlet', 'Re_inlet', 'Fx', 'Fy', 'Cd_inst', 'Cl_inst',
                 'Cd_mid', 'Cl_mid', 'U_cyl', 'Re_cyl', 'E', 'Z']
        self.d = {k: np.full((n, B), np.nan) for k in names}
        n_probes = len(self.explicit_probes) if self.explicit_probes else 5
        self.d['probe_u'] = np.full((n, B, n_probes), np.nan)
        self.d['probe_v'] = np.full((n, B, n_probes), np.nan)
        Nx, Ny = self.grid.Nx, self.grid.Ny
        wrow = torch.full((1, 1, Nx // 2 + 1), 2.0, device=s['qh'].device,
                          dtype=torch.float64)
        wrow[..., 0] = 1.0
        if Nx % 2 == 0:
            wrow[..., -1] = 1.0
        self._parseval_w = wrow
        self._alloc_done = True

    def _setup_geometry(self, chi):
        """Locate the obstacle from the smoothed chi actually applied; place the
        upstream window (1.5*L ahead, |y-yc|<=L/2, solid points excluded) and the
        wake probes (xc+{1,2,3}L, yc) and (xc+L, yc+-L/2) -- or explicit ones."""
        g, L = self.grid, self.length
        dx, dy = float(g.dx), float(g.dy)
        c2 = chi[0] if chi.dim() == 3 else chi
        tot = float(c2.sum())
        ys = torch.arange(g.Ny, device=c2.device, dtype=torch.float64) * dy
        xs = torch.arange(g.Nx, device=c2.device, dtype=torch.float64) * dx
        xc = float((c2.sum(dim=0) * xs).sum() / tot)
        yc = float((c2.sum(dim=1) * ys).sum() / tot)
        self.meta['obstacle_centroid_xy'] = [xc, yc]

        ix_up = int(round((xc - 1.5 * L) / dx)) % g.Nx
        iy_lo = int(round((yc - 0.5 * L) / dy))
        iy_hi = int(round((yc + 0.5 * L) / dy)) + 1
        self._upwin = ((iy_lo, iy_hi), ix_up)
        col_chi = c2[iy_lo:iy_hi, ix_up]
        keep = (col_chi < 0.5)
        self._upwin_keep = None if bool(keep.all()) else keep
        self.meta['upstream_window'] = dict(
            x=ix_up * dx, y_range=[iy_lo * dy, (iy_hi - 1) * dy],
            excluded_solid_points=int((~keep).sum()))

        if self.explicit_probes:
            coords = self.explicit_probes
        else:
            coords = [(xc + 1 * L, yc), (xc + 2 * L, yc), (xc + 3 * L, yc),
                      (xc + 1 * L, yc + 0.5 * L), (xc + 1 * L, yc - 0.5 * L)]
        self._probe_idx = [(int(round(y / dy)) % self.grid.Ny,
                            int(round(x / dx)) % self.grid.Nx) for x, y in coords]
        self.meta['probes_xy_requested'] = [list(p) for p in coords]
        self.meta['probes_ij_used'] = [list(p) for p in self._probe_idx]
        self._geom_ready = True

    def flush(self):
        if not self._alloc_done or self.i == 0:
            return
        n = self.i
        payload = dict(t=self.t_arr[:n], step=self.step_arr[:n],
                       meta=json.dumps(self.meta))
        for k, v in self.d.items():
            payload[k] = v[:n]
        # Write through an open file handle: np.savez APPENDS '.npz' to any
        # str/Path filename not already ending in it, so savez('x.npz.tmp')
        # silently writes 'x.npz.tmp.npz' and the os.replace below dies with
        # FileNotFoundError at the first flush (observed: Gate-1 recorder
        # arms, jobs 1828232-34, 2026-07-09, step 20000 = flush_every*rate).
        # A file object is used verbatim; atomic-replace semantics preserved.
        tmp = self.out + '.tmp'
        with open(tmp, 'wb') as fh:
            np.savez(fh, **payload)
        os.replace(tmp, self.out)
        self._since_flush = 0
