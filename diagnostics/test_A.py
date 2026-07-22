#!/usr/bin/env python
"""Test A (Phase 1): for BARE, ANALYTIC(r3anal), NN(closure) arms, at TRUTH and
NN-AUGMENTED states, compare the J_NN power-iteration rho (frozen certificate)
to the MEASURED rollout error growth (finite-amplitude two-trajectory divergence,
base evolving = a real rollout). Reports amplitude AND enstrophy (amp^2)."""
import os, sys, json, math
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import nn_amplification as A
import rollout_aposteriori as ra
from qg.solver.grid.cartesian import CartesianGrid
from qg.solver.opt.derivative import Derivative
from qg.solver.opt.basis import to_spectral

QG='/gdata/projects/ml_scope/Closure_modeling/QG-closure'
CK=Path(f'{QG}/qg-simple-package-stable/src/qg/training/data/ensemble_N5_7lag/training_runs/rollout_ft_w31p3_certv2/best.pt')
R=Path('data/ensemble_N5_7lag/FRC-kf4/sweep_dT_5em3')
IC=912; DT=5.0e-3; DEV=[0,10,20,30]; device='cuda' if torch.cuda.is_available() else 'cpu'; N_ITER=150
torch.set_grad_enabled(True)
m=json.loads((R/'manifest.json').read_text())
Nx,Ny=int(m['Nx']),int(m['Ny']); Lx,Ly=float(m['Lx']),float(m['Ly'])
nu=float(m['nu']); mu=float(m.get('mu',0.)); beta=float(m.get('beta',0.))
grid=CartesianGrid(Nx=Nx,Ny=Ny,Lx=Lx,Ly=Ly,device=device,precision='float64')
derivative=Derivative(grid)
for a in ('dx','dy','laplacian','inv_laplacian','alias_mask'):
    if hasattr(derivative,a): setattr(derivative,a,getattr(derivative,a).to(device))
L_hat=A.build_L_hat(derivative,nu,mu,beta).to(device)
fc=m.get('forcing') if m.get('has_forcing') else None
F_phys=A.build_forcing(grid,fc,device,torch.float64); F_hat=to_spectral(F_phys) if F_phys is not None else None
ra._DX,ra._DY=Lx/Nx,Ly/Ny; ra._LX,ra._LY=Lx,Ly
model,mname,S=A.load_deriv_model(CK,m,DT,device,nn_float64=True)
for p in model.parameters(): p.requires_grad_(False)
inf=(['omega_0']+[f'omega_m{k}' for k in range(1,S)]+['psi_0']+[f'psi_m{k}' for k in range(1,S)])
ic_om,ic_ps=A.load_ic_stack(R,IC,m,S,derivative,device)
aug={}
for an,ra_arm in (('bare','bare'),('r3anal','r3anal'),('closure','closure')):
    st=A.build_stepper(ra_arm,ic_om,ic_ps,DT,derivative,L_hat,F_hat,device,model,inf,True,closure_apply='folded')
    aug[an]=A.make_aug_step(st,derivative,F_hat,S)

def measured_rollout(aug_step, base, n_steps=120, eps_rel=1e-2, seed=1):
    """Finite-amplitude two-trajectory divergence; base advances under aug_step
    (a real rollout). Returns asymptotic per-step AMPLITUDE growth + #steps."""
    stack=base.detach().clone()
    g=torch.Generator().manual_seed(seed)
    d=torch.randn(S,stack.shape[1],stack.shape[2],generator=g,dtype=stack.dtype).to(stack.device)
    d0=eps_rel*float(stack.abs().mean().clamp_min(1e-30)); d=d/d.norm()*d0
    logs=[]
    with torch.no_grad():
        for n in range(n_steps):
            y0=aug_step(stack); yp=aug_step(stack+d)
            new=torch.cat([y0[None],stack[:-1]],0); newp=torch.cat([yp[None],(stack+d)[:-1]],0)
            d=newp-new; gg=float(d.norm())/d0
            if not math.isfinite(gg) or gg==0: break
            logs.append(math.log(gg)); d=d/d.norm()*d0; stack=new
            if not torch.isfinite(stack).all(): break
    if len(logs)<6: return float('nan'),len(logs)
    tail=logs[len(logs)//2:]
    return math.exp(sum(tail)/len(tail)), len(logs)

states=A.roll_closure_states(aug['closure'] and A.build_stepper('closure',ic_om,ic_ps,DT,derivative,L_hat,F_hat,device,model,inf,True,closure_apply='folded'),ic_om,ic_ps,S,DEV,derivative,F_hat,device)
print(f"\n===== TEST A (Phase 1): J_NN rho vs MEASURED rollout growth =====",flush=True)
print(f"model={mname} S={S} DT={DT} ic={IC} member=FRC-kf4 nu={nu} mu={mu} beta={beta}",flush=True)
print(f"{'arm':>8} {'state':>7} {'JNN_rho':>9} {'meas_amp':>9} {'meas_enst':>10} {'steps':>6}",flush=True)
for arm in ('bare','r3anal','closure'):
    for s in DEV:
        if s not in states:
            print(f"{arm:>8} {('dev'+str(s)):>7}  (state unavailable)",flush=True); continue
        stk=states[s].to(device)
        rj=A.power_iterate(A.FrozenOperator(aug[arm],stk),n_iter=N_ITER,device=device)['rho']
        ma,ns=measured_rollout(aug[arm],stk)
        kind='truth' if s==0 else f'dev{s}'
        enst = ma*ma if math.isfinite(ma) else float('nan')
        print(f"{arm:>8} {kind:>7} {rj:9.4f} {ma:9.4f} {enst:10.4f} {ns:6d}",flush=True)
print("KEY: closure J_NN should ramp to ~1.8 (amp); measured amp should match (~sqrt(2.85)=1.69); enst~2.85. bare/r3anal ~1 (stable).",flush=True)
