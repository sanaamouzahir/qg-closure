"""
Von Neumann (frozen-coefficient) linear stability for the QG IMEX scheme as
implemented:  qh^{n+1} = CN2(qh^n, AB_p(N), dt, Lhat)
  CN2:  qh^{n+1} = [ qh^n + dt(0.5*Lhat*qh^n + S) ] / (1 - 0.5*dt*Lhat)
  S   = AB_p extrapolation of the source N = -J(psi,omega) + F  (dealiased)
  Lhat(k) = -nu*|k|^2 - mu + i*B*kx/|k|^2     (implicit, A-stable part)

Linearize the explicit advection about a frozen background velocity U:
  N' = -U.grad omega'   ->  per-mode eigenvalue  lam_N = -i (U.k)
Worst case over advection direction/background:  |U.k| = Umax*|k|.

Per-mode recurrence (AB level p, weights w_0..w_{p-1}, w_0 multiplies N^n):
  D     = 1 - 0.5*dt*Lhat
  alpha = (1 + 0.5*dt*Lhat)/D
  gamma = dt*lam_N / D
  qh^{n+1} = (alpha + gamma*w_0) qh^n + gamma*w_1 qh^{n-1} + ... + gamma*w_{p-1} qh^{n-p+1}
Companion matrix C (pxp); stable iff spectral radius rho(C) <= 1 (+tol).
"""
import numpy as np

AB_W = {2:[3/2,-1/2], 3:[23/12,-4/3,5/12], 4:[55/24,-59/24,37/24,-3/8]}

def kgrid(Nx,Ny,Lx,Ly):
    kx = (2*np.pi/Lx)*np.arange(0, Nx//2+1)              # rfftfreq layout
    ky = (2*np.pi/Ly)*np.fft.fftfreq(Ny)*Ny              # fftfreq integers
    KX,KY = np.meshgrid(kx, ky, indexing='xy')
    KX=KX.ravel(); KY=KY.ravel()
    ksq = KX**2+KY**2
    knyq = min(kx.max(), abs(ky).max())
    kcut = np.sqrt(2)*(1-1/3)*knyq                       # code's dealias cutoff
    keep = np.sqrt(ksq) <= kcut
    return KX[keep], KY[keep], ksq[keep], kcut

def Lhat(KX,KY,ksq,nu,mu,B):
    out = -nu*ksq - mu + 0j
    nz = ksq>0
    out[nz] = out[nz] + 1j*B*KX[nz]/ksq[nz]
    return out

def max_spectral_radius(dt, U, p, KX,KY,ksq,L):
    w = AB_W[p]
    kmag = np.sqrt(ksq)
    D = 1 - 0.5*dt*L
    alpha = (1+0.5*dt*L)/D
    worst = np.zeros(len(L))
    for sgn in (+1.0,-1.0):
        lamN = sgn*1j*U*kmag                 # = -i (U.k), worst-case |U.k|=U|k|
        gamma = dt*lamN/D
        if p==2:                              # closed-form 2x2
            a = alpha + gamma*w[0]; b = gamma*w[1]
            disc = np.sqrt(a*a + 4*b + 0j)
            r = np.maximum(np.abs((a+disc)/2), np.abs((a-disc)/2))
        else:                                 # batched companion eigvals
            M=len(L); C=np.zeros((M,p,p),dtype=complex)
            C[:,0,0]=alpha+gamma*w[0]
            for j in range(1,p): C[:,0,j]=gamma*w[j]
            for j in range(1,p): C[:,j,j-1]=1.0
            r = np.max(np.abs(np.linalg.eigvals(C)),axis=1)
        worst = np.maximum(worst, r)
    return worst.max()

def dt_max(U,p,KX,KY,ksq,L,tol=1e-8,lo=1e-6,hi=1.0):
    if max_spectral_radius(lo,U,p,KX,KY,ksq,L) > 1+tol: return float('nan')
    for _ in range(60):
        mid=np.sqrt(lo*hi)
        if max_spectral_radius(mid,U,p,KX,KY,ksq,L) <= 1+tol: lo=mid
        else: hi=mid
    return lo

CASES = {
 'forced  (512^2, 4pi, nu=1.025e-4, mu=0.02, B=1)':
    dict(Nx=512,Ny=512,Lx=4*np.pi,Ly=4*np.pi,nu=1.025e-4,mu=0.02,B=1.0),
 'decaying(256^2, 2pi, nu=1.025e-5, mu=0,    B=0)':
    dict(Nx=256,Ny=256,Lx=2*np.pi,Ly=2*np.pi,nu=1.025e-5,mu=0.0,B=0.0),
}
for name,c in CASES.items():
    KX,KY,ksq,kcut = kgrid(c['Nx'],c['Ny'],c['Lx'],c['Ly'])
    L = Lhat(KX,KY,ksq,c['nu'],c['mu'],c['B'])
    # subsample modes for the search (smooth boundary); keep highest-|k| shell fully
    idx = np.argsort(np.sqrt(ksq)); idx = np.unique(np.r_[idx[::7], idx[-3000:]])
    KXs,KYs,ksqs,Ls = KX[idx],KY[idx],ksq[idx],L[idx]
    print(f"\n=== {name}   |k|_cut(dealias)={kcut:.1f}, |k|_max kept={np.sqrt(ksq).max():.1f} ===")
    print(f"{'Umax':>6} | {'AB2CN2':>10} {'AB3CN2':>10} {'AB4CN2':>10}   (max stable dt)")
    for U in (0.5,1.0,2.0,5.0):
        row=[dt_max(U,p,KXs,KYs,ksqs,Ls) for p in (2,3,4)]
        print(f"{U:>6.1f} | {row[0]:>10.2e} {row[1]:>10.2e} {row[2]:>10.2e}")
    # CFL constant C in dt_max ~ C/(Umax*kcut) at U=1
    d2=dt_max(1.0,2,KXs,KYs,ksqs,Ls)
    print(f"  -> AB2 CFL const C = dt_max*Umax*kcut = {d2*1.0*kcut:.3f}  (at Umax=1)")
