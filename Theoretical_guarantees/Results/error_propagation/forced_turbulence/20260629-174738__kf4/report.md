# error_propagation -- forced_turbulence -- kf4

*20260629-174738*  |  host `ibgpu-compute-0-0.local`  |  git `unknown`  |  rc 0  |  34.0s

## Command
```
python closure_error_propagation.py ../data/ensemble_N5/FRC-kf4/forced_turbulence_dT_5em3 --scheme ab2cn2 --orders 3 4 --n-samples 32 --device cuda
```

## Note
per-op eps -> delta; default eps

## How to read this
How per-operator errors on the learned N-derivatives propagate to delta.
- The loss-on-derivatives (~3%/op) is NOT the closure error; the closure
  error is the L^k-weighted combination reported here.
- Watch L^{p-k}: a 3% error on N_dot inside R4 (L^2, amplified) hurts far
  more than 3% on N_ddot in R3 (L^0). RMS = independent; L1 = worst-case
  aligned; correlated = errors aligned with the terms.

## Output
```
/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-env/lib/python3.11/site-packages/torch/cuda/__init__.py:58: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
  ! forced member but no --forcing given: using F=0 (N-derivative high-k structure is cascade-dominated, so magnitudes are representative; pass --forcing for the exact field).

member = forced_turbulence_dT_5em3   scheme=ab2cn2  orders=[3, 4]
  grid 512x512  L=12.57  nu=0.000102 mu=0.02 beta=1  Delta_T=0.005  (avg over 32 samples)
  eps: N_dot=0.026  N_ddot=0.030  N_3dot=0.040

  ||delta|| (full assembled closure) = 9.9825e-06

  order           term  L^pow  |coef|  ||weighted term||    eps  err contrib  /||delta||
  --------------------------------------------------------------------------------------
      3      L^1 N_dot      1       1         4.0072e-07  0.026   1.0419e-08       0.10%
      3     L^0 N_ddot      0       5         9.9195e-06  0.030   2.9759e-07       2.98%
      4      L^2 N_dot      2       2         2.1577e-09  0.026   5.6101e-11       0.00%
      4     L^1 N_ddot      1       4         1.7499e-08  0.030   5.2496e-10       0.01%
      4     L^0 N_3dot      0       1         2.7554e-08  0.040   1.1022e-09       0.01%
  --------------------------------------------------------------------------------------
  closure error / ||delta||:
     RMS (independent errors)         = 2.98%
     L1  (worst case, all aligned)    = 3.10%
     correlated field (err aligned w/ terms) = 2.98%

  explicit error fields (||.||/||delta||):
     dR3 = (dT^3/12) [ +1 L^1(0.026*N_dot) -5 L^0(0.030*N_ddot) ]   ->  2.98%
     dR4 = (dT^4/24) [ +2 L^2(0.026*N_dot) -4 L^1(0.030*N_ddot) +1 L^0(0.040*N_3dot) ]   ->  0.01%
```
