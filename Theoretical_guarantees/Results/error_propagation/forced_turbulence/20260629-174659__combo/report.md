# error_propagation -- forced_turbulence -- combo

*20260629-174659*  |  host `ibgpu-compute-0-0.local`  |  git `unknown`  |  rc 0  |  38.1s

## Command
```
python closure_error_propagation.py ../data/ensemble_N5/FRC-combo/forced_turbulence_dT_5em3 --scheme ab2cn2 --orders 3 4 --n-samples 32 --device cuda
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
  grid 512x512  L=12.57  nu=4e-05 mu=0.02 beta=0.5  Delta_T=0.005  (avg over 32 samples)
  eps: N_dot=0.026  N_ddot=0.030  N_3dot=0.040

  ||delta|| (full assembled closure) = 7.7753e-05

  order           term  L^pow  |coef|  ||weighted term||    eps  err contrib  /||delta||
  --------------------------------------------------------------------------------------
      3      L^1 N_dot      1       1         9.5276e-07  0.026   2.4772e-08       0.03%
      3     L^0 N_ddot      0       5         7.7676e-05  0.030   2.3303e-06       3.00%
      4      L^2 N_dot      2       2         2.2183e-09  0.026   5.7677e-11       0.00%
      4     L^1 N_ddot      1       4         6.0630e-08  0.030   1.8189e-09       0.00%
      4     L^0 N_3dot      0       1         3.1441e-07  0.040   1.2576e-08       0.02%
  --------------------------------------------------------------------------------------
  closure error / ||delta||:
     RMS (independent errors)         = 3.00%
     L1  (worst case, all aligned)    = 3.05%
     correlated field (err aligned w/ terms) = 3.00%

  explicit error fields (||.||/||delta||):
     dR3 = (dT^3/12) [ +1 L^1(0.026*N_dot) -5 L^0(0.030*N_ddot) ]   ->  3.00%
     dR4 = (dT^4/24) [ +2 L^2(0.026*N_dot) -4 L^1(0.030*N_ddot) +1 L^0(0.040*N_3dot) ]   ->  0.02%
```
