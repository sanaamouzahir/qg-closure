# convergence_radius -- forced_turbulence -- Re25k-combo-kf4

*20260629-173037*  |  host `ibgpu-compute-0-0.local`  |  git `unknown`  |  rc 0  |  27.5s

## Command
```
python convergence_radius.py --sources ../data/ensemble_N5/FRC-Re25k/forced_turbulence_dT_5em3 ../data/ensemble_N5/FRC-combo/forced_turbulence_dT_5em3 ../data/ensemble_N5/FRC-kf4/forced_turbulence_dT_5em3 --max-order 5 --n-samples 32 --wall-k 7 --device cuda --dtype float64
```

## Note
3 deep survivors; outer dT* + inner k=7 wall

## How to read this
Two NESTED Delta_T walls per member.
- **OUTER dT\*** (root test / Cauchy-Hadamard, the headline): radius of the
  modified-equation series. Past it, adding higher analytic R_p DIVERGES --
  no finite-order closure helps. Set by the cascade's N-derivative growth
  (smaller dT\* <- faster growth <- higher Re / lower beta).
- **INNER stencil wall**: where a fixed-lag backward FD stops recovering
  omega_ddot. Hit FIRST (inside dT\*). Below it the binding limit is finite
  lags, so more lags help; between inner and outer, more lags help and the
  series still converges; past dT\* nothing finite does.
- The ratio column is a Cauchy-Hadamard BRACKET, not an estimate; trust the
  root value. Both are finite-p (p=3..6) estimates -> ~20-30%.

## Output
```
/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-env/lib/python3.11/site-packages/torch/cuda/__init__.py:58: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]

=== FRC-Re25k ===  nu=4.00e-05 mu=2.00e-02 beta=1  Re~2.5e+04  grid=512x512
  series-coefficient magnitudes  ||c_p|| = ||R_p||/D_p  (median over anchors):
    p=3: ||c_p|| = 3.3312e+03    root  c_p^(-1/p) = 6.6958e-02
    p=4: ||c_p|| = 2.4561e+04    root  c_p^(-1/p) = 7.9880e-02
    p=5: ||c_p|| = 1.4315e+06    root  c_p^(-1/p) = 5.8728e-02
    p=6: ||c_p|| = 1.2389e+07    root  c_p^(-1/p) = 6.5739e-02
  dT* (root test / Cauchy-Hadamard, median over p) = 6.635e-02
  ratio-test BRACKET (sandwiches dT*, not an estimate): [1.716e-02, 1.356e-01]  (spread = non-geometric c_p)
  N-derivative growth  ||N^(m+1)||/||N^(m)||  (median):
    m=0->1: 34.2x
    m=1->2: 57.8x
    m=2->3: 72.3x
    m=3->4: 83.5x
    m=4->5: 92.6x
  inner stencil wall (k=7, omega_ddot relL2 vs truth, thr=0.5):
    dt=0.005: relL2=8.844e-03
    dt=0.01: relL2=1.902e-01
    dt=0.015: relL2=8.695e-01
    dt=0.02: relL2=2.379e+00
    -> inner wall ~ 1.294e-02
  >>> OUTER dT* (series radius) = 6.635e-02   |   INNER stencil wall (k=7) ~ 1.294e-02

=== FRC-combo ===  nu=4.00e-05 mu=2.00e-02 beta=0.5  Re~2.5e+04  grid=512x512
  series-coefficient magnitudes  ||c_p|| = ||R_p||/D_p  (median over anchors):
    p=3: ||c_p|| = 3.2901e+02    root  c_p^(-1/p) = 1.4485e-01
    p=4: ||c_p|| = 1.2180e+03    root  c_p^(-1/p) = 1.6927e-01
    p=5: ||c_p|| = 3.6920e+04    root  c_p^(-1/p) = 1.2205e-01
    p=6: ||c_p|| = 1.7655e+05    root  c_p^(-1/p) = 1.3351e-01
  dT* (root test / Cauchy-Hadamard, median over p) = 1.392e-01
  ratio-test BRACKET (sandwiches dT*, not an estimate): [3.299e-02, 2.701e-01]  (spread = non-geometric c_p)
  N-derivative growth  ||N^(m+1)||/||N^(m)||  (median):
    m=0->1: 14.7x
    m=1->2: 26.8x
    m=2->3: 36.8x
    m=3->4: 45x
    m=4->5: 51.1x
  inner stencil wall (k=7, omega_ddot relL2 vs truth, thr=0.5):
    dt=0.005: relL2=1.633e-03
    dt=0.01: relL2=8.474e-03
    dt=0.015: relL2=5.392e-02
    dt=0.02: relL2=1.940e-01
    -> inner wall > (beyond reach) 2.000e-02
  >>> OUTER dT* (series radius) = 1.392e-01   |   INNER stencil wall (k=7) ~ 2.000e-02

=== FRC-kf4 ===  nu=1.02e-04 mu=2.00e-02 beta=1  Re~9.76e+03  grid=512x512
  series-coefficient magnitudes  ||c_p|| = ||R_p||/D_p  (median over anchors):
    p=3: ||c_p|| = 1.0299e+02    root  c_p^(-1/p) = 2.1334e-01
    p=4: ||c_p|| = 2.8114e+02    root  c_p^(-1/p) = 2.4421e-01
    p=5: ||c_p|| = 6.5986e+03    root  c_p^(-1/p) = 1.7223e-01
    p=6: ||c_p|| = 2.5718e+04    root  c_p^(-1/p) = 1.8406e-01
  dT* (root test / Cauchy-Hadamard, median over p) = 1.987e-01
  ratio-test BRACKET (sandwiches dT*, not an estimate): [4.261e-02, 3.663e-01]  (spread = non-geometric c_p)
  N-derivative growth  ||N^(m+1)||/||N^(m)||  (median):
    m=0->1: 9.65x
    m=1->2: 19.3x
    m=2->3: 27.5x
    m=3->4: 33.4x
    m=4->5: 37x
  inner stencil wall (k=7, omega_ddot relL2 vs truth, thr=0.5):
    dt=0.005: relL2=3.616e-03
    dt=0.01: relL2=2.416e-03
    dt=0.015: relL2=1.478e-02
    dt=0.02: relL2=5.629e-02
    -> inner wall > (beyond reach) 2.000e-02
  >>> OUTER dT* (series radius) = 1.987e-01   |   INNER stencil wall (k=7) ~ 2.000e-02

========= per-member walls (outer series radius / inner stencil) =========
  member             OUTER dT* (root)     INNER wall k=7
  FRC-Re25k                 6.635e-02          1.294e-02
  FRC-combo                 1.392e-01          2.000e-02
  FRC-kf4                   1.987e-01          2.000e-02
Two NESTED walls. OUTER dT* (root test): past it adding higher analytic Rp
diverges -- no finite-order closure helps. INNER stencil wall: where a fixed-
lag backward FD stops recovering omega_ddot, hit FIRST (inside dT*). At dt below
the inner wall the binding limit is finite lags -> more lags help (the FD-check
result); between inner and outer, more lags help but the series still converges;
past dT* nothing finite does. NOTE both are finite-p estimates, trust ~20-30%.
```
