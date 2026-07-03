# convergence_radius -- forced_turbulence -- Re25k-combo-kf4

*20260629-170130*  |  host `mseas.mit.edu`  |  git `unknown`  |  rc 1  |  30.9s

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
Traceback (most recent call last):
  File "/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-simple-package-stable/src/qg/training/Theoretical_guarantees/convergence_radius.py", line 284, in <module>
    main()
  File "/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-simple-package-stable/src/qg/training/Theoretical_guarantees/convergence_radius.py", line 267, in main
    summary.append(run_source(src, args.max_order, args.n_samples,
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-simple-package-stable/src/qg/training/Theoretical_guarantees/convergence_radius.py", line 169, in run_source
    grid = CartesianGrid(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, device=device, precision=dtype)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-simple-package-stable/src/qg/solver/grid/cartesian.py", line 31, in __init__
    self.x = torch.arange(-self.Lx/2, self.Lx/2, self.dx, device=self.device)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-env/lib/python3.11/site-packages/torch/cuda/__init__.py", line 314, in _lazy_init
    torch._C._cuda_init()
RuntimeError: Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a driver from http://www.nvidia.com/Download/index.aspx
```
