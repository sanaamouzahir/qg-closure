# fd_depth_check -- forced_turbulence -- Re25k-combo-kf4

*20260629-163456*  |  host `mseas.mit.edu`  |  git `unknown`  |  rc 1  |  17.0s

## Command
```
python fd_depth_check.py --sources ../data/ensemble_N5/FRC-Re25k/forced_turbulence_dT_5em3 ../data/ensemble_N5/FRC-combo/forced_turbulence_dT_5em3 ../data/ensemble_N5/FRC-kf4/forced_turbulence_dT_5em3 --target-dts 5e-3 1e-2 1.5e-2 --depths 4 7 --n-samples 48 --device cuda --dtype float64
```

## Note
4 vs 7 lag N_ddot floor

## How to read this
Does going 4->7 lags justify regenerating the deep sources?
- Compare the **N_ddot** value at k=7 vs k=4 for dt=1e-2 and 1.5e-2.
- >2x lower -> TRUNCATION-limited; regeneration justified.
- ~equal/worse -> temporal UNDER-RESOLUTION; more lags won't move the
  plateau; report the validated dt<=1e-2 range.
- Faithfulness check: the k=4 N_ddot at the coarse dts should sit near the
  trained plateau (~0.4-0.6) -- if it does, the k=7 column is trustworthy.

## Output
```
/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-env/lib/python3.11/site-packages/torch/cuda/__init__.py:58: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
Traceback (most recent call last):
  File "/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-simple-package-stable/src/qg/training/Theoretical_guarantees/fd_depth_check.py", line 270, in <module>
    main()
  File "/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-simple-package-stable/src/qg/training/Theoretical_guarantees/fd_depth_check.py", line 260, in main
    run_source(src, args.target_dts, args.depths, args.n_samples,
  File "/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-simple-package-stable/src/qg/training/Theoretical_guarantees/fd_depth_check.py", line 192, in run_source
    grid = CartesianGrid(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, device=device, precision=dtype)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-simple-package-stable/src/qg/solver/grid/cartesian.py", line 31, in __init__
    self.x = torch.arange(-self.Lx/2, self.Lx/2, self.dx, device=self.device)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-env/lib/python3.11/site-packages/torch/cuda/__init__.py", line 314, in _lazy_init
    torch._C._cuda_init()
RuntimeError: Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a driver from http://www.nvidia.com/Download/index.aspx
```
