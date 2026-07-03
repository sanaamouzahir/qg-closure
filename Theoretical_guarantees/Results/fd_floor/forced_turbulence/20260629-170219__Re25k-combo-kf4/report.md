# fd_floor -- forced_turbulence -- Re25k-combo-kf4

*20260629-170219*  |  host `mseas.mit.edu`  |  git `unknown`  |  rc 1  |  14.9s

## Command
```
python temporal_fd_floor_deep.py --sources ../data/ensemble_N5/FRC-Re25k/forced_turbulence_dT_5em3 ../data/ensemble_N5/FRC-combo/forced_turbulence_dT_5em3 ../data/ensemble_N5/FRC-kf4/forced_turbulence_dT_5em3 --target-dts 5e-3 1e-2 1.5e-2 --n-list 3 4 5 6 7 --n-samples 48 --device cuda --dtype float64
```

## Note
per-order temporal-FD floor, n_time sweep

## How to read this
Temporal-FD floor with NO model in the loop (perfect spatial ops).
- floor(n=4) ~ the trained val % -> the 4-snapshot TIME stencil is the wall;
  the corrector cannot beat it -> build the 7-snapshot set.
- floor(n=4) << trained val % -> temporal stencil has headroom; the plateau
  is model capacity / spatial path -> the corrector is the right lever.
- The n=4 -> n=7 drop is the predicted payoff of rebuilding.

## Output
```
/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-env/lib/python3.11/site-packages/torch/cuda/__init__.py:58: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
Traceback (most recent call last):
  File "/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-simple-package-stable/src/qg/training/Theoretical_guarantees/temporal_fd_floor_deep.py", line 216, in <module>
    main()
  File "/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-simple-package-stable/src/qg/training/Theoretical_guarantees/temporal_fd_floor_deep.py", line 208, in main
    run_source(src, args.target_dts, sorted(args.n_list), args.n_samples,
  File "/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-simple-package-stable/src/qg/training/Theoretical_guarantees/temporal_fd_floor_deep.py", line 150, in run_source
    grid = CartesianGrid(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, device=device, precision=dtype)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-simple-package-stable/src/qg/solver/grid/cartesian.py", line 31, in __init__
    self.x = torch.arange(-self.Lx/2, self.Lx/2, self.dx, device=self.device)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-env/lib/python3.11/site-packages/torch/cuda/__init__.py", line 314, in _lazy_init
    torch._C._cuda_init()
RuntimeError: Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a driver from http://www.nvidia.com/Download/index.aspx
```
