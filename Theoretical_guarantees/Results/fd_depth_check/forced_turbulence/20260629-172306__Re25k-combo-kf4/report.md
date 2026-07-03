# fd_depth_check -- forced_turbulence -- Re25k-combo-kf4

*20260629-172306*  |  host `ibgpu-compute-0-0.local`  |  git `unknown`  |  rc 0  |  34.1s

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

=== forced_turbulence_dT_5em3 ===  grid=512x512 Lx=12.5664 M=28 marks  dt_fine=5.000e-03  nu=4.00e-05 mu=2.00e-02 beta=1  forced=True  dealias=True
    target dt -> j (marks back per lag): 0.005->1, 0.01->2, 0.015->3
          dt | k=4: omega_ddot   N_ddot | k=7: omega_ddot   N_ddot
    --------------------------------------------------------------
       0.005 | k=4: 1.023e-01   1.486e-01 | k=7: 9.386e-03   1.951e-02
        0.01 | k=4: 3.461e-01   4.997e-01 | k=7: 2.050e-01   4.114e-01
       0.015 | k=4: 6.998e-01   9.870e-01 | k=7: 8.830e-01   1.510e+00

=== forced_turbulence_dT_5em3 ===  grid=512x512 Lx=12.5664 M=28 marks  dt_fine=5.000e-03  nu=4.00e-05 mu=2.00e-02 beta=0.5  forced=True  dealias=True
    target dt -> j (marks back per lag): 0.005->1, 0.01->2, 0.015->3
          dt | k=4: omega_ddot   N_ddot | k=7: omega_ddot   N_ddot
    --------------------------------------------------------------
       0.005 | k=4: 2.011e-02   3.346e-02 | k=7: 1.990e-03   1.319e-03
        0.01 | k=4: 7.757e-02   1.257e-01 | k=7: 6.320e-03   1.414e-02
       0.015 | k=4: 1.697e-01   2.730e-01 | k=7: 4.052e-02   9.195e-02

=== forced_turbulence_dT_5em3 ===  grid=512x512 Lx=12.5664 M=28 marks  dt_fine=5.000e-03  nu=1.02e-04 mu=2.00e-02 beta=1  forced=True  dealias=True
    target dt -> j (marks back per lag): 0.005->1, 0.01->2, 0.015->3
          dt | k=4: omega_ddot   N_ddot | k=7: omega_ddot   N_ddot
    --------------------------------------------------------------
       0.005 | k=4: 1.233e-02   2.147e-02 | k=7: 3.792e-03   2.824e-03
        0.01 | k=4: 4.730e-02   8.396e-02 | k=7: 2.207e-03   4.408e-03
       0.015 | k=4: 1.036e-01   1.833e-01 | k=7: 1.373e-02   3.075e-02

VERDICT GUIDE: compare N_ddot at k=7 vs k=4 for dt=1e-2 and 1.5e-2.
  >2x lower   -> truncation-limited; regeneration justified.
  ~equal/worse -> temporal under-resolution; more lags will NOT move the
                  plateau. Do not regenerate; report validated dt<=1e-2.
```
