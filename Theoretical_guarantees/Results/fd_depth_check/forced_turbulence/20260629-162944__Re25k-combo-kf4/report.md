# fd_depth_check -- forced_turbulence -- Re25k-combo-kf4

*20260629-162944*  |  host `mseas.mit.edu`  |  git `unknown`  |  rc 2  |  0.1s

## Command
```
python fd_depth_check.py --sources data/ensemble_N5/FRC-Re25k/forced_turbulence_dT_5em3 data/ensemble_N5/FRC-combo/forced_turbulence_dT_5em3 data/ensemble_N5/FRC-kf4/forced_turbulence_dT_5em3 --target-dts 5e-3 1e-2 1.5e-2 --depths 4 7 --n-samples 48 --device cuda --dtype float64
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
python: can't open file '/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-simple-package-stable/src/qg/training/fd_depth_check.py': [Errno 2] No such file or directory
```
