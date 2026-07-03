# fd_floor -- forced_turbulence -- Re25k-combo-kf4

*20260629-163514*  |  host `mseas.mit.edu`  |  git `unknown`  |  rc 2  |  0.6s

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
python: can't open file '/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-simple-package-stable/src/qg/training/Theoretical_guarantees/temporal_fd_floor_deep.py': [Errno 2] No such file or directory
```
