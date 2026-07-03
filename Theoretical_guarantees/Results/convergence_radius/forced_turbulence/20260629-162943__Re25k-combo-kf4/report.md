# convergence_radius -- forced_turbulence -- Re25k-combo-kf4

*20260629-162943*  |  host `mseas.mit.edu`  |  git `unknown`  |  rc 2  |  0.1s

## Command
```
python convergence_radius.py --sources data/ensemble_N5/FRC-Re25k/forced_turbulence_dT_5em3 data/ensemble_N5/FRC-combo/forced_turbulence_dT_5em3 data/ensemble_N5/FRC-kf4/forced_turbulence_dT_5em3 --max-order 5 --n-samples 32 --wall-k 7 --device cuda --dtype float64
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
python: can't open file '/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-simple-package-stable/src/qg/training/convergence_radius.py': [Errno 2] No such file or directory
```
