# wallv2 event gate — piff_fpc_gjs_wallv2 vs piff_fpc_gjs_ylp75

verdict: **REGRESSED**

```
wallv2 event gate — NEW piff_fpc_gjs_wallv2 vs BASELINE piff_fpc_gjs_ylp75
bars (pre-registered, Sanaa 2026-07-18): z_true_median < 3.0; worst0.1pct_SS_share <= 0.5*baseline; mean-prediction r2 >= baseline - 0.005

member | z_new | Z<3 | ss_new | ss_base | SS<=.5b | r2_new | r2_base | R2>=b-.005 | r2_act_new | r2_act_base
----------------------------------------------------------------------------------------------------
FPC-const | 4.20 | FAIL | 0.6932 | 0.9522 | FAIL | 0.9649 | 0.8984 | ok | 0.965 | 0.898
FPC-ou | 4.67 | FAIL | 0.6781 | 0.9444 | FAIL | 0.9513 | 0.8852 | ok | 0.951 | 0.885
FPC-ramp | 7.96 | FAIL | 0.7588 | 0.9612 | FAIL | 0.9422 | 0.8340 | ok | 0.942 | 0.834
FPC-sine | 6.27 | FAIL | 0.7216 | 0.9354 | FAIL | 0.9383 | 0.8663 | ok | 0.938 | 0.866
FPC-telS-A | 8.12 | FAIL | 0.7778 | 0.9627 | FAIL | 0.9395 | 0.8308 | ok | 0.940 | 0.831

members: 5; all-bar pass: 0; Z+SS pass: 0; R2 pass: 5
VERDICT: REGRESSED (exit 3)
```
