# wallv2 event gate — piff_cape_gjs_wallv2 vs piff_cape_gjs_ylp75

verdict: **REGRESSED**

```
wallv2 event gate — NEW piff_cape_gjs_wallv2 vs BASELINE piff_cape_gjs_ylp75
bars (pre-registered, Sanaa 2026-07-18): z_true_median < 3.0; worst0.1pct_SS_share <= 0.5*baseline; mean-prediction r2 >= baseline - 0.005

member | z_new | Z<3 | ss_new | ss_base | SS<=.5b | r2_new | r2_base | R2>=b-.005 | r2_act_new | r2_act_base
----------------------------------------------------------------------------------------------------
FPCape-const | 1.85 | ok | 0.6476 | 0.6474 | FAIL | 0.9832 | 0.9843 | ok | 0.983 | 0.984
FPCape-ou | 2.43 | ok | 0.7219 | 0.7137 | FAIL | 0.9795 | 0.9824 | ok | 0.980 | 0.982
FPCape-ramp | 2.58 | ok | 0.6767 | 0.6655 | FAIL | 0.9757 | 0.9806 | ok | 0.976 | 0.981
FPCape-sine | 13.30 | FAIL | 0.8969 | 0.9257 | FAIL | 0.9309 | 0.8886 | ok | 0.931 | 0.889
FPCape-tel | 2.61 | ok | 0.6920 | 0.6700 | FAIL | 0.9768 | 0.9816 | ok | 0.977 | 0.982

members: 5; all-bar pass: 0; Z+SS pass: 0; R2 pass: 5
VERDICT: REGRESSED (exit 3)
```
