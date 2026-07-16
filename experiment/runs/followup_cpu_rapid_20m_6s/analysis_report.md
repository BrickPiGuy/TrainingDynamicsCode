# TinyStories XPU Statistical Analysis

Input file: `telemetry_derived.csv`
Seeds: 6
Intervals: 21
Observations: 126
Duplicate seed/interval rows removed: 0

## Normality Diagnostics

Shapiro-Wilk tests were computed per interval; minimum p-value: 0.1477.

## Sphericity

Mauchly sphericity result: `SpherResults(spher=True, W=np.float64(8.502379612308738e+48), chi2=np.float64(208.4291965034767), dof=209, pval=np.float64(27.80138661428326))`.

## Repeated-Measures ANOVA: val_loss

| Source         |          SS |   DF |          MS |       F |          p_unc |     p_GG_corr |        ng2 |        eps |   sphericity |       W_spher |   p_spher |
|:---------------|------------:|-----:|------------:|--------:|---------------:|--------------:|-----------:|-----------:|-------------:|--------------:|----------:|
| interval_index | 159.547     |   20 | 7.97733     | 10030.9 |   9.29035e-156 |   1.03672e-25 |   0.999295 |   0.154692 |            1 |   8.50238e+48 |   27.8014 |
| Error          |   0.0795278 |  100 | 0.000795278 |   nan   | nan            | nan           | nan        | nan        |          nan | nan           |  nan      |

## Repeated-Measures ANOVA: val_ppl

| Source         |              SS |   DF |            MS |       F |          p_unc |     p_GG_corr |       ng2 |         eps |   sphericity |       W_spher |   p_spher |
|:---------------|----------------:|-----:|--------------:|--------:|---------------:|--------------:|----------:|------------:|-------------:|--------------:|----------:|
| interval_index |     1.01814e+08 |   20 |   5.09068e+06 | 21061.9 |   7.35692e-172 |   2.51177e-10 |   0.99975 |   0.0504018 |            1 |   1.37947e-42 |         1 |
| Error          | 24170           |  100 | 241.7         |   nan   | nan            | nan           | nan       | nan         |          nan | nan           |       nan |

## Repeated-Measures ANOVA: rolling_volatility

| Source         |          SS |   DF |          MS |       F |          p_unc |     p_GG_corr |        ng2 |        eps |   sphericity |       W_spher |   p_spher |
|:---------------|------------:|-----:|------------:|--------:|---------------:|--------------:|-----------:|-----------:|-------------:|--------------:|----------:|
| interval_index | 101.935     |   18 | 5.66304     | 15390.5 |   9.27763e-149 |   2.63525e-21 |   0.999537 |   0.132316 |            1 |   1.38122e+45 |   1.79035 |
| Error          |   0.0331161 |   90 | 0.000367957 |   nan   | nan            | nan           | nan        | nan        |          nan | nan           | nan       |

## Linear Mixed Effects Robustness Check

Mixed-effects model formula: `val_loss ~ C(interval_index)`, grouped by seed.

```text
               Mixed Linear Model Regression Results
====================================================================
Model:                  MixedLM     Dependent Variable:     val_loss
No. Observations:       126         Method:                 ML      
No. Groups:             6           Scale:                  0.0007  
Min. group size:        21          Log-Likelihood:         275.9623
Max. group size:        21          Converged:              Yes     
Mean group size:        21.0                                        
--------------------------------------------------------------------
                        Coef.  Std.Err.    z     P>|z| [0.025 0.975]
--------------------------------------------------------------------
Intercept                8.355    0.012  684.529 0.000  8.331  8.379
C(interval_index)[T.1]  -4.660    0.015 -313.548 0.000 -4.689 -4.631
C(interval_index)[T.2]  -5.287    0.015 -355.731 0.000 -5.316 -5.258
C(interval_index)[T.3]  -5.447    0.015 -366.493 0.000 -5.476 -5.418
C(interval_index)[T.4]  -5.556    0.015 -373.786 0.000 -5.585 -5.526
C(interval_index)[T.5]  -5.547    0.015 -373.241 0.000 -5.577 -5.518
C(interval_index)[T.6]  -5.463    0.015 -367.546 0.000 -5.492 -5.434
C(interval_index)[T.7]  -5.458    0.015 -367.217 0.000 -5.487 -5.429
C(interval_index)[T.8]  -5.300    0.015 -356.583 0.000 -5.329 -5.271
C(interval_index)[T.9]  -5.126    0.015 -344.858 0.000 -5.155 -5.096
C(interval_index)[T.10] -5.096    0.015 -342.834 0.000 -5.125 -5.066
C(interval_index)[T.11] -4.927    0.015 -331.522 0.000 -4.957 -4.898
C(interval_index)[T.12] -4.827    0.015 -324.749 0.000 -4.856 -4.798
C(interval_index)[T.13] -4.820    0.015 -324.273 0.000 -4.849 -4.791
C(interval_index)[T.14] -4.681    0.015 -314.922 0.000 -4.710 -4.652
C(interval_index)[T.15] -4.573    0.015 -307.678 0.000 -4.602 -4.544
C(interval_index)[T.16] -4.592    0.015 -308.927 0.000 -4.621 -4.562
C(interval_index)[T.17] -4.527    0.015 -304.585 0.000 -4.556 -4.498
C(interval_index)[T.18] -4.453    0.015 -299.582 0.000 -4.482 -4.424
C(interval_index)[T.19] -4.485    0.015 -301.755 0.000 -4.514 -4.456
C(interval_index)[T.20] -4.454    0.015 -299.683 0.000 -4.483 -4.425
Group Var                0.000    0.006                             
====================================================================

```
