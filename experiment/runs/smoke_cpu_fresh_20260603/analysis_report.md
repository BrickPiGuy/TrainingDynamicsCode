# TinyStories XPU Statistical Analysis

Input file: `telemetry_derived.csv`
Seeds: 2
Intervals: 5
Observations: 10
Duplicate seed/interval rows removed: 0

## Normality Diagnostics

Fewer than three seeds per interval; Shapiro-Wilk tests were not computed.

## Sphericity

Not enough complete seeds and intervals for a sphericity test.

## Repeated-Measures ANOVA: val_loss

| Source         |          SS |   DF |          MS |       F |         p_unc |        ng2 |    eps |
|:---------------|------------:|-----:|------------:|--------:|--------------:|-----------:|-------:|
| interval_index | 17.5404     |    4 | 4.3851      | 5033.88 |   1.18327e-07 |   0.999535 |   0.25 |
| Error          |  0.00348447 |    4 | 0.000871118 |  nan    | nan           | nan        | nan    |

## Repeated-Measures ANOVA: val_ppl

| Source         |            SS |   DF |            MS |       F |         p_unc |        ng2 |    eps |
|:---------------|--------------:|-----:|--------------:|--------:|--------------:|-----------:|-------:|
| interval_index |   2.51336e+07 |    4 |   6.28339e+06 | 55281.5 |   9.81615e-10 |   0.999982 |   0.25 |
| Error          | 454.647       |    4 | 113.662       |   nan   | nan           | nan        | nan    |

## Repeated-Measures ANOVA: rolling_volatility

| Source         |         SS |   DF |         MS |       F |        p_unc |        ng2 |   eps |
|:---------------|-----------:|-----:|-----------:|--------:|-------------:|-----------:|------:|
| interval_index | 0.569217   |    2 | 0.284609   | 271.797 |   0.00366573 |   0.996284 |   0.5 |
| Error          | 0.00209427 |    2 | 0.00104714 | nan     | nan          | nan        | nan   |

## Linear Mixed Effects Robustness Check

Mixed-effects model skipped: fewer than three seeds often leads to singular random-effects covariance in this design.
