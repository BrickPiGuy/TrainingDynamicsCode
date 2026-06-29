# TinyStories XPU Statistical Analysis

Input file: `telemetry_derived.csv`
Seeds: 2
Intervals: 21
Observations: 42
Duplicate seed/interval rows removed: 0

## Normality Diagnostics

Fewer than three seeds per interval; Shapiro-Wilk tests were not computed.

## Sphericity

Not enough complete seeds and intervals for a sphericity test.

## Repeated-Measures ANOVA: val_loss

| Source         |         SS |   DF |         MS |       F |         p_unc |        ng2 |    eps |
|:---------------|-----------:|-----:|-----------:|--------:|--------------:|-----------:|-------:|
| interval_index | 53.4371    |   20 | 2.67185    | 4652.04 |   1.93837e-32 |   0.999747 |   0.05 |
| Error          |  0.0114868 |   20 | 0.00057434 |  nan    | nan           | nan        | nan    |

## Repeated-Measures ANOVA: val_ppl

| Source         |              SS |   DF |            MS |       F |         p_unc |        ng2 |    eps |
|:---------------|----------------:|-----:|--------------:|--------:|--------------:|-----------:|-------:|
| interval_index |     3.36833e+07 |   20 |   1.68416e+06 | 3343.16 |   5.26792e-31 |   0.999683 |   0.05 |
| Error          | 10075.3         |   20 | 503.764       |  nan    | nan           | nan        | nan    |

## Repeated-Measures ANOVA: rolling_volatility

| Source         |         SS |   DF |          MS |       F |         p_unc |        ng2 |         eps |
|:---------------|-----------:|-----:|------------:|--------:|--------------:|-----------:|------------:|
| interval_index | 33.6041    |   18 | 1.8669      | 3770.62 |   1.57111e-28 |   0.999687 |   0.0555556 |
| Error          |  0.0089121 |   18 | 0.000495117 |  nan    | nan           | nan        | nan         |

## Linear Mixed Effects Robustness Check

Mixed-effects model skipped: fewer than three seeds often leads to singular random-effects covariance in this design.
