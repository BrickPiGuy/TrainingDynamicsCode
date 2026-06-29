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

| Source         |          SS |   DF |          MS |       F |         p_unc |    p_GG_corr |        ng2 |    eps |   sphericity |       W_spher |   p_spher |
|:---------------|------------:|-----:|------------:|--------:|--------------:|-------------:|-----------:|-------:|-------------:|--------------:|----------:|
| interval_index | 17.5404     |    4 | 4.3851      | 5033.88 |   1.18327e-07 |   0.00897222 |   0.999535 |   0.25 |            1 |   6.05104e+09 |   1.70973 |
| Error          |  0.00348447 |    4 | 0.000871118 |  nan    | nan           | nan          | nan        | nan    |          nan | nan           | nan       |

## Repeated-Measures ANOVA: val_ppl

| Source         |            SS |   DF |            MS |       F |         p_unc |    p_GG_corr |        ng2 |    eps |   sphericity |       W_spher |   p_spher |
|:---------------|--------------:|-----:|--------------:|--------:|--------------:|-------------:|-----------:|-------:|-------------:|--------------:|----------:|
| interval_index |   2.51336e+07 |    4 |   6.28339e+06 | 55281.5 |   9.81615e-10 |   0.00270762 |   0.999982 |   0.25 |            1 |   2.72405e-06 |         1 |
| Error          | 454.647       |    4 | 113.662       |   nan   | nan           | nan          | nan        | nan    |          nan | nan           |       nan |

## Repeated-Measures ANOVA: rolling_volatility

Repeated-measures ANOVA failed for `rolling_volatility`: `division by zero`.

## Linear Mixed Effects Robustness Check

Mixed-effects model failed: `Singular matrix`.
