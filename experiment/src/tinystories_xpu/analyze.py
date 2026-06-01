from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels.formula.api as smf
from scipy import stats


def analyze(run_dir: str | Path) -> None:
    run_dir = Path(run_dir).expanduser().resolve()
    input_path = run_dir / "telemetry_derived.csv"
    if not input_path.exists():
        input_path = run_dir / "metrics_all.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing metrics file in {run_dir}")

    df = pd.read_csv(input_path)
    for column in ["seed", "interval_index", "val_loss", "val_ppl", "rolling_volatility"]:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["seed", "interval_index", "val_loss"])
    df["seed"] = df["seed"].astype(int).astype(str)
    df["interval_index"] = df["interval_index"].astype(int)

    duplicate_mask = df.duplicated(subset=["seed", "interval_index"], keep=False)
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count:
        # Keep the latest observation per seed/interval when runs were restarted.
        if "tokens_seen" in df.columns:
            df["tokens_seen"] = pd.to_numeric(df["tokens_seen"], errors="coerce")
            df = df.sort_values(["seed", "interval_index", "tokens_seen"])
        else:
            df = df.sort_values(["seed", "interval_index"])
        df = df.drop_duplicates(subset=["seed", "interval_index"], keep="last")

    report_lines = [
        "# TinyStories XPU Statistical Analysis",
        "",
        f"Input file: `{input_path.name}`",
        f"Seeds: {df['seed'].nunique()}",
        f"Intervals: {df['interval_index'].nunique()}",
        f"Observations: {len(df)}",
        f"Duplicate seed/interval rows removed: {duplicate_count}",
        "",
    ]

    common = _common_interval_data(df)
    if common["seed"].nunique() < 2 or common["interval_index"].nunique() < 2:
        report_lines.append("Not enough complete repeated-measures data for ANOVA.")
        _write_report(run_dir, report_lines)
        return

    report_lines.extend(_normality_report(common))
    report_lines.extend(_sphericity_report(common))
    report_lines.extend(_rm_anova_report(run_dir, common, "val_loss"))
    if "val_ppl" in common:
        report_lines.extend(_rm_anova_report(run_dir, common, "val_ppl"))
    if "rolling_volatility" in common and common["rolling_volatility"].notna().sum() > 0:
        volatility = common.dropna(subset=["rolling_volatility"])
        if volatility["seed"].nunique() >= 2 and volatility["interval_index"].nunique() >= 2:
            report_lines.extend(_rm_anova_report(run_dir, volatility, "rolling_volatility"))
    report_lines.extend(_mixed_model_report(common))
    _pairwise_tests(run_dir, common)
    _write_report(run_dir, report_lines)
    print(f"Analysis report: {run_dir / 'analysis_report.md'}")


def _common_interval_data(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby("interval_index")["seed"].nunique()
    required = df["seed"].nunique()
    complete_intervals = counts[counts == required].index
    return df[df["interval_index"].isin(complete_intervals)].copy()


def _normality_report(df: pd.DataFrame) -> list[str]:
    lines = ["## Normality Diagnostics", ""]
    rows = []
    for interval, interval_df in df.groupby("interval_index", sort=True):
        values = interval_df["val_loss"].dropna()
        if len(values) >= 3:
            stat, p_value = stats.shapiro(values)
            rows.append({"interval_index": interval, "shapiro_w": stat, "p_value": p_value})
    if rows:
        normality = pd.DataFrame(rows)
        min_p = normality["p_value"].min()
        lines.append(f"Shapiro-Wilk tests were computed per interval; minimum p-value: {min_p:.4g}.")
    else:
        lines.append("Fewer than three seeds per interval; Shapiro-Wilk tests were not computed.")
    lines.append("")
    return lines


def _sphericity_report(df: pd.DataFrame) -> list[str]:
    lines = ["## Sphericity", ""]
    wide = df.pivot(index="seed", columns="interval_index", values="val_loss").dropna(axis=1)
    if wide.shape[0] >= 3 and wide.shape[1] >= 3:
        try:
            sphericity = pg.sphericity(wide)
            lines.append(f"Mauchly sphericity result: `{sphericity}`.")
        except Exception as exc:
            lines.append(f"Sphericity test failed: `{exc}`.")
    else:
        lines.append("Not enough complete seeds and intervals for a sphericity test.")
    lines.append("")
    return lines


def _rm_anova_report(run_dir: Path, df: pd.DataFrame, dv: str) -> list[str]:
    lines = [f"## Repeated-Measures ANOVA: {dv}", ""]
    try:
        result = pg.rm_anova(
            data=df,
            dv=dv,
            within="interval_index",
            subject="seed",
            detailed=True,
            correction=True,
        )
        result.to_csv(run_dir / f"rm_anova_{dv}.csv", index=False)
        lines.append(result.to_markdown(index=False))
    except Exception as exc:
        lines.append(f"Repeated-measures ANOVA failed for `{dv}`: `{exc}`.")
    lines.append("")
    return lines


def _mixed_model_report(df: pd.DataFrame) -> list[str]:
    lines = ["## Linear Mixed Effects Robustness Check", ""]
    try:
        model = smf.mixedlm("val_loss ~ C(interval_index)", df, groups=df["seed"])
        result = model.fit(reml=False, method="lbfgs", maxiter=500)
        lines.append("Mixed-effects model formula: `val_loss ~ C(interval_index)`, grouped by seed.")
        lines.append("")
        lines.append("```text")
        lines.append(str(result.summary()))
        lines.append("```")
    except Exception as exc:
        lines.append(f"Mixed-effects model failed: `{exc}`.")
    lines.append("")
    return lines


def _pairwise_tests(run_dir: Path, df: pd.DataFrame) -> None:
    try:
        pairs = pg.pairwise_tests(
            data=df,
            dv="val_loss",
            within="interval_index",
            subject="seed",
            padjust="bonf",
        )
        pairs.to_csv(run_dir / "pairwise_val_loss_bonferroni.csv", index=False)
    except Exception as exc:
        (run_dir / "pairwise_val_loss_bonferroni.error.txt").write_text(str(exc), encoding="utf-8")


def _write_report(run_dir: Path, lines: list[str]) -> None:
    (run_dir / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated-measures analysis for TinyStories telemetry.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing metrics_all.csv.")
    args = parser.parse_args()
    analyze(args.run_dir)


if __name__ == "__main__":
    main()
