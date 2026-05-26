from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tinystories_xpu_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def _load_telemetry_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return config.get("telemetry", {})


def derive_metrics(run_dir: str | Path) -> None:
    run_dir = Path(run_dir).expanduser().resolve()
    metrics_path = run_dir / "metrics_all.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    telemetry_config = _load_telemetry_config(run_dir)
    window = int(telemetry_config.get("rolling_window", 5))
    spike_min_delta = float(telemetry_config.get("spike_min_delta", 0.01))
    spike_z = float(telemetry_config.get("spike_z_threshold", 2.0))
    stable_quantile = float(telemetry_config.get("stable_volatility_quantile", 0.25))
    stable_spike_max = float(telemetry_config.get("stable_spike_rate_max", 0.0))
    stable_backslide_max = float(telemetry_config.get("stable_backslide_frequency_max", 1))

    df = pd.read_csv(metrics_path)
    numeric_columns = [
        "seed",
        "interval_index",
        "tokens_seen",
        "step",
        "train_loss",
        "val_loss",
        "val_ppl",
        "elapsed_seconds",
        "tokens_per_second",
        "learning_rate",
        "epoch_equivalent",
    ]
    for column in numeric_columns:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.sort_values(["seed", "tokens_seen"]).reset_index(drop=True)

    derived = []
    for seed, seed_df in df.groupby("seed", sort=True):
        seed_df = seed_df.copy()
        delta = seed_df["val_loss"].diff()
        rolling_mean = delta.rolling(window=window, min_periods=2).mean()
        rolling_std = delta.rolling(window=window, min_periods=2).std(ddof=0)
        threshold = (rolling_mean + spike_z * rolling_std).fillna(spike_min_delta)
        threshold = np.maximum(threshold, spike_min_delta)

        seed_df["val_loss_delta"] = delta
        seed_df["rolling_volatility"] = delta.rolling(window=window, min_periods=2).std()
        seed_df["backslide"] = (delta > 0).fillna(False).astype(int)
        seed_df["backslide_magnitude"] = delta.clip(lower=0).fillna(0.0)
        seed_df["spike_threshold"] = threshold
        seed_df["spike"] = (delta > threshold).fillna(False).astype(int)
        seed_df["spike_rate"] = seed_df["spike"].rolling(window=window, min_periods=1).mean()
        seed_df["backslide_frequency"] = (
            seed_df["backslide"].rolling(window=window, min_periods=1).sum()
        )
        derived.append(seed_df)

    derived_df = pd.concat(derived, ignore_index=True)
    between_seed = (
        derived_df.groupby("interval_index", as_index=False)
        .agg(
            between_seed_val_loss_mean=("val_loss", "mean"),
            between_seed_val_loss_variance=("val_loss", "var"),
            between_seed_val_loss_std=("val_loss", "std"),
            seeds_observed=("seed", "nunique"),
        )
    )
    derived_df = derived_df.merge(between_seed, on="interval_index", how="left")

    finite_volatility = derived_df["rolling_volatility"].replace([np.inf, -np.inf], np.nan).dropna()
    volatility_cutoff = (
        float(finite_volatility.quantile(stable_quantile)) if not finite_volatility.empty else np.nan
    )
    derived_df["stable_phase_candidate"] = (
        (derived_df["rolling_volatility"] <= volatility_cutoff)
        & (derived_df["spike_rate"] <= stable_spike_max)
        & (derived_df["backslide_frequency"] <= stable_backslide_max)
    ).astype(int)

    derived_path = run_dir / "telemetry_derived.csv"
    derived_df.to_csv(derived_path, index=False)

    interval_summary = (
        derived_df.groupby("interval_index", as_index=False)
        .agg(
            tokens_seen_mean=("tokens_seen", "mean"),
            val_loss_mean=("val_loss", "mean"),
            val_loss_std=("val_loss", "std"),
            val_ppl_mean=("val_ppl", "mean"),
            rolling_volatility_mean=("rolling_volatility", "mean"),
            spike_rate_mean=("spike_rate", "mean"),
            backslide_frequency_mean=("backslide_frequency", "mean"),
            stable_phase_share=("stable_phase_candidate", "mean"),
            between_seed_val_loss_variance=("between_seed_val_loss_variance", "mean"),
        )
    )
    interval_summary.to_csv(run_dir / "interval_summary.csv", index=False)
    _write_plots(run_dir, derived_df, interval_summary)

    print(f"Derived telemetry: {derived_path}")
    print(f"Interval summary: {run_dir / 'interval_summary.csv'}")


def _write_plots(run_dir: Path, derived_df: pd.DataFrame, interval_summary: pd.DataFrame) -> None:
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for seed, seed_df in derived_df.groupby("seed", sort=True):
        plt.plot(seed_df["tokens_seen"], seed_df["val_loss"], marker="o", linewidth=1.5, label=f"seed {seed}")
    plt.xlabel("Cumulative training tokens")
    plt.ylabel("Validation loss")
    plt.title("Validation loss by seed")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "validation_loss_by_seed.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(
        interval_summary["tokens_seen_mean"],
        interval_summary["rolling_volatility_mean"],
        marker="o",
        label="rolling volatility",
    )
    plt.plot(
        interval_summary["tokens_seen_mean"],
        interval_summary["spike_rate_mean"],
        marker="s",
        label="spike rate",
    )
    plt.xlabel("Cumulative training tokens")
    plt.ylabel("Derived stability metric")
    plt.title("Training stability metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "stability_metrics.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive stability telemetry from training metrics.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing metrics_all.csv.")
    args = parser.parse_args()
    derive_metrics(args.run_dir)


if __name__ == "__main__":
    main()
