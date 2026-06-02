from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tokenizers import Tokenizer

from tinystories_xpu.config import load_config, repo_path
from tinystories_xpu.model import LlamaConfig, LlamaLanguageModel
from tinystories_xpu.utils import (
    autocast_for,
    cosine_lr,
    load_json,
    save_json,
    select_device,
    set_seed,
    synchronize,
    system_fingerprint,
)


class TokenBatcher:
    def __init__(
        self,
        path: Path,
        dtype: str,
        block_size: int,
        start_offset: int = 0,
    ) -> None:
        self.data = np.memmap(path, dtype=np.dtype(dtype), mode="r")
        if self.data.size <= block_size + 1:
            raise ValueError(f"Token cache {path} is too small for block_size={block_size}")
        self.block_size = block_size
        self.max_start = int(self.data.size - block_size - 1)
        self.cursor = int(start_offset % self.max_start)

    def next_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        starts = (self.cursor + np.arange(batch_size, dtype=np.int64) * self.block_size) % self.max_start
        self.cursor = int((self.cursor + batch_size * self.block_size) % self.max_start)
        x = np.stack([self.data[start : start + self.block_size] for start in starts])
        y = np.stack([self.data[start + 1 : start + self.block_size + 1] for start in starts])
        x_tensor = torch.from_numpy(x.astype(np.int64, copy=False)).to(device, non_blocking=True)
        y_tensor = torch.from_numpy(y.astype(np.int64, copy=False)).to(device, non_blocking=True)
        return x_tensor, y_tensor


@torch.no_grad()
def evaluate(
    model: LlamaLanguageModel,
    val_path: Path,
    dtype: str,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    experiment = config["experiment"]
    training = config["training"]
    model_config = config["model"]
    batcher = TokenBatcher(
        val_path,
        dtype,
        int(model_config["block_size"]),
        start_offset=int(experiment.get("eval_seed", 0)),
    )
    losses: list[float] = []
    for _ in range(int(experiment["eval_batches"])):
        x, y = batcher.next_batch(int(training["micro_batch_size"]), device)
        with autocast_for(device, str(training.get("precision", "fp32"))):
            _, loss = model(x, y)
        if loss is None:
            raise RuntimeError("Evaluation loss was not computed.")
        losses.append(float(loss.detach().cpu()))
    model.train()
    val_loss = float(np.mean(losses))
    val_ppl = float(math.exp(min(val_loss, 20.0)))
    return val_loss, val_ppl


def write_metric(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_status(path: Path, payload: dict[str, Any]) -> None:
    save_json(path, payload)


def save_checkpoint(
    seed_dir: Path,
    model: LlamaLanguageModel,
    optimizer: torch.optim.Optimizer,
    seed: int,
    step: int,
    tokens_seen: int,
    keep_last: int,
) -> None:
    checkpoint_dir = seed_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"seed_{seed}_tokens_{tokens_seen}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "seed": seed,
            "step": step,
            "tokens_seen": tokens_seen,
        },
        checkpoint_path,
    )
    checkpoints = sorted(checkpoint_dir.glob("seed_*_tokens_*.pt"), key=lambda item: item.stat().st_mtime)
    for old_path in checkpoints[:-keep_last]:
        old_path.unlink(missing_ok=True)


def _checkpoint_tokens(path: Path) -> int:
    match = re.search(r"_tokens_(\d+)\.pt$", path.name)
    if not match:
        return -1
    return int(match.group(1))


def _latest_checkpoint(seed_dir: Path, seed: int) -> Path | None:
    checkpoint_dir = seed_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return None
    candidates = list(checkpoint_dir.glob(f"seed_{seed}_tokens_*.pt"))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (_checkpoint_tokens(item), item.stat().st_mtime))
    return candidates[-1]


def _latest_metrics_row(metrics_path: Path, seed: int) -> dict[str, Any] | None:
    if not metrics_path.exists():
        return None
    latest_row: dict[str, Any] | None = None
    latest_tokens = -1
    latest_interval = -1
    with metrics_path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                row_seed = int(row.get("seed", "-1"))
                row_tokens = int(float(row.get("tokens_seen", "-1")))
                row_interval = int(row.get("interval_index", "-1"))
            except ValueError:
                continue
            if row_seed != seed:
                continue
            if row_tokens > latest_tokens or (row_tokens == latest_tokens and row_interval > latest_interval):
                latest_row = row
                latest_tokens = row_tokens
                latest_interval = row_interval
    return latest_row


def run_seed(config: dict[str, Any], seed: int, run_dir: Path, meta: dict[str, Any]) -> Path:
    experiment = config["experiment"]
    training = config["training"]
    data_config = config["data"]
    model_config = dict(config["model"])
    model_config["vocab_size"] = int(meta["vocab_size"])
    set_seed(seed)
    device = select_device(str(training.get("device", "auto")))

    seed_dir = run_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = seed_dir / "metrics.csv"
    run_status_path = run_dir / "status.json"
    seed_status_path = seed_dir / "status.json"

    target_tokens = int(experiment["target_tokens"])
    checkpoint_tokens = int(experiment["checkpoint_tokens"])
    checkpoint_every_tokens = int(training.get("checkpoint_every_tokens", target_tokens + 1))
    keep_last = int(training.get("keep_last_checkpoints", 2))
    micro_batch_size = int(training["micro_batch_size"])
    grad_accum = int(training["gradient_accumulation_steps"])
    tokens_per_micro = micro_batch_size * int(model_config["block_size"])
    tokens_per_step = tokens_per_micro * grad_accum

    step = 0
    tokens_seen = 0
    resumed_from_checkpoint = False

    train_path = Path(meta["train"]["path"])
    val_path = Path(meta["validation"]["path"])
    dtype = str(meta["train"]["dtype"])
    batcher = TokenBatcher(train_path, dtype, int(model_config["block_size"]), start_offset=0)
    llama_config = LlamaConfig.from_dict(model_config)
    model = LlamaLanguageModel(llama_config).to(device)
    if bool(training.get("compile", False)):
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        betas=(float(training["beta1"]), float(training["beta2"])),
        weight_decay=float(training["weight_decay"]),
    )

    latest_checkpoint = _latest_checkpoint(seed_dir, seed)
    if latest_checkpoint is not None:
        payload = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        step = int(payload.get("step", 0))
        tokens_seen = int(payload.get("tokens_seen", 0))
        resumed_from_checkpoint = True
        print(
            f"seed={seed} resumed checkpoint={latest_checkpoint.name} "
            f"step={step} tokens={tokens_seen:,}"
        )

    write_status(
        seed_status_path,
        {
            "stage": "starting",
            "seed": seed,
            "step": step,
            "tokens_seen": tokens_seen,
            "target_tokens": target_tokens,
            "updated_at": time.time(),
        },
    )
    write_status(
        run_status_path,
        {
            "stage": "seed_starting",
            "seed": seed,
            "step": step,
            "tokens_seen": tokens_seen,
            "target_tokens": target_tokens,
            "updated_at": time.time(),
        },
    )

    batcher = TokenBatcher(
        train_path,
        dtype,
        int(model_config["block_size"]),
        start_offset=tokens_seen,
    )

    fingerprint = system_fingerprint(device)
    save_json(
        seed_dir / "run_meta.json",
        {
            "seed": seed,
            "parameter_count": model.parameter_count()
            if hasattr(model, "parameter_count")
            else sum(p.numel() for p in model.parameters()),
            "system": fingerprint,
            "resumed_from_checkpoint": resumed_from_checkpoint,
            "resume_tokens": tokens_seen,
        },
    )

    last_metrics_row = _latest_metrics_row(metrics_path, seed)
    if last_metrics_row is None:
        interval_index = 0
        next_eval_tokens = 0 if tokens_seen == 0 else tokens_seen
        last_eval_tokens = 0
    else:
        last_logged_tokens = int(float(last_metrics_row["tokens_seen"]))
        interval_index = int(last_metrics_row["interval_index"]) + 1
        last_eval_tokens = last_logged_tokens
        if tokens_seen <= last_logged_tokens:
            next_eval_tokens = last_logged_tokens + checkpoint_tokens
        else:
            # If we resumed from a checkpoint past the last logged eval, evaluate immediately.
            next_eval_tokens = tokens_seen

    if checkpoint_every_tokens > 0:
        next_checkpoint_tokens = checkpoint_every_tokens
        while next_checkpoint_tokens <= tokens_seen:
            next_checkpoint_tokens += checkpoint_every_tokens
    else:
        next_checkpoint_tokens = target_tokens + 1

    recent_losses: list[float] = []
    start_time = time.perf_counter()
    last_eval_time = start_time

    print(
        f"seed={seed} parameters={sum(p.numel() for p in model.parameters()):,} "
        f"device={device} tokens_per_step={tokens_per_step:,}"
    )

    while tokens_seen <= target_tokens:
        if tokens_seen >= next_eval_tokens:
            write_status(
                seed_status_path,
                {
                    "stage": "evaluating",
                    "seed": seed,
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "next_eval_tokens": next_eval_tokens,
                    "updated_at": time.time(),
                },
            )
            synchronize(device)
            val_loss, val_ppl = evaluate(model, val_path, dtype, config, device)
            now = time.perf_counter()
            elapsed = now - start_time
            interval_elapsed = now - last_eval_time
            interval_tokens = tokens_seen - last_eval_tokens
            tokens_per_second = interval_tokens / interval_elapsed if interval_elapsed > 0 else 0.0
            train_loss = float(np.mean(recent_losses)) if recent_losses else np.nan
            row = {
                "run_name": experiment["run_name"],
                "seed": seed,
                "interval_index": interval_index,
                "tokens_seen": tokens_seen,
                "target_checkpoint_tokens": next_eval_tokens,
                "step": step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_ppl": val_ppl,
                "elapsed_seconds": elapsed,
                "tokens_per_second": tokens_per_second,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_equivalent": tokens_seen / max(1, int(meta["train"]["tokens"])),
                "device": str(device),
                "precision": str(training.get("precision", "fp32")),
            }
            write_metric(metrics_path, row)
            write_status(
                seed_status_path,
                {
                    "stage": "checkpointed",
                    "seed": seed,
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "interval_index": interval_index,
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "tokens_per_second": tokens_per_second,
                    "updated_at": time.time(),
                },
            )
            write_status(
                run_status_path,
                {
                    "stage": "seed_checkpointed",
                    "seed": seed,
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "interval_index": interval_index,
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "tokens_per_second": tokens_per_second,
                    "updated_at": time.time(),
                },
            )
            print(
                f"seed={seed} interval={interval_index} tokens={tokens_seen:,} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} ppl={val_ppl:.2f}"
            )
            recent_losses = []
            interval_index += 1
            last_eval_time = now
            last_eval_tokens = tokens_seen
            next_eval_tokens += checkpoint_tokens

        if tokens_seen >= target_tokens:
            break

        lr = cosine_lr(
            tokens_seen,
            target_tokens,
            int(training.get("warmup_tokens", 0)),
            float(training["learning_rate"]),
            float(training["min_learning_rate"]),
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for _ in range(grad_accum):
            x, y = batcher.next_batch(micro_batch_size, device)
            with autocast_for(device, str(training.get("precision", "fp32"))):
                _, loss = model(x, y)
            if loss is None:
                raise RuntimeError("Training loss was not computed.")
            loss = loss / grad_accum
            loss.backward()
            loss_value += float(loss.detach().cpu())
            tokens_seen += tokens_per_micro

        if float(training.get("grad_clip", 0.0)) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["grad_clip"]))
        optimizer.step()
        step += 1
        recent_losses.append(loss_value)

        if tokens_seen >= next_checkpoint_tokens:
            save_checkpoint(seed_dir, model, optimizer, seed, step, tokens_seen, keep_last)
            write_status(
                seed_status_path,
                {
                    "stage": "checkpoint_saved",
                    "seed": seed,
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "next_checkpoint_tokens": next_checkpoint_tokens,
                    "updated_at": time.time(),
                },
            )
            next_checkpoint_tokens += checkpoint_every_tokens

        log_every = int(training.get("log_every_steps", 0))
        if log_every and step % log_every == 0:
            print(f"seed={seed} step={step} tokens={tokens_seen:,} loss={loss_value:.4f} lr={lr:.2e}")

    if last_eval_tokens != tokens_seen:
        write_status(
            seed_status_path,
            {
                "stage": "final_evaluating",
                "seed": seed,
                "step": step,
                "tokens_seen": tokens_seen,
                "updated_at": time.time(),
            },
        )
        synchronize(device)
        val_loss, val_ppl = evaluate(model, val_path, dtype, config, device)
        now = time.perf_counter()
        train_loss = float(np.mean(recent_losses)) if recent_losses else np.nan
        row = {
            "run_name": experiment["run_name"],
            "seed": seed,
            "interval_index": interval_index,
            "tokens_seen": tokens_seen,
            "target_checkpoint_tokens": min(next_eval_tokens, target_tokens),
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "elapsed_seconds": now - start_time,
            "tokens_per_second": (tokens_seen - last_eval_tokens) / max(now - last_eval_time, 1e-9),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_equivalent": tokens_seen / max(1, int(meta["train"]["tokens"])),
            "device": str(device),
            "precision": str(training.get("precision", "fp32")),
        }
        write_metric(metrics_path, row)
        write_status(
            seed_status_path,
            {
                "stage": "finalized",
                "seed": seed,
                "step": step,
                "tokens_seen": tokens_seen,
                "val_loss": val_loss,
                "val_ppl": val_ppl,
                "updated_at": time.time(),
            },
        )
        write_status(
            run_status_path,
            {
                "stage": "seed_finalized",
                "seed": seed,
                "step": step,
                "tokens_seen": tokens_seen,
                "val_loss": val_loss,
                "val_ppl": val_ppl,
                "updated_at": time.time(),
            },
        )
        print(
            f"seed={seed} final tokens={tokens_seen:,} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} ppl={val_ppl:.2f}"
        )

    save_checkpoint(seed_dir, model, optimizer, seed, step, tokens_seen, keep_last)
    write_status(
        seed_status_path,
        {
            "stage": "done",
            "seed": seed,
            "step": step,
            "tokens_seen": tokens_seen,
            "updated_at": time.time(),
        },
    )
    return metrics_path


def combine_metrics(run_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(run_dir.glob("seed_*/metrics.csv")):
        with metrics_path.open("r", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        return

    deduped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row.get("seed", ""), row.get("interval_index", ""), row.get("tokens_seen", ""))
        deduped[key] = row
    rows = list(deduped.values())

    def _sort_key(row: dict[str, Any]) -> tuple[int, int, int]:
        try:
            seed = int(row.get("seed", "0"))
        except ValueError:
            seed = 0
        try:
            interval = int(row.get("interval_index", "0"))
        except ValueError:
            interval = 0
        try:
            tokens = int(float(row.get("tokens_seen", "0")))
        except ValueError:
            tokens = 0
        return (seed, interval, tokens)

    rows.sort(key=_sort_key)

    output_path = run_dir / "metrics_all.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train(config_path: str | Path, seeds_override: list[int] | None = None) -> None:
    config = load_config(config_path)
    experiment = config["experiment"]
    data_config = config["data"]
    run_dir = repo_path(config, experiment["output_dir"]) / experiment["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    write_status(
        run_dir / "status.json",
        {
            "stage": "starting",
            "run_name": experiment["run_name"],
            "updated_at": time.time(),
        },
    )
    shutil.copy2(config["_config_path"], run_dir / "config.yaml")

    meta_path = repo_path(config, data_config["cache_dir"]) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing prepared data cache: {meta_path}. Run tinystories-prepare first."
        )
    meta = load_json(meta_path)
    tokenizer_path = Path(meta["tokenizer_path"])
    if tokenizer_path.exists():
        Tokenizer.from_file(str(tokenizer_path))

    seeds = seeds_override if seeds_override is not None else [int(seed) for seed in experiment["seeds"]]
    for seed in seeds:
        run_seed(config, int(seed), run_dir, meta)
    combine_metrics(run_dir)
    write_status(
        run_dir / "status.json",
        {
            "stage": "done",
            "run_name": experiment["run_name"],
            "metrics_path": str(run_dir / "metrics_all.csv"),
            "updated_at": time.time(),
        },
    )
    print(f"Training complete. Metrics: {run_dir / 'metrics_all.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TinyStories Llama-style model on torch.xpu.")
    parser.add_argument("--config", required=True, help="Path to YAML experiment config.")
    parser.add_argument("--seeds", nargs="*", type=int, help="Optional seed override.")
    args = parser.parse_args()
    train(args.config, args.seeds)


if __name__ == "__main__":
    main()
