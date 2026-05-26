from __future__ import annotations

import json
import math
import os
import platform
import random
import socket
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        return torch.device("cpu")
    if requested == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError(
                "torch.xpu is not available. Install Intel GPU drivers and "
                "PyTorch XPU wheels, then rerun tinystories-verify-xpu."
            )
        return torch.device("xpu")
    return torch.device(requested)


def precision_dtype(precision: str) -> torch.dtype | None:
    precision = precision.lower()
    if precision in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if precision in {"fp16", "float16"}:
        return torch.float16
    if precision in {"fp32", "float32", "none"}:
        return None
    raise ValueError(f"Unsupported precision: {precision}")


def autocast_for(device: torch.device, precision: str):
    dtype = precision_dtype(precision)
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)


def synchronize(device: torch.device) -> None:
    if device.type == "xpu":
        torch.xpu.synchronize()


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def system_fingerprint(device: torch.device) -> dict[str, Any]:
    xpu_name = None
    xpu_count = 0
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        xpu_count = torch.xpu.device_count()
        try:
            xpu_name = torch.xpu.get_device_name(0)
        except Exception:
            xpu_name = "unknown-xpu"

    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "xpu_available": bool(hasattr(torch, "xpu") and torch.xpu.is_available()),
        "xpu_count": xpu_count,
        "xpu_name": xpu_name,
        "pid": os.getpid(),
    }


def cosine_lr(
    tokens_seen: int,
    target_tokens: int,
    warmup_tokens: int,
    learning_rate: float,
    min_learning_rate: float,
) -> float:
    if warmup_tokens > 0 and tokens_seen < warmup_tokens:
        return learning_rate * max(1, tokens_seen) / warmup_tokens
    if target_tokens <= warmup_tokens:
        return min_learning_rate
    decay_ratio = (tokens_seen - warmup_tokens) / (target_tokens - warmup_tokens)
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_learning_rate + coeff * (learning_rate - min_learning_rate)


def batched(iterable: Iterator[Any], batch_size: int) -> Iterator[list[Any]]:
    batch: list[Any] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
