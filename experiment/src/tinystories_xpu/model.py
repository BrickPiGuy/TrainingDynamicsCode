from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LlamaConfig:
    vocab_size: int
    block_size: int = 512
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    multiple_of: int = 256
    rope_base: float = 10000.0

    @classmethod
    def from_dict(cls, values: dict) -> "LlamaConfig":
        return cls(**{key: values[key] for key in cls.__annotations__ if key in values})


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(q: torch.Tensor, k: torch.Tensor, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, seq_len, head_dim = q.shape
    device = q.device
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    cos = torch.repeat_interleave(angles.cos(), 2, dim=-1).to(dtype=q.dtype)
    sin = torch.repeat_interleave(angles.sin(), 2, dim=-1).to(dtype=q.dtype)
    cos = cos.view(1, 1, seq_len, head_dim)
    sin = sin.view(1, 1, seq_len, head_dim)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.rope_base = config.rope_base
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, self.rope_base)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(~self.causal_mask[:, :, :seq_len, :seq_len], torch.finfo(att.dtype).min)
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.resid_dropout(self.o_proj(y))


class FeedForward(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        hidden_dim = int(2 * (4 * config.n_embd) / 3)
        hidden_dim = config.multiple_of * math.ceil(hidden_dim / config.multiple_of)
        self.gate_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class LlamaLanguageModel(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd)
        self.output = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds block size {self.config.block_size}")
        x = self.dropout(self.tok_embeddings(idx))
        for layer in self.layers:
            x = layer(x)
        logits = self.output(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
