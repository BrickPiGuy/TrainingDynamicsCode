from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tqdm import tqdm

from tinystories_xpu.config import load_config, repo_path
from tinystories_xpu.utils import save_json


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def dataset_texts(config: dict, split: str, limit: int | None = None) -> Iterable[str]:
    data_config = config["data"]
    dataset = load_dataset(
        data_config["dataset_name"],
        split=split,
        streaming=bool(data_config.get("streaming", False)),
    )
    text_field = data_config.get("text_field", "text")
    for index, row in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        text = row.get(text_field)
        if text:
            yield text


def train_tokenizer(config: dict) -> Tokenizer:
    data_config = config["data"]
    tokenizer_path = repo_path(config, data_config["tokenizer_path"])
    if tokenizer_path.exists():
        return Tokenizer.from_file(str(tokenizer_path))

    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=int(data_config["tokenizer_vocab_size"]),
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    limit = data_config.get("tokenizer_train_documents")
    texts = dataset_texts(config, data_config["train_split"], limit=limit)
    tokenizer.train_from_iterator(texts, trainer=trainer, length=limit)
    tokenizer.save(str(tokenizer_path))
    return tokenizer


def encode_split(config: dict, tokenizer: Tokenizer, split_name: str, output_name: str) -> dict:
    data_config = config["data"]
    cache_dir = repo_path(config, data_config["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(data_config.get("dtype", "uint16"))
    output_path = cache_dir / output_name
    eos_id = tokenizer.token_to_id("<eos>")
    append_eos = bool(data_config.get("append_eos", True))
    document_limit = data_config.get("prepare_document_limit")

    count = 0
    docs = 0
    with output_path.open("wb") as handle:
        progress = tqdm(dataset_texts(config, split_name, limit=document_limit), desc=f"encoding {split_name}")
        for text in progress:
            ids = tokenizer.encode(text).ids
            if append_eos and eos_id is not None:
                ids.append(eos_id)
            if not ids:
                continue
            array = np.asarray(ids, dtype=dtype)
            array.tofile(handle)
            count += int(array.size)
            docs += 1
            progress.set_postfix(tokens=count, docs=docs)

    return {
        "split": split_name,
        "path": str(output_path),
        "tokens": count,
        "documents": docs,
        "dtype": str(dtype),
    }


def prepare(config_path: str | Path) -> None:
    config = load_config(config_path)
    data_config = config["data"]
    tokenizer = train_tokenizer(config)
    cache_dir = repo_path(config, data_config["cache_dir"])
    train_meta = encode_split(config, tokenizer, data_config["train_split"], "train.bin")
    val_meta = encode_split(config, tokenizer, data_config["validation_split"], "validation.bin")

    meta = {
        "dataset_name": data_config["dataset_name"],
        "tokenizer_path": str(repo_path(config, data_config["tokenizer_path"])),
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS},
        "train": train_meta,
        "validation": val_meta,
    }
    save_json(cache_dir / "meta.json", meta)
    print(f"Prepared TinyStories cache at {cache_dir}")
    print(f"Train tokens: {train_meta['tokens']:,}")
    print(f"Validation tokens: {val_meta['tokens']:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TinyStories token cache.")
    parser.add_argument("--config", required=True, help="Path to YAML experiment config.")
    args = parser.parse_args()
    prepare(args.config)


if __name__ == "__main__":
    main()
