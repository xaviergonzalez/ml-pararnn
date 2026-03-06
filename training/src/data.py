"""
Data loading for SlimPajama (without Books3).
Streams data from HuggingFace and tokenizes on-the-fly.
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
from omegaconf import DictConfig


class SlimPajamaDataset(IterableDataset):
    """Streaming dataset for SlimPajama that excludes Books3 and packs sequences."""

    def __init__(self, cfg: DictConfig, split: str = "train", seq_length: int = 2048):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.seq_length = seq_length
        self.exclude_sources = list(cfg.data.get("exclude_sources", []))

    def _get_tokenizer(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.data.tokenizer,
            cache_dir=self.cfg.data.cache_dir,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def __iter__(self):
        from datasets import load_dataset
        tokenizer = self._get_tokenizer()

        ds = load_dataset(
            self.cfg.data.path,
            split=self.split,
            streaming=True,
            cache_dir=self.cfg.data.cache_dir,
        )

        # Filter out Books3
        if self.exclude_sources:
            ds = ds.filter(
                lambda x: not any(src in x.get("meta", {}).get("redpajama_set_name", "")
                                  for src in self.exclude_sources)
            )

        # Shuffle with buffer
        ds = ds.shuffle(seed=42, buffer_size=10000)

        # Pack tokens into fixed-length sequences
        buffer = []
        for example in ds:
            tokens = tokenizer(
                example["text"],
                truncation=False,
                add_special_tokens=False,
            )["input_ids"]
            buffer.extend(tokens)

            while len(buffer) >= self.seq_length + 1:
                chunk = buffer[:self.seq_length + 1]
                buffer = buffer[self.seq_length + 1:]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                targets = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "targets": targets}


class ValidationDataset(IterableDataset):
    """Validation split of SlimPajama for perplexity measurement."""

    def __init__(self, cfg: DictConfig, seq_length: int = 2048, max_samples: int = 500):
        super().__init__()
        self.cfg = cfg
        self.seq_length = seq_length
        self.max_samples = max_samples

    def __iter__(self):
        from datasets import load_dataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.data.tokenizer,
            cache_dir=self.cfg.data.cache_dir,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        ds = load_dataset(
            self.cfg.data.path,
            split="validation",
            streaming=True,
            cache_dir=self.cfg.data.cache_dir,
        )

        buffer = []
        count = 0
        for example in ds:
            if count >= self.max_samples:
                break
            tokens = tokenizer(example["text"], truncation=False, add_special_tokens=False)["input_ids"]
            buffer.extend(tokens)
            while len(buffer) >= self.seq_length + 1 and count < self.max_samples:
                chunk = buffer[:self.seq_length + 1]
                buffer = buffer[self.seq_length + 1:]
                yield {
                    "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                    "targets": torch.tensor(chunk[1:], dtype=torch.long),
                }
                count += 1


def build_train_dataloader(cfg: DictConfig):
    dataset = SlimPajamaDataset(cfg, split="train", seq_length=cfg.seq_length)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size_per_gpu,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )


def build_val_dataloader(cfg: DictConfig):
    dataset = ValidationDataset(cfg, seq_length=cfg.seq_length)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size_per_gpu,
        num_workers=2,
        pin_memory=True,
    )
