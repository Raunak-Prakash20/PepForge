"""Training and Evaluation Loop for PepForge-AI Generative Model.

Includes:
- Autoregressive causal PeptideDataset with target shifting
- AdamW Optimizer with Cosine Annealing Learning Rate Schedule
- Perplexity (PPL) tracking on Validation sets
- Model Checkpoint Persistence
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import DataLoader, Dataset

from pepforge.data.cleaner import CleanedPeptideRecord
from pepforge.filters.bro5_rules import calculate_net_charge
from pepforge.models.generator import PepForgeGenerator
from pepforge.models.tokenizer import PeptideTokenizer


class PeptideDataset(Dataset):
    """PyTorch Dataset for conditioned autoregressive sequence modeling."""

    def __init__(
        self,
        records: List[CleanedPeptideRecord],
        tokenizer: PeptideTokenizer,
        max_seq_len: int = 64,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._prepare()

    def _prepare(self) -> None:
        for rec in self.records:
            charge = calculate_net_charge(rec.sequence)
            prefix = self.tokenizer.build_condition_prefix(
                target_organism=rec.target_organism,
                gram_type=rec.gram_type,
                target_charge=charge,
                target_length=rec.length,
            )

            # Full sequence tokens: BOS + condition + AAs + EOS
            full_ids = self.tokenizer.encode(
                sequence=rec.sequence,
                condition_prefix=prefix,
                add_bos=True,
                add_eos=True,
            )

            if len(full_ids) > self.max_seq_len:
                full_ids = full_ids[:self.max_seq_len]

            # Shift for autoregressive LM:
            # input_ids: full_ids[:-1]
            # target_ids: full_ids[1:]
            input_ids = full_ids[:-1]
            target_ids = full_ids[1:]

            # Padding
            pad_len = self.max_seq_len - len(input_ids)
            mask = [1] * len(input_ids) + [0] * pad_len
            input_padded = input_ids + [self.tokenizer.pad_id] * pad_len
            target_padded = target_ids + [self.tokenizer.pad_id] * pad_len

            self.samples.append((
                torch.tensor(input_padded, dtype=torch.long),
                torch.tensor(target_padded, dtype=torch.long),
                torch.tensor(mask, dtype=torch.long),
            ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def train_epoch(
    model: PepForgeGenerator,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch and return average cross-entropy loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for inputs, targets, masks in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        _, loss = model(inputs, targets=targets, attention_mask=masks)

        if loss is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate_epoch(
    model: PepForgeGenerator,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on validation loader. Returns (val_loss, perplexity)."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for inputs, targets, masks in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        masks = masks.to(device)

        _, loss = model(inputs, targets=targets, attention_mask=masks)
        if loss is not None:
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    ppl = math.exp(min(avg_loss, 20.0))  # Safeguard against overflow
    return avg_loss, ppl


def save_checkpoint(
    model: PepForgeGenerator,
    tokenizer: PeptideTokenizer,
    checkpoint_dir: Path,
    epoch: int,
    val_loss: float,
) -> None:
    """Save trained weights and configuration."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / "pepforge_model.pt"

    torch.save({
        "epoch": epoch,
        "val_loss": val_loss,
        "model_state_dict": model.state_dict(),
        "model_config": {
            "vocab_size": model.vocab_size,
            "d_model": model.d_model,
            "max_seq_len": model.max_seq_len,
        },
    }, ckpt_path)
