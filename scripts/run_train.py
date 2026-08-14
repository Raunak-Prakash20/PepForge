#!/usr/bin/env python3
"""Train the Conditioned Autoregressive Peptide Generator.

Usage:
    python scripts/run_train.py --epochs 20 --lr 0.001
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from rich.console import Console
from rich.progress import track
import torch
from torch.utils.data import DataLoader

from pepforge.data.cleaner import CleanedPeptideRecord
from pepforge.models.generator import PepForgeGenerator
from pepforge.models.tokenizer import PeptideTokenizer
from pepforge.models.trainer import PeptideDataset, evaluate_epoch, save_checkpoint, train_epoch

console = Console()


def load_records_from_json(json_path: Path) -> list[CleanedPeptideRecord]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [CleanedPeptideRecord(**item) for item in data]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PepForge-AI Generator")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--d_model", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--n_layers", type=int, default=4, help="Transformer layers")
    parser.add_argument("--n_heads", type=int, default=4, help="Attention heads")
    args = parser.parse_args()

    console.print("\n[bold cyan]===========================================================[/bold cyan]")
    console.print("[bold green]   PepForge-AI: Conditioned Transformer Training Engine    [/bold green]")
    console.print("[bold cyan]===========================================================[/bold cyan]\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[yellow]-> Runtime Device:[/yellow] [bold]{device}[/bold]")

    train_path = Path("data/processed/train.json")
    val_path = Path("data/processed/val.json")

    if not train_path.exists():
        console.print("[bold red]Error: data/processed/train.json not found! Run scripts/run_ingest.py first.[/bold red]")
        return

    train_records = load_records_from_json(train_path)
    val_records = load_records_from_json(val_path)

    tokenizer = PeptideTokenizer()
    console.print(f"[green][OK] Tokenizer initialized (Vocab size: {tokenizer.vocab_size})[/green]")

    train_ds = PeptideDataset(train_records, tokenizer, max_seq_len=48)
    val_ds = PeptideDataset(val_records, tokenizer, max_seq_len=48)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = PepForgeGenerator(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_model * 4,
        max_seq_len=64,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    console.print(f"[green][OK] Model architecture: {args.n_layers} layers, {args.n_heads} heads ({total_params:,} parameters)[/green]\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    checkpoint_dir = Path("models")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_ppl = evaluate_epoch(model, val_loader, device)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, tokenizer, checkpoint_dir, epoch, val_loss)
            star = "[bold yellow]*[/bold yellow]"
        else:
            star = " "

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            console.print(
                f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} {star}"
            )

    console.print(f"\n[bold green][OK] Training complete. Best checkpoint saved to {checkpoint_dir / 'pepforge_model.pt'}[/bold green]\n")


if __name__ == "__main__":
    main()
