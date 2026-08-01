#!/usr/bin/env python3
"""Run PepForge-AI Data Ingestion & Homology Clustering Pipeline.

Usage:
    python scripts/run_ingest.py
"""

from __future__ import annotations

import json
from pathlib import Path
from rich.console import Console
from rich.table import Table

from pepforge.data.cleaner import deduplicate_records
from pepforge.data.clustering import cluster_records, split_clustered_dataset
from pepforge.data.downloader import export_dataset_fasta, export_dataset_json, load_seed_records

console = Console()


def main() -> None:
    console.print("\n[bold cyan]===========================================================[/bold cyan]")
    console.print("[bold green]    PepForge-AI: Data Ingestion & CD-HIT Clustering Engine   [/bold green]")
    console.print("[bold cyan]===========================================================[/bold cyan]\n")

    # 1. Ingestion
    console.print("[yellow]-> Step 1: Loading and canonicalizing experimental records...[/yellow]")
    records = load_seed_records()
    console.print(f"   [green][OK] Loaded {len(records)} clean non-redundant records.[/green]")

    # 2. Sequence Clustering
    console.print("\n[yellow]-> Step 2: Running Greedy Sequence Clustering (80% identity)...[/yellow]")
    clustered = cluster_records(records, identity_threshold=0.80)
    num_clusters = len(set(c.cluster_id for c in clustered))
    console.print(f"   [green][OK] Clustered into {num_clusters} distinct homology families.[/green]")

    # 3. Cluster-level Splitting
    console.print("\n[yellow]-> Step 3: Performing cluster-aware Train/Val/Test partitioning...[/yellow]")
    split = split_clustered_dataset(clustered, train_ratio=0.80, val_ratio=0.10, test_ratio=0.10)

    # 4. Table Summary
    table = Table(title="Partition Summary", header_style="bold magenta", show_lines=True)
    table.add_column("Split", style="bold yellow", width=12)
    table.add_column("Clusters", justify="right", width=12)
    table.add_column("Sequences", justify="right", width=12)
    table.add_column("Ratio (%)", justify="right", width=12)

    total_seqs = len(records)
    table.add_row("Train", str(split.cluster_counts["train_clusters"]), str(split.cluster_counts["train_sequences"]), f"{split.cluster_counts['train_sequences']/total_seqs*100:.1f}%")
    table.add_row("Validation", str(split.cluster_counts["val_clusters"]), str(split.cluster_counts["val_sequences"]), f"{split.cluster_counts['val_sequences']/total_seqs*100:.1f}%")
    table.add_row("Test", str(split.cluster_counts["test_clusters"]), str(split.cluster_counts["test_sequences"]), f"{split.cluster_counts['test_sequences']/total_seqs*100:.1f}%")
    console.print(table)

    # 5. Export Processed Datasets
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    export_dataset_json([c.record for c in split.train], processed_dir / "train.json")
    export_dataset_json([c.record for c in split.val], processed_dir / "val.json")
    export_dataset_json([c.record for c in split.test], processed_dir / "test.json")
    export_dataset_fasta([c.record for c in clustered], processed_dir / "all_clustered.fasta")

    with open(processed_dir / "cluster_metadata.json", "w", encoding="utf-8") as f:
        json.dump(split.cluster_counts, f, indent=2)

    console.print(f"\n[bold green][OK] Processed datasets exported to: {processed_dir.resolve()}[/bold green]\n")


if __name__ == "__main__":
    main()
