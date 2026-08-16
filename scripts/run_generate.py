#!/usr/bin/env python3
"""Run End-to-End Conditioned Generation and Complete 5-Stage In Silico Screening.

Usage:
    python scripts/run_generate.py --target "Pseudomonas aeruginosa" --num_candidates 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
import torch

from pepforge.models.generator import PepForgeGenerator
from pepforge.models.tokenizer import PeptideTokenizer
from pepforge.pipeline.screening_funnel import ScreeningFunnel

console = Console()


def load_trained_model(
    checkpoint_path: Path,
    tokenizer: PeptideTokenizer,
    device: torch.device,
) -> PepForgeGenerator:
    """Load trained model weights from checkpoint or initialize architecture."""
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        cfg = ckpt.get("model_config", {})
        model = PepForgeGenerator(
            vocab_size=tokenizer.vocab_size,
            d_model=cfg.get("d_model", 128),
            max_seq_len=cfg.get("max_seq_len", 64),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        console.print(f"[green][OK] Loaded trained weights from: {checkpoint_path}[/green]")
    else:
        console.print("[yellow]Notice: Checkpoint not found. Initializing model with default weights.[/yellow]")
        model = PepForgeGenerator(
            vocab_size=tokenizer.vocab_size,
            d_model=128,
            max_seq_len=64,
        ).to(device)

    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and screen antimicrobial peptide candidates")
    parser.add_argument("--target", type=str, default="Pseudomonas aeruginosa", help="Target bacterial pathogen")
    parser.add_argument("--gram", type=str, default="negative", help="Gram status (negative/positive)")
    parser.add_argument("--charge", type=float, default=3.5, help="Target net charge")
    parser.add_argument("--length", type=int, default=18, help="Target sequence length")
    parser.add_argument("--num_candidates", type=int, default=30, help="Number of sequences to generate")
    parser.add_argument("--temperature", type=float, default=0.85, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.90, help="Nucleus sampling top-p")
    args = parser.parse_args()

    console.print("\n[bold cyan]===========================================================[/bold cyan]")
    console.print("[bold green]    PepForge-AI: Target-Conditioned De Novo Lead Discovery   [/bold green]")
    console.print("[bold cyan]===========================================================[/bold cyan]\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = PeptideTokenizer()
    model = load_trained_model(Path("models/pepforge_model.pt"), tokenizer, device)

    # 1. Condition Prefix
    prefix = tokenizer.build_condition_prefix(
        target_organism=args.target,
        gram_type=args.gram,
        target_charge=args.charge,
        target_length=args.length,
    )
    console.print(f"\n[bold yellow]-> Target Pathogen:[/bold yellow] [bold]{args.target}[/bold] (Gram-{args.gram})")
    console.print(f"[bold yellow]-> Conditioning Prefix:[/bold yellow] {' '.join(prefix)}")
    console.print(f"[bold yellow]-> Generating {args.num_candidates} candidate sequences...[/bold yellow]")

    raw_candidates = model.generate_batch(
        tokenizer=tokenizer,
        condition_prefix=prefix,
        num_samples=args.num_candidates,
        min_generate_len=max(10, args.length - 4),
        max_generate_len=args.length + 4,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
    )
    # Deduplicate generated batch
    unique_candidates = list(dict.fromkeys(raw_candidates))
    console.print(f"   [green][OK] Generated {len(unique_candidates)} unique candidate sequences.[/green]")

    # 2. Automated 5-Stage Screening Funnel
    console.print("\n[yellow]-> Running candidates through Complete 5-Stage In Silico Screening Funnel...[/yellow]")
    funnel = ScreeningFunnel()
    evaluations = funnel.screen_batch(unique_candidates, target_organism=args.target)

    passing_leads = [e for e in evaluations if e.overall_passed]
    console.print(f"   [green][OK] {len(passing_leads)} / {len(unique_candidates)} passed all 5 stages (bRo5, ADMET, 3D, FTO).[/green]\n")

    # 3. Rank and Display Leads
    table = Table(
        title=f"PepForge-AI 5-Stage Discovery Dossier: {args.target}",
        header_style="bold magenta",
        show_lines=True,
    )
    table.add_column("Rank", justify="center", width=6)
    table.add_column("Sequence", style="bold green", width=22)
    table.add_column("Len", justify="right", width=5)
    table.add_column("Charge", justify="right", width=8)
    table.add_column("Hydro %", justify="right", width=9)
    table.add_column("Pred. HC50", justify="right", width=12)
    table.add_column("pLDDT", justify="right", width=8)
    table.add_column("Helix %", justify="right", width=9)
    table.add_column("Max Id %", justify="right", width=10)
    table.add_column("Status", justify="center", width=14)

    # Sort passing leads by Selectivity Index descending, then by pLDDT descending
    passing_leads.sort(
        key=lambda e: (
            -(e.admet_toxicity_metrics["selectivity_index"] or 0),
            -e.structure_metrics["predicted_plddt"],
        )
    )

    display_leads = passing_leads[:10] if passing_leads else evaluations[:8]

    for idx, lead in enumerate(display_leads, 1):
        b = lead.bro5_metrics
        t = lead.admet_toxicity_metrics
        s = lead.structure_metrics
        n = lead.novelty_metrics

        rank_str = f"#{idx}" if lead.overall_passed else f"[dim]#{idx}[/dim]"
        seq_style = "bold green" if lead.overall_passed else "dim yellow"
        status_str = "[bold green]PASS[/bold green]" if lead.overall_passed else f"[dim red]FAIL ({lead.rejection_stage})[/dim red]"

        table.add_row(
            rank_str,
            f"[{seq_style}]{lead.sequence}[/{seq_style}]",
            str(b["length"]),
            f"{b['net_charge_ph74']:+.1f}",
            f"{b['hydrophobic_ratio_pct']:.0f}%",
            f"{t['predicted_hc50_ug_ml']:.0f} ug/mL",
            f"{s['predicted_plddt']:.1f}",
            f"{s['helical_content_pct']:.0f}%",
            f"{n['max_sequence_identity_pct']:.0f}%",
            status_str,
        )

    console.print(table)

    # 4. Save Detailed Dossier
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    dossier_path = results_dir / "leads_dossier.json"
    with open(dossier_path, "w", encoding="utf-8") as f:
        json.dump([e.model_dump() for e in evaluations], f, indent=2)

    console.print(f"\n[bold green][OK] Complete 5-Stage discovery dossier saved to: {dossier_path}[/bold green]\n")


if __name__ == "__main__":
    main()
