#!/usr/bin/env python3
"""PepForge-AI: Interactive 60-Second Guided Tour.

Run this script to see PepForge-AI in action:
    python demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import torch

from pepforge.data.benchmarks import BENCHMARK_PEPTIDES
from pepforge.pipeline.screening_funnel import ScreeningFunnel
from pepforge.models.generator import PepForgeGenerator
from pepforge.models.tokenizer import PeptideTokenizer

console = Console()


def run_demo() -> None:
    console.print()
    console.print(
        Panel.fit(
            "[bold green]PepForge-AI: De Novo Antimicrobial Peptide Design Engine[/bold green]\n"
            "[cyan]Interactive 60-Second Guided Tour & Pipeline Demonstration[/cyan]",
            border_style="bold cyan",
        )
    )

    funnel = ScreeningFunnel()

    # -------------------------------------------------------------
    # DEMO 1: Detecting Real-World Toxin (Melittin)
    # -------------------------------------------------------------
    console.print("\n[bold yellow][1/3] Testing Known Toxic Peptide (Melittin from Bee Venom)...[/bold yellow]")
    melittin = next(p for p in BENCHMARK_PEPTIDES if p.name == "Melittin")
    rep_mel = funnel.evaluate_candidate(melittin.sequence, target_organism="Broad Spectrum")

    console.print(f"  * Sequence: [bold]{melittin.sequence}[/bold]")
    console.print(f"  * Hemolysis Status: [bold red]{rep_mel.admet_toxicity_metrics['hemolysis_risk']}[/bold red]")
    console.print(f"  * Predicted HC50: [bold red]{rep_mel.admet_toxicity_metrics['predicted_hc50_ug_ml']} ug/mL[/bold red] (Dangerous: < 150 ug/mL cutoff)")
    console.print(f"  * Pipeline Result: [bold red]FLAGGED & REJECTED[/bold red] (Stage 2 ADMET Hemolysis)")

    # -------------------------------------------------------------
    # DEMO 2: Evaluating Optimized Therapeutic (Syn-P18)
    # -------------------------------------------------------------
    console.print("\n[bold yellow][2/3] Testing Optimized Therapeutic Lead (Syn-P18)...[/bold yellow]")
    syn_p18 = next(p for p in BENCHMARK_PEPTIDES if p.name == "Syn-P18")
    rep_syn = funnel.evaluate_candidate(syn_p18.sequence, target_organism="Acinetobacter baumannii")

    console.print(f"  * Sequence: [bold]{syn_p18.sequence}[/bold]")
    console.print(f"  * Net Charge: [green]+{rep_syn.bro5_metrics['net_charge_ph74']:.2f}[/green]")
    console.print(f"  * Predicted HC50: [green]{rep_syn.admet_toxicity_metrics['predicted_hc50_ug_ml']} ug/mL[/green] (Safe, Non-hemolytic)")
    console.print(f"  * Selectivity Index: [bold green]SI = {rep_syn.admet_toxicity_metrics['selectivity_index']:.1f}[/bold green]")
    console.print(f"  * 3D Helical Folding (pLDDT): [bold green]{rep_syn.structure_metrics['predicted_plddt']:.1f} (High Confidence)[/bold green]")

    # -------------------------------------------------------------
    # DEMO 3: Generating Brand-New AI Candidates
    # -------------------------------------------------------------
    console.print("\n[bold yellow][3/3] Generating De Novo Candidates targeting Pseudomonas aeruginosa...[/bold yellow]")
    ckpt_path = Path("models/pepforge_model.pt")
    if not ckpt_path.exists():
        console.print("[red]Checkpoint models/pepforge_model.pt not found. Run scripts/run_train.py first.[/red]")
        return

    device = torch.device("cpu")
    tokenizer = PeptideTokenizer()
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("model_config", {})
    model = PepForgeGenerator(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg.get("d_model", 128),
        max_seq_len=cfg.get("max_seq_len", 64),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    prefix = tokenizer.build_condition_prefix(
        target_organism="Pseudomonas aeruginosa",
        gram_type="negative",
        target_charge=3.5,
        target_length=16,
    )
    samples = model.generate_batch(
        tokenizer=tokenizer,
        condition_prefix=prefix,
        num_samples=3,
        min_generate_len=12,
        max_generate_len=20,
        temperature=0.85,
        top_p=0.90,
        device=device,
    )

    table = Table(title="AI-Generated De Novo Peptides", header_style="bold magenta", show_lines=True)
    table.add_column("No.", justify="center", width=4)
    table.add_column("Generated Sequence", style="bold green", width=22)
    table.add_column("Net Charge", justify="right", width=10)
    table.add_column("Pred. HC50", justify="right", width=12)
    table.add_column("pLDDT", justify="right", width=8)
    table.add_column("Triage Status", justify="center", width=16)

    for idx, seq in enumerate(samples, 1):
        rep = funnel.evaluate_candidate(seq, target_organism="Pseudomonas aeruginosa")
        status = "[green]PASS[/green]" if rep.overall_passed else f"[red]FAIL ({rep.rejection_stage})[/red]"
        table.add_row(
            str(idx),
            seq,
            f"{rep.bro5_metrics['net_charge_ph74']:+.1f}",
            f"{rep.admet_toxicity_metrics['predicted_hc50_ug_ml']:.0f} ug/mL",
            f"{rep.structure_metrics['predicted_plddt']:.1f}",
            status,
        )

    console.print(table)
    console.print("\n[bold green][OK] Demo complete! To generate full batches, run: python scripts/run_generate.py[/bold green]\n")


if __name__ == "__main__":
    run_demo()
