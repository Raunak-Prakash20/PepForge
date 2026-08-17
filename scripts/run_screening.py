#!/usr/bin/env python3
"""Run PepForge-AI 5-Stage Screening Funnel on Benchmark and Candidate Peptides.

Usage:
    python scripts/run_screening.py
"""

from __future__ import annotations

import json
from pathlib import Path
from rich.console import Console
from rich.table import Table

from pepforge.data.benchmarks import BENCHMARK_PEPTIDES
from pepforge.pipeline.screening_funnel import ScreeningFunnel

console = Console()


def main() -> None:
    console.print(
        "\n[bold cyan]===========================================================[/bold cyan]"
    )
    console.print(
        "[bold green]   PepForge-AI: 5-Stage In Silico Screening Funnel Demo   [/bold green]"
    )
    console.print(
        "[bold cyan]===========================================================[/bold cyan]\n"
    )

    funnel = ScreeningFunnel()

    table = Table(
        title="5-Stage Screening Funnel Evaluation Results",
        header_style="bold magenta",
        show_lines=True,
    )
    table.add_column("Peptide Name", style="bold yellow", width=18)
    table.add_column("Len", justify="right", width=5)
    table.add_column("Charge", justify="right", width=8)
    table.add_column("Hydro %", justify="right", width=9)
    table.add_column("Boman", justify="right", width=8)
    table.add_column("uH", justify="right", width=7)
    table.add_column("Pred. HC50", justify="right", width=12)
    table.add_column("pLDDT", justify="right", width=8)
    table.add_column("Max Id %", justify="right", width=10)
    table.add_column("Status", justify="center", width=16)

    evaluations = []

    for bench in BENCHMARK_PEPTIDES:
        eval_res = funnel.evaluate_candidate(
            sequence=bench.sequence,
            target_organism=bench.target_organism,
            assumed_mic_ug_ml=bench.experimental_mic_ug_ml or 4.0,
        )
        evaluations.append(eval_res)

        # Status badge
        if eval_res.overall_passed:
            status_str = "[bold green]PASS[/bold green]"
        else:
            status_str = f"[dim red]FAIL ({eval_res.rejection_stage})[/dim red]"

        b = eval_res.bro5_metrics
        t = eval_res.admet_toxicity_metrics
        s = eval_res.structure_metrics
        n = eval_res.novelty_metrics

        table.add_row(
            bench.name,
            str(b["length"]),
            f"{b['net_charge_ph74']:+.1f}",
            f"{b['hydrophobic_ratio_pct']:.0f}%",
            f"{b['boman_index_kcal_mol']:.2f}",
            f"{b['hydrophobic_moment']:.2f}",
            f"{t['predicted_hc50_ug_ml']:.0f} ug/mL",
            f"{s['predicted_plddt']:.1f}",
            f"{n['max_sequence_identity_pct']:.0f}%",
            status_str,
        )

    console.print(table)

    # Save summary report to results/
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    report_file = results_dir / "benchmark_screening_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump([e.model_dump() for e in evaluations], f, indent=2)

    console.print(f"\n[bold green][OK] Detailed 5-Stage diagnostic report saved to: {report_file}[/bold green]\n")


if __name__ == "__main__":
    main()
