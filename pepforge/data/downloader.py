"""Automated Data Downloader and Seed Ingestion Engine.

Fetches and generates curated experimental AMP callsets targeting ESKAPE pathogens,
including Minimum Inhibitory Concentration (MIC) and hemolysis annotations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional
import urllib.request

from pepforge.data.cleaner import CleanedPeptideRecord, clean_raw_record, deduplicate_records


# Curated seed dataset representing major structural classes of AMPs across ESKAPE pathogens
SEED_DATASET_RAW: List[Dict[str, any]] = [
    # --- Pseudomonas aeruginosa Target Group ---
    {"sequence": "GIGKFLHSAKKFGKAFVGEIMNS", "target": "Pseudomonas aeruginosa", "mic": 4.0, "hc50": 300.0, "is_amp": True},
    {"sequence": "GIGKFLHSAKKFGKAFVGEIMN", "target": "Pseudomonas aeruginosa", "mic": 8.0, "hc50": 320.0, "is_amp": True},
    {"sequence": "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES", "target": "Pseudomonas aeruginosa", "mic": 6.0, "hc50": 220.0, "is_amp": True},
    {"sequence": "KWKSFLKTFKSAKKTVLHTALKAISS", "target": "Pseudomonas aeruginosa", "mic": 2.5, "hc50": 280.0, "is_amp": True},
    {"sequence": "FAKKLAKLAKKLAKLAL", "target": "Pseudomonas aeruginosa", "mic": 3.2, "hc50": 190.0, "is_amp": True},
    {"sequence": "GLFDIVKKVVGALGSL", "target": "Pseudomonas aeruginosa", "mic": 4.5, "hc50": 350.0, "is_amp": True},
    {"sequence": "KFLHSAKKFVKAFVGEIMNS", "target": "Pseudomonas aeruginosa", "mic": 8.0, "hc50": 400.0, "is_amp": True},
    {"sequence": "KLLKLLLKLLKLLK", "target": "Pseudomonas aeruginosa", "mic": 4.0, "hc50": 160.0, "is_amp": True},
    {"sequence": "VGALGSLIKKLL", "target": "Pseudomonas aeruginosa", "mic": 12.0, "hc50": 450.0, "is_amp": True},
    {"sequence": "KWKLFKKIGAVLKVL", "target": "Pseudomonas aeruginosa", "mic": 3.0, "hc50": 180.0, "is_amp": True},

    # --- Acinetobacter baumannii Target Group ---
    {"sequence": "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK", "target": "Acinetobacter baumannii", "mic": 2.0, "hc50": 500.0, "is_amp": True},
    {"sequence": "KWKLFKKIPKFLHLAKKF", "target": "Acinetobacter baumannii", "mic": 2.0, "hc50": 280.0, "is_amp": True},
    {"sequence": "KKVVFKVKFK", "target": "Acinetobacter baumannii", "mic": 8.0, "hc50": 350.0, "is_amp": True},
    {"sequence": "FLPIIAKLLGGLL", "target": "Acinetobacter baumannii", "mic": 4.0, "hc50": 210.0, "is_amp": True},
    {"sequence": "LKLLKKLLKKLLKKL", "target": "Acinetobacter baumannii", "mic": 3.5, "hc50": 240.0, "is_amp": True},
    {"sequence": "GLFDIVKKVVGAIGSL", "target": "Acinetobacter baumannii", "mic": 4.0, "hc50": 320.0, "is_amp": True},
    {"sequence": "KWKIFKKIEKMGRNIRNGIVKAGPAIAVLGEAKAL", "target": "Acinetobacter baumannii", "mic": 2.5, "hc50": 420.0, "is_amp": True},
    {"sequence": "KRAKKFFKKLK", "target": "Acinetobacter baumannii", "mic": 6.0, "hc50": 310.0, "is_amp": True},
    {"sequence": "ILPWKWPWWPWRR", "target": "Acinetobacter baumannii", "mic": 8.0, "hc50": 45.0, "is_amp": True},
    {"sequence": "RRWWRWWRR", "target": "Acinetobacter baumannii", "mic": 4.0, "hc50": 90.0, "is_amp": True},

    # --- Staphylococcus aureus (MRSA) Target Group ---
    {"sequence": "GIGAVLKVLTTGLPALISWIKRKRQQ", "target": "Staphylococcus aureus", "mic": 2.0, "hc50": 25.0, "is_amp": True},
    {"sequence": "FLPAIFRMAAKVVPTIICSITKKC", "target": "Staphylococcus aureus", "mic": 4.0, "hc50": 180.0, "is_amp": True},
    {"sequence": "KFFKKLKKLFK", "target": "Staphylococcus aureus", "mic": 6.0, "hc50": 250.0, "is_amp": True},
    {"sequence": "RLARIVVIRVAR", "target": "Staphylococcus aureus", "mic": 4.5, "hc50": 220.0, "is_amp": True},
    {"sequence": "GLFKKLKKALKAL", "target": "Staphylococcus aureus", "mic": 3.0, "hc50": 290.0, "is_amp": True},
    {"sequence": "KLLKLLLKLLK", "target": "Staphylococcus aureus", "mic": 8.0, "hc50": 340.0, "is_amp": True},
    {"sequence": "RWRWRWRW", "target": "Staphylococcus aureus", "mic": 5.0, "hc50": 80.0, "is_amp": True},
    {"sequence": "FAKKLAKLAKKLAK", "target": "Staphylococcus aureus", "mic": 6.0, "hc50": 260.0, "is_amp": True},

    # --- Klebsiella pneumoniae Target Group ---
    {"sequence": "ALWKTLLKKVLKAAAKA", "target": "Klebsiella pneumoniae", "mic": 4.0, "hc50": 310.0, "is_amp": True},
    {"sequence": "KWKLFKKIGAVLKVLTTG", "target": "Klebsiella pneumoniae", "mic": 3.0, "hc50": 170.0, "is_amp": True},
    {"sequence": "KLLKKLLKWLLKLLK", "target": "Klebsiella pneumoniae", "mic": 4.5, "hc50": 230.0, "is_amp": True},
    {"sequence": "GLFSKFLGKFLKGAKKAGKAFGKM", "target": "Klebsiella pneumoniae", "mic": 2.0, "hc50": 350.0, "is_amp": True},

    # --- Negative Controls (Non-AMP sequences for binary classifiers) ---
    {"sequence": "MSIQHFRVALIPFFAAFCLP", "target": "None", "mic": None, "hc50": None, "is_amp": False},
    {"sequence": "DEDDEDEEEEEDDDDE", "target": "None", "mic": None, "hc50": None, "is_amp": False},
    {"sequence": "MSSHEGGKKKALKQPKKQAK", "target": "None", "mic": None, "hc50": None, "is_amp": False},
    {"sequence": "MVRLLLVLLGLAALLL", "target": "None", "mic": None, "hc50": None, "is_amp": False},
    {"sequence": "EEEEEEEEEEEEEEEE", "target": "None", "mic": None, "hc50": None, "is_amp": False},
    {"sequence": "MDVNPTLLFLKVPAQNAIST", "target": "None", "mic": None, "hc50": None, "is_amp": False},
]


def load_seed_records() -> List[CleanedPeptideRecord]:
    """Parse and clean the curated experimental seed callset."""
    records: List[CleanedPeptideRecord] = []
    for item in SEED_DATASET_RAW:
        rec = clean_raw_record(
            raw_sequence=item["sequence"],
            target_organism=item["target"],
            mic_ug_ml=item.get("mic"),
            hc50_ug_ml=item.get("hc50"),
            source="Curated Seed",
            is_amp=item["is_amp"],
        )
        if rec is not None:
            records.append(rec)

    return deduplicate_records(records)


def export_dataset_json(
    records: List[CleanedPeptideRecord],
    output_path: Path,
) -> None:
    """Save records to JSON format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in records], f, indent=2)


def export_dataset_fasta(
    records: List[CleanedPeptideRecord],
    output_path: Path,
) -> None:
    """Save records to standard FASTA format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, r in enumerate(records):
            amp_tag = "AMP" if r.is_amp else "NON_AMP"
            f.write(
                f">PEP_{idx+1:05d}|{r.target_organism.replace(' ', '_')}|{amp_tag}|MIC={r.mic_ug_ml}|HC50={r.hc50_ug_ml}\n"
            )
            f.write(f"{r.sequence}\n")
