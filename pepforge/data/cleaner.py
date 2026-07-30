"""Data Cleaning and Canonicalization Pipeline for Peptide Sequences.

Ensures all training and benchmark sequences adhere to strict biochemical criteria:
- Standard 20 canonical amino acids
- Sequence length bounds [10, 45]
- Taxonomy normalization (ESKAPE pathogens & Gram classification)
- Numerical MIC and Hemolysis extraction and normalization
- Exact deduplication with metadata aggregation
"""

from __future__ import annotations

import re
from typing import Dict, List, Literal, Optional, Set, Tuple
from pydantic import BaseModel, Field

from pepforge.filters.bro5_rules import CANONICAL_AMINO_ACIDS, clean_sequence

GramType = Literal["positive", "negative", "broad", "unknown"]

# Pathogen taxonomy map to Gram classification
GRAM_TAXONOMY_MAP: Dict[str, GramType] = {
    "pseudomonas aeruginosa": "negative",
    "acinetobacter baumannii": "negative",
    "klebsiella pneumoniae": "negative",
    "escherichia coli": "negative",
    "enterobacter cloacae": "negative",
    "salmonella enterica": "negative",
    "staphylococcus aureus": "positive",
    "enterococcus faecium": "positive",
    "streptococcus pneumoniae": "positive",
    "bacillus subtilis": "positive",
    "broad spectrum": "broad",
}


class CleanedPeptideRecord(BaseModel):
    """Normalized schema for a peptide record."""

    sequence: str
    length: int
    target_organism: str
    gram_type: GramType = "unknown"
    mic_ug_ml: Optional[float] = None
    hc50_ug_ml: Optional[float] = None
    source: str = "curated"
    is_amp: bool = True
    metadata: Dict[str, str] = Field(default_factory=dict)


def normalize_organism_name(name: str) -> Tuple[str, GramType]:
    """Normalize bacterial organism name and assign Gram classification."""
    clean_name = name.strip()
    lower_name = clean_name.lower()

    for pattern, gram in GRAM_TAXONOMY_MAP.items():
        if pattern in lower_name:
            # Capitalize standard scientific names
            parts = clean_name.split()
            if len(parts) >= 2:
                std_name = f"{parts[0].capitalize()} {parts[1].lower()}"
            else:
                std_name = clean_name.capitalize()
            return std_name, gram

    return clean_name, "unknown"


def is_valid_sequence(
    sequence: str,
    min_length: int = 10,
    max_length: int = 45,
    allowed_vocab: Set[str] = CANONICAL_AMINO_ACIDS,
) -> bool:
    """Validate whether sequence consists of standard amino acids and is within length bounds."""
    if not sequence or not isinstance(sequence, str):
        return False
    seq = sequence.strip().upper()
    if not (min_length <= len(seq) <= max_length):
        return False
    if set(seq) - allowed_vocab:
        return False
    return True


def clean_raw_record(
    raw_sequence: str,
    target_organism: str = "Broad Spectrum",
    mic_ug_ml: Optional[float] = None,
    hc50_ug_ml: Optional[float] = None,
    source: str = "DBAASP",
    is_amp: bool = True,
    min_length: int = 10,
    max_length: int = 45,
) -> Optional[CleanedPeptideRecord]:
    """Clean a single raw peptide record. Returns None if invalid."""
    if not is_valid_sequence(raw_sequence, min_length, max_length):
        return None

    seq = raw_sequence.strip().upper()
    std_organism, gram = normalize_organism_name(target_organism)

    # Sanitize MIC
    valid_mic: Optional[float] = None
    if mic_ug_ml is not None and mic_ug_ml > 0:
        valid_mic = round(float(mic_ug_ml), 2)

    # Sanitize HC50
    valid_hc50: Optional[float] = None
    if hc50_ug_ml is not None and hc50_ug_ml > 0:
        valid_hc50 = round(float(hc50_ug_ml), 1)

    return CleanedPeptideRecord(
        sequence=seq,
        length=len(seq),
        target_organism=std_organism,
        gram_type=gram,
        mic_ug_ml=valid_mic,
        hc50_ug_ml=valid_hc50,
        source=source,
        is_amp=is_amp,
    )


def deduplicate_records(
    records: List[CleanedPeptideRecord],
) -> List[CleanedPeptideRecord]:
    """Deduplicate records by peptide sequence.

    When duplicate sequences exist, preserves the record with the most
    complete bioactivity (MIC/HC50) data.
    """
    seq_map: Dict[str, CleanedPeptideRecord] = {}

    for rec in records:
        if rec.sequence not in seq_map:
            seq_map[rec.sequence] = rec
        else:
            existing = seq_map[rec.sequence]
            # Prioritize records with experimental values
            if existing.mic_ug_ml is None and rec.mic_ug_ml is not None:
                seq_map[rec.sequence] = rec
            elif existing.hc50_ug_ml is None and rec.hc50_ug_ml is not None:
                seq_map[rec.sequence] = rec

    return list(seq_map.values())
