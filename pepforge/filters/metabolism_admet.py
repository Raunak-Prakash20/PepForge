"""ADMET Metabolism Filter: Proteolytic Cleavage & Metabolic Stability Mapper.

Natural linear peptides have short in vivo half-lives due to rapid enzymatic
degradation by serum and tissue endopeptidases (Trypsin, Chymotrypsin, Elastase, Pepsin).
This module calculates:
- Protease cleavage site locations and frequency
- Cleavage Site Index (CSI)
- In silico stability category & estimated relative half-life
- Automated recommendations for chemical terminal capping (N-acetylation, C-amidation)
"""

from __future__ import annotations

import re
from typing import Dict, List, NamedTuple, Tuple

from pepforge.filters.bro5_rules import clean_sequence


# Protease cleavage regex rules (Cleaves after group 1)
# Standard rules based on Expasy PeptideCutter
PROTEASE_RULES: Dict[str, str] = {
    # Trypsin: cleaves C-term to K or R, unless followed by P
    "trypsin": r"([KR])(?=[^P]|$)",
    # Chymotrypsin (high specificity): cleaves C-term to F, Y, or W, unless followed by P
    "chymotrypsin_high": r"([FYW])(?=[^P]|$)",
    # Chymotrypsin (low specificity): also cleaves after L, M
    "chymotrypsin_low": r"([FYWLM])(?=[^P]|$)",
    # Pepsin (pH > 2): cleaves between hydrophobic/aromatic residues
    "pepsin": r"([FLWY])(?=[FLWY])",
    # Thermolysin: cleaves N-term to A, F, I, L, M, V (cleaves before group 1)
    "thermolysin": r"(?=[AFILMV])",
}

# N-end rule stability classification (mammalian reticulocyte system)
# Categorized by N-terminal amino acid stability:
STABILIZING_N_TERMINAL = {"G", "A", "S", "T", "V", "P", "M"}
DESTABILIZING_N_TERMINAL = {"R", "K", "L", "F", "Y", "W", "D", "E"}


class CleavageSite(NamedTuple):
    enzyme: str
    position: int  # 1-indexed residue position after which cleavage occurs
    motif: str


class MetabolismResult(NamedTuple):
    passed: bool
    cleavage_site_index: float  # Cleavage sites per 100 amino acids
    total_cleavage_sites: int
    cleavage_sites: List[CleavageSite]
    trypsin_sites_count: int
    chymotrypsin_sites_count: int
    estimated_stability: str  # "High", "Moderate", "Low"
    recommended_modifications: List[str]
    failure_reasons: List[str]


def identify_cleavage_sites(sequence: str) -> List[CleavageSite]:
    """Identify vulnerable endopeptidase cleavage sites along the peptide."""
    seq = clean_sequence(sequence)
    sites: List[CleavageSite] = []

    # 1. Trypsin
    for match in re.finditer(PROTEASE_RULES["trypsin"], seq):
        pos = match.start() + 1
        motif_start = max(0, match.start() - 1)
        motif_end = min(len(seq), match.start() + 2)
        motif = seq[motif_start:motif_end]
        sites.append(CleavageSite(enzyme="Trypsin", position=pos, motif=motif))

    # 2. Chymotrypsin (high specificity)
    for match in re.finditer(PROTEASE_RULES["chymotrypsin_high"], seq):
        pos = match.start() + 1
        motif_start = max(0, match.start() - 1)
        motif_end = min(len(seq), match.start() + 2)
        motif = seq[motif_start:motif_end]
        sites.append(CleavageSite(enzyme="Chymotrypsin", position=pos, motif=motif))

    # 3. Pepsin
    for match in re.finditer(PROTEASE_RULES["pepsin"], seq):
        pos = match.start() + 1
        motif_start = max(0, match.start() - 1)
        motif_end = min(len(seq), match.start() + 2)
        motif = seq[motif_start:motif_end]
        sites.append(CleavageSite(enzyme="Pepsin", position=pos, motif=motif))

    # Sort by position
    sites.sort(key=lambda s: s.position)
    return sites


def calculate_cleavage_site_index(sequence: str, total_sites: int) -> float:
    """Calculate the Cleavage Site Index (CSI).

    CSI represents the normalized frequency of vulnerable cleavage sites per 100 residues:
    CSI = (total_sites / sequence_length) * 100
    """
    seq = clean_sequence(sequence)
    length = len(seq)
    if length == 0:
        return 0.0
    return round((total_sites / length) * 100.0, 2)


def evaluate_metabolism_admet(
    sequence: str,
    max_cleavage_site_index: float = 18.0,
    max_trypsin_sites: int = 3,
) -> MetabolismResult:
    """Evaluate metabolic stability and protease resistance of the peptide.

    Checks endopeptidase cleavage vulnerability, generates terminal capping
    protection strategies, and flags rapid clearance risks.
    """
    seq = clean_sequence(sequence)
    sites = identify_cleavage_sites(seq)
    total_sites = len(sites)
    csi = calculate_cleavage_site_index(seq, total_sites)

    trypsin_count = sum(1 for s in sites if s.enzyme == "Trypsin")
    chymotrypsin_count = sum(1 for s in sites if s.enzyme == "Chymotrypsin")

    failures: List[str] = []
    modifications: List[str] = []

    # Threshold checks
    if csi > max_cleavage_site_index:
        failures.append(
            f"Cleavage Site Index {csi:.1f} exceeds threshold {max_cleavage_site_index:.1f}"
        )

    if trypsin_count > max_trypsin_sites:
        failures.append(
            f"Trypsin cleavage sites ({trypsin_count}) exceed allowed limit ({max_trypsin_sites})"
        )

    # N-terminal stability assessment
    n_term_aa = seq[0]
    if n_term_aa in DESTABILIZING_N_TERMINAL:
        modifications.append(
            f"N-terminal Acetylation (Ac-): Protects destabilizing '{n_term_aa}' from aminopeptidases."
        )
    else:
        modifications.append(
            "N-terminal Acetylation (Ac-): Recommended for general exopeptidase blockade."
        )

    # C-terminal amidation assessment
    modifications.append(
        "C-terminal Amidation (-NH2): Recommended to eliminate C-terminal negative charge and block carboxypeptidases."
    )

    if csi > 20.0:
        modifications.append(
            "Head-to-tail cyclization or D-amino acid substitution recommended to disrupt endopeptidase recognition."
        )

    # Stability categorization
    if csi <= 10.0 and trypsin_count <= 2:
        stability = "High"
    elif csi <= 18.0:
        stability = "Moderate"
    else:
        stability = "Low (Rapid Clearance)"

    passed = len(failures) == 0

    return MetabolismResult(
        passed=passed,
        cleavage_site_index=csi,
        total_cleavage_sites=total_sites,
        cleavage_sites=sites,
        trypsin_sites_count=trypsin_count,
        chymotrypsin_sites_count=chymotrypsin_count,
        estimated_stability=stability,
        recommended_modifications=modifications,
        failure_reasons=failures,
    )
