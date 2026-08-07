"""Stage 4: 3D Structure & Amphipathic Folding Filter.

Validates in silico 3D folding confidence and secondary structure:
- Predicted Local Distance Difference Test (pLDDT, 0 to 100 scale)
- Secondary structure prediction (Alpha-helical content %, Beta-sheet %, Coil %)
- Helical Wheel facial segregation verification
"""

from __future__ import annotations

import math
from typing import Dict, List, NamedTuple, Optional

from pepforge.filters.bro5_rules import clean_sequence
from pepforge.utils.helical_wheel import analyze_helical_wheel

# Chou-Fasman alpha-helix conformational parameters (Pa values)
CHOU_FASMAN_PA: Dict[str, float] = {
    "E": 1.51, "M": 1.45, "A": 1.42, "L": 1.21, "K": 1.16,
    "F": 1.13, "Q": 1.11, "W": 1.08, "I": 1.08, "V": 1.06,
    "D": 1.01, "H": 1.00, "R": 0.98, "T": 0.83, "S": 0.77,
    "C": 0.70, "Y": 0.69, "N": 0.67, "P": 0.57, "G": 0.57,
}

# Chou-Fasman beta-sheet conformational parameters (Pb values)
CHOU_FASMAN_PB: Dict[str, float] = {
    "M": 1.67, "V": 1.70, "I": 1.60, "C": 1.19, "Y": 1.47,
    "F": 1.38, "Q": 1.10, "L": 1.30, "T": 1.19, "W": 1.37,
    "A": 0.83, "R": 0.93, "G": 0.75, "D": 0.54, "K": 0.74,
    "S": 0.75, "H": 0.87, "N": 0.89, "P": 0.55, "E": 0.37,
}


class StructureResult(NamedTuple):
    passed: bool
    predicted_plddt: float  # [0, 100]
    helical_content_pct: float
    beta_sheet_content_pct: float
    random_coil_content_pct: float
    is_amphipathic_helix: bool
    segregation_score: float
    failure_reasons: List[str]


def estimate_secondary_structure(sequence: str) -> Tuple[float, float, float]:
    """Estimate alpha-helical, beta-sheet, and random coil content percentages.

    Uses Chou-Fasman conformational potential scoring with windowed smoothing.
    """
    seq = clean_sequence(sequence)
    length = len(seq)
    if length == 0:
        return 0.0, 0.0, 100.0

    helix_scores = [CHOU_FASMAN_PA[aa] for aa in seq]
    sheet_scores = [CHOU_FASMAN_PB[aa] for aa in seq]

    # Sliding window of 4 residues for alpha-helix nucleation
    helix_count = 0
    sheet_count = 0

    window = 4
    for i in range(len(seq)):
        w_start = max(0, i - 2)
        w_end = min(length, i + 3)
        avg_helix = sum(helix_scores[w_start:w_end]) / (w_end - w_start)
        avg_sheet = sum(sheet_scores[w_start:w_end]) / (w_end - w_start)

        # Proline is a strong helix breaker unless at N-terminus
        is_proline = (seq[i] == "P" and i > 2)

        if not is_proline and avg_helix > 1.05 and avg_helix > avg_sheet:
            helix_count += 1
        elif avg_sheet > 1.05 and avg_sheet >= avg_helix:
            sheet_count += 1

    helix_pct = round((helix_count / length) * 100.0, 1)
    sheet_pct = round((sheet_count / length) * 100.0, 1)
    coil_pct = round(max(0.0, 100.0 - helix_pct - sheet_pct), 1)

    return helix_pct, sheet_pct, coil_pct


def predict_plddt_confidence(
    sequence: str,
    helical_content_pct: float,
    segregation_score: float,
) -> float:
    """Predict in silico pLDDT structural confidence score [0, 100].

    In ESMFold and AlphaFold, short peptides typically have higher pLDDT when:
    - They have strong secondary structure propensity (e.g. well-defined alpha-helix).
    - They have amphipathic segregation that stabilizes the helical fold through hydrophobic packing.
    - They do not contain multiple destabilizing proline/glycine kinks in the middle.
    """
    seq = clean_sequence(sequence)
    length = len(seq)

    # Base pLDDT for peptides
    base_plddt = 55.0

    # Bonus for helical propensity
    helix_bonus = 0.25 * helical_content_pct

    # Bonus for amphipathic facial segregation
    amphipathic_bonus = 15.0 * segregation_score

    # Penalty for proline/glycine kinks in the middle (positions 3 to L-3)
    middle_segment = seq[2:-2] if length > 4 else ""
    breaker_count = middle_segment.count("P") + 0.5 * middle_segment.count("G")
    kink_penalty = 6.0 * breaker_count

    score = base_plddt + helix_bonus + amphipathic_bonus - kink_penalty
    # Bounded in [30.0, 95.0]
    plddt = max(30.0, min(95.0, score))
    return round(plddt, 1)


def evaluate_structure(
    sequence: str,
    min_plddt: float = 70.0,
    min_helical_content_pct: float = 45.0,
) -> StructureResult:
    """Evaluate candidate peptide against Stage 4 3D Structural Criteria."""
    seq = clean_sequence(sequence)
    helix_pct, sheet_pct, coil_pct = estimate_secondary_structure(seq)

    wheel_analysis = analyze_helical_wheel(seq)
    plddt = predict_plddt_confidence(seq, helix_pct, wheel_analysis.segregation_score)

    failures: List[str] = []

    if plddt < min_plddt:
        failures.append(
            f"Predicted pLDDT ({plddt:.1f}) is below structural confidence cutoff ({min_plddt:.1f})"
        )

    if helix_pct < min_helical_content_pct:
        failures.append(
            f"Helical content ({helix_pct:.1f}%) is below minimum threshold ({min_helical_content_pct:.1f}%)"
        )

    if not wheel_analysis.is_amphipathically_segregated:
        failures.append(
            f"Lacks clear amphipathic facial segregation on helical axis (score: {wheel_analysis.segregation_score:.2f})"
        )

    passed = len(failures) == 0

    return StructureResult(
        passed=passed,
        predicted_plddt=plddt,
        helical_content_pct=helix_pct,
        beta_sheet_content_pct=sheet_pct,
        random_coil_content_pct=coil_pct,
        is_amphipathic_helix=wheel_analysis.is_amphipathically_segregated,
        segregation_score=wheel_analysis.segregation_score,
        failure_reasons=failures,
    )
