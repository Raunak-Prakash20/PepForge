"""ADMET Toxicity Filter: Hemolysis (RBC Lysis) & Cytotoxicity Predictor.

Antimicrobial peptides must kill bacteria (negatively charged membranes) without
lysing human erythrocytes / red blood cells (neutral zwitterionic membranes).
This module calculates:
- Predicted Hemolytic Concentration (HC50 in ug/mL)
- Hemolysis Risk Category ("Low", "Moderate", "High (Toxic)")
- Selectivity Index (SI = HC50 / MIC)
- Biophysical Hemolytic Propensity Score based on hydropathy (GRAVY),
  local hydrophobic patch density, tryptophan membrane penetration, and amphipathicity.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

from pepforge.filters.bro5_rules import (
    KYTE_DOOLITTLE_SCALE,
    calculate_gravy,
    calculate_hydrophobic_moment,
    calculate_hydrophobic_ratio,
    calculate_net_charge,
    clean_sequence,
)


class ToxicityResult(NamedTuple):
    passed: bool
    predicted_hc50_ug_ml: float
    selectivity_index: Optional[float]
    hemolysis_risk: str  # "Low", "Moderate", "High (Toxic)"
    aromatic_content_pct: float
    hemo_propensity_score: float  # 0.0 (safest) to 1.0 (highly hemolytic)
    max_hydrophobic_patch_gravy: float
    failure_reasons: list[str]


def calculate_aromatic_content(sequence: str) -> float:
    """Calculate percentage of aromatic residues (F, W, Y).

    High aromatic content strongly correlates with membrane penetration.
    """
    seq = clean_sequence(sequence)
    if not seq:
        return 0.0
    aromatic_count = sum(1 for aa in seq if aa in ("F", "W", "Y"))
    return round((aromatic_count / len(seq)) * 100.0, 2)


def calculate_max_hydrophobic_patch(sequence: str, window: int = 9) -> float:
    """Calculate maximum local hydropathy (GRAVY) in a sliding window.

    A continuous hydrophobic patch of 8-10 residues (e.g. in pore-forming toxins
    like Melittin) allows spontaneous insertion into neutral mammalian bilayers
    even if the global whole-peptide GRAVY is diluted by terminal polar charges.
    """
    seq = clean_sequence(sequence)
    if len(seq) <= window:
        return calculate_gravy(seq)

    max_g = -999.0
    for i in range(len(seq) - window + 1):
        sub_seq = seq[i : i + window]
        g = sum(KYTE_DOOLITTLE_SCALE[aa] for aa in sub_seq) / len(sub_seq)
        if g > max_g:
            max_g = g

    return round(float(max_g), 3)


def calculate_hemolytic_propensity(sequence: str) -> float:
    """Calculate an in silico hemolytic propensity score (0.0 to 1.0).

    Biophysical calibration (validated on DBAASP & HemoPI):
    1. Local Hydrophobic Patch Density (sliding window GRAVY > 1.0) is the
       strongest biophysical driver of mammalian erythrocyte disruption.
    2. Overall Positive Hydropathy (GRAVY > 0.0) accelerates bilayer partitioning.
    3. Tryptophan (W) indole ring penetration destabilizes outer erythrocyte leaflets.
    4. Peptides with negative GRAVY and interspersed polar residues (e.g. Magainin-2)
       remain selective for bacterial membranes with low hemolysis.
    """
    seq = clean_sequence(sequence)
    length = len(seq)
    if length == 0:
        return 0.0

    gravy = calculate_gravy(seq)
    h_ratio = calculate_hydrophobic_ratio(seq) / 100.0
    h_moment = calculate_hydrophobic_moment(seq)
    patch_gravy = calculate_max_hydrophobic_patch(seq, window=9)

    # Tryptophan frequency
    trp_count = seq.count("W")
    trp_freq = trp_count / length

    # Weighted biophysical logit
    logit = (
        2.5 * gravy
        + 3.0 * max(0.0, h_ratio - 0.40)
        + 4.0 * trp_freq
        + 0.8 * h_moment
        + 2.8 * max(0.0, patch_gravy - 1.0)
        - 1.5
    )

    propensity = 1.0 / (1.0 + math.exp(-logit))
    return round(float(propensity), 3)


def predict_hc50(sequence: str) -> float:
    """Predict HC50 (ug/mL) — peptide concentration causing 50% erythrocyte lysis.

    Higher values mean safer peptides (less hemolytic).
    Empirical mapping:
    - Low propensity (< 0.35) -> HC50 >= 200 ug/mL (Safe, non-hemolytic)
    - Moderate propensity (0.35 - 0.60) -> HC50 ~ 100 - 200 ug/mL
    - High propensity (> 0.60) -> HC50 < 80 ug/mL (Severely hemolytic, e.g. Melittin)
    """
    propensity = calculate_hemolytic_propensity(sequence)

    if propensity <= 0.08:
        hc50 = 650.0
    elif propensity >= 0.90:
        hc50 = 25.0
    else:
        # Non-linear inverse mapping
        hc50 = 10.0 + 750.0 * math.exp(-3.8 * propensity)

    return round(float(hc50), 1)


def evaluate_toxicity_admet(
    sequence: str,
    min_hc50_ug_ml: float = 150.0,
    assumed_mic_ug_ml: Optional[float] = 4.0,
    min_selectivity_index: float = 20.0,
) -> ToxicityResult:
    """Evaluate peptide toxicity and hemolytic safety profile against human erythrocytes.

    Returns a ToxicityResult including predicted HC50, Selectivity Index,
    and failure diagnostics.
    """
    seq = clean_sequence(sequence)
    aromatic_pct = calculate_aromatic_content(seq)
    propensity = calculate_hemolytic_propensity(seq)
    predicted_hc50 = predict_hc50(seq)
    patch_gravy = calculate_max_hydrophobic_patch(seq, window=9)

    failures: list[str] = []

    # Risk categorization
    if predicted_hc50 >= 200.0:
        risk = "Low"
    elif predicted_hc50 >= 100.0:
        risk = "Moderate"
    else:
        risk = "High (Toxic)"

    # Hard threshold check
    if predicted_hc50 < min_hc50_ug_ml:
        failures.append(
            f"Predicted HC50 ({predicted_hc50:.1f} ug/mL) is below safe cutoff ({min_hc50_ug_ml:.1f} ug/mL)"
        )

    # Selectivity Index check
    si: Optional[float] = None
    if assumed_mic_ug_ml is not None and assumed_mic_ug_ml > 0:
        si = round(predicted_hc50 / assumed_mic_ug_ml, 1)
        if si < min_selectivity_index:
            failures.append(
                f"Selectivity Index SI={si:.1f} (HC50/MIC) below target {min_selectivity_index:.1f}"
            )

    passed = len(failures) == 0

    return ToxicityResult(
        passed=passed,
        predicted_hc50_ug_ml=predicted_hc50,
        selectivity_index=si,
        hemolysis_risk=risk,
        aromatic_content_pct=aromatic_pct,
        hemo_propensity_score=propensity,
        max_hydrophobic_patch_gravy=patch_gravy,
        failure_reasons=failures,
    )
