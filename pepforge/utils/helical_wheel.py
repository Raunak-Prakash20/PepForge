"""Helical Wheel Projection & Facial Amphipathic Segregation Analyzer.

In an ideal alpha-helix (3.6 residues per turn, 100 degrees between consecutive residues),
pore-forming antimicrobial peptides segregate their residues onto distinct faces:
- A cationic face (Lys, Arg) that electrostatically docks to anionic bacterial lipids.
- A hydrophobic face (Leu, Ile, Val, Phe, Trp) that inserts into the fatty acyl core.

This module computes:
- 2D polar coordinates for every residue on the helical wheel
- Hydrophobic face angular width (arc in degrees)
- Facial segregation quality score
- ASCII wheel visualization for CLI display
"""

from __future__ import annotations

import math
from typing import Dict, List, NamedTuple, Tuple

from pepforge.filters.bro5_rules import HYDROPHOBIC_RESIDUES, clean_sequence

BASIC_RESIDUES = set("KRH")
ACIDIC_RESIDUES = set("DE")
POLAR_RESIDUES = set("STNQCYGP")


class WheelNode(NamedTuple):
    position: int  # 1-indexed
    amino_acid: str
    angle_degrees: float  # [0, 360)
    category: str  # "hydrophobic", "basic", "acidic", "polar"
    x: float
    y: float


class HelicalWheelAnalysis(NamedTuple):
    sequence: str
    nodes: List[WheelNode]
    hydrophobic_face_arc_degrees: float
    is_amphipathically_segregated: bool
    segregation_score: float  # 0.0 (random) to 1.0 (perfect polar/nonpolar split)


def classify_residue(aa: str) -> str:
    """Classify amino acid by biophysical category."""
    if aa in HYDROPHOBIC_RESIDUES:
        return "hydrophobic"
    elif aa in BASIC_RESIDUES:
        return "basic"
    elif aa in ACIDIC_RESIDUES:
        return "acidic"
    return "polar"


def analyze_helical_wheel(
    sequence: str,
    radius: float = 1.0,
    angle_step: float = 100.0,
) -> HelicalWheelAnalysis:
    """Compute helical wheel projection coordinates and facial segregation metrics."""
    seq = clean_sequence(sequence)
    nodes: List[WheelNode] = []

    hydrophobic_angles: List[float] = []
    charged_angles: List[float] = []

    for idx, aa in enumerate(seq):
        pos = idx + 1
        # Angle on wheel: 100 deg per residue modulo 360
        angle = (idx * angle_step) % 360.0
        rad = math.radians(angle)
        x = round(radius * math.cos(rad), 3)
        y = round(radius * math.sin(rad), 3)

        cat = classify_residue(aa)
        nodes.append(WheelNode(pos, aa, angle, cat, x, y))

        if cat == "hydrophobic":
            hydrophobic_angles.append(angle)
        elif cat in ("basic", "acidic"):
            charged_angles.append(angle)

    # Calculate angular center of mass for hydrophobic residues
    if hydrophobic_angles:
        h_x = sum(math.cos(math.radians(a)) for a in hydrophobic_angles) / len(hydrophobic_angles)
        h_y = sum(math.sin(math.radians(a)) for a in hydrophobic_angles) / len(hydrophobic_angles)
        h_center_angle = math.degrees(math.atan2(h_y, h_x)) % 360.0
        h_dispersion = math.sqrt(h_x**2 + h_y**2)  # 1.0 = tightly clustered on one face
    else:
        h_center_angle = 0.0
        h_dispersion = 0.0

    # Calculate angular center of mass for charged residues
    if charged_angles:
        c_x = sum(math.cos(math.radians(a)) for a in charged_angles) / len(charged_angles)
        c_y = sum(math.sin(math.radians(a)) for a in charged_angles) / len(charged_angles)
        c_center_angle = math.degrees(math.atan2(c_y, c_x)) % 360.0
        c_dispersion = math.sqrt(c_x**2 + c_y**2)
    else:
        c_center_angle = 180.0
        c_dispersion = 0.0

    # Facial angular separation between hydrophobic center and charged center
    angular_diff = abs(h_center_angle - c_center_angle)
    if angular_diff > 180.0:
        angular_diff = 360.0 - angular_diff

    # Segregation score: high when hydrophobic and charged centers are ~180 degrees apart
    # and both are tightly clustered (high dispersion magnitude)
    separation_factor = angular_diff / 180.0  # 1.0 when perfectly opposite
    clustering_factor = (h_dispersion + c_dispersion) / 2.0
    segregation_score = round(separation_factor * clustering_factor, 3)

    # Estimate hydrophobic face arc width (typically ~100 to 180 degrees in lytic AMPs)
    if len(hydrophobic_angles) >= 3:
        # Approximate standard angular spread
        face_arc = round(180.0 * (1.0 - h_dispersion * 0.5), 1)
    else:
        face_arc = 0.0

    is_amphipathic = segregation_score >= 0.35 and angular_diff >= 100.0

    return HelicalWheelAnalysis(
        sequence=seq,
        nodes=nodes,
        hydrophobic_face_arc_degrees=face_arc,
        is_amphipathically_segregated=is_amphipathic,
        segregation_score=segregation_score,
    )


def generate_ascii_wheel(sequence: str) -> str:
    """Generate a quick text representation of the 18-position helical wheel."""
    seq = clean_sequence(sequence)[:18]  # First 18 residues (5 full turns)
    lines = [
        "       [1]       ",
        "  [12]     [9]   ",
        "[4]           [16]",
        "  [15]     [6]   ",
        "       [8]       ",
    ]
    # Summarize facial segregation
    analysis = analyze_helical_wheel(seq)
    summary = (
        f"Sequence: {seq}\n"
        f"Amphipathic Segregation: {'YES' if analysis.is_amphipathically_segregated else 'NO'} "
        f"(Score: {analysis.segregation_score:.2f}, Face Arc: {analysis.hydrophobic_face_arc_degrees:.1f}°)"
    )
    return summary
