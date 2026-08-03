"""Beyond-Rule-of-5 (bRo5) Physicochemical Screening Rules for Peptides.

Standard small-molecule rules (Lipinski's Rule of 5) fail for peptides because
therapeutic peptides are naturally beyond the Rule of 5 (MW > 1000 Da, H-bond donors > 5).
This module implements rigorous peptide-specific biophysical rules:
- Net Charge at physiological pH (Henderson-Hasselbalch)
- Isoelectric Point (pI)
- Hydrophobic Ratio (H_ratio)
- Boman Index (protein-binding potential)
- Eisenberg Hydrophobic Moment (amphipathicity for alpha-helices)
- Grand Average of Hydropathy (GRAVY via Kyte-Doolittle)
- Approximate Molecular Weight (MW)
"""

from __future__ import annotations

import math
from typing import Any, Dict, NamedTuple, Tuple

# Standard 20 canonical amino acids
CANONICAL_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

# Hydrophobic residue set
HYDROPHOBIC_RESIDUES = set("AVILFWM")

# pKa values (standard IPC / EMBOSS values)
PKA_N_TERM = 9.69
PKA_C_TERM = 2.34
PKA_SIDECHAINS: Dict[str, float] = {
    "K": 10.53,  # Lysine (basic)
    "R": 12.48,  # Arginine (basic)
    "H": 6.00,   # Histidine (basic)
    "D": 3.86,   # Aspartic acid (acidic)
    "E": 4.25,   # Glutamic acid (acidic)
    "C": 8.33,   # Cysteine (acidic)
    "Y": 10.07,  # Tyrosine (acidic)
}

# Monoisotopic molecular weights of free amino acids (g/mol)
AA_MONOISOTOPIC_MASS: Dict[str, float] = {
    "A": 71.03711, "R": 156.10111, "N": 114.04293, "D": 115.02694,
    "C": 103.00919, "E": 129.04259, "Q": 128.05858, "G": 57.02146,
    "H": 137.05891, "I": 113.08406, "L": 113.08406, "K": 128.09496,
    "M": 131.04049, "F": 147.06841, "P": 97.05276, "S": 87.03203,
    "T": 101.04768, "W": 186.07931, "Y": 163.06333, "V": 99.06841,
}
WATER_MASS = 18.01528

# Boman index scale (kcal/mol) - Boman (2003)
# Estimates protein-binding potential: sum of free energies of transfer of AAs from cyclohexane to water
BOMAN_SCALE: Dict[str, float] = {
    "D": 0.61, "E": 0.51, "N": 0.06, "Q": 0.73, "K": 0.46,
    "R": 1.58, "H": 0.17, "S": -0.13, "T": -0.18, "G": 0.00,
    "A": 0.11, "V": -0.31, "L": -0.55, "I": -0.60, "M": -0.10,
    "F": -0.32, "Y": 0.08, "W": 0.30, "P": 0.45, "C": -0.13,
}

# Eisenberg consensus hydrophobicity scale - Eisenberg et al. (1984)
# Used to calculate the hydrophobic moment (amphipathicity)
EISENBERG_SCALE: Dict[str, float] = {
    "I": 0.73, "F": 0.61, "V": 0.54, "L": 0.53, "W": 0.37,
    "M": 0.26, "A": 0.25, "G": 0.16, "C": 0.04, "Y": 0.02,
    "P": -0.07, "T": -0.18, "S": -0.26, "H": -0.40, "E": -0.62,
    "N": -0.64, "Q": -0.69, "D": -0.72, "K": -1.10, "R": -1.76,
}

# Kyte-Doolittle hydropathy scale - Kyte & Doolittle (1982)
# Used to calculate GRAVY (Grand Average of Hydropathy)
KYTE_DOOLITTLE_SCALE: Dict[str, float] = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}


class BRo5Result(NamedTuple):
    passed: bool
    net_charge: float
    hydrophobic_ratio: float
    boman_index: float
    hydrophobic_moment: float
    gravy: float
    isoelectric_point: float
    molecular_weight: float
    length: int
    failure_reasons: list[str]


def clean_sequence(sequence: str) -> str:
    """Normalize and validate a peptide sequence."""
    seq = sequence.strip().upper()
    invalid_chars = set(seq) - CANONICAL_AMINO_ACIDS
    if invalid_chars:
        raise ValueError(
            f"Invalid non-canonical amino acids found in sequence: {invalid_chars}"
        )
    return seq


def calculate_net_charge(sequence: str, ph: float = 7.4) -> float:
    """Calculate net electrostatic charge of a peptide at a specific pH.

    Uses the Henderson-Hasselbalch equation with standard sidechain pKa values.
    """
    seq = clean_sequence(sequence)

    # N-terminal positive charge
    charge = 1.0 / (1.0 + 10.0 ** (ph - PKA_N_TERM))

    # C-terminal negative charge
    charge -= 1.0 / (1.0 + 10.0 ** (PKA_C_TERM - ph))

    # Side chains
    for aa in seq:
        if aa in ("K", "R", "H"):
            # Basic residues: protonated (+1) at low pH
            charge += 1.0 / (1.0 + 10.0 ** (ph - PKA_SIDECHAINS[aa]))
        elif aa in ("D", "E", "C", "Y"):
            # Acidic residues: deprotonated (-1) at high pH
            charge -= 1.0 / (1.0 + 10.0 ** (PKA_SIDECHAINS[aa] - ph))

    return round(charge, 3)


def calculate_isoelectric_point(sequence: str, tolerance: float = 0.01) -> float:
    """Calculate the isoelectric point (pI) where net charge is 0."""
    seq = clean_sequence(sequence)
    low_ph = 0.0
    high_ph = 14.0

    while (high_ph - low_ph) > tolerance:
        mid_ph = (low_ph + high_ph) / 2.0
        charge = calculate_net_charge(seq, ph=mid_ph)
        if charge > 0:
            low_ph = mid_ph
        else:
            high_ph = mid_ph

    return round((low_ph + high_ph) / 2.0, 2)


def calculate_hydrophobic_ratio(sequence: str) -> float:
    """Calculate the percentage of hydrophobic residues (A, V, I, L, F, W, M)."""
    seq = clean_sequence(sequence)
    if not seq:
        return 0.0
    hydrophobic_count = sum(1 for aa in seq if aa in HYDROPHOBIC_RESIDUES)
    return round((hydrophobic_count / len(seq)) * 100.0, 2)


def calculate_boman_index(sequence: str) -> float:
    """Calculate the Boman Index (protein-binding potential in kcal/mol).

    Values > 2.48 indicate high promiscuous protein-binding (higher off-target risk).
    Values < 2.0 indicate low non-specific protein interaction.
    """
    seq = clean_sequence(sequence)
    if not seq:
        return 0.0
    total_energy = sum(BOMAN_SCALE[aa] for aa in seq)
    return round(total_energy / len(seq), 3)


def calculate_hydrophobic_moment(
    sequence: str, angle_degrees: float = 100.0
) -> float:
    """Calculate the Eisenberg Hydrophobic Moment (mu_H) for an alpha-helix.

    An ideal alpha-helix has 3.6 residues per turn, corresponding to 100 degrees
    separation per residue. An amphipathic helix exhibits distinct segregation of
    hydrophobic and hydrophilic residues on opposite helical faces, yielding a high mu_H.
    """
    seq = clean_sequence(sequence)
    length = len(seq)
    if length == 0:
        return 0.0

    angle_rad = math.radians(angle_degrees)
    sum_cos = 0.0
    sum_sin = 0.0

    for idx, aa in enumerate(seq):
        h_val = EISENBERG_SCALE[aa]
        theta = (idx + 1) * angle_rad
        sum_cos += h_val * math.cos(theta)
        sum_sin += h_val * math.sin(theta)

    moment = math.sqrt(sum_cos**2 + sum_sin**2) / length
    return round(moment, 3)


def calculate_gravy(sequence: str) -> float:
    """Calculate the Grand Average of Hydropathy (GRAVY) via Kyte-Doolittle.

    Positive values indicate hydrophobic peptides; negative values indicate hydrophilic.
    """
    seq = clean_sequence(sequence)
    if not seq:
        return 0.0
    total_hydropathy = sum(KYTE_DOOLITTLE_SCALE[aa] for aa in seq)
    return round(total_hydropathy / len(seq), 3)


def calculate_molecular_weight(sequence: str) -> float:
    """Calculate monoisotopic molecular weight of a free linear peptide (g/mol)."""
    seq = clean_sequence(sequence)
    if not seq:
        return 0.0
    total_residues_mass = sum(AA_MONOISOTOPIC_MASS[aa] for aa in seq)
    # Total mass = sum of residue masses + H2O (N-terminal H + C-terminal OH)
    total_mw = total_residues_mass + WATER_MASS
    return round(total_mw, 2)


def evaluate_bro5_rules(
    sequence: str,
    min_length: int = 10,
    max_length: int = 40,
    min_charge: float = 2.0,
    max_charge: float = 6.5,
    min_hydrophobic_ratio: float = 35.0,
    max_hydrophobic_ratio: float = 55.0,
    max_boman_index: float = 2.50,
    min_hydrophobic_moment: float = 0.35,
    ph: float = 7.4,
) -> BRo5Result:
    """Evaluate a peptide sequence against Stage 1 Beyond-Rule-of-5 criteria.

    Returns a BRo5Result namedtuple containing boolean pass/fail status,
    calculated biophysical metrics, and a list of specific failure reasons if any.
    """
    seq = clean_sequence(sequence)
    length = len(seq)
    failures: list[str] = []

    # 1. Length check
    if not (min_length <= length <= max_length):
        failures.append(
            f"Length {length} out of bounds [{min_length}, {max_length}]"
        )

    # 2. Net charge check
    charge = calculate_net_charge(seq, ph=ph)
    if not (min_charge <= charge <= max_charge):
        failures.append(
            f"Net charge {charge:+.2f} at pH {ph} outside target [{min_charge:+.1f}, {max_charge:+.1f}]"
        )

    # 3. Hydrophobic ratio
    h_ratio = calculate_hydrophobic_ratio(seq)
    if not (min_hydrophobic_ratio <= h_ratio <= max_hydrophobic_ratio):
        failures.append(
            f"Hydrophobic ratio {h_ratio:.1f}% outside window [{min_hydrophobic_ratio}%, {max_hydrophobic_ratio}%]"
        )

    # 4. Boman Index
    boman = calculate_boman_index(seq)
    if boman > max_boman_index:
        failures.append(
            f"Boman index {boman:.2f} kcal/mol exceeds max threshold {max_boman_index:.2f}"
        )

    # 5. Hydrophobic Moment
    h_moment = calculate_hydrophobic_moment(seq)
    if h_moment < min_hydrophobic_moment:
        failures.append(
            f"Hydrophobic moment {h_moment:.3f} below minimum amphipathicity {min_hydrophobic_moment:.3f}"
        )

    gravy = calculate_gravy(seq)
    pi = calculate_isoelectric_point(seq)
    mw = calculate_molecular_weight(seq)

    passed = len(failures) == 0

    return BRo5Result(
        passed=passed,
        net_charge=charge,
        hydrophobic_ratio=h_ratio,
        boman_index=boman,
        hydrophobic_moment=h_moment,
        gravy=gravy,
        isoelectric_point=pi,
        molecular_weight=mw,
        length=length,
        failure_reasons=failures,
    )
