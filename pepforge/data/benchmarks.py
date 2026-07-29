"""Curated Benchmark Peptides for Validation, Regression Testing, and Calibration.

Contains validated experimental reference peptides:
- Standard non-hemolytic AMPs (Magainin-2, Cecropin A, LL-37)
- Severely hemolytic cytotoxic peptide (Melittin from bee venom - positive toxicity control)
- Aromatic-rich peptide (Indolicidin)
- Negative non-AMP control sequences (Cytoplasmic hydrophobic fragments & poly-acidic peptides)
"""

from typing import Dict, List, NamedTuple, Optional


class BenchmarkPeptide(NamedTuple):
    name: str
    sequence: str
    target_organism: str
    is_amp: bool
    is_hemolytic: bool
    experimental_mic_ug_ml: Optional[float]
    experimental_hc50_ug_ml: Optional[float]
    source_notes: str


BENCHMARK_PEPTIDES: List[BenchmarkPeptide] = [
    BenchmarkPeptide(
        name="Magainin-2",
        sequence="GIGKFLHSAKKFGKAFVGEIMNS",
        target_organism="Pseudomonas aeruginosa",
        is_amp=True,
        is_hemolytic=False,
        experimental_mic_ug_ml=4.0,
        experimental_hc50_ug_ml=300.0,
        source_notes="Xenopus laevis skin. Classic amphipathic alpha-helical AMP with low hemolysis.",
    ),
    BenchmarkPeptide(
        name="Melittin",
        sequence="GIGAVLKVLTTGLPALISWIKRKRQQ",
        target_organism="Broad Spectrum",
        is_amp=True,
        is_hemolytic=True,  # Severe hemolysis
        experimental_mic_ug_ml=2.0,
        experimental_hc50_ug_ml=8.0,
        source_notes="Apis mellifera venom. High antimicrobial activity but lethal mammalian hemolysis.",
    ),
    BenchmarkPeptide(
        name="Cecropin A",
        sequence="KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",
        target_organism="Acinetobacter baumannii",
        is_amp=True,
        is_hemolytic=False,
        experimental_mic_ug_ml=2.5,
        experimental_hc50_ug_ml=500.0,
        source_notes="Hyalophora cecropia. Highly selective against Gram-negative pathogens.",
    ),
    BenchmarkPeptide(
        name="LL-37",
        sequence="LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
        target_organism="Pseudomonas aeruginosa",
        is_amp=True,
        is_hemolytic=False,
        experimental_mic_ug_ml=6.0,
        experimental_hc50_ug_ml=220.0,
        source_notes="Human cathelicidin. Primary innate defense against opportunistic pathogens.",
    ),
    BenchmarkPeptide(
        name="Indolicidin",
        sequence="ILPWKWPWWPWRR",
        target_organism="Staphylococcus aureus",
        is_amp=True,
        is_hemolytic=True,
        experimental_mic_ug_ml=4.0,
        experimental_hc50_ug_ml=45.0,
        source_notes="Bovine neutrophil peptide. Extremely high tryptophan content (38.5%).",
    ),
    BenchmarkPeptide(
        name="Syn-P18",
        sequence="KWKLFKKIPKFLHLAKKF",
        target_organism="Acinetobacter baumannii",
        is_amp=True,
        is_hemolytic=False,
        experimental_mic_ug_ml=2.0,
        experimental_hc50_ug_ml=280.0,
        source_notes="Engineered Cecropin-Magainin hybrid with optimized net charge and amphipathicity.",
    ),
    BenchmarkPeptide(
        name="Neg-Hydrophobic-Transmembrane",
        sequence="MSIQHFRVALIPFFAAFCLP",
        target_organism="None",
        is_amp=False,
        is_hemolytic=False,
        experimental_mic_ug_ml=None,
        experimental_hc50_ug_ml=None,
        source_notes="Hydrophobic non-AMP intracellular fragment. Lacks cationic charge for bacterial attraction.",
    ),
    BenchmarkPeptide(
        name="Neg-PolyAcidic",
        sequence="DEDDEDEEEEEDDDDE",
        target_organism="None",
        is_amp=False,
        is_hemolytic=False,
        experimental_mic_ug_ml=None,
        experimental_hc50_ug_ml=None,
        source_notes="Poly-anionic sequence. Net charge is -16, electrostatically repelled by bacterial membranes.",
    ),
]


def get_benchmark_by_name(name: str) -> Optional[BenchmarkPeptide]:
    """Retrieve a benchmark peptide by its common name."""
    for p in BENCHMARK_PEPTIDES:
        if p.name.lower() == name.lower():
            return p
    return None
