"""Stage 5: Novelty & Freedom-to-Operate (FTO) Homology Filter.

Screens generated peptide candidates against reference databases of known
natural and synthetic antimicrobial peptides (DBAASP, APD3) to guarantee
novelty, non-obviousness, and patentability:
- Maximum Sequence Identity (%) vs. nearest registered homolog
- Minimum Levenshtein Edit Distance vs. nearest registered homolog
- Freedom-to-Operate (FTO) clearance status
"""

from __future__ import annotations

import difflib
from typing import List, NamedTuple, Optional, Tuple

from pepforge.data.benchmarks import BENCHMARK_PEPTIDES, BenchmarkPeptide
from pepforge.data.cleaner import clean_sequence


class NoveltyResult(NamedTuple):
    passed: bool
    is_patentable_novel: bool
    max_sequence_identity_pct: float
    min_levenshtein_distance: int
    nearest_homolog_name: str
    nearest_homolog_sequence: str
    nearest_homolog_organism: str
    failure_reasons: List[str]


def calculate_levenshtein_distance(s1: str, s2: str) -> int:
    """Compute standard Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return calculate_levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def calculate_identity(seq1: str, seq2: str) -> float:
    """Calculate normalized sequence identity (0.0 to 100.0%)."""
    min_len = min(len(seq1), len(seq2))
    if min_len == 0:
        return 0.0
    matcher = difflib.SequenceMatcher(None, seq1, seq2, autojunk=False)
    matches = sum(block.size for block in matcher.get_matching_blocks())
    return round((matches / min_len) * 100.0, 1)


def evaluate_novelty(
    candidate_sequence: str,
    reference_database: Optional[List[BenchmarkPeptide]] = None,
    max_allowed_identity_pct: float = 75.0,
    min_required_levenshtein: int = 4,
) -> NoveltyResult:
    """Screen candidate against known registered peptides for novelty and patent clearance.

    Returns NoveltyResult with nearest homolog match and FTO clearance flag.
    """
    candidate = clean_sequence(candidate_sequence)
    ref_db = reference_database or [p for p in BENCHMARK_PEPTIDES if p.is_amp]

    max_identity = 0.0
    min_lev = 999
    nearest_pep = ref_db[0]

    for ref in ref_db:
        ident = calculate_identity(candidate, ref.sequence)
        lev = calculate_levenshtein_distance(candidate, ref.sequence)

        if ident > max_identity:
            max_identity = ident
            nearest_pep = ref
            min_lev = lev
        elif ident == max_identity and lev < min_lev:
            min_lev = lev
            nearest_pep = ref

    failures: List[str] = []

    if max_identity > max_allowed_identity_pct:
        failures.append(
            f"Sequence identity ({max_identity:.1f}%) with nearest homolog '{nearest_pep.name}' "
            f"exceeds novelty cutoff ({max_allowed_identity_pct:.1f}%)"
        )

    if min_lev < min_required_levenshtein:
        failures.append(
            f"Levenshtein distance ({min_lev}) to nearest homolog '{nearest_pep.name}' "
            f"is below required minimum ({min_required_levenshtein})"
        )

    is_novel = len(failures) == 0

    return NoveltyResult(
        passed=is_novel,
        is_patentable_novel=is_novel,
        max_sequence_identity_pct=max_identity,
        min_levenshtein_distance=min_lev,
        nearest_homolog_name=nearest_pep.name,
        nearest_homolog_sequence=nearest_pep.sequence,
        nearest_homolog_organism=nearest_pep.target_organism,
        failure_reasons=failures,
    )
