"""Unit tests for Stage 4 (3D Structure & Helical Wheel) and Stage 5 (Novelty / FTO)."""

import pytest

from pepforge.data.benchmarks import get_benchmark_by_name
from pepforge.filters.novelty_check import (
    calculate_identity,
    calculate_levenshtein_distance,
    evaluate_novelty,
)
from pepforge.filters.structure_esmfold import (
    estimate_secondary_structure,
    evaluate_structure,
    predict_plddt_confidence,
)
from pepforge.utils.helical_wheel import analyze_helical_wheel, generate_ascii_wheel


def test_helical_wheel_facial_segregation():
    """Verify facial segregation on canonical amphipathic alpha-helices."""
    # Syn-P18 is engineered with a strict amphipathic alpha-helical split
    syn_p18 = get_benchmark_by_name("Syn-P18")
    assert syn_p18 is not None

    wheel = analyze_helical_wheel(syn_p18.sequence)
    assert wheel.is_amphipathically_segregated
    assert wheel.segregation_score >= 0.35
    assert len(wheel.nodes) == len(syn_p18.sequence)

    # ASCII representation smoke check
    ascii_rep = generate_ascii_wheel(syn_p18.sequence)
    assert "Amphipathic Segregation: YES" in ascii_rep


def test_structure_chou_fasman_and_plddt():
    """Verify secondary structure estimation and pLDDT scoring."""
    # Poly-alanine/leucine is strongly alpha-helical
    helical_seq = "AAAKAAALAAAKAAAL"
    helix_pct, sheet_pct, coil_pct = estimate_secondary_structure(helical_seq)
    assert helix_pct >= 50.0

    # Multiple prolines break alpha-helices and lower pLDDT
    kinky_seq = "AAPPAAPPAAPPPP"
    k_helix, _, _ = estimate_secondary_structure(kinky_seq)
    plddt_helical = predict_plddt_confidence(helical_seq, helix_pct, 0.7)
    plddt_kinky = predict_plddt_confidence(kinky_seq, k_helix, 0.2)
    assert plddt_helical > plddt_kinky


def test_levenshtein_distance():
    """Verify string edit distance computation."""
    assert calculate_levenshtein_distance("ABCD", "ABCD") == 0
    assert calculate_levenshtein_distance("ABCD", "ABCE") == 1
    assert calculate_levenshtein_distance("ABCD", "EFGH") == 4
    assert calculate_levenshtein_distance("ABC", "ABCD") == 1


def test_novelty_catches_existing_peptides():
    """Verify Stage 5 rejects exact or near-identical copies of known registered AMPs."""
    magainin = get_benchmark_by_name("Magainin-2")
    assert magainin is not None

    # Exact duplicate must be rejected with 100% identity
    res_exact = evaluate_novelty(magainin.sequence)
    assert not res_exact.passed
    assert res_exact.max_sequence_identity_pct == 100.0
    assert res_exact.min_levenshtein_distance == 0
    assert res_exact.nearest_homolog_name == "Magainin-2"

    # Novel de novo designed sequence should pass FTO clearance
    novel_seq = "KLAKLAKKWLAKLAK"
    res_novel = evaluate_novelty(novel_seq)
    assert res_novel.max_sequence_identity_pct < 75.0
    assert res_novel.min_levenshtein_distance >= 4
    assert res_novel.passed
