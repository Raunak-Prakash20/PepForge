"""Unit tests for PepForge-AI Physicochemical (bRo5) & ADMET Filters."""

import pytest

from pepforge.data.benchmarks import BENCHMARK_PEPTIDES, get_benchmark_by_name
from pepforge.filters.bro5_rules import (
    calculate_boman_index,
    calculate_gravy,
    calculate_hydrophobic_moment,
    calculate_hydrophobic_ratio,
    calculate_isoelectric_point,
    calculate_molecular_weight,
    calculate_net_charge,
    evaluate_bro5_rules,
)
from pepforge.filters.metabolism_admet import (
    evaluate_metabolism_admet,
    identify_cleavage_sites,
)
from pepforge.filters.toxicity_admet import (
    calculate_aromatic_content,
    evaluate_toxicity_admet,
    predict_hc50,
)
from pepforge.pipeline.screening_funnel import ScreeningFunnel


def test_net_charge_and_pi():
    """Verify Henderson-Hasselbalch charge calculations."""
    # Poly-lysine must be highly basic (approx +3.99 at pH 7.4 accounting for C-term -1)
    charge_k4 = calculate_net_charge("KKKK", ph=7.4)
    assert charge_k4 >= 3.9

    # Poly-aspartate must be highly acidic
    charge_d4 = calculate_net_charge("DDDD", ph=7.4)
    assert charge_d4 <= -3.9

    # Neutral peptide pI near neutral
    pi_g = calculate_isoelectric_point("GGGG")
    assert 5.0 <= pi_g <= 6.5


def test_hydrophobic_ratio():
    """Verify percentage calculation for hydrophobic residues (AVILFWM)."""
    assert calculate_hydrophobic_ratio("AAAA") == 100.0
    assert calculate_hydrophobic_ratio("KKKK") == 0.0
    assert calculate_hydrophobic_ratio("ALKR") == 50.0


def test_hydrophobic_moment():
    """Verify Eisenberg hydrophobic moment for amphipathic alpha-helices."""
    # Magainin-2 is a canonical amphipathic helix
    magainin = get_benchmark_by_name("Magainin-2")
    assert magainin is not None
    moment = calculate_hydrophobic_moment(magainin.sequence)
    assert moment >= 0.25


def test_trypsin_proline_rule():
    """Verify that Trypsin does NOT cleave when followed by Proline (P)."""
    # K followed by A should cleave
    sites_cleavable = identify_cleavage_sites("AKAA")
    trypsin_cleavable = [s for s in sites_cleavable if s.enzyme == "Trypsin"]
    assert len(trypsin_cleavable) == 1

    # K followed by P should NOT cleave
    sites_blocked = identify_cleavage_sites("AKPA")
    trypsin_blocked = [s for s in sites_blocked if s.enzyme == "Trypsin"]
    assert len(trypsin_blocked) == 0


def test_toxicity_melittin_vs_magainin():
    """Verify that Melittin (bee venom) is flagged as high hemolysis risk compared to Magainin."""
    melittin = get_benchmark_by_name("Melittin")
    magainin = get_benchmark_by_name("Magainin-2")
    assert melittin is not None and magainin is not None

    hc50_melittin = predict_hc50(melittin.sequence)
    hc50_magainin = predict_hc50(magainin.sequence)

    # Melittin should have significantly lower HC50 (more toxic) than Magainin
    assert hc50_melittin < hc50_magainin
    assert hc50_melittin < 100.0  # Known severe hemolytic peptide
    assert hc50_magainin > 200.0  # Magainin is non-hemolytic at therapeutic concentration


def test_screening_funnel_rejects_negative_controls():
    """Verify the screening funnel properly rejects non-AMP negative controls."""
    funnel = ScreeningFunnel()

    # Poly-acidic peptide should fail Stage 1 due to negative charge
    neg_acidic = get_benchmark_by_name("Neg-PolyAcidic")
    assert neg_acidic is not None
    res_acidic = funnel.evaluate_candidate(neg_acidic.sequence)
    assert not res_acidic.overall_passed
    assert not res_acidic.stages_passed["stage_1_bro5"]

    # Hydrophobic transmembrane should fail due to lack of charge and high hydropathy
    neg_tm = get_benchmark_by_name("Neg-Hydrophobic-Transmembrane")
    assert neg_tm is not None
    res_tm = funnel.evaluate_candidate(neg_tm.sequence)
    assert not res_tm.overall_passed


def test_screening_funnel_accepts_optimized_amp():
    """Verify that an optimized AMP (Syn-P18) passes the screening criteria."""
    funnel = ScreeningFunnel()
    syn_p18 = get_benchmark_by_name("Syn-P18")
    assert syn_p18 is not None

    res = funnel.evaluate_candidate(syn_p18.sequence, target_organism=syn_p18.target_organism)
    assert res.bro5_metrics["net_charge_ph74"] >= 2.0
    assert res.admet_toxicity_metrics["hemolysis_risk"] in ("Low", "Moderate")
    assert len(res.synthesis_recommendations) > 0
