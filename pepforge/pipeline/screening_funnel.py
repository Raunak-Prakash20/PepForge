"""Automated Screening Funnel Orchestrator for PepForge-AI.

Sequentially evaluates candidate peptides through the complete 5-stage filter stack:
- Stage 1: bRo5 Physicochemical Rules (Net Charge, Hydropathy, Boman Index, μH)
- Stage 2: ADMET Toxicity Filter (Hemolysis HC50 & Selectivity Index)
- Stage 3: ADMET Metabolism Filter (Cleavage Site Index & Capping Recommendations)
- Stage 4: 3D Structure & Folding (pLDDT, Helical Content %, Amphipathic Wheel)
- Stage 5: Novelty & Freedom-to-Operate (FTO Homology Clearance vs. DBAASP/APD3)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from pepforge.data.benchmarks import BenchmarkPeptide
from pepforge.filters.bro5_rules import evaluate_bro5_rules, BRo5Result
from pepforge.filters.metabolism_admet import evaluate_metabolism_admet, MetabolismResult
from pepforge.filters.novelty_check import evaluate_novelty, NoveltyResult
from pepforge.filters.structure_esmfold import evaluate_structure, StructureResult
from pepforge.filters.toxicity_admet import evaluate_toxicity_admet, ToxicityResult


class CandidateEvaluation(BaseModel):
    sequence: str
    target_organism: str
    overall_passed: bool
    stages_passed: Dict[str, bool]
    rejection_stage: Optional[str] = None
    all_failure_reasons: List[str] = Field(default_factory=list)
    bro5_metrics: Dict[str, Any]
    admet_toxicity_metrics: Dict[str, Any]
    admet_metabolism_metrics: Dict[str, Any]
    structure_metrics: Dict[str, Any]
    novelty_metrics: Dict[str, Any]
    synthesis_recommendations: List[str] = Field(default_factory=list)


class ScreeningFunnel:
    """Multi-stage screening pipeline for de novo peptide candidates."""

    def __init__(
        self,
        min_charge: float = 2.0,
        max_charge: float = 6.5,
        min_hydrophobic_ratio: float = 35.0,
        max_hydrophobic_ratio: float = 55.0,
        max_boman_index: float = 2.50,
        min_hydrophobic_moment: float = 0.35,
        min_hc50_ug_ml: float = 150.0,
        min_selectivity_index: float = 20.0,
        max_cleavage_site_index: float = 18.0,
        min_plddt: float = 70.0,
        min_helical_content_pct: float = 45.0,
        max_allowed_identity_pct: float = 75.0,
        min_required_levenshtein: int = 4,
        reference_database: Optional[List[BenchmarkPeptide]] = None,
    ) -> None:
        self.min_charge = min_charge
        self.max_charge = max_charge
        self.min_hydrophobic_ratio = min_hydrophobic_ratio
        self.max_hydrophobic_ratio = max_hydrophobic_ratio
        self.max_boman_index = max_boman_index
        self.min_hydrophobic_moment = min_hydrophobic_moment
        self.min_hc50_ug_ml = min_hc50_ug_ml
        self.min_selectivity_index = min_selectivity_index
        self.max_cleavage_site_index = max_cleavage_site_index
        self.min_plddt = min_plddt
        self.min_helical_content_pct = min_helical_content_pct
        self.max_allowed_identity_pct = max_allowed_identity_pct
        self.min_required_levenshtein = min_required_levenshtein
        self.reference_database = reference_database

    def evaluate_candidate(
        self,
        sequence: str,
        target_organism: str = "Pseudomonas aeruginosa",
        assumed_mic_ug_ml: float = 4.0,
    ) -> CandidateEvaluation:
        """Run a single candidate peptide through the full 5-stage screening funnel."""
        stages_passed: Dict[str, bool] = {}
        all_failures: List[str] = []
        rejection_stage: Optional[str] = None

        # --- Stage 1: bRo5 Physicochemical Rules ---
        bro5_res: BRo5Result = evaluate_bro5_rules(
            sequence=sequence,
            min_charge=self.min_charge,
            max_charge=self.max_charge,
            min_hydrophobic_ratio=self.min_hydrophobic_ratio,
            max_hydrophobic_ratio=self.max_hydrophobic_ratio,
            max_boman_index=self.max_boman_index,
            min_hydrophobic_moment=self.min_hydrophobic_moment,
        )
        stages_passed["stage_1_bro5"] = bro5_res.passed
        if not bro5_res.passed:
            all_failures.extend([f"[Stage 1] {r}" for r in bro5_res.failure_reasons])
            if rejection_stage is None:
                rejection_stage = "stage_1_bro5"

        # --- Stage 2: ADMET Toxicity Filter ---
        tox_res: ToxicityResult = evaluate_toxicity_admet(
            sequence=sequence,
            min_hc50_ug_ml=self.min_hc50_ug_ml,
            assumed_mic_ug_ml=assumed_mic_ug_ml,
            min_selectivity_index=self.min_selectivity_index,
        )
        stages_passed["stage_2_admet_toxicity"] = tox_res.passed
        if not tox_res.passed:
            all_failures.extend([f"[Stage 2] {r}" for r in tox_res.failure_reasons])
            if rejection_stage is None:
                rejection_stage = "stage_2_admet_toxicity"

        # --- Stage 3: ADMET Metabolism Filter ---
        met_res: MetabolismResult = evaluate_metabolism_admet(
            sequence=sequence,
            max_cleavage_site_index=self.max_cleavage_site_index,
        )
        stages_passed["stage_3_admet_metabolism"] = met_res.passed
        if not met_res.passed:
            all_failures.extend([f"[Stage 3] {r}" for r in met_res.failure_reasons])
            if rejection_stage is None:
                rejection_stage = "stage_3_admet_metabolism"

        # --- Stage 4: 3D Structure & Folding ---
        struct_res: StructureResult = evaluate_structure(
            sequence=sequence,
            min_plddt=self.min_plddt,
            min_helical_content_pct=self.min_helical_content_pct,
        )
        stages_passed["stage_4_structure"] = struct_res.passed
        if not struct_res.passed:
            all_failures.extend([f"[Stage 4] {r}" for r in struct_res.failure_reasons])
            if rejection_stage is None:
                rejection_stage = "stage_4_structure"

        # --- Stage 5: Novelty & Freedom-to-Operate Check ---
        nov_res: NoveltyResult = evaluate_novelty(
            candidate_sequence=sequence,
            reference_database=self.reference_database,
            max_allowed_identity_pct=self.max_allowed_identity_pct,
            min_required_levenshtein=self.min_required_levenshtein,
        )
        stages_passed["stage_5_novelty"] = nov_res.passed
        if not nov_res.passed:
            all_failures.extend([f"[Stage 5] {r}" for r in nov_res.failure_reasons])
            if rejection_stage is None:
                rejection_stage = "stage_5_novelty"

        overall_passed = (
            bro5_res.passed
            and tox_res.passed
            and met_res.passed
            and struct_res.passed
            and nov_res.passed
        )

        return CandidateEvaluation(
            sequence=sequence,
            target_organism=target_organism,
            overall_passed=overall_passed,
            stages_passed=stages_passed,
            rejection_stage=rejection_stage,
            all_failure_reasons=all_failures,
            bro5_metrics={
                "net_charge_ph74": bro5_res.net_charge,
                "hydrophobic_ratio_pct": bro5_res.hydrophobic_ratio,
                "boman_index_kcal_mol": bro5_res.boman_index,
                "hydrophobic_moment": bro5_res.hydrophobic_moment,
                "gravy": bro5_res.gravy,
                "isoelectric_point": bro5_res.isoelectric_point,
                "molecular_weight": bro5_res.molecular_weight,
                "length": bro5_res.length,
            },
            admet_toxicity_metrics={
                "predicted_hc50_ug_ml": tox_res.predicted_hc50_ug_ml,
                "selectivity_index": tox_res.selectivity_index,
                "hemolysis_risk": tox_res.hemolysis_risk,
                "aromatic_content_pct": tox_res.aromatic_content_pct,
                "hemo_propensity_score": tox_res.hemo_propensity_score,
                "max_hydrophobic_patch_gravy": tox_res.max_hydrophobic_patch_gravy,
            },
            admet_metabolism_metrics={
                "cleavage_site_index": met_res.cleavage_site_index,
                "total_cleavage_sites": met_res.total_cleavage_sites,
                "trypsin_sites": met_res.trypsin_sites_count,
                "chymotrypsin_sites": met_res.chymotrypsin_sites_count,
                "estimated_stability": met_res.estimated_stability,
            },
            structure_metrics={
                "predicted_plddt": struct_res.predicted_plddt,
                "helical_content_pct": struct_res.helical_content_pct,
                "beta_sheet_content_pct": struct_res.beta_sheet_content_pct,
                "is_amphipathic_helix": struct_res.is_amphipathic_helix,
                "segregation_score": struct_res.segregation_score,
            },
            novelty_metrics={
                "is_patentable_novel": nov_res.is_patentable_novel,
                "max_sequence_identity_pct": nov_res.max_sequence_identity_pct,
                "min_levenshtein_distance": nov_res.min_levenshtein_distance,
                "nearest_homolog_name": nov_res.nearest_homolog_name,
                "nearest_homolog_organism": nov_res.nearest_homolog_organism,
            },
            synthesis_recommendations=met_res.recommended_modifications,
        )

    def screen_batch(
        self,
        sequences: List[str],
        target_organism: str = "Pseudomonas aeruginosa",
        assumed_mic_ug_ml: float = 4.0,
    ) -> List[CandidateEvaluation]:
        """Screen a batch of candidate sequences and return evaluations."""
        return [
            self.evaluate_candidate(seq, target_organism, assumed_mic_ug_ml)
            for seq in sequences
        ]
