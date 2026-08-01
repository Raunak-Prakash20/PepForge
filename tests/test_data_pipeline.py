"""Unit tests for PepForge-AI Data Cleaning, Ingestion, and CD-HIT Clustering."""

import pytest

from pepforge.data.cleaner import (
    CleanedPeptideRecord,
    clean_raw_record,
    deduplicate_records,
    is_valid_sequence,
    normalize_organism_name,
)
from pepforge.data.clustering import (
    calculate_pairwise_identity,
    cluster_records,
    split_clustered_dataset,
)
from pepforge.data.downloader import load_seed_records


def test_is_valid_sequence():
    """Verify validation of canonical amino acids and length limits."""
    # Valid standard 16-mer
    assert is_valid_sequence("GLFDIVKKVVGALGSL")

    # Invalid non-canonical amino acid 'X'
    assert not is_valid_sequence("GLFDIVXKVVGALGSL")

    # Too short (< 10 AAs)
    assert not is_valid_sequence("GLFDI")

    # Too long (> 45 AAs)
    assert not is_valid_sequence("A" * 50)


def test_normalize_organism_name():
    """Verify bacterial name normalization and Gram typing."""
    std_name, gram = normalize_organism_name("pseudomonas aeruginosa")
    assert std_name == "Pseudomonas aeruginosa"
    assert gram == "negative"

    std_name, gram = normalize_organism_name("Staphylococcus aureus (MRSA)")
    assert "Staphylococcus aureus" in std_name
    assert gram == "positive"


def test_deduplication():
    """Verify that duplicate sequences are resolved and bioactivity retained."""
    rec1 = CleanedPeptideRecord(
        sequence="GLFDIVKKVVGALGSL",
        length=16,
        target_organism="Pseudomonas aeruginosa",
        mic_ug_ml=None,
    )
    rec2 = CleanedPeptideRecord(
        sequence="GLFDIVKKVVGALGSL",
        length=16,
        target_organism="Pseudomonas aeruginosa",
        mic_ug_ml=4.0,
    )

    deduped = deduplicate_records([rec1, rec2])
    assert len(deduped) == 1
    assert deduped[0].mic_ug_ml == 4.0


def test_pairwise_identity():
    """Verify sequence identity calculation."""
    # Identical
    assert calculate_pairwise_identity("ACDEFGHIKL", "ACDEFGHIKL") == 1.0

    # 1 mismatch out of 10
    ident = calculate_pairwise_identity("ACDEFGHIKA", "ACDEFGHIKL")
    assert ident >= 0.90


def test_homology_clustering_groups_homologs():
    """Verify homologous sequences group into the same cluster at 80% threshold."""
    records = [
        CleanedPeptideRecord(
            sequence="GIGKFLHSAKKFGKAFVGEIMNS",  # Magainin-2 (23 AAs)
            length=23,
            target_organism="Pseudomonas aeruginosa",
        ),
        CleanedPeptideRecord(
            sequence="GIGKFLHSAKKFGKAFVGEIMN",  # 22 AAs (identical prefix)
            length=22,
            target_organism="Pseudomonas aeruginosa",
        ),
        CleanedPeptideRecord(
            sequence="RRWWRWWRR",  # Short Trp-Arg peptide (completely distinct)
            length=9,
            target_organism="Staphylococcus aureus",
        ),
    ]

    clustered = cluster_records(records, identity_threshold=0.80)
    assert len(clustered) == 3

    # Magainin homologs must share the same cluster_id
    mag1_cid = next(c.cluster_id for c in clustered if len(c.record.sequence) == 23)
    mag2_cid = next(c.cluster_id for c in clustered if len(c.record.sequence) == 22)
    rrw_cid = next(c.cluster_id for c in clustered if c.record.sequence == "RRWWRWWRR")

    assert mag1_cid == mag2_cid
    assert mag1_cid != rrw_cid


def test_cluster_split_has_zero_leakage():
    """Verify cluster-aware split guarantees zero cluster overlap across partitions."""
    seed_records = load_seed_records()
    clustered = cluster_records(seed_records, identity_threshold=0.80)

    split = split_clustered_dataset(
        clustered, train_ratio=0.80, val_ratio=0.10, test_ratio=0.10
    )

    train_cids = {item.cluster_id for item in split.train}
    val_cids = {item.cluster_id for item in split.val}
    test_cids = {item.cluster_id for item in split.test}

    # Strict mutual exclusivity of clusters
    assert len(train_cids.intersection(val_cids)) == 0
    assert len(train_cids.intersection(test_cids)) == 0
    assert len(val_cids.intersection(test_cids)) == 0
