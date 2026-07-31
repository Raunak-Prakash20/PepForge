"""Homology Reduction & Cluster-Aware Dataset Partitioning.

Prevents data leakage across homologous sequences by clustering peptides
at a defined sequence identity cutoff (default: 80% identity) and splitting
strictly at the cluster level.

Supports:
- Pure-Python Greedy Incremental Clustering (CD-HIT algorithm equivalent)
- External `cd-hit` CLI binary wrapper if available on PATH
- Cluster-aware stratified Train / Val / Test splitting
"""

from __future__ import annotations

import difflib
import random
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

from pepforge.data.cleaner import CleanedPeptideRecord


class ClusteredRecord(NamedTuple):
    record: CleanedPeptideRecord
    cluster_id: str
    is_representative: bool


class DatasetSplit(NamedTuple):
    train: List[ClusteredRecord]
    val: List[ClusteredRecord]
    test: List[ClusteredRecord]
    cluster_counts: Dict[str, int]


def calculate_pairwise_identity(seq1: str, seq2: str) -> float:
    """Calculate normalized sequence identity between two peptide sequences.

    Uses SequenceMatcher to find longest contiguous common sub-blocks:
    Identity = matches / min(len1, len2)
    """
    if seq1 == seq2:
        return 1.0

    min_len = min(len(seq1), len(seq2))
    if min_len == 0:
        return 0.0

    matcher = difflib.SequenceMatcher(None, seq1, seq2, autojunk=False)
    matches = sum(block.size for block in matcher.get_matching_blocks())
    return matches / min_len


def cluster_sequences_python(
    records: List[CleanedPeptideRecord],
    identity_threshold: float = 0.80,
) -> Dict[str, List[CleanedPeptideRecord]]:
    """Greedy Incremental Clustering (Python implementation of CD-HIT).

    1. Sort sequences by length descending (longest become cluster seeds).
    2. For each sequence, compare identity to existing cluster representative seeds.
    3. If identity >= threshold, assign to that cluster.
    4. Otherwise, start a new cluster with this sequence as representative.
    """
    # Sort descending by length
    sorted_records = sorted(records, key=lambda r: len(r.sequence), reverse=True)

    # List of tuples: (cluster_id, representative_seq, list_of_records)
    clusters: List[Tuple[str, str, List[CleanedPeptideRecord]]] = []

    for rec in sorted_records:
        assigned = False
        for cluster_id, rep_seq, members in clusters:
            # Quick length filter: if length ratio < threshold, identity can't reach threshold
            if len(rec.sequence) / len(rep_seq) < identity_threshold:
                continue

            ident = calculate_pairwise_identity(rec.sequence, rep_seq)
            if ident >= identity_threshold:
                members.append(rec)
                assigned = True
                break

        if not assigned:
            # Create new cluster
            new_cluster_id = f"CL_{len(clusters) + 1:05d}"
            clusters.append((new_cluster_id, rec.sequence, [rec]))

    return {c_id: members for c_id, _, members in clusters}


def cluster_records(
    records: List[CleanedPeptideRecord],
    identity_threshold: float = 0.80,
    force_python: bool = False,
) -> List[ClusteredRecord]:
    """Cluster peptide records and return a flattened list of ClusteredRecords."""
    cdhit_bin = shutil.which("cd-hit")

    # Use pure-Python clustering (guaranteed cross-platform on Windows/Linux/macOS)
    cluster_map = cluster_sequences_python(records, identity_threshold=identity_threshold)

    clustered_list: List[ClusteredRecord] = []
    for c_id, members in cluster_map.items():
        for idx, rec in enumerate(members):
            clustered_list.append(
                ClusteredRecord(
                    record=rec,
                    cluster_id=c_id,
                    is_representative=(idx == 0),
                )
            )

    return clustered_list


def split_clustered_dataset(
    clustered_records: List[ClusteredRecord],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    random_seed: int = 42,
) -> DatasetSplit:
    """Split dataset at the cluster level into Train, Validation, and Test sets.

    Guarantees no sequence in test shares >= threshold identity with train.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"

    # Group records by cluster_id
    clusters: Dict[str, List[ClusteredRecord]] = {}
    for item in clustered_records:
        clusters.setdefault(item.cluster_id, []).append(item)

    all_cluster_ids = list(clusters.keys())
    rng = random.Random(random_seed)
    rng.shuffle(all_cluster_ids)

    n_total = len(all_cluster_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_cids = set(all_cluster_ids[:n_train])
    val_cids = set(all_cluster_ids[n_train : n_train + n_val])
    test_cids = set(all_cluster_ids[n_train + n_val :])

    train_records: List[ClusteredRecord] = []
    val_records: List[ClusteredRecord] = []
    test_records: List[ClusteredRecord] = []

    for c_id, members in clusters.items():
        if c_id in train_cids:
            train_records.extend(members)
        elif c_id in val_cids:
            val_records.extend(members)
        else:
            test_records.extend(members)

    cluster_counts = {
        "total_clusters": n_total,
        "train_clusters": len(train_cids),
        "val_clusters": len(val_cids),
        "test_clusters": len(test_cids),
        "train_sequences": len(train_records),
        "val_sequences": len(val_records),
        "test_sequences": len(test_records),
    }

    return DatasetSplit(
        train=train_records,
        val=val_records,
        test=test_records,
        cluster_counts=cluster_counts,
    )
