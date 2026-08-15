"""Unit tests for PepForge-AI Tokenizer, Generator Transformer, and Sampling Engine."""

import pytest
import torch

from pepforge.filters.bro5_rules import CANONICAL_AMINO_ACIDS
from pepforge.models.generator import PepForgeGenerator
from pepforge.models.tokenizer import PeptideTokenizer
from pepforge.models.trainer import PeptideDataset, train_epoch
from pepforge.data.cleaner import CleanedPeptideRecord


def test_tokenizer_vocab_and_encoding():
    """Verify tokenizer vocab construction and round-trip decoding."""
    tokenizer = PeptideTokenizer()
    assert tokenizer.vocab_size >= 35

    prefix = tokenizer.build_condition_prefix(
        target_organism="Pseudomonas aeruginosa",
        gram_type="negative",
        target_charge=3.5,
        target_length=16,
    )
    assert "[TARGET:P_AERUGINOSA]" in prefix
    assert "[GRAM:NEGATIVE]" in prefix
    assert "<sep>" in prefix

    # Encode and decode sequence
    original_seq = "GLFDIVKKVVGALGSL"
    token_ids = tokenizer.encode(original_seq, condition_prefix=prefix, add_bos=True, add_eos=True)
    decoded_seq = tokenizer.decode(token_ids, strip_special=True)
    assert decoded_seq == original_seq


def test_generator_forward_and_loss():
    """Verify transformer forward pass shapes and loss calculation."""
    tokenizer = PeptideTokenizer()
    model = PepForgeGenerator(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=32,
    )

    batch_size = 2
    seq_len = 16
    dummy_input = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len))
    dummy_targets = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len))

    logits, loss = model(dummy_input, targets=dummy_targets)
    assert logits.shape == (batch_size, seq_len, tokenizer.vocab_size)
    assert loss is not None
    assert loss.item() > 0.0


def test_generator_autoregressive_sampling():
    """Verify that sampling produces valid canonical peptide sequences."""
    tokenizer = PeptideTokenizer()
    model = PepForgeGenerator(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=40,
    )

    prefix = tokenizer.build_condition_prefix(
        target_organism="Pseudomonas aeruginosa",
        gram_type="negative",
        target_charge=3.0,
        target_length=15,
    )

    generated = model.generate_candidate(
        tokenizer=tokenizer,
        condition_prefix=prefix,
        max_generate_len=20,
        temperature=0.8,
    )

    assert isinstance(generated, str)
    assert len(generated) > 0
    # Every generated character must be a canonical amino acid
    for aa in generated:
        assert aa in CANONICAL_AMINO_ACIDS


def test_training_smoke_step():
    """Verify that one training step executes and computes gradients."""
    tokenizer = PeptideTokenizer()
    model = PepForgeGenerator(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=40,
    )

    records = [
        CleanedPeptideRecord(sequence="GIGKFLHSAKKFGKAFVGEIMNS", length=23, target_organism="Pseudomonas aeruginosa"),
        CleanedPeptideRecord(sequence="KWKLFKKIPKFLHLAKKF", length=18, target_organism="Acinetobacter baumannii"),
    ]

    dataset = PeptideDataset(records, tokenizer, max_seq_len=40)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    loss = train_epoch(model, loader, optimizer, torch.device("cpu"))
    assert loss > 0.0
