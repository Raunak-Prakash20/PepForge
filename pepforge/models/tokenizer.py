"""Conditioned Peptide Sequence Tokenizer for PepForge-AI.

Supports:
- Canonical 20 amino acid vocabulary
- Functional prefix conditioning tokens (Target pathogen, Gram status, Charge range, Length range)
- PyTorch tensor encoding and string decoding with padding and truncation
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import torch

from pepforge.filters.bro5_rules import CANONICAL_AMINO_ACIDS

# Special tokens
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SEP_TOKEN = "<sep>"

# Conditioning tokens
TARGET_TOKENS = [
    "[TARGET:P_AERUGINOSA]",
    "[TARGET:A_BAUMANNII]",
    "[TARGET:S_AUREUS]",
    "[TARGET:K_PNEUMONIAE]",
    "[TARGET:BROAD_SPECTRUM]",
]

GRAM_TOKENS = [
    "[GRAM:POSITIVE]",
    "[GRAM:NEGATIVE]",
    "[GRAM:BROAD]",
]

CHARGE_TOKENS = [
    "[CHARGE:LOW]",   # Charge +1.0 to +2.5
    "[CHARGE:MID]",   # Charge +2.5 to +5.0
    "[CHARGE:HIGH]",  # Charge > +5.0
]

LENGTH_TOKENS = [
    "[LEN:SHORT]",  # 10 to 16 AAs
    "[LEN:MID]",    # 17 to 25 AAs
    "[LEN:LONG]",   # 26 to 45 AAs
]


class PeptideTokenizer:
    """Tokenizer mapping amino acids and conditioning tokens to discrete integer IDs."""

    def __init__(self) -> None:
        self.special_tokens = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, SEP_TOKEN]
        self.condition_tokens = TARGET_TOKENS + GRAM_TOKENS + CHARGE_TOKENS + LENGTH_TOKENS
        self.amino_acid_tokens = sorted(list(CANONICAL_AMINO_ACIDS))

        self.vocab: List[str] = (
            self.special_tokens + self.condition_tokens + self.amino_acid_tokens
        )
        self.token_to_id: Dict[str, int] = {tok: idx for idx, tok in enumerate(self.vocab)}
        self.id_to_token: Dict[int, str] = {idx: tok for idx, tok in enumerate(self.vocab)}

        # Cached special token IDs
        self.pad_id = self.token_to_id[PAD_TOKEN]
        self.bos_id = self.token_to_id[BOS_TOKEN]
        self.eos_id = self.token_to_id[EOS_TOKEN]
        self.unk_id = self.token_to_id[UNK_TOKEN]
        self.sep_id = self.token_to_id[SEP_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def build_condition_prefix(
        self,
        target_organism: str = "Pseudomonas aeruginosa",
        gram_type: str = "negative",
        target_charge: float = 3.5,
        target_length: int = 18,
    ) -> List[str]:
        """Construct the prompt conditioning token prefix."""
        # 1. Target organism
        lower_t = target_organism.lower()
        if "pseudomonas" in lower_t:
            target_tok = "[TARGET:P_AERUGINOSA]"
        elif "acinetobacter" in lower_t:
            target_tok = "[TARGET:A_BAUMANNII]"
        elif "staphylococcus" in lower_t:
            target_tok = "[TARGET:S_AUREUS]"
        elif "klebsiella" in lower_t:
            target_tok = "[TARGET:K_PNEUMONIAE]"
        else:
            target_tok = "[TARGET:BROAD_SPECTRUM]"

        # 2. Gram status
        if gram_type.lower() == "positive":
            gram_tok = "[GRAM:POSITIVE]"
        elif gram_type.lower() == "negative":
            gram_tok = "[GRAM:NEGATIVE]"
        else:
            gram_tok = "[GRAM:BROAD]"

        # 3. Charge
        if target_charge < 2.5:
            charge_tok = "[CHARGE:LOW]"
        elif target_charge <= 5.0:
            charge_tok = "[CHARGE:MID]"
        else:
            charge_tok = "[CHARGE:HIGH]"

        # 4. Length
        if target_length <= 16:
            len_tok = "[LEN:SHORT]"
        elif target_length <= 25:
            len_tok = "[LEN:MID]"
        else:
            len_tok = "[LEN:LONG]"

        return [target_tok, gram_tok, charge_tok, len_tok, SEP_TOKEN]

    def encode(
        self,
        sequence: str,
        condition_prefix: Optional[List[str]] = None,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        """Encode a peptide sequence into token IDs."""
        tokens: List[str] = []
        if add_bos:
            tokens.append(BOS_TOKEN)

        if condition_prefix:
            tokens.extend(condition_prefix)

        # Amino acid tokens
        for char in sequence.strip().upper():
            if char in CANONICAL_AMINO_ACIDS:
                tokens.append(char)
            else:
                tokens.append(UNK_TOKEN)

        if add_eos:
            tokens.append(EOS_TOKEN)

        return [self.token_to_id.get(tok, self.unk_id) for tok in tokens]

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        strip_special: bool = True,
    ) -> str:
        """Decode token IDs back into an amino acid sequence string."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        res_chars: List[str] = []
        for tid in token_ids:
            tok = self.id_to_token.get(tid, UNK_TOKEN)
            if strip_special:
                if tok in self.special_tokens or tok in self.condition_tokens:
                    continue
            res_chars.append(tok)

        return "".join(res_chars)

    def batch_encode(
        self,
        sequences: List[str],
        conditions: Optional[List[List[str]]] = None,
        max_length: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode batch of sequences to padded PyTorch tensor and attention mask."""
        batch_ids: List[List[int]] = []
        for idx, seq in enumerate(sequences):
            cond = conditions[idx] if conditions else None
            ids = self.encode(seq, condition_prefix=cond, add_bos=True, add_eos=True)
            batch_ids.append(ids)

        actual_max = min(max_length, max(len(ids) for ids in batch_ids))
        padded_ids = []
        attention_masks = []

        for ids in batch_ids:
            if len(ids) > actual_max:
                truncated = ids[:actual_max]
                mask = [1] * actual_max
            else:
                pad_len = actual_max - len(ids)
                truncated = ids + [self.pad_id] * pad_len
                mask = [1] * len(ids) + [0] * pad_len

            padded_ids.append(truncated)
            attention_masks.append(mask)

        return torch.tensor(padded_ids, dtype=torch.long), torch.tensor(
            attention_masks, dtype=torch.long
        )
