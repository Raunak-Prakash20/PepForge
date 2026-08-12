"""Conditioned Autoregressive Transformer Decoder for De Novo Peptide Generation.

Implements a causal decoder-only architecture:
- Learned Token & Positional Embeddings
- Pre-LayerNorm Causal Multi-Head Self-Attention Blocks
- GELU Feedforward Networks with Residual Skip Connections
- Nucleus (Top-p) & Temperature-Controlled Autoregressive Sampling
- Repetition Penalty to prevent monotonic token repeats
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from pepforge.models.tokenizer import PeptideTokenizer


class MultiHeadAttention(nn.Module):
    """Causal Multi-Head Attention with Pre-LayerNorm."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask (strictly lower-triangular)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device),
            diagonal=1,
        )
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        # External padding mask if provided
        if attn_mask is not None:
            # attn_mask shape: (batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(context)


class TransformerBlock(nn.Module):
    """Transformer Decoder Block with Pre-LayerNorm and GELU MLP."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-LN Self-Attention
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        # Pre-LN Feedforward
        x = x + self.mlp(self.ln2(x))
        return x


class PepForgeGenerator(nn.Module):
    """Conditioned Autoregressive Transformer Decoder for Peptide Generation."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        max_seq_len: int = 80,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize parameters with normal distribution scaled by depth."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len = input_ids.shape
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds max {self.max_seq_len}"

        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        h = self.token_embedding(input_ids) + self.pos_embedding(positions)
        h = self.drop(h)

        mask = None
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)

        for block in self.blocks:
            h = block(h, attn_mask=mask)

        h = self.ln_f(h)
        logits = self.lm_head(h)

        loss = None
        if targets is not None:
            # Shifted cross-entropy: predict targets using previous token logits
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=0,  # PAD token
            )

        return logits, loss

    @torch.no_grad()
    def generate_candidate(
        self,
        tokenizer: PeptideTokenizer,
        condition_prefix: List[str],
        min_generate_len: int = 10,
        max_generate_len: int = 30,
        temperature: float = 0.85,
        top_p: float = 0.90,
        repetition_penalty: float = 1.2,
        device: torch.device = torch.device("cpu"),
    ) -> str:
        """Autoregressively sample a novel peptide sequence given conditioning tokens."""
        self.eval()
        # Encode condition prefix
        prefix_ids = tokenizer.encode(
            sequence="",
            condition_prefix=condition_prefix,
            add_bos=True,
            add_eos=False,
        )
        curr_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)

        for _ in range(max_generate_len):
            # Forward pass
            logits, _ = self.forward(curr_ids)
            next_token_logits = logits[0, -1, :].clone()

            # Prevent premature EOS before reaching min_generate_len
            gen_len = curr_ids.shape[1] - len(prefix_ids)
            if gen_len < min_generate_len:
                next_token_logits[tokenizer.eos_id] = float("-inf")

            # Apply repetition penalty to recently generated tokens
            for past_id in set(curr_ids[0].tolist()):
                if past_id in tokenizer.token_to_id.values():
                    if next_token_logits[past_id] < 0:
                        next_token_logits[past_id] *= repetition_penalty
                    else:
                        next_token_logits[past_id] /= repetition_penalty

            # Mask out condition and pad tokens from being generated inside sequence
            for cond_tok in tokenizer.condition_tokens + [tokenizer.pad_id, tokenizer.bos_id, tokenizer.sep_id]:
                if isinstance(cond_tok, str):
                    tid = tokenizer.token_to_id[cond_tok]
                else:
                    tid = cond_tok
                next_token_logits[tid] = float("-inf")

            # Temperature scaling
            scaled_logits = next_token_logits / max(temperature, 1e-4)

            # Top-p (nucleus) filtering
            sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            scaled_logits[indices_to_remove] = float("-inf")

            # Sample next token
            probs = F.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if next_token.item() == tokenizer.eos_id:
                break

            curr_ids = torch.cat([curr_ids, next_token.unsqueeze(0)], dim=1)

        # Decode only the generated amino acid segment (excluding condition tokens)
        generated_seq = tokenizer.decode(curr_ids[0], strip_special=True)
        return generated_seq

    @torch.no_grad()
    def generate_batch(
        self,
        tokenizer: PeptideTokenizer,
        condition_prefix: List[str],
        num_samples: int = 10,
        min_generate_len: int = 10,
        max_generate_len: int = 30,
        temperature: float = 0.85,
        top_p: float = 0.90,
        repetition_penalty: float = 1.2,
        device: torch.device = torch.device("cpu"),
    ) -> List[str]:
        """Generate a batch of candidate sequences."""
        candidates: List[str] = []
        for _ in range(num_samples):
            seq = self.generate_candidate(
                tokenizer=tokenizer,
                condition_prefix=condition_prefix,
                min_generate_len=min_generate_len,
                max_generate_len=max_generate_len,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                device=device,
            )
            if len(seq) >= 10:  # Minimum valid length
                candidates.append(seq)
        return candidates
