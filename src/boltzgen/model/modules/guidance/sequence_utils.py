"""Sequence conversion utilities for guidance module.

Provides functions to convert between:
- BoltzGen res_type logits/indices
- Amino acid strings (1-letter codes)
- Straight-through estimator for gradient flow
"""

from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from boltzgen.data import const


# Build mapping from BoltzGen token indices to 1-letter AA codes
# canonical_tokens are at indices [canonicals_offset, canonicals_offset + 20)
_TOKEN_IDX_TO_ONE_LETTER = {}
for i, three_letter in enumerate(const.canonical_tokens):
    token_idx = i + const.canonicals_offset
    one_letter = const.prot_token_to_letter.get(three_letter, "X")
    _TOKEN_IDX_TO_ONE_LETTER[token_idx] = one_letter

# Reverse mapping: 1-letter to token index
_ONE_LETTER_TO_TOKEN_IDX = {v: k for k, v in _TOKEN_IDX_TO_ONE_LETTER.items()}

# Number of canonical amino acids
NUM_CANONICAL_AAS = len(const.canonical_tokens)  # 20


def get_canonical_mask(num_tokens: int = const.num_tokens) -> Tensor:
    """Get a boolean mask for canonical amino acid positions in token space.
    
    Args:
        num_tokens: Total number of tokens in vocabulary
        
    Returns:
        Boolean tensor [num_tokens] where True = canonical AA
    """
    mask = torch.zeros(num_tokens, dtype=torch.bool)
    start = const.canonicals_offset
    end = start + NUM_CANONICAL_AAS
    mask[start:end] = True
    return mask


def logits_to_aa_indices(
    logits: Tensor,
    mask: Optional[Tensor] = None,
    canonical_only: bool = True,
) -> Tensor:
    """Convert res_type logits to amino acid token indices.
    
    Args:
        logits: Tensor of shape [B, L, num_tokens] or [L, num_tokens]
        mask: Optional boolean mask [B, L] or [L] for valid positions
        canonical_only: If True, only consider canonical AAs (indices 2-21)
        
    Returns:
        Tensor of token indices [B, L] or [L]
    """
    if canonical_only:
        # Only look at canonical AA logits
        start = const.canonicals_offset
        end = start + NUM_CANONICAL_AAS
        canonical_logits = logits[..., start:end]
        # Argmax within canonicals, then shift back to full token space
        indices = canonical_logits.argmax(dim=-1) + start
    else:
        indices = logits.argmax(dim=-1)
    
    return indices


def aa_indices_to_string(
    indices: Tensor,
    mask: Optional[Tensor] = None,
) -> Union[str, List[str]]:
    """Convert token indices to amino acid string(s).
    
    Args:
        indices: Tensor of shape [B, L] or [L]
        mask: Optional boolean mask [B, L] or [L] for valid positions
              Masked positions are excluded from output string
              
    Returns:
        Single string if input is [L], list of strings if [B, L]
    """
    single_sequence = indices.dim() == 1
    if single_sequence:
        indices = indices.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)
    
    batch_size = indices.shape[0]
    sequences = []
    
    for b in range(batch_size):
        seq_indices = indices[b]
        if mask is not None:
            seq_indices = seq_indices[mask[b]]
        
        aa_list = []
        for idx in seq_indices.tolist():
            aa = _TOKEN_IDX_TO_ONE_LETTER.get(idx, "X")
            aa_list.append(aa)
        
        sequences.append("".join(aa_list))
    
    if single_sequence:
        return sequences[0]
    return sequences


def logits_to_aa_string(
    logits: Tensor,
    mask: Optional[Tensor] = None,
    canonical_only: bool = True,
) -> Union[str, List[str]]:
    """Convert res_type logits directly to amino acid string(s).
    
    This is a convenience function combining logits_to_aa_indices 
    and aa_indices_to_string.
    
    Args:
        logits: Tensor of shape [B, L, num_tokens] or [L, num_tokens]
        mask: Optional boolean mask [B, L] or [L] for valid positions
        canonical_only: If True, only consider canonical AAs
        
    Returns:
        Single string if input is [L, num_tokens], list of strings if [B, L, num_tokens]
    """
    indices = logits_to_aa_indices(logits, mask, canonical_only)
    return aa_indices_to_string(indices, mask)


def aa_string_to_indices(
    sequence: Union[str, List[str]],
    device: Optional[torch.device] = None,
) -> Tensor:
    """Convert amino acid string(s) to token indices.
    
    Args:
        sequence: Single AA string or list of AA strings
        device: Target device for output tensor
        
    Returns:
        Tensor of shape [L] or [B, L] with token indices
    """
    single_sequence = isinstance(sequence, str)
    if single_sequence:
        sequences = [sequence]
    else:
        sequences = sequence
    
    batch_indices = []
    for seq in sequences:
        indices = []
        for aa in seq:
            idx = _ONE_LETTER_TO_TOKEN_IDX.get(aa.upper(), const.token_ids["UNK"])
            indices.append(idx)
        batch_indices.append(indices)
    
    # Handle variable length sequences by padding
    max_len = max(len(seq) for seq in batch_indices)
    padded = []
    for indices in batch_indices:
        padded.append(indices + [const.token_ids["<pad>"]] * (max_len - len(indices)))
    
    tensor = torch.tensor(padded, dtype=torch.long, device=device)
    
    if single_sequence:
        return tensor.squeeze(0)
    return tensor


def straight_through_softmax(
    logits: Tensor,
    temperature: float = 1.0,
    dim: int = -1,
) -> Tensor:
    """Straight-through estimator for discrete token selection.
    
    Forward pass: Returns one-hot of argmax (discrete selection)
    Backward pass: Gradients flow through softmax (continuous)
    
    This allows us to:
    1. Get discrete token indices for predictors that need them
    2. Still backpropagate gradients to the logits
    
    Args:
        logits: Input logits tensor [..., num_classes]
        temperature: Softmax temperature (lower = sharper)
        dim: Dimension to apply softmax/argmax
        
    Returns:
        One-hot tensor same shape as input, with gradient path through softmax
    """
    # Soft version (differentiable)
    soft = F.softmax(logits / temperature, dim=dim)
    
    # Hard version (non-differentiable)
    indices = logits.argmax(dim=dim)
    hard = F.one_hot(indices, num_classes=logits.shape[dim]).float()
    
    # Move hard tensor to same device/dtype as soft
    hard = hard.to(soft)
    
    # STE trick: use hard in forward, soft gradient in backward
    # hard - soft.detach() + soft
    # Forward: soft.detach() and soft cancel, leaving hard
    # Backward: hard has no grad, soft.detach() has no grad, only soft contributes
    return hard - soft.detach() + soft


def straight_through_gumbel_softmax(
    logits: Tensor,
    temperature: float = 1.0,
    dim: int = -1,
) -> Tensor:
    """Gumbel-softmax with straight-through estimator.
    
    Like straight_through_softmax but adds Gumbel noise for stochastic sampling.
    Useful for exploration during guidance.
    
    Args:
        logits: Input logits tensor [..., num_classes]
        temperature: Softmax temperature
        dim: Dimension to apply softmax/argmax
        
    Returns:
        One-hot tensor with gradient path through Gumbel-softmax
    """
    # Sample Gumbel noise
    gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    
    # Soft Gumbel-softmax
    soft = F.softmax((logits + gumbels) / temperature, dim=dim)
    
    # Hard version
    indices = soft.argmax(dim=dim)
    hard = F.one_hot(indices, num_classes=logits.shape[dim]).float()
    hard = hard.to(soft)
    
    # STE trick
    return hard - soft.detach() + soft


def extract_canonical_logits(logits: Tensor) -> Tensor:
    """Extract only the canonical amino acid logits from full token logits.
    
    Args:
        logits: Full token logits [..., num_tokens]
        
    Returns:
        Canonical AA logits [..., 20]
    """
    start = const.canonicals_offset
    end = start + NUM_CANONICAL_AAS
    return logits[..., start:end]
