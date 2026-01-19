"""Core utilities for guidance - sequence extraction from model output.

These are used by GeometricGuidance internally and may be useful for 
downstream processing of guided outputs.
"""

from typing import Dict, Tuple
import torch
from torch import Tensor

from boltzgen.data import const


def extract_sequence_from_sample(sample: Dict[str, Tensor]) -> Tuple[str, str]:
    """Extract sequence from a sample dict (after res_from_atom14).
    
    Args:
        sample: Feature dict with res_type and design_mask
        
    Returns:
        Tuple of (full_sequence, designed_sequence)
        - full_sequence: lowercase=template, uppercase=designed
        - designed_sequence: only the designed residues
    """
    token_ids = torch.argmax(sample["res_type"], dim=-1)
    design_mask = sample.get("design_mask", torch.zeros_like(token_ids, dtype=torch.bool)).bool()
    
    full_seq = ""
    designed_seq = ""
    for i, tid in enumerate(token_ids.tolist()):
        token_name = const.tokens[tid] if 0 <= tid < len(const.tokens) else "UNK"
        letter = const.prot_token_to_letter.get(token_name, "X")
        
        if design_mask[i]:
            full_seq += letter.upper()
            designed_seq += letter.upper()
        else:
            full_seq += letter.lower()
    
    return full_seq, designed_seq
