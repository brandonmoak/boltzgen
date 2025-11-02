#!/usr/bin/env python3
"""Test sequence extraction utility."""

import torch
from boltzgen.utils.sequence_extraction import extract_sequence_from_res_type
from boltzgen.data import const


def test_sequence_extraction():
    """Test extracting sequences from residue type logits."""
    # Create mock res_type logits (batch=1, num_tokens=5, num_token_types=len(const.tokens))
    batch_size = 1
    num_tokens = 5
    num_token_types = len(const.tokens)
    
    # Create logits with high confidence for specific residues
    res_type_logits = torch.zeros(batch_size, num_tokens, num_token_types)
    
    # Set specific residues: ALA, ARG, ASN, ASP, CYS
    target_residues = ["ALA", "ARG", "ASN", "ASP", "CYS"]
    for i, res_name in enumerate(target_residues):
        token_id = const.token_ids[res_name]
        res_type_logits[0, i, token_id] = 10.0  # High logit
    
    # Create masks
    token_pad_mask = torch.ones(batch_size, num_tokens, dtype=torch.bool)
    design_mask = torch.ones(batch_size, num_tokens, dtype=torch.bool)
    mol_type = torch.full((batch_size, num_tokens), const.chain_type_ids["PROTEIN"], dtype=torch.long)
    
    # Extract sequence
    sequences = extract_sequence_from_res_type(
        res_type_logits,
        token_pad_mask,
        design_mask=design_mask,
        mol_type=mol_type,
    )
    
    # Verify
    expected = "".join([const.prot_token_to_letter[r] for r in target_residues])
    assert sequences[0] == expected, f"Expected {expected}, got {sequences[0]}"
    print(f"✓ Sequence extraction works: {sequences[0]}")
    
    # Test with design mask filtering
    design_mask_partial = torch.zeros(batch_size, num_tokens, dtype=torch.bool)
    design_mask_partial[0, 0:3] = True  # Only first 3 residues
    sequences_partial = extract_sequence_from_res_type(
        res_type_logits,
        token_pad_mask,
        design_mask=design_mask_partial,
        mol_type=mol_type,
    )
    expected_partial = "".join([const.prot_token_to_letter[r] for r in target_residues[:3]])
    assert sequences_partial[0] == expected_partial
    print(f"✓ Design mask filtering works: {sequences_partial[0]}")
    
    print("\n✅ All sequence extraction tests passed!")


if __name__ == "__main__":
    test_sequence_extraction()

