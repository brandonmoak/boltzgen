"""Utilities for extracting amino acid sequences from model predictions."""

from typing import Optional

import torch

from boltzgen.data import const


def extract_sequence_from_res_type(
    res_type_logits: torch.Tensor,  # Shape: (batch, num_tokens, num_token_types)
    token_pad_mask: torch.Tensor,  # Shape: (batch, num_tokens)
    design_mask: Optional[torch.Tensor] = None,  # Shape: (batch, num_tokens)
    mol_type: Optional[torch.Tensor] = None,  # Shape: (batch, num_tokens)
) -> list[str]:
    """Extract amino acid sequence string from predicted residue type logits.
    
    Parameters
    ----------
    res_type_logits : torch.Tensor
        Predicted residue type logits from the model.
        Shape: (batch, num_tokens, num_token_types)
    token_pad_mask : torch.Tensor
        Mask indicating which tokens are valid (not padding).
        Shape: (batch, num_tokens)
    design_mask : torch.Tensor, optional
        Mask indicating which tokens are being designed.
        If provided, only extracts sequence for designed residues.
        Shape: (batch, num_tokens)
    mol_type : torch.Tensor, optional
        Molecular type tensor to filter for protein tokens only.
        Shape: (batch, num_tokens)
        
    Returns
    -------
    list[str]
        List of amino acid sequences (one per batch item).
        Each sequence is a string of single-letter amino acid codes.
        
    Notes
    -----
    - Only extracts canonical amino acids (20 standard AAs)
    - Unknown/non-protein tokens are converted to 'X'
    - Returns sequences for all batch items
    """
    batch_size = res_type_logits.shape[0]
    
    # Get predicted token IDs (argmax over token types)
    predicted_token_ids = torch.argmax(res_type_logits, dim=-1)  # (batch, num_tokens)
    
    # Convert token IDs to token names
    sequences = []
    for batch_idx in range(batch_size):
        res_type_batch = res_type_logits[batch_idx]  # (num_tokens, num_token_types)
        token_mask_batch = token_pad_mask[batch_idx]  # (num_tokens,)
        design_mask_batch = design_mask[batch_idx] if design_mask is not None else None
        mol_type_batch = mol_type[batch_idx] if mol_type is not None else None
        token_ids_batch = predicted_token_ids[batch_idx]  # (num_tokens,)
        
        sequence_letters = []
        for token_idx in range(res_type_batch.shape[0]):
            # Check if token is valid (not padding)
            if not token_mask_batch[token_idx].item():
                continue
                
            # Check if we should include this token (design mask filter)
            if design_mask_batch is not None:
                if not design_mask_batch[token_idx].item():
                    continue
                    
            # Check if token is protein (mol_type filter)
            if mol_type_batch is not None:
                token_mol_type = mol_type_batch[token_idx].item()
                if token_mol_type != const.chain_type_ids["PROTEIN"]:
                    continue
            
            # Get token name from token ID
            token_id = token_ids_batch[token_idx].item()
            token_name = const.tokens[token_id]
            
            # Convert token name to single-letter amino acid code
            if token_name in const.prot_token_to_letter:
                aa_letter = const.prot_token_to_letter[token_name]
            else:
                # Unknown or non-protein token
                aa_letter = "X"
            
            # Skip gap characters ("-") as they're not valid amino acids
            # Also skip if we got "-" from the mapping
            if aa_letter != "-":
                sequence_letters.append(aa_letter)
        
        sequence = "".join(sequence_letters)
        sequences.append(sequence)
    
    return sequences


def extract_designed_chain_sequence(
    res_type_logits: torch.Tensor,
    feats: dict[str, torch.Tensor],
) -> Optional[str]:
    """Extract amino acid sequence for the designed chain(s).
    
    This is a convenience function that uses the standard feature dictionary
    format used throughout BoltzGen.
    
    Parameters
    ----------
    res_type_logits : torch.Tensor
        Predicted residue type logits from the model.
        Shape: (batch, num_tokens, num_token_types)
    feats : dict[str, torch.Tensor]
        Feature dictionary containing:
        - token_pad_mask: Mask for valid tokens
        - design_mask: Mask for designed residues
        - mol_type: Molecular type tensor
        
    Returns
    -------
    Optional[str]
        Amino acid sequence string, or None if no valid sequence found.
        Only returns the first batch item's sequence.
    """
    sequences = extract_sequence_from_res_type(
        res_type_logits=res_type_logits,
        token_pad_mask=feats["token_pad_mask"],
        design_mask=feats.get("design_mask"),
        mol_type=feats.get("mol_type"),
    )
    
    if sequences:
        return sequences[0]
    return None

