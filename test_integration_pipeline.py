#!/usr/bin/env python3
"""Integration test: Sequence → Stability → Bias pipeline."""

import torch
from boltzgen.utils.sequence_extraction import extract_designed_chain_sequence
from boltzgen.data import const

try:
    from boltzgen.model.modules.stability_predictor import TAPEStabilityPredictor
    from boltzgen.model.modules.property_steering import CombinedPropertySteering
    TAPE_AVAILABLE = True
except ImportError:
    TAPE_AVAILABLE = False


def test_full_pipeline():
    """Test full pipeline: sequence extraction → stability → bias generation."""
    print("Testing full integration pipeline:\n")
    
    # 1. Create mock res_type predictions
    test_sequence = "MKTAYIAKQR"
    num_tokens = len(test_sequence)
    
    print(f"1. Creating mock predictions for sequence: {test_sequence}")
    # Create logits that decode to our test sequence
    res_type_logits = torch.zeros(1, num_tokens, len(const.tokens))
    for i, aa_letter in enumerate(test_sequence):
        token_name = const.prot_letter_to_token[aa_letter]
        token_id = const.token_ids[token_name]
        res_type_logits[0, i, token_id] = 10.0
    
    feats = {
        "token_pad_mask": torch.ones(1, num_tokens, dtype=torch.bool),
        "design_mask": torch.ones(1, num_tokens, dtype=torch.bool),
        "mol_type": torch.full((1, num_tokens), const.chain_type_ids["PROTEIN"], dtype=torch.long),
    }
    
    # 2. Extract sequence
    print("2. Extracting sequence from logits...")
    sequence = extract_designed_chain_sequence(res_type_logits, feats)
    assert sequence == test_sequence, f"Expected {test_sequence}, got {sequence}"
    print(f"   ✓ Sequence extracted: {sequence}")
    
    if not TAPE_AVAILABLE:
        print("\n⚠ TAPE not available - skipping stability prediction and bias generation")
        print("  Install with: pip install tape-proteins")
        return
    
    # 3. Predict stability
    print("3. Predicting stability with TAPE...")
    predictor = TAPEStabilityPredictor(device="cpu")
    stability = predictor(sequence)
    print(f"   ✓ Stability prediction: {stability:.4f}")
    
    # 4. Generate bias
    print("4. Generating property steering bias...")
    target_stability = 0.7
    steering = CombinedPropertySteering(
        target_property=target_stability,
        bias_weight=1.0,
        temperature=1.0,
        atom_decoder_depth=3,
        atom_decoder_heads=4,
        token_transformer_depth=6,
        token_transformer_heads=8,
        use_atom_bias=True,
        use_token_bias=False,
    )
    
    num_atoms = 150  # Approximate for a 10-residue protein
    biases = steering(
        predicted_property=stability,
        num_atoms=num_atoms,
        num_tokens=num_tokens,
        batch_size=1,
        device=torch.device("cpu"),
    )
    
    print(f"   ✓ Bias generated")
    print(f"     Bias shape: {biases['atom_dec_bias'].shape}")
    print(f"     Bias mean: {biases['atom_dec_bias'].mean():.4f}")
    
    # Verify bias direction
    error = stability - target_stability
    expected_bias_sign = -1 if error > 0 else (1 if error < 0 else 0)
    actual_bias_sign = 1 if biases['atom_dec_bias'].mean() > 0 else (-1 if biases['atom_dec_bias'].mean() < 0 else 0)
    assert actual_bias_sign == expected_bias_sign or abs(error) < 0.001, \
        f"Bias direction incorrect: error={error:.4f}, sign={actual_bias_sign}, expected={expected_bias_sign}"
    print(f"   ✓ Bias direction correct (error: {error:.4f})")
    
    print("\n✅ Full pipeline test passed!")


if __name__ == "__main__":
    test_full_pipeline()

