#!/usr/bin/env python3
"""Test property steering bias generation."""

import torch
from boltzgen.model.modules.property_steering import PropertySteering, CombinedPropertySteering


def test_property_steering():
    """Test property steering bias computation."""
    target_stability = 0.7
    bias_weight = 1.0
    temperature = 1.0
    
    # Test PropertySteering for atom_dec_bias
    steering = PropertySteering(
        target_property=target_stability,
        bias_weight=bias_weight,
        temperature=temperature,
        atom_decoder_depth=3,
        atom_decoder_heads=4,
        bias_type="atom_dec_bias",
    )
    
    # Test with different predicted values
    test_cases = [
        (0.5, "below target - should be positive bias"),
        (0.7, "at target - should be zero bias"),
        (0.9, "above target - should be negative bias"),
    ]
    
    num_atoms = 100
    batch_size = 1
    
    print("Testing PropertySteering bias generation:")
    for predicted, description in test_cases:
        bias = steering(
            predicted_property=predicted,
            num_atoms=num_atoms,
            batch_size=batch_size,
            device=torch.device("cpu"),
        )
        
        # Check bias sign and magnitude
        expected_sign = -1 if predicted > target_stability else (1 if predicted < target_stability else 0)
        actual_sign = 1 if bias.sum() > 0 else (-1 if bias.sum() < 0 else 0)
        
        error = predicted - target_stability
        expected_bias_value = -error * bias_weight / temperature
        
        print(f"\n  {description}")
        print(f"    Predicted: {predicted}, Target: {target_stability}, Error: {error:.4f}")
        print(f"    Bias shape: {bias.shape}")
        print(f"    Bias mean: {bias.mean():.4f} (expected: {expected_bias_value:.4f})")
        
        # Allow small tolerance for numerical precision
        assert abs(bias.mean() - expected_bias_value) < 0.001, \
            f"Bias value mismatch: got {bias.mean():.4f}, expected {expected_bias_value:.4f}"
        assert actual_sign == expected_sign or abs(error) < 0.001, \
            f"Bias sign mismatch: got {actual_sign}, expected {expected_sign}"
        print(f"    ✓ Correct bias direction and magnitude")
    
    # Test CombinedPropertySteering
    print("\nTesting CombinedPropertySteering:")
    combined = CombinedPropertySteering(
        target_property=target_stability,
        bias_weight=bias_weight,
        temperature=temperature,
        atom_decoder_depth=3,
        atom_decoder_heads=4,
        token_transformer_depth=6,
        token_transformer_heads=8,
        use_atom_bias=True,
        use_token_bias=True,
    )
    
    biases = combined(
        predicted_property=0.6,
        num_atoms=100,
        num_tokens=50,
        batch_size=1,
        device=torch.device("cpu"),
    )
    
    assert "atom_dec_bias" in biases
    assert "token_trans_bias" in biases
    print(f"  ✓ CombinedPropertySteering generates both bias types")
    print(f"    atom_dec_bias shape: {biases['atom_dec_bias'].shape}")
    print(f"    token_trans_bias shape: {biases['token_trans_bias'].shape}")
    
    # Verify both biases have correct sign
    error = 0.6 - target_stability  # -0.1 (below target, positive bias expected)
    assert biases['atom_dec_bias'].mean() > 0, "atom_dec_bias should be positive when below target"
    assert biases['token_trans_bias'].mean() > 0, "token_trans_bias should be positive when below target"
    print(f"  ✓ Both biases have correct direction (error: {error:.4f})")
    
    print("\n✅ All property steering tests passed!")


if __name__ == "__main__":
    test_property_steering()

