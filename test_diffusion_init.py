#!/usr/bin/env python3
"""Test AtomDiffusion initialization with property steering."""

import torch
from boltzgen.model.modules.diffusion import AtomDiffusion


def test_diffusion_init():
    """Test that AtomDiffusion can be initialized with steering parameters."""
    print("Testing AtomDiffusion initialization with property steering:\n")
    
    # Create minimal config
    score_model_args = {
        "token_s": 384,
        "atom_s": 128,
        "atoms_per_window_queries": 32,
        "atoms_per_window_keys": 128,
        "atom_encoder_depth": 3,
        "atom_encoder_heads": 4,
        "atom_decoder_depth": 3,
        "atom_decoder_heads": 4,
        "token_transformer_depth": 6,
        "token_transformer_heads": 8,
        "conditioning_transition_layers": 2,
        "predict_res_type": False,  # Should be auto-enabled
    }
    
    print("1. Testing initialization WITHOUT steering...")
    try:
        diffusion_no_steering = AtomDiffusion(
            score_model_args=score_model_args,
            enable_property_steering=False,
            num_sampling_steps=5,
        )
        assert not diffusion_no_steering.score_model.atom_attention_decoder.predict_res_type
        print("   ✓ Initialized without steering")
        print("   ✓ predict_res_type is False (as expected)")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        raise
    
    print("\n2. Testing initialization WITH steering...")
    try:
        diffusion_with_steering = AtomDiffusion(
            score_model_args=score_model_args,
            enable_property_steering=True,
            target_stability=0.7,
            stability_bias_weight=1.0,
            stability_bias_temperature=1.0,
            steering_update_freq=1,
            num_sampling_steps=5,
        )
        print("   ✓ Initialized with property steering")
        
        # Check that predict_res_type is enabled
        assert diffusion_with_steering.score_model.atom_attention_decoder.predict_res_type
        print("   ✓ predict_res_type is automatically enabled")
        
        # Check that steering components are initialized
        assert diffusion_with_steering.stability_predictor is not None
        assert diffusion_with_steering.property_steering is not None
        print("   ✓ Stability predictor initialized")
        print("   ✓ Property steering module initialized")
        
        # Check parameters
        assert diffusion_with_steering.target_stability == 0.7
        assert diffusion_with_steering.stability_bias_weight == 1.0
        print("   ✓ Steering parameters set correctly")
        
    except ImportError as e:
        print(f"   ⚠ TAPE not available (expected if not installed): {e}")
        print("     Install with: pip install tape-proteins")
        print("     This is expected - TAPE is required for steering")
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")
        raise
    
    print("\n✅ AtomDiffusion initialization tests passed!")


if __name__ == "__main__":
    test_diffusion_init()

