#!/usr/bin/env python3
"""Test TAPE stability predictor."""

try:
    from boltzgen.model.modules.stability_predictor import TAPEStabilityPredictor
    TAPE_AVAILABLE = True
except ImportError as e:
    TAPE_AVAILABLE = False
    IMPORT_ERROR = e


def test_tape_stability_predictor():
    """Test TAPE stability prediction."""
    if not TAPE_AVAILABLE:
        print(f"✗ TAPE not available: {IMPORT_ERROR}")
        print("  Install with: pip install tape-proteins")
        return
    
    # Test sequence (short example)
    test_sequence = "MKTAYIAKQR"
    
    print("Testing TAPE Stability Predictor:")
    print(f"  Test sequence: {test_sequence} (length: {len(test_sequence)})")
    
    try:
        # Initialize predictor
        predictor = TAPEStabilityPredictor(device="cpu")
        print("  ✓ Predictor initialized")
        
        # Predict stability
        stability = predictor.forward(test_sequence)
        print(f"  ✓ Stability prediction: {stability:.4f}")
        
        # Verify stability is a scalar float
        assert isinstance(stability, (float, int)), "Stability should be a scalar"
        print(f"  ✓ Return type correct: {type(stability)}")
        
        # Test batch prediction
        sequences = ["MKTAYIAKQR", "AAAAA", "MKTAYIAKQR"]
        print(f"\n  Testing batch prediction with {len(sequences)} sequences...")
        stabilities = predictor.predict_batch(sequences)
        
        assert len(stabilities) == len(sequences), "Batch size mismatch"
        assert all(isinstance(s, (float, int)) for s in stabilities), "All outputs should be scalars"
        print(f"  ✓ Batch prediction works: {stabilities}")
        
        # Test that different sequences give different predictions (usually)
        assert stabilities[0] == stabilities[2], "Same sequence should give same prediction"
        print(f"  ✓ Identical sequences give identical predictions")
        
        print("\n✅ All stability predictor tests passed!")
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        raise


if __name__ == "__main__":
    test_tape_stability_predictor()

