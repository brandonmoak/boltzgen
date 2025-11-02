# Testing Strategy for Property Steering Implementation

This document outlines a step-by-step approach to test the property steering implementation.

## Testing Order (Bottom-Up)

### 1. Sequence Extraction Utility
**File**: `src/boltzgen/utils/sequence_extraction.py`

**Test**: Extract sequences from mock residue type logits

```python
# test_sequence_extraction.py
import torch
from boltzgen.utils.sequence_extraction import extract_sequence_from_res_type
from boltzgen.data import const

def test_sequence_extraction():
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

if __name__ == "__main__":
    test_sequence_extraction()
```

**Run**:
```bash
python test_sequence_extraction.py
```

---

### 2. TAPE Stability Predictor
**File**: `src/boltzgen/model/modules/stability_predictor.py`

**Test**: Load TAPE model and predict stability

```python
# test_stability_predictor.py
import torch
from boltzgen.model.modules.stability_predictor import TAPEStabilityPredictor

def test_tape_stability_predictor():
    # Test sequence
    test_sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHTKTYPAIAVKQALDYPSVLSFDSFNFIRVSDFFTDKVQQVTVKQALDVEAVSRTLNDEFNGKVLVDLLRK"]
    
    # Initialize predictor
    predictor = TAPEStabilityPredictor(device="cpu")
    
    # Predict stability
    stability = predictor.forward(test_sequence)
    
    print(f"✓ TAPE stability predictor works")
    print(f"  Sequence length: {len(test_sequence)}")
    print(f"  Predicted stability: {stability:.4f}")
    
    # Test batch prediction
    sequences = ["MKTAYIAKQR", "AAAAA", "MKTAYIAKQR"]
    stabilities = predictor.predict_batch(sequences)
    print(f"✓ Batch prediction works: {stabilities}")

if __name__ == "__main__":
    test_tape_stability_predictor()
```

**Prerequisites**:
```bash
pip install tape-proteins
```

**Run**:
```bash
python test_stability_predictor.py
```

---

### 3. Property Steering Bias Generation
**File**: `src/boltzgen/model/modules/property_steering.py`

**Test**: Generate bias terms from property predictions

```python
# test_property_steering.py
import torch
from boltzgen.model.modules.property_steering import PropertySteering, CombinedPropertySteering

def test_property_steering():
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
        
        print(f"✓ {description}")
        print(f"  Predicted: {predicted}, Target: {target_stability}")
        print(f"  Bias range: [{bias.min():.4f}, {bias.max():.4f}], Mean: {bias.mean():.4f}")
        assert actual_sign == expected_sign or abs(predicted - target_stability) < 0.001
    
    # Test CombinedPropertySteering
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
    print(f"✓ CombinedPropertySteering works")
    print(f"  atom_dec_bias shape: {biases['atom_dec_bias'].shape}")
    print(f"  token_trans_bias shape: {biases['token_trans_bias'].shape}")

if __name__ == "__main__":
    test_property_steering()
```

**Run**:
```bash
python test_property_steering.py
```

---

### 4. Integration Test: Sequence → Stability → Bias
**File**: Combined test

**Test**: Full pipeline from sequence to bias

```python
# test_full_pipeline.py
import torch
from boltzgen.utils.sequence_extraction import extract_designed_chain_sequence
from boltzgen.model.modules.stability_predictor import TAPEStabilityPredictor
from boltzgen.model.modules.property_steering import CombinedPropertySteering
from boltzgen.data import const

def test_full_pipeline():
    # 1. Create mock res_type predictions
    test_sequence = "MKTAYIAKQR"
    num_tokens = len(test_sequence)
    
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
    sequence = extract_designed_chain_sequence(res_type_logits, feats)
    assert sequence == test_sequence, f"Expected {test_sequence}, got {sequence}"
    print(f"✓ Sequence extraction: {sequence}")
    
    # 3. Predict stability
    predictor = TAPEStabilityPredictor(device="cpu")
    stability = predictor(sequence)
    print(f"✓ Stability prediction: {stability:.4f}")
    
    # 4. Generate bias
    steering = CombinedPropertySteering(
        target_property=0.7,
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
    
    print(f"✓ Bias generation works")
    print(f"  Bias shape: {biases['atom_dec_bias'].shape}")
    print(f"  Bias mean: {biases['atom_dec_bias'].mean():.4f}")
    
    # Verify bias direction
    error = stability - 0.7
    expected_bias_sign = -1 if error > 0 else (1 if error < 0 else 0)
    actual_bias_sign = 1 if biases['atom_dec_bias'].mean() > 0 else (-1 if biases['atom_dec_bias'].mean() < 0 else 0)
    assert actual_bias_sign == expected_bias_sign or abs(error) < 0.001
    print(f"✓ Bias direction correct (error: {error:.4f})")

if __name__ == "__main__":
    test_full_pipeline()
```

**Run**:
```bash
python test_full_pipeline.py
```

---

### 5. Integration with Denoising (Single Step)
**Test**: Test that biases are correctly merged into diffusion conditioning

```python
# test_denoising_integration.py
import torch
from boltzgen.model.modules.diffusion import AtomDiffusion

def test_denoising_bias_injection():
    """Test that property bias is correctly injected into diffusion_conditioning."""
    
    # This is a minimal test - you'll need actual model weights
    # For now, just test that the code path exists and doesn't crash
    
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
        "predict_res_type": True,
    }
    
    # Try to initialize with steering enabled
    # Note: This will fail if TAPE is not installed, which is expected
    try:
        diffusion = AtomDiffusion(
            score_model_args=score_model_args,
            enable_property_steering=True,
            target_stability=0.7,
            stability_bias_weight=1.0,
            stability_bias_temperature=1.0,
            num_sampling_steps=5,
        )
        print("✓ AtomDiffusion initialized with property steering")
        
        # Check that predict_res_type is enabled
        assert diffusion.score_model.atom_attention_decoder.predict_res_type
        print("✓ predict_res_type is enabled")
        
    except ImportError as e:
        print(f"⚠ TAPE not available (expected): {e}")
        print("  Install with: pip install tape-proteins")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        raise

if __name__ == "__main__":
    test_denoising_bias_injection()
```

---

### 6. End-to-End Test with Mock Data
**Test**: Full denoising step with property steering

```python
# test_end_to_end.py
"""
End-to-end test with minimal mock data.
This requires:
1. A trained BoltzGen checkpoint
2. TAPE installed
3. A simple design spec YAML
"""

def test_end_to_end_steering():
    """Run a single design with property steering enabled."""
    
    # This would require:
    # 1. Loading a checkpoint
    # 2. Creating a design spec with steering enabled
    # 3. Running one denoising step
    # 4. Verifying stability predictions are computed
    # 5. Verifying biases are applied
    
    print("End-to-end test requires:")
    print("1. BoltzGen checkpoint")
    print("2. TAPE installed")
    print("3. Design spec YAML with steering enabled")
    print("\nExample config:")
    print("""
    diffusion_process_args:
      enable_property_steering: true
      target_stability: 0.7
      stability_bias_weight: 1.0
      stability_bias_temperature: 1.0
      steering_update_freq: 1
    """)

if __name__ == "__main__":
    test_end_to_end_steering()
```

---

## Manual Testing with Actual Model

### Step 1: Install TAPE
```bash
pip install tape-proteins
```

### Step 2: Create Test Config
Modify `config/design.yaml` to add steering parameters:

```yaml
override:
  diffusion_process_args:
    enable_property_steering: true
    target_stability: 0.7
    stability_bias_weight: 1.0
    stability_bias_temperature: 1.0
    steering_update_freq: 5  # Update every 5 steps
    use_atom_bias: true
    use_token_bias: false
```

### Step 3: Run Single Design
```bash
# Use a simple example
boltzgen run example/vanilla_peptide_with_target_binding_site/beetletert.yaml \
  --checkpoint <path_to_checkpoint> \
  --output test_steering_output \
  --diffusion_samples 1 \
  --sampling_steps 50  # Fewer steps for testing
```

### Step 4: Verify
- Check logs for stability predictions
- Verify sequences are being extracted
- Verify no errors in bias computation
- Compare outputs with/without steering

---

## Debugging Tips

1. **Add logging**: In `_compute_property_bias`, add print statements:
   ```python
   print(f"Step {step_idx}: Sequence={sequence[:20]}..., Stability={predicted_stability:.4f}, Error={error:.4f}")
   ```

2. **Check bias shapes**: Verify bias tensors match expected shapes:
   ```python
   print(f"Bias shape: {bias.shape}, Expected: (batch, dim1, dim2, heads)")
   ```

3. **Test with minimal steps**: Start with `sampling_steps=5` and `steering_update_freq=1` to see immediate feedback

4. **Compare outputs**: Generate with steering on/off and compare stability distributions

---

## Expected Behavior

When working correctly, you should see:
1. ✅ Sequences extracted at each steering update step
2. ✅ Stability predictions in reasonable range (typically -2 to 2 for TAPE)
3. ✅ Bias terms computed and merged without shape mismatches
4. ✅ Generated sequences show stability closer to target than without steering
5. ✅ No crashes or silent failures

