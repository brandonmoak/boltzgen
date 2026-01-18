# BoltzGen Guidance Implementation Summary

## Overview

This document summarizes the implementation of a **policy-guided diffusion system** for BoltzGen that allows steering sequence generation toward desired properties (e.g., stability, expression, hydrophobicity) using external predictor models.

## What Was Built

### Core Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Sequence Utilities | `src/boltzgen/model/modules/guidance/sequence_utils.py` | Convert between logits, token indices, and AA strings; implements Straight-Through Estimator (STE) for gradient flow |
| Predictor Interface | `src/boltzgen/model/modules/guidance/predictor.py` | Abstract base class `SequencePredictor` for property predictors |
| Mock Predictors | `src/boltzgen/model/modules/guidance/mock_predictor.py` | Test predictors: `HydrophobicityPredictor`, `AminoAcidFrequencyPredictor`, `ConstantPredictor`, `SequenceLengthPredictor` |
| Guidance Module | `src/boltzgen/model/modules/guidance/guidance.py` | Core `DiffusionGuidance` class that computes steering gradients |
| Package Init | `src/boltzgen/model/modules/guidance/__init__.py` | Exports all public classes/functions |

### Model Integration

Modified files:
- `src/boltzgen/model/modules/diffusion.py` - Added `guidance` parameter to `AtomDiffusion.sample()`
- `src/boltzgen/model/models/boltz.py` - Added `guidance` parameter to `Boltz.forward()` and passes it to `structure_module.sample()`

### Test Suite

| Test File | Tests | Description |
|-----------|-------|-------------|
| `tests/guidance/test_sequence_utils.py` | 30 | Token mapping, string conversion, STE gradients |
| `tests/guidance/test_predictors.py` | 22 | Predictor interface, mock predictors, gradient flow |
| `tests/guidance/test_guidance.py` | 23 | Guidance schedules, gradient computation, toggle |
| `tests/guidance/test_integration.py` | 14 | Parameter passing, backwards compatibility |
| `tests/guidance/test_boltz_integration.py` | 19 | Boltz model integration, signature verification |
| **Total** | **108** | All passing |

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/validate_guidance.py` | Validates guidance mechanism with mock logits (no model needed) |
| `scripts/e2e_guidance_test.py` | Full e2e test with model checkpoint download |

## How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Diffusion Loop                              │
│                                                                     │
│  1. Model predicts res_type_logits from noisy coordinates          │
│  2. DiffusionGuidance receives logits                              │
│  3. STE converts logits → discrete tokens (with gradient path)     │
│  4. SequencePredictor scores the sequence                          │
│  5. Backprop computes gradient w.r.t. logits                       │
│  6. Gradient applied to steer toward higher scores                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Usage Example

```python
from boltzgen.model.modules.guidance import (
    DiffusionGuidance,
    HydrophobicityPredictor,
    create_guidance,
)

# Create predictor (or use your own SequencePredictor subclass)
predictor = HydrophobicityPredictor(higher_is_better=True)

# Create guidance module
guidance = DiffusionGuidance(
    predictor=predictor,
    guidance_scale=1.0,      # Strength of guidance
    temperature=0.5,         # STE temperature (lower = sharper)
    schedule="linear",       # Ramp up guidance over diffusion
    schedule_start=0.3,      # Start guidance at 30% through diffusion
    enabled=True,            # Can toggle on/off
)

# Pass to model forward (guidance flows to structure_module.sample())
output = model.forward(
    feats=feats,
    guidance=guidance,  # NEW parameter
    # ... other args
)
```

## What To Test Next (Requires CUDA)

The implementation is complete but full model testing requires a **CUDA-enabled machine**. The BoltzGen checkpoints contain CUDA-specific torchmetrics that fail to load on Mac/CPU-only systems.

### On a CUDA Machine, Run:

```bash
# 1. Run all unit tests (should pass)
cd /path/to/boltzgen
python -m pytest tests/guidance/ -v

# 2. Run validation script (should pass)
python scripts/validate_guidance.py

# 3. Run full e2e test with model (downloads checkpoint automatically)
python scripts/e2e_guidance_test.py

# 4. Test with actual design spec
# Create a design spec YAML, then run with guidance:
# (Requires adding guidance parameter to CLI - see below)
```

### To Fully Integrate with CLI

The guidance parameter is wired through `Boltz.forward()` but not yet exposed in the CLI. To use guidance in production:

1. Modify `src/boltzgen/task/predict/predict.py` to accept guidance config
2. Update CLI in `src/boltzgen/cli/boltzgen.py` to parse guidance arguments
3. Instantiate guidance in `Predict.run()` and pass to model

### Creating Custom Predictors

To use your own property predictor:

```python
from boltzgen.model.modules.guidance import SequencePredictor
import torch

class MyStabilityPredictor(SequencePredictor):
    def __init__(self, model_path: str):
        super().__init__(higher_is_better=True)
        self.model = torch.load(model_path)
    
    def forward(self, sequences, mask=None):
        # sequences can be:
        # - str or List[str]: amino acid strings
        # - Tensor [B, L]: token indices
        # - Tensor [B, L, num_tokens]: soft logits (for gradient flow)
        
        # Return shape [B] with one score per sequence
        return self.model(sequences)
```

## Key Design Decisions

1. **Straight-Through Estimator (STE)** - Enables gradient flow through discrete token selection
2. **Guidance Schedules** - Delays guidance until logits are meaningful (early diffusion logits are noise)
3. **Toggleable** - `enabled=False` completely disables guidance for A/B testing
4. **Backwards Compatible** - All new parameters have defaults; existing code works unchanged

## Files Changed (Git Diff Summary)

```
Modified:
  src/boltzgen/model/modules/diffusion.py      # Added guidance param to sample()
  src/boltzgen/model/models/boltz.py           # Added guidance param to forward()

Added:
  src/boltzgen/model/modules/guidance/         # New package
    __init__.py
    sequence_utils.py
    predictor.py
    mock_predictor.py
    guidance.py
  tests/guidance/                              # Test suite
    __init__.py
    test_sequence_utils.py
    test_predictors.py
    test_guidance.py
    test_integration.py
    test_boltz_integration.py
  scripts/validate_guidance.py                 # Validation script
  scripts/e2e_guidance_test.py                 # E2E test script
  docs/GUIDANCE_IMPLEMENTATION_SUMMARY.md      # This file
```

## Test Commands Quick Reference

```bash
# All guidance tests
python -m pytest tests/guidance/ -v

# Just the integration tests
python -m pytest tests/guidance/test_boltz_integration.py -v

# Validation (no model needed)
python scripts/validate_guidance.py

# E2E with model (CUDA only)
python scripts/e2e_guidance_test.py

# E2E mock only (works on Mac/CPU)
python scripts/e2e_guidance_test.py --mock-only
```
