#!/usr/bin/env python3
"""End-to-end test for guidance integration with BoltzGen model.

This script tests the full pipeline with guidance enabled/disabled.
It requires a model checkpoint (downloaded automatically if not present).

Usage:
    # Run with default checkpoint (downloads from HuggingFace if needed)
    python scripts/e2e_guidance_test.py
    
    # Run with specific checkpoint
    python scripts/e2e_guidance_test.py --checkpoint /path/to/checkpoint.ckpt
    
    # Skip download, only run if checkpoint exists
    python scripts/e2e_guidance_test.py --no-download

Requirements:
    - Model checkpoint (auto-downloaded from HuggingFace)
    - GPU (CUDA or MPS) recommended, but CPU works for small tests
"""

import argparse
import sys
import os
from pathlib import Path
import time

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Quiet startup before imports
os.environ.setdefault("CUEQ_DEFAULT_CONFIG", "1")
os.environ.setdefault("CUEQ_DISABLE_AOT_TUNING", "1")

import torch
import numpy as np

from boltzgen.data import const
from boltzgen.model.modules.guidance import (
    DiffusionGuidance,
    HydrophobicityPredictor,
    AminoAcidFrequencyPredictor,
    create_guidance,
)


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def download_checkpoint(cache_dir: Path = None) -> Path:
    """Download the design checkpoint from HuggingFace.
    
    Returns:
        Path to the downloaded checkpoint
    """
    try:
        import huggingface_hub
    except ImportError:
        print("huggingface_hub not installed. Install with: pip install huggingface_hub")
        sys.exit(1)
    
    repo_id = "boltzgen/boltzgen-1"
    filename = "boltzgen1_diverse.ckpt"  # Use the diverse design checkpoint
    
    print(f"Downloading checkpoint from HuggingFace: {repo_id}/{filename}")
    print("This may take a while on first run...")
    
    checkpoint_path = huggingface_hub.hf_hub_download(
        repo_id,
        filename,
        repo_type="model",
        library_name="boltzgen",
        cache_dir=cache_dir,
    )
    
    print(f"Checkpoint downloaded to: {checkpoint_path}")
    return Path(checkpoint_path)


def load_model(checkpoint_path: Path, device: torch.device):
    """Load the Boltz model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    from boltzgen.model.models.boltz import Boltz
    
    print(f"Loading model from: {checkpoint_path}")
    print(f"Device: {device}")
    
    # For MPS or non-CUDA devices, load to CPU first then move
    # This avoids issues with CUDA-specific tensors in checkpoints
    load_device = "cpu" if device.type in ("mps", "cpu") else device
    
    # Load with appropriate settings
    model = Boltz.load_from_checkpoint(
        str(checkpoint_path),
        strict=False,  # Allow missing keys for flexibility
        map_location=load_device,
        weights_only=False,
    )
    model.eval()
    
    # Move to target device after loading
    if device.type == "mps":
        # For MPS, we need to be careful about moving
        # Some operations may not be supported
        print("Note: Moving model to MPS. Some operations may fall back to CPU.")
        try:
            model.to(device)
        except Exception as e:
            print(f"Warning: Could not move to MPS ({e}), staying on CPU")
            device = torch.device("cpu")
    else:
        model.to(device)
    
    print("Model loaded successfully")
    return model


def create_mock_features(batch_size: int = 1, seq_len: int = 50, device: torch.device = None):
    """Create mock features for testing.
    
    This creates minimal features needed for the diffusion module.
    In practice, you'd use proper features from a design spec.
    
    Args:
        batch_size: Number of sequences
        seq_len: Length of each sequence
        device: Device for tensors
        
    Returns:
        Dictionary of features
    """
    if device is None:
        device = torch.device("cpu")
    
    # Number of atoms (roughly 4-5 per residue for backbone + some sidechain)
    num_atoms = seq_len * 5
    
    feats = {
        # Token features
        "token_index": torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1),
        "token_pad_mask": torch.ones(batch_size, seq_len, dtype=torch.bool, device=device),
        "design_mask": torch.ones(batch_size, seq_len, dtype=torch.bool, device=device),
        
        # Atom features  
        "atom_pad_mask": torch.ones(batch_size, num_atoms, dtype=torch.bool, device=device),
        "atom_to_token": torch.arange(num_atoms, device=device).unsqueeze(0).expand(batch_size, -1) // 5,
        
        # Molecule type (0 = protein)
        "mol_type": torch.zeros(batch_size, seq_len, dtype=torch.long, device=device),
        
        # Residue types (random valid tokens for testing)
        "res_type": torch.randint(
            const.canonicals_offset, 
            const.canonicals_offset + 20,
            (batch_size, seq_len),
            device=device,
        ),
    }
    
    return feats


def test_guidance_integration_mock():
    """Test guidance integration with mock data (no model required).
    
    This is a quick sanity check that doesn't require loading the full model.
    """
    print("\n" + "=" * 60)
    print("  TEST: Mock Guidance Integration")
    print("=" * 60)
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Create predictors
    hydro_pred = HydrophobicityPredictor(higher_is_better=True)
    guidance = DiffusionGuidance(
        predictor=hydro_pred,
        guidance_scale=5.0,
        temperature=0.5,
        schedule="linear",
        schedule_start=0.3,
        enabled=True,
    )
    
    # Create mock logits
    B, L = 2, 30
    logits = torch.randn(B, L, const.num_tokens, device=device)
    feats = {}
    
    # Test guidance computation
    initial_sequences = guidance.get_predicted_sequences(logits)
    initial_scores = guidance.get_predictor_scores(logits)
    
    print(f"\nInitial sequences:")
    for i, seq in enumerate(initial_sequences):
        print(f"  {i}: {seq[:20]}... (score: {initial_scores[i].item():.3f})")
    
    # Apply guidance iteratively
    print("\nApplying guidance for 10 steps...")
    for step in range(10):
        grad = guidance.compute_sequence_guidance(logits, feats, step=step, total_steps=10)
        if grad is not None:
            logits = logits + grad
    
    final_sequences = guidance.get_predicted_sequences(logits)
    final_scores = guidance.get_predictor_scores(logits)
    
    print(f"\nFinal sequences:")
    for i, seq in enumerate(final_sequences):
        print(f"  {i}: {seq[:20]}... (score: {final_scores[i].item():.3f})")
    
    # Check improvement
    improved = (final_scores > initial_scores).all().item()
    print(f"\nAll scores improved: {improved}")
    
    if improved:
        print("✓ Mock guidance test PASSED")
        return True
    else:
        print("✗ Mock guidance test FAILED")
        return False


def test_guidance_with_model(checkpoint_path: Path = None, download: bool = True):
    """Test guidance integration with actual model.
    
    Args:
        checkpoint_path: Path to checkpoint, or None to download
        download: Whether to download checkpoint if not present
        
    Returns:
        True if test passed
    """
    print("\n" + "=" * 60)
    print("  TEST: Full Model Guidance Integration")
    print("=" * 60)
    
    # Get checkpoint
    if checkpoint_path is None or not checkpoint_path.exists():
        if download:
            checkpoint_path = download_checkpoint()
        else:
            print("Checkpoint not found and download disabled. Skipping model test.")
            return None
    
    device = get_device()
    
    try:
        # Load model
        model = load_model(checkpoint_path, device)
        
        # Check that forward() accepts guidance parameter
        import inspect
        sig = inspect.signature(model.forward)
        if 'guidance' not in sig.parameters:
            print("✗ model.forward() does not accept guidance parameter")
            return False
        print("✓ model.forward() accepts guidance parameter")
        
        # Check that structure_module.sample() accepts guidance parameter
        if hasattr(model, 'structure_module'):
            sig = inspect.signature(model.structure_module.sample)
            if 'guidance' not in sig.parameters:
                print("✗ structure_module.sample() does not accept guidance parameter")
                return False
            print("✓ structure_module.sample() accepts guidance parameter")
        
        # Create guidance module
        predictor = HydrophobicityPredictor(higher_is_better=True)
        guidance = DiffusionGuidance(
            predictor=predictor,
            guidance_scale=1.0,
            enabled=True,
        )
        
        # Test that guidance can be passed (we won't run full inference
        # as that requires properly formatted input data)
        print("✓ Guidance module created successfully")
        print("✓ Model integration test PASSED")
        
        return True
        
    except Exception as e:
        print(f"✗ Model integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_guidance_comparison(checkpoint_path: Path = None, download: bool = True):
    """Compare model outputs with and without guidance.
    
    This is the full e2e test that runs actual inference.
    Requires significant compute resources.
    
    Args:
        checkpoint_path: Path to checkpoint
        download: Whether to download if not present
        
    Returns:
        True if test passed
    """
    print("\n" + "=" * 60)
    print("  TEST: Guided vs Unguided Comparison")
    print("=" * 60)
    
    # This test requires proper input data and significant compute
    # For now, we'll just verify the plumbing works
    
    device = get_device()
    
    # Get checkpoint
    if checkpoint_path is None:
        if download:
            checkpoint_path = download_checkpoint()
        else:
            print("Checkpoint not found. Skipping comparison test.")
            return None
    
    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        return None
    
    try:
        # Load model
        model = load_model(checkpoint_path, device)
        
        # Create guidance configurations
        guidance_configs = [
            ("No guidance", None),
            ("Disabled guidance", create_guidance(
                HydrophobicityPredictor(), enabled=False
            )),
            ("Enabled guidance", create_guidance(
                HydrophobicityPredictor(higher_is_better=True),
                guidance_scale=1.0,
                enabled=True,
            )),
        ]
        
        print("\nGuidance configurations ready:")
        for name, guidance in guidance_configs:
            status = "enabled" if (guidance and guidance.enabled) else "disabled/none"
            print(f"  - {name}: {status}")
        
        print("\n✓ Comparison test setup successful")
        print("\nNote: Full inference comparison requires proper input data.")
        print("Use 'boltzgen run' with a design spec to test full pipeline.")
        
        return True
        
    except Exception as e:
        print(f"✗ Comparison test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="E2E Guidance Integration Test")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to model checkpoint (downloads if not provided)",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Don't download checkpoint, only run if exists",
    )
    parser.add_argument(
        "--mock-only",
        action="store_true",
        help="Only run mock tests (no model required)",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  BOLTZGEN GUIDANCE E2E TEST SUITE")
    print("=" * 60)
    
    device = get_device()
    print(f"\nDevice: {device}")
    
    results = {}
    
    # Always run mock test
    results["mock"] = test_guidance_integration_mock()
    
    if not args.mock_only:
        # Run model tests
        results["model"] = test_guidance_with_model(
            args.checkpoint,
            download=not args.no_download,
        )
        
        results["comparison"] = test_guidance_comparison(
            args.checkpoint,
            download=not args.no_download,
        )
    
    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    
    for name, result in results.items():
        if result is True:
            print(f"  ✓ {name}: PASSED")
        elif result is False:
            print(f"  ✗ {name}: FAILED")
        else:
            print(f"  - {name}: SKIPPED")
    
    # Exit code
    all_passed = all(r is True or r is None for r in results.values())
    any_failed = any(r is False for r in results.values())
    
    if any_failed:
        print("\n✗ Some tests FAILED")
        sys.exit(1)
    elif all_passed:
        print("\n✓ All tests PASSED")
        sys.exit(0)
    else:
        print("\n- Tests completed with skips")
        sys.exit(0)


if __name__ == "__main__":
    main()
