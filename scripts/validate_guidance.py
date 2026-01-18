#!/usr/bin/env python3
"""End-to-end validation script for diffusion guidance.

This script validates that the guidance module correctly steers
sequence generation toward desired properties.

Since full model inference requires GPU and model weights, this script
uses mock scenarios to validate the guidance mechanism itself:

1. Gradient Steering Test: Verifies that guidance gradients point toward
   sequences with higher predictor scores.

2. Iterative Application Test: Simulates applying guidance over multiple
   steps and confirms that predicted sequences improve.

3. Toggle Test: Confirms that disabled guidance produces no effect.

Usage:
    python scripts/validate_guidance.py

"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

from boltzgen.data import const
from boltzgen.model.modules.guidance import (
    DiffusionGuidance,
    HydrophobicityPredictor,
    AminoAcidFrequencyPredictor,
    ConstantPredictor,
    create_guidance,
    logits_to_aa_string,
    straight_through_softmax,
)


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(name: str, passed: bool, details: str = ""):
    """Print test result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {name}")
    if details:
        print(f"         {details}")


class GuidanceValidator:
    """Validates guidance module behavior."""
    
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.results: List[Tuple[str, bool, str]] = []
    
    def validate_gradient_direction(self) -> bool:
        """Test that gradients point toward higher predictor scores.
        
        We create logits that decode to a sequence, compute guidance gradient,
        and verify that applying the gradient increases the predictor score.
        """
        print_header("Test 1: Gradient Direction")
        
        # Create hydrophobicity predictor (higher = more hydrophobic)
        predictor = HydrophobicityPredictor(higher_is_better=True)
        guidance = DiffusionGuidance(
            predictor=predictor,
            guidance_scale=10.0,  # Strong guidance for clear effect
            temperature=0.1,      # Sharp STE
            enabled=True,
        )
        
        # Create random logits
        B, L = 2, 30
        logits = torch.randn(B, L, const.num_tokens, device=self.device)
        
        # Get initial score
        initial_score = guidance.get_predictor_scores(logits).mean()
        initial_sequences = guidance.get_predicted_sequences(logits)
        
        # Compute guidance gradient
        feats = {}
        grad = guidance.compute_sequence_guidance(logits, feats, step=5, total_steps=10)
        
        if grad is None:
            print_result("Gradient computed", False, "Gradient is None")
            return False
        
        print_result("Gradient computed", True, f"shape={grad.shape}")
        
        # Apply gradient to logits (gradient ascent for higher scores)
        # The guidance gradient points toward higher scores
        updated_logits = logits + grad
        
        # Get updated score
        updated_score = guidance.get_predictor_scores(updated_logits).mean()
        updated_sequences = guidance.get_predicted_sequences(updated_logits)
        
        score_improved = updated_score > initial_score
        
        print(f"\n  Initial sequences: {initial_sequences}")
        print(f"  Updated sequences: {updated_sequences}")
        print(f"  Initial score:     {initial_score:.4f}")
        print(f"  Updated score:     {updated_score:.4f}")
        print(f"  Improvement:       {updated_score - initial_score:.4f}")
        
        print_result(
            "Score improves after gradient step", 
            score_improved,
            f"Δ = {(updated_score - initial_score).item():.4f}"
        )
        
        return score_improved
    
    def validate_iterative_guidance(self) -> bool:
        """Test that iteratively applying guidance improves scores.
        
        Simulates multiple steps of guidance application to verify
        that scores consistently improve.
        """
        print_header("Test 2: Iterative Guidance Application")
        
        predictor = HydrophobicityPredictor(higher_is_better=True)
        guidance = DiffusionGuidance(
            predictor=predictor,
            guidance_scale=5.0,
            temperature=0.5,
            enabled=True,
        )
        
        # Start with random logits
        B, L = 1, 20
        logits = torch.randn(B, L, const.num_tokens, device=self.device)
        
        scores = []
        sequences = []
        
        num_steps = 10
        feats = {}
        
        for step in range(num_steps):
            score = guidance.get_predictor_scores(logits).mean().item()
            seq = guidance.get_predicted_sequences(logits)[0]
            scores.append(score)
            sequences.append(seq)
            
            # Apply guidance
            grad = guidance.compute_sequence_guidance(
                logits, feats, step=step, total_steps=num_steps
            )
            if grad is not None:
                logits = logits + grad
        
        # Final score
        final_score = guidance.get_predictor_scores(logits).mean().item()
        final_seq = guidance.get_predicted_sequences(logits)[0]
        scores.append(final_score)
        sequences.append(final_seq)
        
        print(f"\n  Score trajectory:")
        for i, (s, seq) in enumerate(zip(scores, sequences)):
            marker = "→" if i < len(scores) - 1 else "★"
            print(f"    Step {i:2d}: score={s:.4f}  seq={seq[:10]}... {marker}")
        
        # Check if final score > initial score
        improved = scores[-1] > scores[0]
        improvement = scores[-1] - scores[0]
        
        # Check monotonicity (most steps should improve)
        improvements = sum(1 for i in range(1, len(scores)) if scores[i] > scores[i-1])
        monotonic = improvements >= len(scores) // 2
        
        print_result(
            "Overall improvement", 
            improved,
            f"Δ = {improvement:.4f}"
        )
        print_result(
            f"Mostly monotonic ({improvements}/{len(scores)-1} steps improved)",
            monotonic
        )
        
        return improved and monotonic
    
    def validate_toggle_effect(self) -> bool:
        """Test that disabled guidance produces no effect."""
        print_header("Test 3: Guidance Toggle (On/Off)")
        
        predictor = HydrophobicityPredictor()
        
        # Same logits for both
        B, L = 2, 15
        logits = torch.randn(B, L, const.num_tokens, device=self.device)
        feats = {}
        
        # Enabled guidance
        guidance_on = DiffusionGuidance(predictor, guidance_scale=10.0, enabled=True)
        grad_on = guidance_on.compute_sequence_guidance(logits, feats)
        
        # Disabled guidance
        guidance_off = DiffusionGuidance(predictor, guidance_scale=10.0, enabled=False)
        grad_off = guidance_off.compute_sequence_guidance(logits, feats)
        
        on_has_grad = grad_on is not None and grad_on.abs().sum() > 0
        off_no_grad = grad_off is None
        
        print_result("Enabled guidance produces gradient", on_has_grad)
        print_result("Disabled guidance returns None", off_no_grad)
        
        return on_has_grad and off_no_grad
    
    def validate_schedule_effect(self) -> bool:
        """Test that schedule controls when guidance is applied."""
        print_header("Test 4: Guidance Schedule")
        
        predictor = HydrophobicityPredictor()
        
        # Guidance only starts at 50% through diffusion
        guidance = DiffusionGuidance(
            predictor,
            guidance_scale=10.0,
            schedule="linear",
            schedule_start=0.5,
            enabled=True,
        )
        
        B, L = 1, 10
        logits = torch.randn(B, L, const.num_tokens, device=self.device)
        feats = {}
        total_steps = 10
        
        # Early steps (before schedule_start): no gradient
        early_grad = guidance.compute_sequence_guidance(logits, feats, step=2, total_steps=total_steps)
        
        # Late steps (after schedule_start): has gradient
        late_grad = guidance.compute_sequence_guidance(logits, feats, step=8, total_steps=total_steps)
        
        early_zero = early_grad is None
        late_nonzero = late_grad is not None and late_grad.abs().sum() > 0
        
        print(f"\n  Step 2/10 (progress=0.2): grad={early_grad}")
        print(f"  Step 8/10 (progress=0.8): grad_norm={late_grad.norm().item() if late_grad is not None else 0:.4f}")
        
        print_result("Early steps (before schedule_start) have no gradient", early_zero)
        print_result("Late steps (after schedule_start) have gradient", late_nonzero)
        
        return early_zero and late_nonzero
    
    def validate_different_predictors(self) -> bool:
        """Test that different predictors produce different guidance."""
        print_header("Test 5: Different Predictors Produce Different Results")
        
        B, L = 1, 20
        logits = torch.randn(B, L, const.num_tokens, device=self.device)
        feats = {}
        
        # Two different predictors
        hydro_pred = HydrophobicityPredictor(higher_is_better=True)
        freq_pred = AminoAcidFrequencyPredictor(target_aas='K', higher_is_better=True)  # Target lysine
        
        guidance_hydro = DiffusionGuidance(hydro_pred, guidance_scale=5.0, enabled=True)
        guidance_freq = DiffusionGuidance(freq_pred, guidance_scale=5.0, enabled=True)
        
        grad_hydro = guidance_hydro.compute_sequence_guidance(logits, feats)
        grad_freq = guidance_freq.compute_sequence_guidance(logits, feats)
        
        if grad_hydro is None or grad_freq is None:
            print_result("Both produce gradients", False)
            return False
        
        # Gradients should be different
        grad_diff = (grad_hydro - grad_freq).abs().sum()
        grads_different = grad_diff > 1e-6
        
        print(f"\n  Hydrophobicity grad norm: {grad_hydro.norm().item():.4f}")
        print(f"  Frequency (K) grad norm:  {grad_freq.norm().item():.4f}")
        print(f"  Gradient difference:      {grad_diff.item():.4f}")
        
        print_result("Different predictors produce different gradients", grads_different)
        
        return grads_different
    
    def validate_sequence_changes(self) -> bool:
        """Test that guidance actually changes predicted sequences."""
        print_header("Test 6: Guidance Changes Predicted Sequences")
        
        predictor = AminoAcidFrequencyPredictor(target_aas='W', higher_is_better=True)  # Tryptophan
        guidance = DiffusionGuidance(
            predictor,
            guidance_scale=20.0,  # Strong guidance
            temperature=0.1,
            enabled=True,
        )
        
        B, L = 1, 15
        torch.manual_seed(42)  # For reproducibility
        logits = torch.randn(B, L, const.num_tokens, device=self.device)
        feats = {}
        
        initial_seq = guidance.get_predicted_sequences(logits)[0]
        initial_W_count = initial_seq.count('W')
        
        # Apply guidance multiple times
        for _ in range(20):
            grad = guidance.compute_sequence_guidance(logits, feats)
            if grad is not None:
                logits = logits + grad
        
        final_seq = guidance.get_predicted_sequences(logits)[0]
        final_W_count = final_seq.count('W')
        
        print(f"\n  Initial sequence: {initial_seq}")
        print(f"  Final sequence:   {final_seq}")
        print(f"  Initial W count:  {initial_W_count}")
        print(f"  Final W count:    {final_W_count}")
        
        W_increased = final_W_count >= initial_W_count
        seq_changed = initial_seq != final_seq
        
        print_result("Sequence changed", seq_changed)
        print_result("Target AA (W) count increased or maintained", W_increased)
        
        return seq_changed
    
    def run_all(self) -> bool:
        """Run all validation tests."""
        print("\n" + "=" * 60)
        print("  GUIDANCE VALIDATION SUITE")
        print("=" * 60)
        
        tests = [
            ("Gradient Direction", self.validate_gradient_direction),
            ("Iterative Guidance", self.validate_iterative_guidance),
            ("Toggle Effect", self.validate_toggle_effect),
            ("Schedule Effect", self.validate_schedule_effect),
            ("Different Predictors", self.validate_different_predictors),
            ("Sequence Changes", self.validate_sequence_changes),
        ]
        
        results = []
        for name, test_fn in tests:
            try:
                passed = test_fn()
                results.append((name, passed))
            except Exception as e:
                print(f"\n  ✗ EXCEPTION in {name}: {e}")
                results.append((name, False))
        
        # Summary
        print_header("SUMMARY")
        passed = sum(1 for _, p in results if p)
        total = len(results)
        
        for name, p in results:
            status = "✓" if p else "✗"
            print(f"  {status} {name}")
        
        print(f"\n  {passed}/{total} tests passed")
        
        all_passed = passed == total
        if all_passed:
            print("\n  ✓ All validation tests PASSED!")
        else:
            print("\n  ✗ Some validation tests FAILED")
        
        return all_passed


def main():
    """Run validation suite."""
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    validator = GuidanceValidator(device=device)
    success = validator.run_all()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
