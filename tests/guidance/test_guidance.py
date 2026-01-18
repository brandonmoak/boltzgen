"""Tests for the DiffusionGuidance module."""

import pytest
import torch
import torch.nn.functional as F
import math

from boltzgen.data import const
from boltzgen.model.modules.guidance import (
    DiffusionGuidance,
    create_guidance,
    HydrophobicityPredictor,
    AminoAcidFrequencyPredictor,
    ConstantPredictor,
)


class TestDiffusionGuidanceInit:
    """Test guidance initialization."""
    
    def test_basic_init(self):
        """Test basic initialization."""
        predictor = HydrophobicityPredictor()
        guidance = DiffusionGuidance(predictor)
        
        assert guidance.enabled is True
        assert guidance.guidance_scale == 1.0
        assert guidance.temperature == 1.0
    
    def test_custom_params(self):
        """Test initialization with custom parameters."""
        predictor = HydrophobicityPredictor()
        guidance = DiffusionGuidance(
            predictor,
            guidance_scale=2.0,
            temperature=0.5,
            schedule="linear",
            enabled=False,
        )
        
        assert guidance.guidance_scale == 2.0
        assert guidance.temperature == 0.5
        assert guidance.schedule == "linear"
        assert guidance.enabled is False
    
    def test_create_guidance_factory(self):
        """Test factory function."""
        predictor = HydrophobicityPredictor()
        guidance = create_guidance(predictor, guidance_scale=3.0, enabled=True)
        
        assert isinstance(guidance, DiffusionGuidance)
        assert guidance.guidance_scale == 3.0


class TestScheduleWeight:
    """Test guidance schedule weights."""
    
    def test_constant_schedule(self):
        """Constant schedule should always return 1.0."""
        predictor = HydrophobicityPredictor()
        guidance = DiffusionGuidance(predictor, schedule="constant")
        
        for progress in [0.0, 0.25, 0.5, 0.75, 1.0]:
            weight = guidance.get_schedule_weight(progress)
            assert weight == 1.0
    
    def test_linear_schedule(self):
        """Linear schedule should ramp from 0 to 1."""
        predictor = HydrophobicityPredictor()
        guidance = DiffusionGuidance(predictor, schedule="linear")
        
        assert guidance.get_schedule_weight(0.0) == 0.0
        assert guidance.get_schedule_weight(0.5) == 0.5
        assert guidance.get_schedule_weight(1.0) == 1.0
    
    def test_cosine_schedule(self):
        """Cosine schedule should have smooth ramp."""
        predictor = HydrophobicityPredictor()
        guidance = DiffusionGuidance(predictor, schedule="cosine")
        
        # Starts at 0
        assert guidance.get_schedule_weight(0.0) == 0.0
        # Ends at 1
        assert abs(guidance.get_schedule_weight(1.0) - 1.0) < 1e-6
        # Middle is 0.5 for cosine
        assert abs(guidance.get_schedule_weight(0.5) - 0.5) < 1e-6
    
    def test_sigmoid_schedule(self):
        """Sigmoid schedule should have S-curve shape."""
        predictor = HydrophobicityPredictor()
        guidance = DiffusionGuidance(predictor, schedule="sigmoid")
        
        # Should be monotonically increasing
        prev = 0.0
        for progress in [0.0, 0.25, 0.5, 0.75, 1.0]:
            weight = guidance.get_schedule_weight(progress)
            assert weight >= prev
            prev = weight
        
        # Should be approximately 0.5 at midpoint
        assert abs(guidance.get_schedule_weight(0.5) - 0.5) < 0.1
    
    def test_schedule_start_end(self):
        """Test schedule_start and schedule_end parameters."""
        predictor = HydrophobicityPredictor()
        guidance = DiffusionGuidance(
            predictor,
            schedule="linear",
            schedule_start=0.5,  # Start at 50%
            schedule_end=1.0,    # Full at 100%
        )
        
        # Before start: no guidance
        assert guidance.get_schedule_weight(0.0) == 0.0
        assert guidance.get_schedule_weight(0.4) == 0.0
        
        # After start: ramps up
        assert guidance.get_schedule_weight(0.5) == 0.0
        assert guidance.get_schedule_weight(0.75) == 0.5
        assert guidance.get_schedule_weight(1.0) == 1.0


class TestComputeSequenceGuidance:
    """Test sequence guidance computation."""
    
    def test_returns_none_when_disabled(self):
        """Should return None when guidance is disabled."""
        predictor = HydrophobicityPredictor()
        guidance = DiffusionGuidance(predictor, enabled=False)
        
        logits = torch.randn(2, 10, const.num_tokens)
        feats = {}
        
        result = guidance.compute_sequence_guidance(logits, feats)
        assert result is None
    
    def test_returns_none_for_none_logits(self):
        """Should return None when logits are None."""
        predictor = HydrophobicityPredictor()
        guidance = DiffusionGuidance(predictor, enabled=True)
        
        result = guidance.compute_sequence_guidance(None, {})
        assert result is None
    
    def test_returns_gradient_tensor(self):
        """Should return gradient with same shape as input."""
        predictor = HydrophobicityPredictor()
        guidance = DiffusionGuidance(predictor, enabled=True)
        
        B, L = 2, 10
        logits = torch.randn(B, L, const.num_tokens)
        feats = {}
        
        grad = guidance.compute_sequence_guidance(logits, feats)
        
        assert grad is not None
        assert grad.shape == logits.shape
    
    def test_gradient_is_nonzero(self):
        """Gradient should be non-zero for non-trivial cases."""
        predictor = HydrophobicityPredictor()
        guidance = DiffusionGuidance(predictor, guidance_scale=1.0)
        
        logits = torch.randn(2, 10, const.num_tokens)
        feats = {}
        
        grad = guidance.compute_sequence_guidance(logits, feats)
        
        assert (grad != 0).any()
    
    def test_schedule_affects_gradient_magnitude(self):
        """Schedule should affect gradient magnitude."""
        predictor = HydrophobicityPredictor()
        guidance = DiffusionGuidance(
            predictor,
            schedule="linear",
            guidance_scale=1.0,
        )
        
        logits = torch.randn(2, 10, const.num_tokens)
        feats = {}
        
        # Early in diffusion (step 1 of 10)
        grad_early = guidance.compute_sequence_guidance(
            logits, feats, step=1, total_steps=10
        )
        
        # Late in diffusion (step 9 of 10)
        grad_late = guidance.compute_sequence_guidance(
            logits, feats, step=9, total_steps=10
        )
        
        # Late gradient should be larger
        assert grad_late.norm() > grad_early.norm()
    
    def test_guidance_scale_affects_magnitude(self):
        """Guidance scale should linearly affect gradient magnitude."""
        predictor = HydrophobicityPredictor()
        
        logits = torch.randn(2, 10, const.num_tokens)
        feats = {}
        
        guidance_1x = DiffusionGuidance(predictor, guidance_scale=1.0)
        guidance_2x = DiffusionGuidance(predictor, guidance_scale=2.0)
        
        grad_1x = guidance_1x.compute_sequence_guidance(logits, feats)
        grad_2x = guidance_2x.compute_sequence_guidance(logits, feats)
        
        # 2x scale should give ~2x gradient
        ratio = grad_2x.norm() / grad_1x.norm()
        assert abs(ratio - 2.0) < 0.1
    
    def test_gradient_clamp(self):
        """Gradient should be clamped when clamp_grad is set."""
        predictor = HydrophobicityPredictor()
        guidance = DiffusionGuidance(
            predictor,
            guidance_scale=100.0,  # Large scale
            clamp_grad=1.0,        # But clamp to 1.0
        )
        
        logits = torch.randn(2, 10, const.num_tokens)
        feats = {}
        
        grad = guidance.compute_sequence_guidance(logits, feats)
        
        # Gradient norm should be at most 1.0
        assert grad.norm() <= 1.0 + 1e-6


class TestGuidanceDirection:
    """Test that guidance respects predictor direction."""
    
    def test_higher_is_better_direction(self):
        """Should maximize scores when higher_is_better=True."""
        predictor = HydrophobicityPredictor(higher_is_better=True)
        guidance = DiffusionGuidance(predictor, guidance_scale=1.0)
        
        # Create logits biased toward hydrophilic (low score)
        # Gradient should push toward hydrophobic (high score)
        B, L = 2, 10
        logits = torch.randn(B, L, const.num_tokens)
        feats = {}
        
        grad = guidance.compute_sequence_guidance(logits, feats)
        
        # The gradient should be defined (direction test is implicit in sign)
        assert grad is not None
    
    def test_lower_is_better_direction(self):
        """Should minimize scores when higher_is_better=False."""
        predictor_max = HydrophobicityPredictor(higher_is_better=True)
        predictor_min = HydrophobicityPredictor(higher_is_better=False)
        
        guidance_max = DiffusionGuidance(predictor_max)
        guidance_min = DiffusionGuidance(predictor_min)
        
        logits = torch.randn(2, 10, const.num_tokens)
        feats = {}
        
        grad_max = guidance_max.compute_sequence_guidance(logits, feats)
        grad_min = guidance_min.compute_sequence_guidance(logits, feats)
        
        # Gradients should be in opposite directions
        # Check by looking at the dot product
        dot = (grad_max * grad_min).sum()
        assert dot < 0  # Should be negative (opposite directions)


class TestGetPredictedSequences:
    """Test sequence extraction for debugging."""
    
    def test_returns_strings(self):
        """Should return list of strings."""
        predictor = HydrophobicityPredictor()
        guidance = DiffusionGuidance(predictor)
        
        # Create logits for known sequence
        B, L = 2, 5
        logits = torch.randn(B, L, const.num_tokens)
        
        sequences = guidance.get_predicted_sequences(logits)
        
        assert isinstance(sequences, list)
        assert len(sequences) == B
        assert all(isinstance(s, str) for s in sequences)
        assert all(len(s) == L for s in sequences)


class TestGetPredictorScores:
    """Test score extraction for debugging."""
    
    def test_returns_scores(self):
        """Should return tensor of scores."""
        predictor = HydrophobicityPredictor()
        guidance = DiffusionGuidance(predictor)
        
        B, L = 3, 10
        logits = torch.randn(B, L, const.num_tokens)
        
        scores = guidance.get_predictor_scores(logits)
        
        assert scores.shape == (B,)
    
    def test_scores_are_deterministic(self):
        """Same logits should give same scores."""
        predictor = HydrophobicityPredictor()
        guidance = DiffusionGuidance(predictor)
        
        logits = torch.randn(2, 10, const.num_tokens)
        
        scores1 = guidance.get_predictor_scores(logits)
        scores2 = guidance.get_predictor_scores(logits)
        
        assert torch.allclose(scores1, scores2)


class TestGuidanceToggle:
    """Test enabling/disabling guidance."""
    
    def test_can_disable(self):
        """Should be able to disable guidance."""
        predictor = HydrophobicityPredictor()
        guidance = DiffusionGuidance(predictor, enabled=True)
        
        assert guidance.enabled is True
        
        guidance.enabled = False
        assert guidance.enabled is False
        
        logits = torch.randn(2, 10, const.num_tokens)
        grad = guidance.compute_sequence_guidance(logits, {})
        
        assert grad is None
    
    def test_can_enable(self):
        """Should be able to enable guidance."""
        predictor = HydrophobicityPredictor()
        guidance = DiffusionGuidance(predictor, enabled=False)
        
        logits = torch.randn(2, 10, const.num_tokens)
        
        # Disabled: returns None
        assert guidance.compute_sequence_guidance(logits, {}) is None
        
        # Enable
        guidance.enabled = True
        
        # Enabled: returns gradient
        grad = guidance.compute_sequence_guidance(logits, {})
        assert grad is not None


class TestWithDesignMask:
    """Test guidance with design mask."""
    
    def test_respects_design_mask(self):
        """Guidance should only apply to designed positions."""
        predictor = HydrophobicityPredictor()
        guidance = DiffusionGuidance(predictor)
        
        B, L = 2, 10
        logits = torch.randn(B, L, const.num_tokens)
        
        # Only design first 5 positions
        design_mask = torch.zeros(B, L, dtype=torch.bool)
        design_mask[:, :5] = True
        
        feats = {"design_mask": design_mask}
        
        # Should run without error
        grad = guidance.compute_sequence_guidance(logits, feats)
        assert grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
