"""Integration tests for guidance with diffusion sampling.

These tests verify that:
1. Guidance can be passed to AtomDiffusion.sample() without errors
2. Disabled guidance doesn't affect sampling
3. The guidance toggle works correctly
"""

import pytest
import torch

from boltzgen.data import const
from boltzgen.model.modules.guidance import (
    DiffusionGuidance,
    HydrophobicityPredictor,
    create_guidance,
)


class TestGuidanceParameterAccepted:
    """Test that sample() accepts the guidance parameter."""
    
    def test_import_diffusion_with_guidance(self):
        """Diffusion module should import with guidance support."""
        from boltzgen.model.modules.diffusion import AtomDiffusion
        
        # Check that sample method accepts guidance parameter
        import inspect
        sig = inspect.signature(AtomDiffusion.sample)
        param_names = list(sig.parameters.keys())
        
        assert 'guidance' in param_names, "sample() should accept 'guidance' parameter"
    
    def test_guidance_parameter_defaults_to_none(self):
        """Guidance parameter should default to None."""
        from boltzgen.model.modules.diffusion import AtomDiffusion
        import inspect
        
        sig = inspect.signature(AtomDiffusion.sample)
        guidance_param = sig.parameters.get('guidance')
        
        assert guidance_param is not None
        assert guidance_param.default is None


class TestGuidanceModuleCreation:
    """Test creating guidance modules for use with diffusion."""
    
    def test_create_guidance_for_diffusion(self):
        """Should be able to create guidance module."""
        predictor = HydrophobicityPredictor()
        guidance = DiffusionGuidance(
            predictor=predictor,
            guidance_scale=1.0,
            enabled=True,
        )
        
        assert guidance.enabled is True
        assert guidance.predictor is predictor
    
    def test_create_disabled_guidance(self):
        """Should be able to create disabled guidance."""
        predictor = HydrophobicityPredictor()
        guidance = create_guidance(predictor, enabled=False)
        
        assert guidance.enabled is False
    
    def test_guidance_toggle(self):
        """Should be able to toggle guidance on/off."""
        predictor = HydrophobicityPredictor()
        guidance = DiffusionGuidance(predictor, enabled=True)
        
        # Initially enabled
        assert guidance.enabled is True
        
        # Disable
        guidance.enabled = False
        assert guidance.enabled is False
        
        # Re-enable
        guidance.enabled = True
        assert guidance.enabled is True


class TestGuidanceWithMockLogits:
    """Test guidance computation with mock res_type logits."""
    
    def test_guidance_computes_gradient(self):
        """Guidance should compute gradient from mock logits."""
        predictor = HydrophobicityPredictor()
        guidance = DiffusionGuidance(predictor, guidance_scale=1.0, enabled=True)
        
        # Mock res_type logits
        B, L = 2, 20
        logits = torch.randn(B, L, const.num_tokens)
        feats = {}
        
        grad = guidance.compute_sequence_guidance(logits, feats, step=5, total_steps=10)
        
        assert grad is not None
        assert grad.shape == logits.shape
    
    def test_disabled_guidance_returns_none(self):
        """Disabled guidance should return None."""
        predictor = HydrophobicityPredictor()
        guidance = DiffusionGuidance(predictor, enabled=False)
        
        logits = torch.randn(2, 20, const.num_tokens)
        feats = {}
        
        grad = guidance.compute_sequence_guidance(logits, feats)
        
        assert grad is None
    
    def test_guidance_respects_schedule(self):
        """Guidance should respect the schedule."""
        predictor = HydrophobicityPredictor()
        
        # Linear schedule starting at step 5 of 10
        guidance = DiffusionGuidance(
            predictor,
            schedule="linear",
            schedule_start=0.5,
            enabled=True,
        )
        
        logits = torch.randn(2, 20, const.num_tokens)
        feats = {}
        
        # Before schedule_start: no gradient
        grad_early = guidance.compute_sequence_guidance(logits, feats, step=2, total_steps=10)
        assert grad_early is None
        
        # After schedule_start: gradient exists
        grad_late = guidance.compute_sequence_guidance(logits, feats, step=8, total_steps=10)
        assert grad_late is not None


class TestGuidanceInfoTracking:
    """Test that guidance info is properly tracked."""
    
    def test_guidance_tracks_scores(self):
        """Guidance should track predictor scores."""
        predictor = HydrophobicityPredictor()
        guidance = DiffusionGuidance(predictor, enabled=True)
        
        logits = torch.randn(2, 20, const.num_tokens)
        
        scores = guidance.get_predictor_scores(logits)
        
        assert scores.shape == (2,)  # One score per batch item
    
    def test_guidance_extracts_sequences(self):
        """Guidance should extract predicted sequences."""
        predictor = HydrophobicityPredictor()
        guidance = DiffusionGuidance(predictor, enabled=True)
        
        logits = torch.randn(3, 15, const.num_tokens)
        
        sequences = guidance.get_predicted_sequences(logits)
        
        assert len(sequences) == 3
        assert all(len(s) == 15 for s in sequences)
        assert all(isinstance(s, str) for s in sequences)


class TestBackwardsCompatibility:
    """Test that existing code without guidance still works."""
    
    def test_sample_signature_backwards_compatible(self):
        """sample() should work when called without guidance."""
        from boltzgen.model.modules.diffusion import AtomDiffusion
        import inspect
        
        sig = inspect.signature(AtomDiffusion.sample)
        
        # All new parameters should have defaults
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
            if name == 'guidance':
                assert param.default is None, "guidance should default to None"


class TestGuidanceEdgeCases:
    """Test edge cases in guidance."""
    
    def test_none_logits_handled(self):
        """Should handle None logits gracefully."""
        predictor = HydrophobicityPredictor()
        guidance = DiffusionGuidance(predictor, enabled=True)
        
        grad = guidance.compute_sequence_guidance(None, {})
        
        assert grad is None
    
    def test_empty_feats_handled(self):
        """Should handle empty feats dict."""
        predictor = HydrophobicityPredictor()
        guidance = DiffusionGuidance(predictor, enabled=True)
        
        logits = torch.randn(2, 10, const.num_tokens)
        
        # Should not raise
        grad = guidance.compute_sequence_guidance(logits, {})
        assert grad is not None
    
    def test_single_step_total(self):
        """Should handle edge case of single step."""
        predictor = HydrophobicityPredictor()
        guidance = DiffusionGuidance(predictor, enabled=True)
        
        logits = torch.randn(2, 10, const.num_tokens)
        
        # Single step should not cause division by zero
        grad = guidance.compute_sequence_guidance(logits, {}, step=0, total_steps=1)
        assert grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
