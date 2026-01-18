"""Integration tests for guidance with Boltz model.

These tests verify that guidance is properly wired through the model
without requiring a checkpoint or running full inference.
"""

import pytest
import inspect
from unittest.mock import Mock, patch, MagicMock
import torch

from boltzgen.data import const
from boltzgen.model.modules.guidance import (
    DiffusionGuidance,
    HydrophobicityPredictor,
    create_guidance,
)


class TestBoltzForwardSignature:
    """Test that Boltz.forward() accepts guidance parameter."""
    
    def test_boltz_forward_has_guidance_parameter(self):
        """Boltz.forward() should have a guidance parameter."""
        from boltzgen.model.models.boltz import Boltz
        
        sig = inspect.signature(Boltz.forward)
        param_names = list(sig.parameters.keys())
        
        assert 'guidance' in param_names, "Boltz.forward() should accept 'guidance'"
    
    def test_guidance_defaults_to_none(self):
        """Guidance parameter should default to None."""
        from boltzgen.model.models.boltz import Boltz
        
        sig = inspect.signature(Boltz.forward)
        guidance_param = sig.parameters.get('guidance')
        
        assert guidance_param is not None
        assert guidance_param.default is None, "guidance should default to None"
    
    def test_guidance_is_keyword_only_or_has_default(self):
        """Guidance should be safe to omit (backwards compatible)."""
        from boltzgen.model.models.boltz import Boltz
        
        sig = inspect.signature(Boltz.forward)
        guidance_param = sig.parameters.get('guidance')
        
        # Should have a default value for backwards compatibility
        assert guidance_param.default is not inspect.Parameter.empty, \
            "guidance should have a default value for backwards compatibility"


class TestAtomDiffusionSampleSignature:
    """Test that AtomDiffusion.sample() accepts guidance parameter."""
    
    def test_sample_has_guidance_parameter(self):
        """AtomDiffusion.sample() should have a guidance parameter."""
        from boltzgen.model.modules.diffusion import AtomDiffusion
        
        sig = inspect.signature(AtomDiffusion.sample)
        param_names = list(sig.parameters.keys())
        
        assert 'guidance' in param_names, "sample() should accept 'guidance'"
    
    def test_sample_guidance_defaults_to_none(self):
        """sample() guidance parameter should default to None."""
        from boltzgen.model.modules.diffusion import AtomDiffusion
        
        sig = inspect.signature(AtomDiffusion.sample)
        guidance_param = sig.parameters.get('guidance')
        
        assert guidance_param is not None
        assert guidance_param.default is None


class TestGuidanceWiring:
    """Test that guidance is properly wired through the model."""
    
    def test_guidance_module_can_be_created(self):
        """Should be able to create a guidance module."""
        predictor = HydrophobicityPredictor()
        guidance = DiffusionGuidance(predictor, guidance_scale=1.0, enabled=True)
        
        assert guidance is not None
        assert guidance.enabled is True
    
    def test_disabled_guidance_module(self):
        """Should be able to create disabled guidance."""
        predictor = HydrophobicityPredictor()
        guidance = create_guidance(predictor, guidance_scale=1.0, enabled=False)
        
        assert guidance is not None
        assert guidance.enabled is False
    
    def test_guidance_toggle(self):
        """Should be able to toggle guidance on/off."""
        predictor = HydrophobicityPredictor()
        guidance = DiffusionGuidance(predictor, enabled=True)
        
        assert guidance.enabled is True
        guidance.enabled = False
        assert guidance.enabled is False
        guidance.enabled = True
        assert guidance.enabled is True


class TestGuidanceWithMockModel:
    """Test guidance integration using mocked model components."""
    
    def test_guidance_computes_without_model(self):
        """Guidance should compute gradients from mock logits."""
        predictor = HydrophobicityPredictor()
        guidance = DiffusionGuidance(predictor, guidance_scale=1.0, enabled=True)
        
        # Mock res_type logits as would come from model
        B, L = 2, 50
        logits = torch.randn(B, L, const.num_tokens)
        feats = {}
        
        grad = guidance.compute_sequence_guidance(logits, feats, step=5, total_steps=10)
        
        assert grad is not None
        assert grad.shape == logits.shape
    
    def test_disabled_guidance_returns_none(self):
        """Disabled guidance should return None gradient."""
        predictor = HydrophobicityPredictor()
        guidance = DiffusionGuidance(predictor, enabled=False)
        
        logits = torch.randn(2, 50, const.num_tokens)
        feats = {}
        
        grad = guidance.compute_sequence_guidance(logits, feats)
        
        assert grad is None


class TestBackwardsCompatibility:
    """Test that existing code without guidance still works."""
    
    def test_boltz_forward_signature_allows_omitting_guidance(self):
        """Calling forward() without guidance should be valid."""
        from boltzgen.model.models.boltz import Boltz
        
        sig = inspect.signature(Boltz.forward)
        
        # Get all required parameters (no default)
        required_params = [
            name for name, param in sig.parameters.items()
            if param.default is inspect.Parameter.empty and name != 'self'
        ]
        
        # guidance should NOT be in required params
        assert 'guidance' not in required_params, \
            "guidance should not be required (breaks backwards compatibility)"
    
    def test_sample_signature_allows_omitting_guidance(self):
        """Calling sample() without guidance should be valid."""
        from boltzgen.model.modules.diffusion import AtomDiffusion
        
        sig = inspect.signature(AtomDiffusion.sample)
        
        required_params = [
            name for name, param in sig.parameters.items()
            if param.default is inspect.Parameter.empty and name != 'self'
        ]
        
        assert 'guidance' not in required_params


class TestGuidanceInfoInOutput:
    """Test that guidance info is included in sample() output."""
    
    def test_guidance_info_structure(self):
        """Guidance info should have expected structure."""
        predictor = HydrophobicityPredictor()
        guidance = DiffusionGuidance(predictor, enabled=True)
        
        # Simulate what happens in diffusion loop
        logits = torch.randn(1, 20, const.num_tokens)
        feats = {}
        
        grad = guidance.compute_sequence_guidance(logits, feats, step=0, total_steps=10)
        
        # Should be able to get scores and sequences for logging
        scores = guidance.get_predictor_scores(logits)
        sequences = guidance.get_predicted_sequences(logits)
        
        assert scores.shape == (1,)
        assert len(sequences) == 1
        assert isinstance(sequences[0], str)


class TestGuidanceSchedules:
    """Test various guidance schedule configurations."""
    
    @pytest.mark.parametrize("schedule", ["constant", "linear", "cosine", "sigmoid"])
    def test_schedule_types(self, schedule):
        """All schedule types should work."""
        predictor = HydrophobicityPredictor()
        guidance = DiffusionGuidance(
            predictor,
            guidance_scale=1.0,
            schedule=schedule,
            enabled=True,
        )
        
        logits = torch.randn(1, 10, const.num_tokens)
        feats = {}
        
        # Should not raise
        grad = guidance.compute_sequence_guidance(logits, feats, step=5, total_steps=10)
        # May be None for some schedules if before schedule_start
        # but should not raise
    
    def test_schedule_start_end(self):
        """Schedule start/end should control when guidance is applied."""
        predictor = HydrophobicityPredictor()
        guidance = DiffusionGuidance(
            predictor,
            schedule="linear",
            schedule_start=0.5,  # Start at 50%
            schedule_end=1.0,
            enabled=True,
        )
        
        logits = torch.randn(1, 10, const.num_tokens)
        feats = {}
        
        # Before schedule_start (step 2 of 10 = 20%)
        grad_early = guidance.compute_sequence_guidance(logits, feats, step=2, total_steps=10)
        assert grad_early is None
        
        # After schedule_start (step 8 of 10 = 80%)
        grad_late = guidance.compute_sequence_guidance(logits, feats, step=8, total_steps=10)
        assert grad_late is not None


class TestGuidanceGradientClamping:
    """Test gradient clamping functionality."""
    
    def test_clamp_grad_limits_magnitude(self):
        """Gradient clamping should limit gradient magnitude."""
        predictor = HydrophobicityPredictor()
        
        # Very low clamp value
        guidance = DiffusionGuidance(
            predictor,
            guidance_scale=100.0,  # High scale to produce large gradients
            clamp_grad=0.1,        # But clamp them
            enabled=True,
        )
        
        logits = torch.randn(1, 20, const.num_tokens)
        feats = {}
        
        grad = guidance.compute_sequence_guidance(logits, feats)
        
        if grad is not None:
            assert grad.norm().item() <= 0.1 + 1e-6  # Allow small numerical error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
