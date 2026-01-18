"""Tests for predictor interface and mock predictors."""

import pytest
import torch
import torch.nn.functional as F

from boltzgen.data import const
from boltzgen.model.modules.guidance import (
    SequencePredictor,
    EnsemblePredictor,
    HydrophobicityPredictor,
    SequenceLengthPredictor,
    ConstantPredictor,
    AminoAcidFrequencyPredictor,
    straight_through_softmax,
    aa_string_to_indices,
)


class TestHydrophobicityPredictor:
    """Test hydrophobicity predictor."""
    
    def test_string_input(self):
        """Test with string input."""
        pred = HydrophobicityPredictor()
        score = pred("AAAA")
        
        assert score.shape == (1,)
        assert isinstance(score, torch.Tensor)
    
    def test_batch_string_input(self):
        """Test with batch of strings."""
        pred = HydrophobicityPredictor()
        scores = pred(["AAAA", "MMMM", "KKKK"])
        
        assert scores.shape == (3,)
    
    def test_tensor_input_indices(self):
        """Test with tensor indices."""
        pred = HydrophobicityPredictor()
        indices = aa_string_to_indices(["AAAA", "IIII"])
        scores = pred(indices)
        
        assert scores.shape == (2,)
    
    def test_tensor_input_one_hot(self):
        """Test with one-hot tensor."""
        pred = HydrophobicityPredictor()
        indices = aa_string_to_indices("AAAA")
        one_hot = F.one_hot(indices, num_classes=const.num_tokens).float()
        one_hot = one_hot.unsqueeze(0)  # Add batch dim
        
        scores = pred(one_hot)
        assert scores.shape == (1,)
    
    def test_hydrophobic_vs_hydrophilic(self):
        """Hydrophobic sequences should score higher."""
        pred = HydrophobicityPredictor(higher_is_better=True)
        
        # Isoleucine (I) is very hydrophobic
        # Lysine (K) is very hydrophilic
        hydrophobic_score = pred("IIIIII")
        hydrophilic_score = pred("KKKKKK")
        
        assert hydrophobic_score > hydrophilic_score
    
    def test_gradient_flow_with_ste(self):
        """Gradients should flow through STE to logits."""
        pred = HydrophobicityPredictor()
        
        # Create logits that require grad
        B, L = 2, 10
        logits = torch.randn(B, L, const.num_tokens, requires_grad=True)
        
        # Apply STE to get differentiable one-hot
        soft = F.softmax(logits, dim=-1)
        hard = F.one_hot(logits.argmax(dim=-1), num_classes=const.num_tokens).float()
        one_hot_ste = hard - soft.detach() + soft
        
        # Get score
        scores = pred(one_hot_ste)
        loss = scores.sum()
        loss.backward()
        
        # Gradients should exist
        assert logits.grad is not None
        assert (logits.grad != 0).any()
    
    def test_mask_handling(self):
        """Mask should exclude positions from average."""
        pred = HydrophobicityPredictor()
        
        indices = aa_string_to_indices("IIIKK")  # I=hydrophobic, K=hydrophilic
        
        # Full sequence
        full_score = pred(indices.unsqueeze(0))
        
        # Mask out the K's (last 2 positions)
        mask = torch.tensor([[True, True, True, False, False]])
        masked_score = pred(indices.unsqueeze(0), mask=mask)
        
        # Masked score should be higher (only I's counted)
        assert masked_score > full_score


class TestSequenceLengthPredictor:
    """Test length predictor."""
    
    def test_string_input(self):
        """Test with string input."""
        pred = SequenceLengthPredictor(target_length=10)
        
        score_exact = pred("A" * 10)
        score_short = pred("A" * 5)
        
        # Exact match should have highest score (0)
        assert score_exact > score_short
    
    def test_target_length(self):
        """Score should be highest at target length."""
        pred = SequenceLengthPredictor(target_length=20)
        
        scores = []
        for length in [10, 15, 20, 25, 30]:
            score = pred("A" * length)
            scores.append(score.item())
        
        # 20 should have the highest score (index 2)
        assert scores.index(max(scores)) == 2


class TestConstantPredictor:
    """Test constant predictor."""
    
    def test_returns_constant(self):
        """Should always return the same value."""
        pred = ConstantPredictor(value=42.0)
        
        assert pred("A").item() == 42.0
        assert pred("AAAAAAA").item() == 42.0
        assert pred(["A", "AA", "AAA"]).tolist() == [42.0, 42.0, 42.0]
    
    def test_no_gradient(self):
        """Constant predictor returns values independent of input."""
        pred = ConstantPredictor(value=1.0)
        
        # The constant predictor doesn't depend on input at all
        # So it doesn't have a gradient path - the output tensor
        # is created fresh and doesn't track the input
        logits = torch.randn(2, 5, const.num_tokens, requires_grad=True)
        one_hot = F.one_hot(logits.argmax(dim=-1), num_classes=const.num_tokens).float()
        
        scores = pred(one_hot)
        
        # The scores tensor doesn't require grad because it's not connected to input
        assert not scores.requires_grad
        
        # Different inputs give same output
        scores2 = pred(torch.randn(2, 5, const.num_tokens))
        assert torch.allclose(scores, scores2)


class TestAminoAcidFrequencyPredictor:
    """Test AA frequency predictor."""
    
    def test_target_aa_detection(self):
        """Should correctly count target AAs."""
        pred = AminoAcidFrequencyPredictor(target_aas="K")
        
        # 100% K
        score_all_k = pred("KKKK")
        # 50% K
        score_half_k = pred("KKAA")
        # 0% K
        score_no_k = pred("AAAA")
        
        assert score_all_k > score_half_k > score_no_k
        assert torch.isclose(score_all_k, torch.tensor(1.0), atol=0.01)
        assert torch.isclose(score_no_k, torch.tensor(0.0), atol=0.01)
    
    def test_multiple_target_aas(self):
        """Should count multiple target AAs."""
        pred = AminoAcidFrequencyPredictor(target_aas="KR")  # Positively charged
        
        score_kr = pred("KKRR")  # All target
        score_mixed = pred("KARA")  # 50% target
        score_none = pred("AAAA")  # No target
        
        assert score_kr > score_mixed > score_none
    
    def test_gradient_flow(self):
        """Gradients should flow through to logits."""
        pred = AminoAcidFrequencyPredictor(target_aas="K")
        
        B, L = 2, 10
        logits = torch.randn(B, L, const.num_tokens, requires_grad=True)
        
        # Apply STE
        soft = F.softmax(logits, dim=-1)
        hard = F.one_hot(logits.argmax(dim=-1), num_classes=const.num_tokens).float()
        one_hot_ste = hard - soft.detach() + soft
        
        scores = pred(one_hot_ste)
        loss = scores.sum()
        loss.backward()
        
        assert logits.grad is not None
        assert (logits.grad != 0).any()


class TestEnsemblePredictor:
    """Test ensemble predictor."""
    
    def test_combines_predictors(self):
        """Ensemble should combine predictor scores."""
        pred1 = ConstantPredictor(value=1.0)
        pred2 = ConstantPredictor(value=2.0)
        
        ensemble = EnsemblePredictor([pred1, pred2], weights=[1.0, 1.0])
        score = ensemble("AAA")
        
        # Equal weights: (1.0 + 2.0) / 2 = 1.5
        # But weights are normalized: [0.5, 0.5]
        # So: 0.5 * 1.0 + 0.5 * 2.0 = 1.5
        assert torch.isclose(score, torch.tensor([1.5]), atol=0.01)
    
    def test_weight_normalization(self):
        """Weights should be normalized."""
        pred1 = ConstantPredictor(value=1.0)
        pred2 = ConstantPredictor(value=1.0)
        
        # Weights [2.0, 2.0] should normalize to [0.5, 0.5]
        ensemble = EnsemblePredictor([pred1, pred2], weights=[2.0, 2.0])
        score = ensemble("AAA")
        
        assert torch.isclose(score, torch.tensor([1.0]), atol=0.01)
    
    def test_gradient_flow(self):
        """Gradients should flow through ensemble."""
        pred1 = HydrophobicityPredictor()
        pred2 = AminoAcidFrequencyPredictor(target_aas="K")
        
        ensemble = EnsemblePredictor([pred1, pred2])
        
        logits = torch.randn(2, 5, const.num_tokens, requires_grad=True)
        soft = F.softmax(logits, dim=-1)
        hard = F.one_hot(logits.argmax(dim=-1), num_classes=const.num_tokens).float()
        one_hot_ste = hard - soft.detach() + soft
        
        scores = ensemble(one_hot_ste)
        loss = scores.sum()
        loss.backward()
        
        assert logits.grad is not None
        assert (logits.grad != 0).any()
    
    def test_direction_handling(self):
        """Should handle mixed higher/lower is better."""
        # Maximize hydrophobicity, minimize charged residue frequency
        pred_hydro = HydrophobicityPredictor(higher_is_better=True)
        pred_charge = AminoAcidFrequencyPredictor(target_aas="KR", higher_is_better=False)
        
        ensemble = EnsemblePredictor([pred_hydro, pred_charge])
        
        # Isoleucine: hydrophobic, no charge -> should score high
        # Lysine: hydrophilic, charged -> should score low
        score_i = ensemble("IIII")
        score_k = ensemble("KKKK")
        
        # I should score higher on both metrics
        assert score_i > score_k


class TestPredictorInterface:
    """Test predictor interface properties."""
    
    def test_higher_is_better_default(self):
        """Default should be higher is better."""
        pred = HydrophobicityPredictor()
        assert pred.higher_is_better is True
        assert pred.get_guidance_direction() == 1.0
    
    def test_higher_is_better_false(self):
        """Can set higher_is_better to False."""
        pred = HydrophobicityPredictor(higher_is_better=False)
        assert pred.higher_is_better is False
        assert pred.get_guidance_direction() == -1.0
    
    def test_name_property(self):
        """Predictors should have meaningful names."""
        assert "Hydrophobicity" in HydrophobicityPredictor().name
        assert "Length" in SequenceLengthPredictor().name
        assert "Constant" in ConstantPredictor().name


class TestGradientMagnitudes:
    """Test that gradient magnitudes are reasonable."""
    
    def test_hydrophobicity_gradient_magnitude(self):
        """Gradients should be reasonable magnitude for guidance."""
        pred = HydrophobicityPredictor()
        
        logits = torch.randn(4, 20, const.num_tokens, requires_grad=True)
        
        soft = F.softmax(logits, dim=-1)
        hard = F.one_hot(logits.argmax(dim=-1), num_classes=const.num_tokens).float()
        one_hot_ste = hard - soft.detach() + soft
        
        scores = pred(one_hot_ste)
        loss = scores.sum()
        loss.backward()
        
        grad_norm = logits.grad.norm()
        
        # Gradient should be non-trivial but not exploding
        assert grad_norm > 0.01, "Gradient too small"
        assert grad_norm < 1000, "Gradient too large"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
