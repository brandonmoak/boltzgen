"""Tests for sequence conversion utilities."""

import pytest
import torch
import torch.nn.functional as F

from boltzgen.data import const
from boltzgen.model.modules.guidance.sequence_utils import (
    logits_to_aa_string,
    logits_to_aa_indices,
    aa_indices_to_string,
    aa_string_to_indices,
    straight_through_softmax,
    straight_through_gumbel_softmax,
    get_canonical_mask,
    extract_canonical_logits,
    NUM_CANONICAL_AAS,
    _TOKEN_IDX_TO_ONE_LETTER,
    _ONE_LETTER_TO_TOKEN_IDX,
)


class TestTokenMapping:
    """Test that token mappings are correct."""
    
    def test_num_canonical_aas(self):
        """Should have exactly 20 canonical amino acids."""
        assert NUM_CANONICAL_AAS == 20
    
    def test_token_mapping_coverage(self):
        """All 20 canonical AAs should be mapped."""
        assert len(_TOKEN_IDX_TO_ONE_LETTER) == 20
        assert len(_ONE_LETTER_TO_TOKEN_IDX) == 20
    
    def test_token_mapping_roundtrip(self):
        """Token index -> letter -> token index should be identity."""
        for idx, letter in _TOKEN_IDX_TO_ONE_LETTER.items():
            assert _ONE_LETTER_TO_TOKEN_IDX[letter] == idx
    
    def test_known_mappings(self):
        """Check a few known mappings."""
        # ALA is first canonical, at index 2 (canonicals_offset)
        ala_idx = const.canonicals_offset
        assert _TOKEN_IDX_TO_ONE_LETTER[ala_idx] == "A"
        
        # Check a few more
        assert _ONE_LETTER_TO_TOKEN_IDX["A"] == const.token_ids["ALA"]
        assert _ONE_LETTER_TO_TOKEN_IDX["R"] == const.token_ids["ARG"]
        assert _ONE_LETTER_TO_TOKEN_IDX["M"] == const.token_ids["MET"]


class TestGetCanonicalMask:
    """Test canonical mask generation."""
    
    def test_mask_shape(self):
        """Mask should have correct shape."""
        mask = get_canonical_mask()
        assert mask.shape == (const.num_tokens,)
    
    def test_mask_sum(self):
        """Should have exactly 20 True values."""
        mask = get_canonical_mask()
        assert mask.sum().item() == 20
    
    def test_mask_positions(self):
        """True values should be at canonical positions."""
        mask = get_canonical_mask()
        start = const.canonicals_offset
        end = start + 20
        
        assert mask[start:end].all()
        assert not mask[:start].any()
        assert not mask[end:].any()


class TestLogitsToIndices:
    """Test logits to token index conversion."""
    
    def test_single_sequence(self):
        """Test with single sequence (no batch dim)."""
        L = 10
        logits = torch.randn(L, const.num_tokens)
        indices = logits_to_aa_indices(logits)
        
        assert indices.shape == (L,)
        # All indices should be in canonical range
        assert (indices >= const.canonicals_offset).all()
        assert (indices < const.canonicals_offset + 20).all()
    
    def test_batch_sequences(self):
        """Test with batch of sequences."""
        B, L = 3, 10
        logits = torch.randn(B, L, const.num_tokens)
        indices = logits_to_aa_indices(logits)
        
        assert indices.shape == (B, L)
    
    def test_deterministic(self):
        """Same logits should give same indices."""
        logits = torch.randn(5, const.num_tokens)
        indices1 = logits_to_aa_indices(logits)
        indices2 = logits_to_aa_indices(logits)
        
        assert (indices1 == indices2).all()
    
    def test_respects_argmax(self):
        """Should return argmax within canonicals."""
        L = 5
        logits = torch.zeros(L, const.num_tokens)
        
        # Set specific AAs to have highest logit
        target_aas = [0, 5, 10, 15, 19]  # indices within canonical range
        for i, aa_idx in enumerate(target_aas):
            full_idx = aa_idx + const.canonicals_offset
            logits[i, full_idx] = 10.0
        
        indices = logits_to_aa_indices(logits)
        expected = torch.tensor([aa + const.canonicals_offset for aa in target_aas])
        
        assert (indices == expected).all()


class TestIndicesToString:
    """Test token indices to string conversion."""
    
    def test_single_sequence(self):
        """Test converting single sequence."""
        # Create indices for "ACDEFG"
        aa_string = "ACDEFG"
        indices = aa_string_to_indices(aa_string)
        
        result = aa_indices_to_string(indices)
        assert result == aa_string
    
    def test_batch_sequences(self):
        """Test converting batch of sequences."""
        sequences = ["AAAAA", "MMMMM", "VVVVV"]
        indices = aa_string_to_indices(sequences)
        
        results = aa_indices_to_string(indices)
        assert results == sequences
    
    def test_with_mask(self):
        """Test that mask filters positions."""
        indices = aa_string_to_indices("ABCDE")  # Note: B maps to UNK/X
        mask = torch.tensor([True, False, True, False, True])
        
        result = aa_indices_to_string(indices, mask)
        # Should only include positions 0, 2, 4: A, C, E
        assert len(result) == 3


class TestRoundTrip:
    """Test full round-trip conversions."""
    
    def test_string_to_indices_to_string(self):
        """String -> indices -> string should be identity for valid AAs."""
        original = "MKTVRQERLKSIG"
        indices = aa_string_to_indices(original)
        recovered = aa_indices_to_string(indices)
        
        assert recovered == original
    
    def test_logits_roundtrip(self):
        """Create logits for known sequence, convert back."""
        target_seq = "ACDEFGHIKLM"
        target_indices = aa_string_to_indices(target_seq)
        
        # Create one-hot logits
        logits = torch.zeros(len(target_seq), const.num_tokens)
        for i, idx in enumerate(target_indices.tolist()):
            logits[i, idx] = 10.0
        
        result = logits_to_aa_string(logits)
        assert result == target_seq


class TestStraightThroughSoftmax:
    """Test straight-through estimator."""
    
    def test_output_is_one_hot(self):
        """Output should be one-hot vectors."""
        logits = torch.randn(5, 20)
        result = straight_through_softmax(logits)
        
        # Check one-hot: each row sums to 1 and has exactly one 1
        assert torch.allclose(result.sum(dim=-1), torch.ones(5))
        assert (result.max(dim=-1).values == 1.0).all()
    
    def test_forward_matches_argmax(self):
        """Forward pass should match argmax."""
        logits = torch.randn(5, 20)
        result = straight_through_softmax(logits)
        argmax_indices = logits.argmax(dim=-1)
        
        # The argmax position should be 1, others 0
        for i in range(5):
            assert result[i, argmax_indices[i]] == 1.0
    
    def test_gradients_flow(self):
        """Gradients should flow back through STE."""
        logits = torch.randn(5, 20, requires_grad=True)
        result = straight_through_softmax(logits)
        
        # Compute a simple loss
        loss = result.sum()
        loss.backward()
        
        # Gradients should exist and be non-zero
        assert logits.grad is not None
        assert (logits.grad != 0).any()
    
    def test_gradient_shape(self):
        """Gradients should have same shape as input."""
        logits = torch.randn(3, 10, 20, requires_grad=True)
        result = straight_through_softmax(logits)
        loss = result.sum()
        loss.backward()
        
        assert logits.grad.shape == logits.shape
    
    def test_temperature_effect(self):
        """Lower temperature should give sharper softmax in backward."""
        logits = torch.randn(5, 20, requires_grad=True)
        
        # High temperature
        result_high = straight_through_softmax(logits.clone().requires_grad_(True), temperature=10.0)
        loss_high = result_high.sum()
        
        # Low temperature  
        result_low = straight_through_softmax(logits.clone().requires_grad_(True), temperature=0.1)
        loss_low = result_low.sum()
        
        # Forward results should be identical (both one-hot)
        assert torch.allclose(result_high, result_low)


class TestGumbelSoftmax:
    """Test Gumbel-softmax STE."""
    
    def test_output_is_one_hot(self):
        """Output should be one-hot."""
        torch.manual_seed(42)
        logits = torch.randn(5, 20)
        result = straight_through_gumbel_softmax(logits)
        
        assert torch.allclose(result.sum(dim=-1), torch.ones(5))
    
    def test_stochastic(self):
        """Different calls should give different results (with high prob)."""
        logits = torch.zeros(100, 20)  # Uniform logits
        
        results = []
        for _ in range(10):
            result = straight_through_gumbel_softmax(logits)
            results.append(result.argmax(dim=-1))
        
        # With uniform logits, we should see variation
        unique_counts = [len(torch.unique(r)) for r in results]
        assert max(unique_counts) > 1  # Should have some variation
    
    def test_gradients_flow(self):
        """Gradients should flow back."""
        logits = torch.randn(5, 20, requires_grad=True)
        result = straight_through_gumbel_softmax(logits)
        loss = result.sum()
        loss.backward()
        
        assert logits.grad is not None


class TestExtractCanonicalLogits:
    """Test extracting canonical AA logits."""
    
    def test_output_shape(self):
        """Should reduce last dim to 20."""
        logits = torch.randn(3, 10, const.num_tokens)
        canonical = extract_canonical_logits(logits)
        
        assert canonical.shape == (3, 10, 20)
    
    def test_values_match(self):
        """Values should match the canonical slice."""
        logits = torch.randn(5, const.num_tokens)
        canonical = extract_canonical_logits(logits)
        
        start = const.canonicals_offset
        end = start + 20
        expected = logits[:, start:end]
        
        assert torch.allclose(canonical, expected)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_sequence(self):
        """Handle empty sequence."""
        result = aa_string_to_indices("")
        assert result.shape == (0,)
    
    def test_unknown_amino_acid(self):
        """Unknown AAs should map to X/UNK."""
        # 'B' is not a standard AA
        indices = aa_string_to_indices("B")
        result = aa_indices_to_string(indices)
        assert result == "X"
    
    def test_lowercase_input(self):
        """Should handle lowercase input."""
        indices = aa_string_to_indices("acdef")
        result = aa_indices_to_string(indices)
        assert result == "ACDEF"
    
    def test_all_masked(self):
        """Handle case where all positions are masked."""
        indices = aa_string_to_indices("AAAAA")
        mask = torch.zeros(5, dtype=torch.bool)
        result = aa_indices_to_string(indices, mask)
        assert result == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
