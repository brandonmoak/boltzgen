"""Mock predictors for testing the guidance pipeline.

These simple predictors allow validating that:
1. The guidance pipeline correctly calls predictors
2. Gradients flow back through the STE
3. Guidance actually affects generation
"""

from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from boltzgen.data import const
from boltzgen.model.modules.guidance.predictor import SequencePredictor
from boltzgen.model.modules.guidance.sequence_utils import (
    aa_string_to_indices,
    straight_through_softmax,
    NUM_CANONICAL_AAS,
)


# Hydrophobicity scores for each canonical amino acid (Kyte-Doolittle scale)
# Higher = more hydrophobic
HYDROPHOBICITY_SCORES = {
    "A": 1.8,   # Alanine
    "R": -4.5,  # Arginine
    "N": -3.5,  # Asparagine
    "D": -3.5,  # Aspartic acid
    "C": 2.5,   # Cysteine
    "Q": -3.5,  # Glutamine
    "E": -3.5,  # Glutamic acid
    "G": -0.4,  # Glycine
    "H": -3.2,  # Histidine
    "I": 4.5,   # Isoleucine
    "L": 3.8,   # Leucine
    "K": -3.9,  # Lysine
    "M": 1.9,   # Methionine
    "F": 2.8,   # Phenylalanine
    "P": -1.6,  # Proline
    "S": -0.8,  # Serine
    "T": -0.7,  # Threonine
    "W": -0.9,  # Tryptophan
    "Y": -1.3,  # Tyrosine
    "V": 4.2,   # Valine
}

# Order matches const.canonical_tokens
HYDROPHOBICITY_TENSOR = torch.tensor([
    HYDROPHOBICITY_SCORES[const.prot_token_to_letter[aa]]
    for aa in const.canonical_tokens
])


class HydrophobicityPredictor(SequencePredictor):
    """Mock predictor that scores sequences by average hydrophobicity.
    
    Uses Kyte-Doolittle hydrophobicity scale.
    Higher scores = more hydrophobic sequences.
    
    This predictor is differentiable through soft sequences, making it
    useful for testing gradient flow in the guidance pipeline.
    """
    
    def __init__(self, higher_is_better: bool = True):
        """Initialize hydrophobicity predictor.
        
        Args:
            higher_is_better: If True, guide toward hydrophobic sequences.
                             If False, guide toward hydrophilic sequences.
        """
        super().__init__(higher_is_better=higher_is_better)
        
        # Register hydrophobicity scores as buffer (not a parameter)
        self.register_buffer("scores", HYDROPHOBICITY_TENSOR.clone())
    
    def forward(
        self,
        sequences: Union[str, List[str], Tensor],
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute average hydrophobicity score.
        
        Args:
            sequences: Input as string(s) or tensor
            mask: Optional mask for valid positions
            
        Returns:
            Average hydrophobicity score per sequence [B]
        """
        # Handle string input
        if isinstance(sequences, str):
            sequences = [sequences]
        
        if isinstance(sequences, list):
            # Convert strings to indices
            indices = aa_string_to_indices(sequences, device=self.scores.device)
            # Convert to one-hot for differentiability
            # But for strings, we don't need gradients
            return self._score_from_indices(indices, mask)
        
        # Tensor input
        if sequences.dim() == 2:
            # [B, L] indices - convert to one-hot
            one_hot = F.one_hot(sequences, num_classes=const.num_tokens).float()
            sequences = one_hot
        
        # [B, L, num_tokens] - extract canonical AAs
        return self._score_from_soft(sequences, mask)
    
    def _score_from_indices(
        self,
        indices: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Score from token indices (non-differentiable path)."""
        # Shift indices to canonical range [0, 20)
        canonical_indices = indices - const.canonicals_offset
        canonical_indices = canonical_indices.clamp(0, NUM_CANONICAL_AAS - 1)
        
        # Move scores to same device as input and look up
        scores = self.scores.to(indices.device)
        seq_scores = scores[canonical_indices]  # [B, L]
        
        if mask is not None:
            seq_scores = seq_scores * mask.float()
            avg_scores = seq_scores.sum(dim=-1) / (mask.float().sum(dim=-1) + 1e-8)
        else:
            avg_scores = seq_scores.mean(dim=-1)
        
        return avg_scores
    
    def _score_from_soft(
        self,
        soft_tokens: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Score from soft token probabilities (differentiable path).
        
        This is the key method for gradient-based guidance.
        """
        # Extract canonical AA probabilities [B, L, 20]
        start = const.canonicals_offset
        end = start + NUM_CANONICAL_AAS
        canonical_probs = soft_tokens[..., start:end]
        
        # Move scores to same device as input
        scores = self.scores.to(soft_tokens.device)
        
        # Weighted sum of hydrophobicity scores
        # [B, L, 20] @ [20] -> [B, L]
        seq_scores = (canonical_probs * scores).sum(dim=-1)
        
        if mask is not None:
            seq_scores = seq_scores * mask.float()
            avg_scores = seq_scores.sum(dim=-1) / (mask.float().sum(dim=-1) + 1e-8)
        else:
            avg_scores = seq_scores.mean(dim=-1)
        
        return avg_scores
    
    @property
    def name(self) -> str:
        direction = "high" if self.higher_is_better else "low"
        return f"Hydrophobicity({direction})"


class SequenceLengthPredictor(SequencePredictor):
    """Mock predictor that scores based on sequence length.
    
    Useful for testing basic functionality - doesn't need amino acid content.
    """
    
    def __init__(self, target_length: int = 100, higher_is_better: bool = True):
        """Initialize length predictor.
        
        Args:
            target_length: Desired sequence length
            higher_is_better: If True, score is higher when closer to target
        """
        super().__init__(higher_is_better=higher_is_better)
        self.target_length = target_length
    
    def forward(
        self,
        sequences: Union[str, List[str], Tensor],
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Score based on distance from target length."""
        if isinstance(sequences, str):
            lengths = torch.tensor([len(sequences)], dtype=torch.float32)
        elif isinstance(sequences, list):
            lengths = torch.tensor([len(s) for s in sequences], dtype=torch.float32)
        else:
            # Tensor input
            if mask is not None:
                lengths = mask.float().sum(dim=-1)
            else:
                lengths = torch.tensor(
                    [sequences.shape[-2]] * sequences.shape[0],
                    dtype=torch.float32,
                    device=sequences.device,
                )
        
        # Negative distance from target (so higher = closer to target)
        scores = -torch.abs(lengths - self.target_length)
        return scores
    
    @property
    def name(self) -> str:
        return f"Length(target={self.target_length})"


class ConstantPredictor(SequencePredictor):
    """Predictor that always returns a constant score.
    
    Useful for testing that guidance correctly handles no-gradient cases.
    """
    
    def __init__(self, value: float = 1.0):
        super().__init__(higher_is_better=True)
        self.value = value
    
    def forward(
        self,
        sequences: Union[str, List[str], Tensor],
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if isinstance(sequences, str):
            batch_size = 1
            device = None
        elif isinstance(sequences, list):
            batch_size = len(sequences)
            device = None
        else:
            batch_size = sequences.shape[0]
            device = sequences.device
        
        return torch.full((batch_size,), self.value, device=device)
    
    @property
    def name(self) -> str:
        return f"Constant({self.value})"


class AminoAcidFrequencyPredictor(SequencePredictor):
    """Predictor that scores based on frequency of specific amino acids.
    
    Example: Maximize frequency of charged residues (K, R, D, E)
    """
    
    def __init__(
        self,
        target_aas: str = "KR",  # Target amino acids (1-letter codes)
        higher_is_better: bool = True,
    ):
        """Initialize frequency predictor.
        
        Args:
            target_aas: String of 1-letter AA codes to maximize/minimize
            higher_is_better: If True, maximize frequency of target AAs
        """
        super().__init__(higher_is_better=higher_is_better)
        self.target_aas = target_aas.upper()
        
        # Create weight tensor: 1 for target AAs, 0 for others
        weights = torch.zeros(NUM_CANONICAL_AAS)
        for aa in self.target_aas:
            # Find index in canonical tokens
            three_letter = const.prot_letter_to_token.get(aa)
            if three_letter and three_letter in const.canonical_tokens:
                idx = const.canonical_tokens.index(three_letter)
                weights[idx] = 1.0
        
        self.register_buffer("weights", weights)
    
    def forward(
        self,
        sequences: Union[str, List[str], Tensor],
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute frequency of target amino acids."""
        # Handle string input
        if isinstance(sequences, str):
            sequences = [sequences]
        
        if isinstance(sequences, list):
            # Count target AAs in each sequence
            scores = []
            for seq in sequences:
                count = sum(1 for aa in seq.upper() if aa in self.target_aas)
                freq = count / max(len(seq), 1)
                scores.append(freq)
            return torch.tensor(scores, device=self.weights.device)
        
        # Tensor input [B, L, num_tokens] or [B, L]
        if sequences.dim() == 2:
            one_hot = F.one_hot(sequences, num_classes=const.num_tokens).float()
            sequences = one_hot
        
        # Extract canonical probs and compute weighted sum
        start = const.canonicals_offset
        end = start + NUM_CANONICAL_AAS
        canonical_probs = sequences[..., start:end]  # [B, L, 20]
        
        # Move weights to same device as input
        weights = self.weights.to(sequences.device)
        
        # Frequency of target AAs
        target_probs = (canonical_probs * weights).sum(dim=-1)  # [B, L]
        
        if mask is not None:
            target_probs = target_probs * mask.float()
            freq = target_probs.sum(dim=-1) / (mask.float().sum(dim=-1) + 1e-8)
        else:
            freq = target_probs.mean(dim=-1)
        
        return freq
    
    @property
    def name(self) -> str:
        return f"AAFrequency({self.target_aas})"
