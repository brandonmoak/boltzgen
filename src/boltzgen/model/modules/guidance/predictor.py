"""Base interface for sequence property predictors.

Predictors take amino acid sequences and return property scores.
They are used to guide diffusion toward sequences with desired properties.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union

import torch
from torch import Tensor, nn


class SequencePredictor(nn.Module, ABC):
    """Abstract base class for sequence property predictors.
    
    Predictors should:
    1. Accept amino acid sequences (strings or tensors)
    2. Return scalar property scores (higher = better by default)
    3. Be differentiable if gradients are needed for guidance
    
    Example usage:
        predictor = MyStabilityPredictor()
        score = predictor("MKTVRQERLKSIG")  # Single sequence
        scores = predictor(["AAAA", "MMMM"])  # Batch
    """
    
    def __init__(self, higher_is_better: bool = True):
        """Initialize predictor.
        
        Args:
            higher_is_better: If True, guidance will maximize scores.
                             If False, guidance will minimize scores.
        """
        super().__init__()
        self.higher_is_better = higher_is_better
    
    @abstractmethod
    def forward(
        self,
        sequences: Union[str, List[str], Tensor],
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute property scores for sequences.
        
        Args:
            sequences: Input sequences, one of:
                - str: Single amino acid sequence (e.g., "MKTVRQ")
                - List[str]: Batch of sequences
                - Tensor: Token indices [B, L] or one-hot [B, L, 20]
            mask: Optional boolean mask [B, L] for valid positions.
                  Only used when sequences is a Tensor.
                  
        Returns:
            Tensor of shape [B] with property scores.
            For single sequence input, returns shape [1].
        """
        pass
    
    @property
    def name(self) -> str:
        """Human-readable name for this predictor."""
        return self.__class__.__name__
    
    def get_guidance_direction(self) -> float:
        """Get the sign for guidance gradient.
        
        Returns:
            +1.0 if higher_is_better (maximize score)
            -1.0 if not higher_is_better (minimize score)
        """
        return 1.0 if self.higher_is_better else -1.0


class EnsemblePredictor(SequencePredictor):
    """Combines multiple predictors into a weighted ensemble.
    
    Useful for optimizing multiple properties simultaneously.
    
    Example:
        stability_pred = StabilityPredictor()
        solubility_pred = SolubilityPredictor()
        
        ensemble = EnsemblePredictor(
            predictors=[stability_pred, solubility_pred],
            weights=[1.0, 0.5],  # Prioritize stability
        )
    """
    
    def __init__(
        self,
        predictors: List[SequencePredictor],
        weights: Optional[List[float]] = None,
        higher_is_better: bool = True,
    ):
        """Initialize ensemble predictor.
        
        Args:
            predictors: List of predictor modules
            weights: Optional weights for each predictor (default: equal weights)
            higher_is_better: Direction for the combined score
        """
        super().__init__(higher_is_better=higher_is_better)
        
        self.predictors = nn.ModuleList(predictors)
        
        if weights is None:
            weights = [1.0] * len(predictors)
        
        if len(weights) != len(predictors):
            raise ValueError(
                f"Number of weights ({len(weights)}) must match "
                f"number of predictors ({len(predictors)})"
            )
        
        # Normalize weights
        total = sum(weights)
        self.weights = [w / total for w in weights]
    
    def forward(
        self,
        sequences: Union[str, List[str], Tensor],
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute weighted ensemble score.
        
        Each predictor's score is multiplied by its weight and the
        predictor's guidance direction (to handle mixed higher/lower is better).
        """
        total_score = None
        
        for predictor, weight in zip(self.predictors, self.weights):
            score = predictor(sequences, mask)
            # Account for predictor's direction
            directed_score = score * predictor.get_guidance_direction()
            weighted_score = weight * directed_score
            
            if total_score is None:
                total_score = weighted_score
            else:
                total_score = total_score + weighted_score
        
        return total_score
    
    @property
    def name(self) -> str:
        names = [p.name for p in self.predictors]
        return f"Ensemble({', '.join(names)})"
