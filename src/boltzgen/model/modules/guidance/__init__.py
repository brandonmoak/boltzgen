"""Guidance module for steering diffusion generation with external predictors."""

from boltzgen.model.modules.guidance.sequence_utils import (
    logits_to_aa_string,
    logits_to_aa_indices,
    aa_indices_to_string,
    aa_string_to_indices,
    straight_through_softmax,
    straight_through_gumbel_softmax,
    get_canonical_mask,
    extract_canonical_logits,
)

from boltzgen.model.modules.guidance.predictor import (
    SequencePredictor,
    EnsemblePredictor,
)

from boltzgen.model.modules.guidance.mock_predictor import (
    HydrophobicityPredictor,
    SequenceLengthPredictor,
    ConstantPredictor,
    AminoAcidFrequencyPredictor,
)

from boltzgen.model.modules.guidance.guidance import (
    DiffusionGuidance,
    create_guidance,
)

__all__ = [
    # Sequence utilities
    "logits_to_aa_string",
    "logits_to_aa_indices", 
    "aa_indices_to_string",
    "aa_string_to_indices",
    "straight_through_softmax",
    "straight_through_gumbel_softmax",
    "get_canonical_mask",
    "extract_canonical_logits",
    # Predictor interface
    "SequencePredictor",
    "EnsemblePredictor",
    # Mock predictors
    "HydrophobicityPredictor",
    "SequenceLengthPredictor",
    "ConstantPredictor",
    "AminoAcidFrequencyPredictor",
    # Guidance
    "DiffusionGuidance",
    "create_guidance",
]
