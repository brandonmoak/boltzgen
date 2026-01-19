"""Guidance module for steering diffusion generation with property optimization.

BoltzGen encodes residue identity geometrically using virtual atoms. This module
provides differentiable guidance that works directly with this representation.
"""

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
    GeometricGuidance,
    create_geometric_guidance,
    enable_profiling,
    reset_profiling,
    get_profiling_stats,
)

from boltzgen.model.modules.guidance.utils import (
    extract_sequence_from_sample,
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
    # Mock predictors (for testing)
    "HydrophobicityPredictor",
    "SequenceLengthPredictor",
    "ConstantPredictor",
    "AminoAcidFrequencyPredictor",
    # Geometric guidance
    "GeometricGuidance",
    "create_geometric_guidance",
    # Profiling
    "enable_profiling",
    "reset_profiling", 
    "get_profiling_stats",
    # Output utilities
    "extract_sequence_from_sample",
]
