"""Guidance module for steering diffusion with external predictors.

This module provides the core guidance computation that integrates
sequence predictors with the diffusion sampling process.
"""

from typing import Dict, Optional, Callable, Literal
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from boltzgen.data import const
from boltzgen.model.modules.guidance.predictor import SequencePredictor
from boltzgen.model.modules.guidance.sequence_utils import (
    straight_through_softmax,
    extract_canonical_logits,
    aa_indices_to_string,
    NUM_CANONICAL_AAS,
)


class DiffusionGuidance(nn.Module):
    """Computes guidance gradients for steering diffusion sampling.
    
    This module:
    1. Takes res_type logits from the diffusion model
    2. Applies STE to get discrete tokens with gradient path
    3. Scores sequences using a predictor
    4. Computes gradients to guide the diffusion process
    
    Example usage:
        predictor = HydrophobicityPredictor()
        guidance = DiffusionGuidance(predictor, guidance_scale=1.0)
        
        # In diffusion sampling loop:
        guidance_grad = guidance.compute_guidance(
            atom_coords=atom_coords_noisy,
            res_type_logits=net_out["res_type"],
            sigma=t_hat,
            feats=feats,
        )
        atom_coords_next = atom_coords_next + guidance_grad
    """
    
    def __init__(
        self,
        predictor: SequencePredictor,
        guidance_scale: float = 1.0,
        temperature: float = 1.0,
        schedule: Literal["constant", "linear", "cosine", "sigmoid"] = "constant",
        schedule_start: float = 0.0,  # When to start applying guidance (0-1)
        schedule_end: float = 1.0,    # When guidance reaches full strength
        clamp_grad: Optional[float] = None,  # Max gradient norm
        enabled: bool = True,
    ):
        """Initialize guidance module.
        
        Args:
            predictor: Sequence property predictor
            guidance_scale: Base scale for guidance gradient
            temperature: Temperature for STE softmax (lower = sharper)
            schedule: How guidance scale varies over diffusion steps
                - "constant": Always use guidance_scale
                - "linear": Ramp from 0 to guidance_scale
                - "cosine": Cosine ramp from 0 to guidance_scale  
                - "sigmoid": Sigmoid ramp from 0 to guidance_scale
            schedule_start: Diffusion progress (0-1) when guidance starts
            schedule_end: Diffusion progress (0-1) when guidance reaches full strength
            clamp_grad: If set, clamp gradient norm to this value
            enabled: Whether guidance is active (for A/B testing)
        """
        super().__init__()
        self.predictor = predictor
        self.guidance_scale = guidance_scale
        self.temperature = temperature
        self.schedule = schedule
        self.schedule_start = schedule_start
        self.schedule_end = schedule_end
        self.clamp_grad = clamp_grad
        self.enabled = enabled
    
    def get_schedule_weight(self, progress: float) -> float:
        """Get guidance weight based on diffusion progress.
        
        Args:
            progress: Current progress through diffusion (0 = start, 1 = end)
            
        Returns:
            Weight multiplier for guidance scale
        """
        if progress < self.schedule_start:
            return 0.0
        
        if progress >= self.schedule_end:
            t = 1.0
        else:
            t = (progress - self.schedule_start) / (self.schedule_end - self.schedule_start)
        
        if self.schedule == "constant":
            return 1.0
        elif self.schedule == "linear":
            return t
        elif self.schedule == "cosine":
            return 0.5 * (1 - math.cos(math.pi * t))
        elif self.schedule == "sigmoid":
            # Sigmoid centered at t=0.5, scaled to [0, 1]
            return 1 / (1 + math.exp(-10 * (t - 0.5)))
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")
    
    def compute_guidance(
        self,
        atom_coords: Tensor,
        res_type_logits: Tensor,
        sigma: float,
        feats: Dict[str, Tensor],
        step: int = 0,
        total_steps: int = 1,
    ) -> Optional[Tensor]:
        """Compute guidance gradient for current diffusion step.
        
        Args:
            atom_coords: Current atom coordinates [B, N_atoms, 3]
            res_type_logits: Predicted residue type logits [B, N_tokens, num_tokens]
            sigma: Current noise level
            feats: Feature dictionary (contains masks, etc.)
            step: Current diffusion step
            total_steps: Total number of diffusion steps
            
        Returns:
            Guidance gradient [B, N_atoms, 3] or None if guidance disabled
        """
        if not self.enabled:
            return None
        
        if res_type_logits is None:
            return None
        
        # Compute schedule weight
        progress = step / max(total_steps - 1, 1)
        schedule_weight = self.get_schedule_weight(progress)
        
        if schedule_weight == 0.0:
            return None
        
        # Get design mask if available
        design_mask = feats.get("design_mask", None)
        if design_mask is not None:
            design_mask = design_mask.bool()
        
        # Enable gradients for this computation
        atom_coords_input = atom_coords.detach().clone()
        atom_coords_input.requires_grad_(True)
        
        # We need to connect atom_coords to res_type_logits
        # This is tricky because they're computed separately by the model
        # For now, we compute guidance on the sequence and assume the gradient
        # flows back through the relationship between structure and sequence
        
        # Apply STE to get discrete tokens with gradient path
        # Extract canonical AA logits
        canonical_logits = extract_canonical_logits(res_type_logits)  # [B, L, 20]
        
        # STE: discrete in forward, soft in backward
        soft = F.softmax(canonical_logits / self.temperature, dim=-1)
        hard_indices = canonical_logits.argmax(dim=-1)
        hard = F.one_hot(hard_indices, num_classes=NUM_CANONICAL_AAS).float()
        tokens_ste = hard - soft.detach() + soft  # [B, L, 20]
        
        # Expand back to full token space for predictor
        full_tokens = torch.zeros(
            *res_type_logits.shape[:-1], const.num_tokens,
            device=res_type_logits.device,
            dtype=res_type_logits.dtype,
        )
        start = const.canonicals_offset
        end = start + NUM_CANONICAL_AAS
        full_tokens[..., start:end] = tokens_ste
        
        # Get predictor score
        scores = self.predictor(full_tokens, mask=design_mask)
        
        # Direction: maximize if higher_is_better, minimize otherwise
        direction = self.predictor.get_guidance_direction()
        
        # Compute loss (we want to maximize score * direction)
        loss = -direction * scores.sum()
        
        # Backprop to get gradient w.r.t. logits
        loss.backward()
        
        # The gradient is w.r.t. res_type_logits, but we need it w.r.t. atom_coords
        # This is the fundamental challenge: the predictor operates on sequences,
        # but we want to guide the structure.
        #
        # For now, we'll return None and note this limitation.
        # A proper implementation would need to either:
        # 1. Have a differentiable path from atom_coords to res_type_logits
        # 2. Use a structure-aware predictor
        # 3. Use REINFORCE-style gradient estimation
        #
        # The gradient w.r.t. logits IS available though:
        if res_type_logits.grad is not None:
            logits_grad = res_type_logits.grad
            # Scale and clamp
            effective_scale = self.guidance_scale * schedule_weight
            scaled_grad = effective_scale * logits_grad
            
            if self.clamp_grad is not None:
                grad_norm = scaled_grad.norm()
                if grad_norm > self.clamp_grad:
                    scaled_grad = scaled_grad * (self.clamp_grad / grad_norm)
            
            # Store for potential use
            self._last_logits_grad = scaled_grad
        
        # For structure guidance, we need a different approach
        # Return None for now - the integration point will handle this
        return None
    
    def compute_sequence_guidance(
        self,
        res_type_logits: Tensor,
        feats: Dict[str, Tensor],
        step: int = 0,
        total_steps: int = 1,
    ) -> Optional[Tensor]:
        """Compute guidance gradient for sequence (res_type) logits.
        
        This is useful when the model jointly predicts structure and sequence,
        and we want to guide the sequence prediction directly.
        
        Args:
            res_type_logits: Predicted residue type logits [B, N_tokens, num_tokens]
            feats: Feature dictionary
            step: Current diffusion step
            total_steps: Total number of diffusion steps
            
        Returns:
            Gradient w.r.t. res_type_logits [B, N_tokens, num_tokens] or None
        """
        if not self.enabled:
            return None
        
        if res_type_logits is None:
            return None
        
        # Compute schedule weight
        progress = step / max(total_steps - 1, 1)
        schedule_weight = self.get_schedule_weight(progress)
        
        if schedule_weight == 0.0:
            return None
        
        # Get design mask if available
        design_mask = feats.get("design_mask", None)
        if design_mask is not None:
            design_mask = design_mask.bool()
        
        # Make logits require grad
        logits = res_type_logits.detach().clone()
        logits.requires_grad_(True)
        
        # Apply STE
        canonical_logits = extract_canonical_logits(logits)
        soft = F.softmax(canonical_logits / self.temperature, dim=-1)
        hard_indices = canonical_logits.argmax(dim=-1)
        hard = F.one_hot(hard_indices, num_classes=NUM_CANONICAL_AAS).float()
        tokens_ste = hard - soft.detach() + soft
        
        # Expand to full token space
        full_tokens = torch.zeros(
            *logits.shape[:-1], const.num_tokens,
            device=logits.device,
            dtype=logits.dtype,
        )
        start = const.canonicals_offset
        end = start + NUM_CANONICAL_AAS
        full_tokens[..., start:end] = tokens_ste
        
        # Get predictor score
        scores = self.predictor(full_tokens, mask=design_mask)
        
        # Direction: +1 if higher_is_better, -1 otherwise
        direction = self.predictor.get_guidance_direction()
        
        # We want to move in the direction of increasing (direction * scores)
        # So we compute gradient of scores and scale by direction
        objective = scores.sum()
        objective.backward()
        
        if logits.grad is None:
            return None
        
        # Scale gradient
        # logits.grad = ∂scores/∂logits, points toward increasing scores
        # We want to move toward increasing (direction * scores)
        effective_scale = self.guidance_scale * schedule_weight * direction
        guidance_grad = effective_scale * logits.grad
        
        # Clamp if needed
        if self.clamp_grad is not None:
            grad_norm = guidance_grad.norm()
            if grad_norm > self.clamp_grad:
                guidance_grad = guidance_grad * (self.clamp_grad / grad_norm)
        
        return guidance_grad
    
    def get_predicted_sequences(
        self,
        res_type_logits: Tensor,
        mask: Optional[Tensor] = None,
    ) -> list:
        """Get predicted sequences from logits (for logging/debugging).
        
        Args:
            res_type_logits: Logits [B, L, num_tokens]
            mask: Optional mask [B, L]
            
        Returns:
            List of predicted amino acid sequences
        """
        indices = res_type_logits[..., const.canonicals_offset:const.canonicals_offset + NUM_CANONICAL_AAS].argmax(dim=-1)
        indices = indices + const.canonicals_offset
        return aa_indices_to_string(indices, mask)
    
    def get_predictor_scores(
        self,
        res_type_logits: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Get predictor scores for current sequences (for logging/debugging).
        
        Args:
            res_type_logits: Logits [B, L, num_tokens]
            mask: Optional mask [B, L]
            
        Returns:
            Scores [B]
        """
        with torch.no_grad():
            canonical_logits = extract_canonical_logits(res_type_logits)
            indices = canonical_logits.argmax(dim=-1) + const.canonicals_offset
            return self.predictor(indices, mask=mask)


def create_guidance(
    predictor: SequencePredictor,
    guidance_scale: float = 1.0,
    enabled: bool = True,
    **kwargs,
) -> DiffusionGuidance:
    """Factory function to create guidance module.
    
    Args:
        predictor: Sequence property predictor
        guidance_scale: Base scale for guidance
        enabled: Whether guidance is active
        **kwargs: Additional arguments for DiffusionGuidance
        
    Returns:
        Configured DiffusionGuidance instance
    """
    return DiffusionGuidance(
        predictor=predictor,
        guidance_scale=guidance_scale,
        enabled=enabled,
        **kwargs,
    )
