"""Geometric guidance for steering diffusion with property optimization.

BoltzGen encodes residue identity geometrically using virtual atoms - 14 atoms
per residue where sidechain atoms "collapse" to backbone positions in patterns
that uniquely identify each amino acid. This module provides differentiable
property optimization that works directly with this representation.
"""

from typing import Dict, Optional, Literal
import math
import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from boltzgen.data import const


# Profiling storage (module-level for easy access)
_profiling_enabled = False
_profiling_stats = {
    "compute_score_calls": 0,
    "compute_score_total_ms": 0.0,
    "residues_processed": 0,
    "cdist_ms": 0.0,
    "ste_ms": 0.0,
    "pattern_match_ms": 0.0,
}


def enable_profiling(enabled: bool = True):
    """Enable or disable profiling."""
    global _profiling_enabled
    _profiling_enabled = enabled
    if enabled:
        reset_profiling()


def reset_profiling():
    """Reset profiling statistics."""
    global _profiling_stats
    _profiling_stats = {
        "compute_score_calls": 0,
        "compute_score_total_ms": 0.0,
        "residues_processed": 0,
        "cdist_ms": 0.0,
        "ste_ms": 0.0,
        "pattern_match_ms": 0.0,
    }


def get_profiling_stats() -> dict:
    """Get profiling statistics."""
    stats = _profiling_stats.copy()
    if stats["compute_score_calls"] > 0:
        stats["avg_ms_per_call"] = stats["compute_score_total_ms"] / stats["compute_score_calls"]
    if stats["residues_processed"] > 0:
        stats["avg_ms_per_residue"] = stats["compute_score_total_ms"] / stats["residues_processed"]
    return stats


class GeometricGuidance(nn.Module):
    """Guidance that decodes residue type from atom geometry.
    
    BoltzGen encodes residue identity using virtual atoms - 14 atoms per residue
    where sidechain atoms "collapse" to backbone positions in a pattern that
    uniquely identifies each amino acid type.
    
    This guidance module:
    1. Decodes soft residue probabilities from atom coordinates
    2. Computes a differentiable property score (e.g., hydrophobicity)
    3. Backpropagates to get gradients for steering diffusion
    
    Example usage:
        guidance = GeometricGuidance(
            property_type="hydrophobicity",
            higher_is_better=True,
            guidance_scale=5.0,
        )
        
        # Pass to model.forward()
        out = model.forward(feats=feats, guidance=guidance, ...)
    """
    
    def __init__(
        self,
        property_type: str = "hydrophobicity",
        higher_is_better: bool = True,
        guidance_scale: float = 1.0,
        temperature: float = 0.1,
        schedule: Literal["constant", "linear", "cosine", "sigmoid"] = "constant",
        schedule_start: float = 0.0,
        schedule_end: float = 1.0,
        clamp_grad: Optional[float] = None,
        enabled: bool = True,
    ):
        """Initialize geometric guidance.
        
        Args:
            property_type: Property to optimize ("hydrophobicity")
            higher_is_better: Whether to maximize (True) or minimize (False)
            guidance_scale: Scale for guidance gradient
            temperature: Temperature for soft decoding (lower = sharper)
            schedule: Guidance schedule over diffusion steps
                - "constant": Always use full guidance_scale
                - "linear": Ramp from 0 to guidance_scale
                - "cosine": Cosine ramp from 0 to guidance_scale  
                - "sigmoid": Sigmoid ramp from 0 to guidance_scale
            schedule_start: When to start guidance (0-1 progress)
            schedule_end: When guidance reaches full strength
            clamp_grad: Max gradient norm
            enabled: Whether guidance is active
        """
        super().__init__()
        self.property_type = property_type
        self.higher_is_better = higher_is_better
        self.guidance_scale = guidance_scale
        self.temperature = temperature
        self.schedule = schedule
        self.schedule_start = schedule_start
        self.schedule_end = schedule_end
        self.clamp_grad = clamp_grad
        self.enabled = enabled
        
        self._build_patterns()
    
    def _build_patterns(self):
        """Build the geometric decoding patterns from BoltzGen constants."""
        token_to_count = const.token_to_placement_count
        
        # Canonical AAs in standard order (matches Kyte-Doolittle order)
        self.canonical_aas = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 
                              'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER',
                              'THR', 'TRP', 'TYR', 'VAL']
        
        patterns = []
        for aa in self.canonical_aas:
            if aa in token_to_count:
                patterns.append(token_to_count[aa])
            else:
                patterns.append([0, 0, 0, 0])
        
        self.register_buffer("count_patterns", torch.tensor(patterns, dtype=torch.float32))
        
        # Kyte-Doolittle hydrophobicity scale
        # Positive = hydrophobic, Negative = hydrophilic
        hydro = torch.tensor([
            1.8,   # ALA - hydrophobic
            2.5,   # CYS - hydrophobic
            -3.5,  # ASP - hydrophilic
            -3.5,  # GLU - hydrophilic
            2.8,   # PHE - hydrophobic
            -0.4,  # GLY - neutral
            -3.2,  # HIS - hydrophilic
            4.5,   # ILE - hydrophobic
            -3.9,  # LYS - hydrophilic
            3.8,   # LEU - hydrophobic
            1.9,   # MET - hydrophobic
            -3.5,  # ASN - hydrophilic
            -1.6,  # PRO - slightly hydrophilic
            -3.5,  # GLN - hydrophilic
            -4.5,  # ARG - hydrophilic
            -0.8,  # SER - slightly hydrophilic
            -0.7,  # THR - slightly hydrophilic
            -0.9,  # TRP - slightly hydrophilic
            -1.3,  # TYR - slightly hydrophilic
            4.2,   # VAL - hydrophobic
        ], dtype=torch.float32)
        self.register_buffer("hydrophobicity", hydro)
    
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
            return 1 / (1 + math.exp(-10 * (t - 0.5)))
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")
    
    def compute_geometric_score(
        self,
        coords: Tensor,
        feats: Dict[str, Tensor],
        threshold: float = 0.5,
    ) -> Tensor:
        """Compute property score from coordinates using geometric decoding.
        
        VECTORIZED VERSION: Processes all residues in parallel for efficiency.
        
        This implements a differentiable version of res_from_atom14 that produces
        IDENTICAL hard outputs but allows gradients to flow via straight-through
        estimators (STE).
        
        Args:
            coords: Atom coordinates [B, N_atoms, 3]
            feats: Feature dictionary with atom_to_token, design_mask
            threshold: Distance threshold for counting (default 0.5, same as res_from_atom14)
            
        Returns:
            Score [B] (differentiable w.r.t. coords via STE)
        """
        global _profiling_enabled, _profiling_stats
        
        if _profiling_enabled:
            start_time = time.perf_counter()
            _profiling_stats["compute_score_calls"] += 1
        
        device = coords.device
        B = coords.shape[0]
        
        design_mask = feats.get("design_mask")
        atom_to_token = feats.get("atom_to_token")
        
        if design_mask is None or atom_to_token is None:
            return torch.zeros(B, device=device)
        
        # Handle case where feats have batch=1 but coords have batch=B (diffusion_samples > 1)
        if design_mask.dim() == 1:
            design_mask = design_mask.unsqueeze(0)
        if design_mask.shape[0] == 1 and B > 1:
            design_mask = design_mask.expand(B, -1)
        
        if atom_to_token.dim() == 3:
            atom_to_token_idx = atom_to_token.int().argmax(dim=-1)
        else:
            atom_to_token_idx = atom_to_token
        
        # Expand atom_to_token_idx if needed
        if atom_to_token_idx.shape[0] == 1 and B > 1:
            atom_to_token_idx = atom_to_token_idx.expand(B, -1)
        
        batch_scores = []
        
        for b in range(B):
            design_indices = design_mask[b].nonzero(as_tuple=True)[0]
            
            if len(design_indices) == 0:
                batch_scores.append(torch.zeros(1, device=device))
                continue
            
            # Gather all atom indices for designed residues
            # Build a list of valid residues (those with exactly 14 atoms)
            valid_residue_atoms = []
            for token_idx in design_indices:
                atom_mask = (atom_to_token_idx[b] == token_idx)
                atom_indices = atom_mask.nonzero(as_tuple=True)[0]
                if len(atom_indices) == 14:
                    valid_residue_atoms.append(atom_indices)
            
            if len(valid_residue_atoms) == 0:
                batch_scores.append(torch.zeros(1, device=device))
                continue
            
            if _profiling_enabled:
                _profiling_stats["residues_processed"] += len(valid_residue_atoms)
            
            # Stack all residue atom indices: [N_res, 14]
            all_atom_indices = torch.stack(valid_residue_atoms)
            N_res = all_atom_indices.shape[0]
            
            # Gather coordinates for all residues at once: [N_res, 14, 3]
            all_res_coords = coords[b, all_atom_indices]
            
            # Split into backbone and sidechain
            backbone_coords = all_res_coords[:, :4, :]   # [N_res, 4, 3]
            sidechain_coords = all_res_coords[:, 4:, :]  # [N_res, 10, 3]
            
            # Step 1: Compute distances for all residues at once
            # cdist expects [batch, points1, dim] and [batch, points2, dim]
            # Output: [N_res, 4, 10]
            if _profiling_enabled:
                t0 = time.perf_counter()
            
            distances = torch.cdist(backbone_coords, sidechain_coords)  # [N_res, 4, 10]
            
            if _profiling_enabled:
                _profiling_stats["cdist_ms"] += (time.perf_counter() - t0) * 1000
                t0 = time.perf_counter()
            
            # Step 2: Find closest backbone for each sidechain atom
            # min over backbone dim (dim=1) → [N_res, 10]
            min_dists, hard_argmin = distances.min(dim=1)  # both [N_res, 10]
            
            # NOISE DETECTION: If distances are too large, coordinates are noise
            # Return neutral score (0) instead of biased prediction toward TRP
            # Typical bond lengths are 1-2Å; if mean min dist > 5Å, it's noise
            mean_min_dist = min_dists.mean()
            if mean_min_dist > 5.0:
                # Return zero with gradient connection to coords for proper backprop
                # Use a tiny contribution from coords to maintain gradient flow
                # unsqueeze(0) ensures shape [1] to match other batch_scores entries
                batch_scores.append((coords[b, 0, 0] * 0.0).unsqueeze(0))
                continue
            
            # Step 3: Threshold - atoms beyond threshold don't count
            threshold_mask = (min_dists <= threshold).float()  # [N_res, 10]
            
            # STE for threshold
            soft_threshold = torch.sigmoid((threshold - min_dists) / 0.05)
            threshold_mask_ste = threshold_mask + (soft_threshold - soft_threshold.detach())
            
            # Step 4: Count assignments to each backbone atom
            # Hard one-hot assignment: [N_res, 10, 4]
            hard_one_hot = F.one_hot(hard_argmin, num_classes=4).float()
            
            # STE for argmin: need to transpose distances for softmax over backbone
            # distances is [N_res, 4, 10], transpose to [N_res, 10, 4]
            distances_t = distances.transpose(1, 2)  # [N_res, 10, 4]
            soft_assignment = F.softmax(-distances_t / 0.01, dim=-1)  # [N_res, 10, 4]
            assignment_ste = hard_one_hot + (soft_assignment - soft_assignment.detach())
            
            # Apply threshold mask and sum to get counts
            # threshold_mask_ste: [N_res, 10] → [N_res, 10, 1]
            masked_assignment = assignment_ste * threshold_mask_ste.unsqueeze(-1)  # [N_res, 10, 4]
            soft_counts = masked_assignment.sum(dim=1)  # [N_res, 4]
            
            if _profiling_enabled:
                _profiling_stats["ste_ms"] += (time.perf_counter() - t0) * 1000
                t0 = time.perf_counter()
            
            # Step 5: Match counts to residue type patterns (vectorized)
            # soft_counts: [N_res, 4], count_patterns: [20, 4]
            # Compute L2 distance to each pattern for all residues
            # Move patterns to same device as input
            count_patterns = self.count_patterns.to(soft_counts.device)
            count_diff = (soft_counts.unsqueeze(1) - count_patterns.unsqueeze(0)) ** 2  # [N_res, 20, 4]
            count_dist = count_diff.sum(dim=-1)  # [N_res, 20]
            
            # Hard argmin for pattern matching
            hard_best_pattern = count_dist.argmin(dim=-1)  # [N_res]
            
            # STE: hard in forward, soft in backward
            soft_pattern_probs = F.softmax(-count_dist / self.temperature, dim=-1)  # [N_res, 20]
            hard_pattern_one_hot = F.one_hot(hard_best_pattern, num_classes=20).float()  # [N_res, 20]
            res_probs = hard_pattern_one_hot + (soft_pattern_probs - soft_pattern_probs.detach())
            
            if _profiling_enabled:
                _profiling_stats["pattern_match_ms"] += (time.perf_counter() - t0) * 1000
            
            # Compute property score
            if self.property_type == "hydrophobicity":
                # res_probs: [N_res, 20], hydrophobicity: [20]
                hydro = self.hydrophobicity.to(res_probs.device)
                residue_scores = (res_probs * hydro).sum(dim=-1)  # [N_res]
                batch_scores.append(residue_scores.mean().unsqueeze(0))
            else:
                batch_scores.append(torch.zeros(1, device=device))
        
        if _profiling_enabled:
            _profiling_stats["compute_score_total_ms"] += (time.perf_counter() - start_time) * 1000
        
        return torch.cat(batch_scores)
    
    def get_guidance_direction(self) -> float:
        """Return +1 if higher is better, -1 if lower is better."""
        return 1.0 if self.higher_is_better else -1.0


def create_geometric_guidance(
    property_type: str = "hydrophobicity",
    higher_is_better: bool = True,
    guidance_scale: float = 1.0,
    temperature: float = 0.1,
    schedule: str = "constant",
    schedule_start: float = 0.0,
    enabled: bool = True,
    **kwargs,
) -> GeometricGuidance:
    """Factory function to create geometric guidance.
    
    Args:
        property_type: Property to optimize ("hydrophobicity")
        higher_is_better: Whether to maximize (True) or minimize (False)
        guidance_scale: Base scale for guidance gradient
        temperature: Temperature for soft decoding (lower = sharper)
        schedule: Guidance schedule ("constant", "linear", "cosine", "sigmoid")
        schedule_start: When to start guidance (0-1 progress)
        enabled: Whether guidance is active
        **kwargs: Additional arguments
        
    Returns:
        Configured GeometricGuidance instance
    """
    return GeometricGuidance(
        property_type=property_type,
        higher_is_better=higher_is_better,
        guidance_scale=guidance_scale,
        temperature=temperature,
        schedule=schedule,
        schedule_start=schedule_start,
        enabled=enabled,
        **kwargs,
    )
