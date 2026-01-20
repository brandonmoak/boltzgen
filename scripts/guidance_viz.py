"""Visualization and metrics for guidance comparison experiments.

Keep near notebook for easy editing during development.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor


# =============================================================================
# HYDROPHOBICITY
# =============================================================================

HYDRO_SCALE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8,
    'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}


def compute_hydrophobicity(sequence: str) -> float:
    """Compute mean hydrophobicity for designed (uppercase) residues."""
    designed = [c for c in sequence if c.isupper() and c in HYDRO_SCALE]
    if not designed:
        return 0.0
    return np.mean([HYDRO_SCALE[aa] for aa in designed])


# =============================================================================
# STRUCTURAL METRICS
# =============================================================================

def extract_ca_coords(coords: Tensor, feats: Dict[str, Tensor]) -> Optional[Tensor]:
    """Extract CA coordinates for designed residues."""
    design_mask = feats.get("design_mask")
    atom_to_token = feats.get("atom_to_token")
    
    if design_mask is None or atom_to_token is None:
        return None
    
    if design_mask.dim() == 2:
        design_mask = design_mask[0]
    if atom_to_token.dim() == 3:
        atom_to_token = atom_to_token[0]
    
    designed_tokens = design_mask.nonzero(as_tuple=True)[0]
    if len(designed_tokens) == 0:
        return None
    
    ca_coords_list = []
    # atom_to_token is one-hot [atoms, tokens] - get token index for each atom
    if atom_to_token.dim() == 2:
        atom_to_token_idx = atom_to_token.int().argmax(dim=-1)
    else:
        atom_to_token_idx = atom_to_token
    
    for token_idx in designed_tokens:
        atom_mask = (atom_to_token_idx == token_idx)
        atom_indices = atom_mask.nonzero(as_tuple=True)[0]
        if len(atom_indices) >= 2:
            ca_idx = atom_indices[1]
            ca_coords_list.append(coords[ca_idx])
    
    if not ca_coords_list:
        return None
    
    return torch.stack(ca_coords_list)


def compute_structural_metrics(ca_coords: Tensor) -> Dict[str, float]:
    """Compute Rg, E2E distance, contact density from CA coords."""
    if ca_coords is None or len(ca_coords) < 2:
        return {'radius_of_gyration': None, 'end_to_end_distance': None, 
                'contact_density': None, 'mean_ca_distance': None, 'std_ca_distance': None}
    
    ca = ca_coords.float()
    centroid = ca.mean(dim=0)
    rg = torch.sqrt(((ca - centroid) ** 2).sum(dim=1).mean()).item()
    e2e = torch.norm(ca[-1] - ca[0]).item()
    
    dists = torch.cdist(ca, ca)
    contacts = (dists < 8.0).float()
    contacts.fill_diagonal_(0)
    contact_density = contacts.sum().item() / (2 * len(ca))
    
    triu_idx = torch.triu_indices(len(ca), len(ca), offset=1)
    pairwise = dists[triu_idx[0], triu_idx[1]]
    
    return {
        'radius_of_gyration': rg,
        'end_to_end_distance': e2e,
        'contact_density': contact_density,
        'mean_ca_distance': pairwise.mean().item(),
        'std_ca_distance': pairwise.std().item(),
    }


def compute_lddt(coords1: Tensor, coords2: Tensor, cutoff: float = 15.0) -> Optional[float]:
    """Compute lDDT between two structures."""
    if coords1 is None or coords2 is None or len(coords1) != len(coords2) or len(coords1) < 2:
        return None
    
    c1, c2 = coords1.float(), coords2.float()
    d1, d2 = torch.cdist(c1, c1), torch.cdist(c2, c2)
    
    mask = (d1 < cutoff) & (d1 > 0)
    if mask.sum() == 0:
        return None
    
    diff = torch.abs(d1[mask] - d2[mask])
    scores = [(diff < t).float().mean().item() for t in [0.5, 1.0, 2.0, 4.0]]
    return np.mean(scores)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_comparison(results_unguided: List[Dict], results_guided: List[Dict], 
                    metric: str = 'hydrophobicity', title: str = None):
    """Plot paired comparison between guided and unguided."""
    import matplotlib.pyplot as plt
    
    u_vals = [r.get(metric) for r in results_unguided if r.get(metric) is not None]
    g_vals = [r.get(metric) for r in results_guided if r.get(metric) is not None]
    
    n = min(len(u_vals), len(g_vals))
    if n == 0:
        print(f"No data for {metric}")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for i in range(n):
        ax.plot([0, 1], [u_vals[i], g_vals[i]], 'o-', alpha=0.5, color='gray')
    
    ax.scatter([0], [np.mean(u_vals[:n])], s=200, c='blue', zorder=5, 
               label=f'Unguided mean: {np.mean(u_vals[:n]):.2f}')
    ax.scatter([1], [np.mean(g_vals[:n])], s=200, c='red', zorder=5, 
               label=f'Guided mean: {np.mean(g_vals[:n]):.2f}')
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Unguided', 'Guided'])
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title or f'{metric.replace("_", " ").title()} Comparison')
    ax.legend()
    plt.tight_layout()
    return fig


def plot_hydrophobicity_comparison(results_unguided: List[Dict], results_guided: List[Dict]):
    """Plot both geometric and string-based hydrophobicity side by side."""
    import matplotlib.pyplot as plt
    
    # Check if we have both metrics
    has_string = any('hydrophobicity_string' in r for r in results_unguided + results_guided)
    
    if has_string:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Geometric score
        u_geo = [r.get('hydrophobicity') for r in results_unguided if r.get('hydrophobicity') is not None]
        g_geo = [r.get('hydrophobicity') for r in results_guided if r.get('hydrophobicity') is not None]
        n = min(len(u_geo), len(g_geo))
        
        for i in range(n):
            ax1.plot([0, 1], [u_geo[i], g_geo[i]], 'o-', alpha=0.5, color='gray')
        ax1.scatter([0], [np.mean(u_geo[:n])], s=200, c='blue', zorder=5, 
                   label=f'Unguided: {np.mean(u_geo[:n]):.3f}')
        ax1.scatter([1], [np.mean(g_geo[:n])], s=200, c='red', zorder=5, 
                   label=f'Guided: {np.mean(g_geo[:n]):.3f}')
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(['Unguided', 'Guided'])
        ax1.set_ylabel('Hydrophobicity (Geometric Score)')
        ax1.set_title('Geometric Score (What Guidance Optimizes)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # String-based
        u_str = [r.get('hydrophobicity_string') for r in results_unguided if r.get('hydrophobicity_string') is not None]
        g_str = [r.get('hydrophobicity_string') for r in results_guided if r.get('hydrophobicity_string') is not None]
        n = min(len(u_str), len(g_str))
        
        for i in range(n):
            ax2.plot([0, 1], [u_str[i], g_str[i]], 'o-', alpha=0.5, color='gray')
        ax2.scatter([0], [np.mean(u_str[:n])], s=200, c='blue', zorder=5, 
                   label=f'Unguided: {np.mean(u_str[:n]):.3f}')
        ax2.scatter([1], [np.mean(g_str[:n])], s=200, c='red', zorder=5, 
                   label=f'Guided: {np.mean(g_str[:n]):.3f}')
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(['Unguided', 'Guided'])
        ax2.set_ylabel('Hydrophobicity (String-based)')
        ax2.set_title('String-based (From Decoded Sequence)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    else:
        # Fall back to single plot
        return plot_comparison(results_unguided, results_guided, 'hydrophobicity')


def create_sequence_comparison_html(results_unguided: List[Dict], results_guided: List[Dict],
                                     max_display: int = 10) -> str:
    """Create HTML table comparing sequences."""
    html = ['<table style="font-family: monospace; font-size: 12px;">']
    html.append('<tr><th>ID</th><th>Unguided</th><th>Guided</th><th>Changes</th></tr>')
    
    n = min(len(results_unguided), len(results_guided), max_display)
    
    for i in range(n):
        seq_u = results_unguided[i].get('designed_sequence', '')
        seq_g = results_guided[i].get('designed_sequence', '')
        
        changes = sum(1 for a, b in zip(seq_u, seq_g) if a != b and a != 'X' and b != 'X')
        
        colored_g = ""
        for u, g in zip(seq_u, seq_g):
            if u != g and u != 'X' and g != 'X':
                h_diff = HYDRO_SCALE.get(g, 0) - HYDRO_SCALE.get(u, 0)
                color = '#90EE90' if h_diff > 0 else '#FFB6C1'
                colored_g += f'<span style="background:{color}">{g}</span>'
            else:
                colored_g += g
        
        html.append(f'<tr><td>{i}</td><td>{seq_u}</td><td>{colored_g}</td><td>{changes}</td></tr>')
    
    html.append('</table>')
    return '\n'.join(html)


def print_summary(results_unguided: List[Dict], results_guided: List[Dict]):
    """Print summary statistics."""
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    # Check if we have both geometric and string-based hydrophobicity
    has_string_hydro = any('hydrophobicity_string' in r for r in results_unguided + results_guided)
    
    metrics = ['hydrophobicity', 'radius_of_gyration', 'contact_density']
    if has_string_hydro:
        metrics.append('hydrophobicity_string')
    
    for m in metrics:
        u_vals = [r.get(m) for r in results_unguided if r.get(m) is not None]
        g_vals = [r.get(m) for r in results_guided if r.get(m) is not None]
        
        if u_vals and g_vals:
            label = m.replace('_', ' ').title()
            if m == 'hydrophobicity':
                label = 'Hydrophobicity (Geometric Score)'
            elif m == 'hydrophobicity_string':
                label = 'Hydrophobicity (String-based)'
            
            print(f"\n{label}:")
            print(f"  Unguided: {np.mean(u_vals):.3f} ± {np.std(u_vals):.3f}")
            print(f"  Guided:   {np.mean(g_vals):.3f} ± {np.std(g_vals):.3f}")
            print(f"  Δ:        {np.mean(g_vals) - np.mean(u_vals):+.3f}")


# =============================================================================
# DIAGNOSTIC: RESIDUE TYPE DISTRIBUTION AT DIFFERENT NOISE LEVELS
# =============================================================================

# Standard amino acid order (matching BoltzGen's const.canonical_aa)
AA_ORDER = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
            'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

HYDRO_TENSOR = torch.tensor([
    1.8, -4.5, -3.5, -3.5, 2.5, -3.5, -3.5, -0.4, -3.2, 4.5,
    3.8, -3.9, 1.9, 2.8, -1.6, -0.8, -0.7, -0.9, -1.3, 4.2
])

def analyze_residue_predictions_at_step(
    coords: Tensor,
    feats: Dict[str, Tensor],
    guidance = None,
    step_label: str = "",
) -> Dict:
    """Analyze what residue types are being predicted at a given diffusion step.
    
    This helps diagnose why noisy coordinates produce biased hydrophobicity scores.
    
    Returns:
        Dictionary with residue type distribution and related diagnostics
    """
    if guidance is None:
        try:
            from boltzgen.model.modules.guidance import create_geometric_guidance
            guidance = create_geometric_guidance(
                property_type="hydrophobicity",
                higher_is_better=True,
                guidance_scale=0.0,
                temperature=0.1,
                enabled=False,
            )
            device = coords.device if hasattr(coords, 'device') else 'cpu'
            guidance = guidance.to(device)
        except ImportError:
            return {'error': 'Could not import guidance'}
    
    # Ensure batch dim
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)
    
    # Take first sample for analysis
    if coords.shape[0] > 1:
        coords = coords[0:1]
    
    device = coords.device
    design_mask = feats.get("design_mask")
    atom_to_token = feats.get("atom_to_token")
    
    if design_mask is None or atom_to_token is None:
        return {'error': 'Missing design_mask or atom_to_token'}
    
    # Handle batch dims
    if design_mask.dim() == 1:
        design_mask = design_mask.unsqueeze(0)
    if design_mask.shape[0] == 1 and coords.shape[0] > 1:
        design_mask = design_mask.expand(coords.shape[0], -1)
    
    if atom_to_token.dim() == 3:
        atom_to_token_idx = atom_to_token.int().argmax(dim=-1)
    else:
        atom_to_token_idx = atom_to_token
    if atom_to_token_idx.shape[0] == 1 and coords.shape[0] > 1:
        atom_to_token_idx = atom_to_token_idx.expand(coords.shape[0], -1)
    
    b = 0  # First batch
    design_indices = design_mask[b].nonzero(as_tuple=True)[0]
    
    if len(design_indices) == 0:
        return {'error': 'No designed residues'}
    
    # Gather valid residues (14 atoms each)
    valid_residue_atoms = []
    for token_idx in design_indices:
        atom_mask = (atom_to_token_idx[b] == token_idx)
        atom_indices = atom_mask.nonzero(as_tuple=True)[0]
        if len(atom_indices) == 14:
            valid_residue_atoms.append(atom_indices)
    
    if len(valid_residue_atoms) == 0:
        return {'error': 'No valid 14-atom residues'}
    
    all_atom_indices = torch.stack(valid_residue_atoms)
    N_res = all_atom_indices.shape[0]
    all_res_coords = coords[b, all_atom_indices]  # [N_res, 14, 3]
    
    backbone_coords = all_res_coords[:, :4, :]   # [N_res, 4, 3]
    sidechain_coords = all_res_coords[:, 4:, :]  # [N_res, 10, 3]
    
    # Compute distances
    distances = torch.cdist(backbone_coords, sidechain_coords)  # [N_res, 4, 10]
    
    # Find closest backbone for each sidechain atom
    min_dists, hard_argmin = distances.min(dim=1)  # both [N_res, 10]
    
    mean_min_dist = min_dists.mean().item()
    
    # NOISE DETECTION: Same threshold as guidance module
    is_noise = mean_min_dist > 5.0
    
    # Threshold
    threshold = 0.5
    threshold_mask = (min_dists <= threshold).float()  # [N_res, 10]
    frac_within = threshold_mask.mean().item()
    mean_counts = np.array([0., 0., 0., 0.])
    
    if is_noise:
        # Coordinates are noise - no meaningful prediction possible
        result = {
            'step_label': step_label,
            'n_residues': N_res,
            'is_noise': True,
            'predicted_hydrophobicity': 0.0,  # Neutral when uncertain
            'uniform_baseline': HYDRO_TENSOR.mean().item(),
            'bias': 0.0,
            'pred_distribution': None,
            'top_predictions': [('UNCERTAIN', N_res)],
            'mean_min_distance': mean_min_dist,
            'frac_within_threshold': frac_within,
            'mean_counts': mean_counts,
        }
        return result
    
    # Count assignments to each backbone atom
    import torch.nn.functional as F
    hard_one_hot = F.one_hot(hard_argmin, num_classes=4).float()  # [N_res, 10, 4]
    masked_assignment = hard_one_hot * threshold_mask.unsqueeze(-1)  # [N_res, 10, 4]
    hard_counts = masked_assignment.sum(dim=1)  # [N_res, 4]
    
    # Match to patterns
    count_patterns = guidance.count_patterns.to(device)  # [20, 4]
    count_diff = (hard_counts.unsqueeze(1) - count_patterns.unsqueeze(0)) ** 2
    count_dist = count_diff.sum(dim=-1)  # [N_res, 20]
    
    # Hard prediction
    hard_best_pattern = count_dist.argmin(dim=-1)  # [N_res]
    
    # Distribution of predicted residue types
    pred_counts = torch.zeros(20, device=device)
    for i in range(20):
        pred_counts[i] = (hard_best_pattern == i).sum()
    
    pred_probs = pred_counts / pred_counts.sum()
    
    # Compute statistics
    hydro = HYDRO_TENSOR.to(device)
    expected_hydro = (pred_probs * hydro).sum().item()
    
    # Mean hydrophobicity if uniform distribution
    uniform_hydro = hydro.mean().item()
    
    # Most common predictions
    top_k = 5
    top_indices = pred_counts.argsort(descending=True)[:top_k]
    
    result = {
        'step_label': step_label,
        'n_residues': N_res,
        'is_noise': False,
        'predicted_hydrophobicity': expected_hydro,
        'uniform_baseline': uniform_hydro,
        'bias': expected_hydro - uniform_hydro,
        'pred_distribution': pred_probs.cpu().numpy(),
        'top_predictions': [(AA_ORDER[i], int(pred_counts[i].item())) for i in top_indices],
        'mean_min_distance': mean_min_dist,
        'frac_within_threshold': frac_within,
        'mean_counts': hard_counts.mean(dim=0).cpu().numpy(),  # [4] average counts per backbone atom
    }
    
    return result


def print_residue_prediction_analysis(analysis: Dict):
    """Pretty print the residue prediction analysis."""
    if 'error' in analysis:
        print(f"Error: {analysis['error']}")
        return
    
    print(f"\n{'='*60}")
    print(f"RESIDUE PREDICTION ANALYSIS: {analysis['step_label']}")
    print(f"{'='*60}")
    print(f"Number of designed residues: {analysis['n_residues']}")
    
    # Check if coordinates are noise
    if analysis.get('is_noise', False):
        print(f"\n⚠️  NOISE DETECTED - Coordinates too diffuse for meaningful prediction")
        print(f"   Mean distance: {analysis['mean_min_distance']:.1f} Å (threshold: 5.0 Å)")
        print(f"   Returning NEUTRAL score: 0.0")
        print(f"\nGeometric statistics:")
        print(f"  Mean min distance (SC to BB):      {analysis['mean_min_distance']:.2f} Å")
        print(f"  Fraction within 0.5Å threshold:    {analysis['frac_within_threshold']:.1%}")
        return
    
    print(f"\nHydrophobicity:")
    print(f"  Predicted (from decoded residues): {analysis['predicted_hydrophobicity']:.3f}")
    print(f"  Uniform baseline (if random AA):   {analysis['uniform_baseline']:.3f}")
    print(f"  Bias from uniform:                 {analysis['bias']:+.3f}")
    
    print(f"\nGeometric statistics:")
    print(f"  Mean min distance (SC to BB):      {analysis['mean_min_distance']:.2f} Å")
    print(f"  Fraction within 0.5Å threshold:    {analysis['frac_within_threshold']:.1%}")
    print(f"  Mean counts [N, CA, C, O]:         {analysis['mean_counts']}")
    
    print(f"\nTop {len(analysis['top_predictions'])} predicted residue types:")
    for aa, count in analysis['top_predictions']:
        hydro = HYDRO_SCALE.get(aa, 0)
        print(f"  {aa}: {count:3d} (hydrophobicity: {hydro:+.1f})")


# =============================================================================
# TRAJECTORY HYDROPHOBICITY SCORING
# =============================================================================

def compute_trajectory_hydrophobicity(
    coords_traj: List[Tensor],
    feats: Dict[str, Tensor],
    guidance = None,
    temperature: float = 0.1,
) -> Dict[str, np.ndarray]:
    """Compute hydrophobicity scores for each step in a coordinate trajectory.
    
    This allows comparing how hydrophobicity evolves during diffusion for both
    guided and unguided runs. Computes scores for ALL samples in the batch.
    
    Args:
        coords_traj: List of coordinate tensors [N_atoms, 3] or [B, N_atoms, 3]
        feats: Feature dictionary with design_mask, atom_to_token
        guidance: Optional GeometricGuidance object (creates one if not provided)
        temperature: Temperature for soft argmax (default 0.1)
        
    Returns:
        Dictionary with:
            - steps: array of step indices [N_steps]
            - scores_all: array of scores [N_steps, N_samples]
            - scores_mean: mean score at each step [N_steps]
            - scores_std: std of scores at each step [N_steps]
            - n_samples: number of samples
    """
    if coords_traj is None or len(coords_traj) == 0:
        return {'error': 'No coordinate trajectory provided'}
    
    # Create guidance object if not provided
    if guidance is None:
        try:
            from boltzgen.model.modules.guidance import create_geometric_guidance
            guidance = create_geometric_guidance(
                property_type="hydrophobicity",
                higher_is_better=True,
                guidance_scale=0.0,  # Scale doesn't matter for scoring
                temperature=temperature,
                enabled=False,  # Not actually applying guidance
            )
            # Move to same device as coords
            device = coords_traj[0].device if hasattr(coords_traj[0], 'device') else 'cpu'
            guidance = guidance.to(device)
        except ImportError:
            return {'error': 'Could not import create_geometric_guidance'}
    
    all_scores = []  # Will be [N_steps, N_samples]
    
    with torch.no_grad():
        for i, coords in enumerate(coords_traj):
            # Ensure proper batch dimension
            if coords.dim() == 2:
                coords = coords.unsqueeze(0)
            
            try:
                # Score ALL samples in the batch
                scores = guidance.compute_geometric_score(coords, feats)  # [B]
                all_scores.append(scores.cpu().numpy())
            except Exception as e:
                # If scoring fails, append NaNs
                n_samples = coords.shape[0]
                all_scores.append(np.full(n_samples, np.nan))
    
    # Stack to [N_steps, N_samples]
    scores_arr = np.array(all_scores)
    n_steps, n_samples = scores_arr.shape
    
    # Compute statistics across samples at each step
    scores_mean = np.nanmean(scores_arr, axis=1)
    scores_std = np.nanstd(scores_arr, axis=1)
    
    return {
        'steps': np.arange(n_steps),
        'scores_all': scores_arr,  # [N_steps, N_samples]
        'scores_mean': scores_mean,  # [N_steps]
        'scores_std': scores_std,  # [N_steps]
        'n_samples': n_samples,
        # For backwards compatibility
        'scores': scores_mean,
        'initial_score': float(scores_mean[0]) if len(scores_mean) > 0 else float('nan'),
        'final_score': float(scores_mean[-1]) if len(scores_mean) > 0 else float('nan'),
    }


def plot_trajectory_hydrophobicity_comparison(
    traj_unguided: Dict[str, np.ndarray],
    traj_guided: Dict[str, np.ndarray],
    traj_guided_x0: Optional[Dict[str, np.ndarray]] = None,
    title: str = "Hydrophobicity Through Diffusion",
    show_individual: bool = False,
):
    """Plot hydrophobicity score distributions across diffusion trajectory.
    
    Shows mean trajectory with shaded ±1 std region for all samples.
    
    Args:
        traj_unguided: Output from compute_trajectory_hydrophobicity for unguided run
        traj_guided: Output from compute_trajectory_hydrophobicity for guided run (noisy coords)
        traj_guided_x0: Output from compute_trajectory_hydrophobicity for guided run's 
                        clean predicted coords (x̂_0) - this is what reconstruction guidance optimizes
        title: Plot title
        show_individual: If True, also plot individual sample trajectories as thin lines
        
    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n_samples_str = ""
    
    # Plot unguided (blue)
    if 'scores_mean' in traj_unguided and len(traj_unguided['scores_mean']) > 0:
        steps = traj_unguided['steps']
        mean = traj_unguided['scores_mean']
        std = traj_unguided['scores_std']
        n_samples = traj_unguided.get('n_samples', 1)
        
        # Shaded std region
        ax.fill_between(steps, mean - std, mean + std, color='blue', alpha=0.2)
        # Mean line
        ax.plot(steps, mean, 'b-', linewidth=2, label=f'Unguided (n={n_samples})')
        
        # Individual trajectories
        if show_individual and 'scores_all' in traj_unguided:
            for i in range(traj_unguided['scores_all'].shape[1]):
                ax.plot(steps, traj_unguided['scores_all'][:, i], 'b-', alpha=0.1, linewidth=0.5)
        
        n_samples_str = f"n={n_samples}"
    
    # Plot guided noisy coords (red)
    if 'scores_mean' in traj_guided and len(traj_guided['scores_mean']) > 0:
        steps = traj_guided['steps']
        mean = traj_guided['scores_mean']
        std = traj_guided['scores_std']
        n_samples = traj_guided.get('n_samples', 1)
        
        # Shaded std region
        ax.fill_between(steps, mean - std, mean + std, color='red', alpha=0.2)
        # Mean line
        ax.plot(steps, mean, 'r-', linewidth=2, label=f'Guided noisy (n={n_samples})')
        
        # Individual trajectories
        if show_individual and 'scores_all' in traj_guided:
            for i in range(traj_guided['scores_all'].shape[1]):
                ax.plot(steps, traj_guided['scores_all'][:, i], 'r-', alpha=0.1, linewidth=0.5)
    
    # Plot guided x0 (clean predicted) - this is what we're optimizing! (green)
    if traj_guided_x0 is not None and 'scores_mean' in traj_guided_x0 and len(traj_guided_x0['scores_mean']) > 0:
        steps = traj_guided_x0['steps']
        mean = traj_guided_x0['scores_mean']
        std = traj_guided_x0['scores_std']
        n_samples = traj_guided_x0.get('n_samples', 1)
        
        # Shaded std region
        ax.fill_between(steps, mean - std, mean + std, color='green', alpha=0.2)
        # Mean line - dashed to distinguish
        ax.plot(steps, mean, 'g--', linewidth=2, label=f'Guided x̂₀ (optimized, n={n_samples})')
        
        # Individual trajectories
        if show_individual and 'scores_all' in traj_guided_x0:
            for i in range(traj_guided_x0['scores_all'].shape[1]):
                ax.plot(steps, traj_guided_x0['scores_all'][:, i], 'g--', alpha=0.1, linewidth=0.5)
    
    # Find max step for x-axis
    max_step = 0
    if 'steps' in traj_unguided and len(traj_unguided['steps']) > 0:
        max_step = max(max_step, max(traj_unguided['steps']))
    if 'steps' in traj_guided and len(traj_guided['steps']) > 0:
        max_step = max(max_step, max(traj_guided['steps']))
    if traj_guided_x0 is not None and 'steps' in traj_guided_x0 and len(traj_guided_x0['steps']) > 0:
        max_step = max(max_step, max(traj_guided_x0['steps']))
    
    ax.set_xlabel('Diffusion Step')
    ax.set_ylabel('Hydrophobicity Score')
    ax.set_title(f"{title}\n(mean ± std across all samples)")
    ax.set_xlim(0, max_step if max_step > 0 else 1)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Add final value annotations
    y_offset = 5
    if 'scores_mean' in traj_unguided and len(traj_unguided['scores_mean']) > 0:
        final_mean = traj_unguided['scores_mean'][-1]
        final_std = traj_unguided['scores_std'][-1]
        ax.annotate(f'{final_mean:.2f}±{final_std:.2f}', 
                   xy=(traj_unguided['steps'][-1], final_mean),
                   xytext=(5, y_offset), textcoords='offset points', color='blue', fontsize=9)
    
    if 'scores_mean' in traj_guided and len(traj_guided['scores_mean']) > 0:
        final_mean = traj_guided['scores_mean'][-1]
        final_std = traj_guided['scores_std'][-1]
        ax.annotate(f'{final_mean:.2f}±{final_std:.2f}', 
                   xy=(traj_guided['steps'][-1], final_mean),
                   xytext=(5, -10), textcoords='offset points', color='red', fontsize=9)
    
    if traj_guided_x0 is not None and 'scores_mean' in traj_guided_x0 and len(traj_guided_x0['scores_mean']) > 0:
        final_mean = traj_guided_x0['scores_mean'][-1]
        final_std = traj_guided_x0['scores_std'][-1]
        ax.annotate(f'{final_mean:.2f}±{final_std:.2f}', 
                   xy=(traj_guided_x0['steps'][-1], final_mean),
                   xytext=(5, -25), textcoords='offset points', color='green', fontsize=9)
    
    plt.tight_layout()
    return fig


def print_trajectory_summary(
    traj_unguided: Dict[str, np.ndarray],
    traj_guided: Dict[str, np.ndarray],
):
    """Print summary of trajectory hydrophobicity comparison."""
    print("\n" + "="*60)
    print("TRAJECTORY HYDROPHOBICITY SUMMARY")
    print("="*60)
    
    for label, traj in [("Unguided", traj_unguided), ("Guided", traj_guided)]:
        if 'error' in traj:
            print(f"\n{label}: {traj['error']}")
            continue
            
        if 'scores_mean' not in traj or len(traj['scores_mean']) == 0:
            print(f"\n{label}: No scores available")
            continue
        
        n_samples = traj.get('n_samples', 1)
        initial_mean = traj['scores_mean'][0]
        initial_std = traj['scores_std'][0]
        final_mean = traj['scores_mean'][-1]
        final_std = traj['scores_std'][-1]
            
        print(f"\n{label} (n={n_samples} samples):")
        print(f"  Initial: {initial_mean:.3f} ± {initial_std:.3f}")
        print(f"  Final:   {final_mean:.3f} ± {final_std:.3f}")
        print(f"  Change:  {final_mean - initial_mean:+.3f}")
        
        # Overall trajectory stats
        all_means = traj['scores_mean']
        print(f"  Trajectory range: [{all_means.min():.3f}, {all_means.max():.3f}]")


# =============================================================================
# GUIDANCE DIAGNOSTICS
# =============================================================================

def extract_guidance_info(pred: Dict) -> Optional[List[Dict]]:
    """Extract guidance information from model output."""
    if 'guidance_info' in pred:
        return pred['guidance_info']
    
    # Check if it's in structure_module
    if hasattr(pred, 'get') and 'coords' in pred:
        # Try to get from model if available
        return None
    
    return None


def analyze_guidance_trajectory(guidance_info: List[Dict]) -> Dict[str, np.ndarray]:
    """Extract per-step guidance metrics from guidance_info."""
    if not guidance_info:
        return {}
    
    steps = [g.get('step', i) for i, g in enumerate(guidance_info)]
    sigmas = [g.get('sigma', 0.0) for g in guidance_info]
    grad_norms = [g.get('coord_grad_norm', 0.0) for g in guidance_info]
    scores = [g.get('score', 0.0) for g in guidance_info]
    
    # Compute progress
    if steps:
        max_step = max(steps)
        progress = [s / max(max_step, 1) for s in steps]
    else:
        progress = []
    
    return {
        'steps': np.array(steps),
        'progress': np.array(progress),
        'sigmas': np.array(sigmas),
        'grad_norms': np.array(grad_norms),
        'scores': np.array(scores),
    }


def compute_coordinate_trajectory_metrics(coords_traj: Optional[List[Tensor]], 
                                         atom_mask: Optional[Tensor] = None) -> Dict[str, np.ndarray]:
    """Compute metrics from coordinate trajectory."""
    if coords_traj is None or len(coords_traj) < 2:
        return {}
    
    # Convert to numpy for analysis
    traj = [c.cpu().numpy() if isinstance(c, Tensor) else c for c in coords_traj]
    
    # Compute RMSD between consecutive steps
    coord_changes = []
    for i in range(1, len(traj)):
        if atom_mask is not None:
            mask = atom_mask.cpu().numpy() if isinstance(atom_mask, Tensor) else atom_mask
            if mask.ndim == 1:
                mask = mask[:, None]
            diff = (traj[i] - traj[i-1]) * mask
        else:
            diff = traj[i] - traj[i-1]
        rmsd = np.sqrt(np.mean(diff ** 2))
        coord_changes.append(rmsd)
    
    # Compute RMSD from initial
    rmsd_from_start = []
    for i in range(1, len(traj)):
        if atom_mask is not None:
            mask = atom_mask.cpu().numpy() if isinstance(atom_mask, Tensor) else atom_mask
            if mask.ndim == 1:
                mask = mask[:, None]
            diff = (traj[i] - traj[0]) * mask
        else:
            diff = traj[i] - traj[0]
        rmsd = np.sqrt(np.mean(diff ** 2))
        rmsd_from_start.append(rmsd)
    
    return {
        'coord_change_rmsd': np.array(coord_changes),
        'rmsd_from_start': np.array(rmsd_from_start),
    }


def plot_guidance_analysis(guidance_info: List[Dict], 
                           coords_traj: Optional[List[Tensor]] = None,
                           title: str = "Guidance Analysis"):
    """Comprehensive guidance analysis - single consolidated view.
    
    Shows:
    1. Score trajectory through diffusion
    2. Gradient magnitude with sigma context (dual y-axis)
    3. Relative guidance strength (gradient/sigma ratio)
    4. Gradient vs coordinate change (impact)
    """
    import matplotlib.pyplot as plt
    
    traj_data = analyze_guidance_trajectory(guidance_info)
    if not traj_data or len(traj_data['steps']) == 0:
        print("No guidance info available")
        return None
    
    steps = traj_data['steps']
    grad_norms = traj_data['grad_norms']
    sigmas = traj_data['sigmas']
    scores = traj_data['scores']
    max_step = max(steps) if len(steps) > 0 else 1
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Score trajectory
    ax = axes[0, 0]
    ax.plot(steps, scores, 'b-', linewidth=2, marker='o', markersize=3, alpha=0.8)
    ax.set_xlabel('Diffusion Step')
    ax.set_ylabel('Hydrophobicity Score')
    ax.set_title('Score Through Diffusion')
    ax.set_xlim(0, max_step)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    # Annotate final score
    if len(scores) > 0:
        ax.annotate(f'Final: {scores[-1]:.3f}', xy=(steps[-1], scores[-1]),
                   xytext=(-50, 10), textcoords='offset points', fontsize=9)
    
    # Plot 2: Gradient norm with sigma context (same y-axis for direct comparison)
    ax = axes[0, 1]
    ax.plot(steps, grad_norms, 'r-', linewidth=2, label='Gradient Norm', alpha=0.8)
    ax.plot(steps, sigmas, 'b--', linewidth=2, label='Sigma (noise level)', alpha=0.8)
    ax.set_xlabel('Diffusion Step')
    ax.set_ylabel('Magnitude')
    ax.set_xlim(0, max_step)
    # Use log scale if we have positive values
    all_vals = np.concatenate([grad_norms[grad_norms > 0], sigmas[sigmas > 0]]) if np.any(grad_norms > 0) or np.any(sigmas > 0) else np.array([1])
    if len(all_vals) > 0:
        ax.set_yscale('log')
    ax.legend(loc='upper right')
    ax.set_title('Gradient Magnitude vs Noise Level')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Relative guidance strength (gradient/sigma)
    ax = axes[1, 0]
    ratio = grad_norms / np.maximum(sigmas, 1e-6)
    ax.plot(steps, ratio, 'g-', linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Diffusion Step')
    ax.set_ylabel('Gradient / Sigma')
    ax.set_title('Relative Guidance Strength\n(gradient normalized by noise level)')
    ax.set_xlim(0, max_step)
    ax.grid(True, alpha=0.3)
    if np.any(ratio > 0):
        ax.set_yscale('log')
    
    # Plot 4: Gradient vs coordinate change
    ax = axes[1, 1]
    if coords_traj is not None and len(coords_traj) > 1:
        coord_metrics = compute_coordinate_trajectory_metrics(coords_traj)
        if 'coord_change_rmsd' in coord_metrics and len(coord_metrics['coord_change_rmsd']) > 0:
            coord_changes = coord_metrics['coord_change_rmsd']
            min_len = min(len(grad_norms), len(coord_changes))
            if min_len > 0:
                ax.plot(steps[:min_len], coord_changes[:min_len], 'b-', 
                       linewidth=2, label='Total coord change', alpha=0.8)
                ax.plot(steps[:min_len], grad_norms[:min_len], 'r-', 
                       linewidth=2, label='Guidance gradient', alpha=0.8)
                ax.set_xlabel('Diffusion Step')
                ax.set_ylabel('Magnitude')
                ax.set_title('Guidance Impact vs Denoising')
                ax.set_xlim(0, max(steps[:min_len]) if min_len > 0 else 1)
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')
        else:
            ax.text(0.5, 0.5, 'No coord trajectory data', ha='center', va='center', transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, 'No coord trajectory data', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Guidance Impact vs Total Coord Change')
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


# Keep old functions as aliases for backwards compatibility
def plot_guidance_diagnostics(guidance_info: List[Dict], 
                              coords_traj: Optional[List[Tensor]] = None,
                              title_prefix: str = ""):
    """Deprecated: Use plot_guidance_analysis instead."""
    return plot_guidance_analysis(guidance_info, coords_traj, title=f"{title_prefix}Guidance Analysis")


def plot_gradient_context(guidance_info: List[Dict], 
                          coords_traj: Optional[List[Tensor]] = None,
                          title: str = "Gradient Magnitude in Context"):
    """Deprecated: Use plot_guidance_analysis instead."""
    return plot_guidance_analysis(guidance_info, coords_traj, title=title)


def plot_guidance_comparison(guidance_info_unguided: Optional[List[Dict]],
                            guidance_info_guided: List[Dict],
                            title: str = "Guidance Comparison"):
    """Compare guidance metrics between unguided and guided runs."""
    import matplotlib.pyplot as plt
    
    if not guidance_info_guided:
        print("No guided guidance info available")
        return None
    
    traj_guided = analyze_guidance_trajectory(guidance_info_guided)
    traj_unguided = analyze_guidance_trajectory(guidance_info_unguided) if guidance_info_unguided else {}
    
    # Find max step for consistent x-axis
    max_step = max(traj_guided['steps']) if len(traj_guided['steps']) > 0 else 1
    if traj_unguided and len(traj_unguided.get('steps', [])) > 0:
        max_step = max(max_step, max(traj_unguided['steps']))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Score comparison
    ax = axes[0, 0]
    if traj_unguided and len(traj_unguided.get('steps', [])) > 0:
        ax.plot(traj_unguided['steps'], traj_unguided['scores'], 'o-', 
               label='Unguided', linewidth=2, markersize=4, alpha=0.7)
    ax.plot(traj_guided['steps'], traj_guided['scores'], 'o-', 
           label='Guided', linewidth=2, markersize=4, color='red')
    ax.set_xlabel('Diffusion Step')
    ax.set_ylabel('Hydrophobicity Score')
    ax.set_title('Score Trajectory Comparison')
    ax.set_xlim(0, max_step)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 2: Gradient norms (guided only)
    ax = axes[0, 1]
    ax.plot(traj_guided['steps'], traj_guided['grad_norms'], 'o-', 
           linewidth=2, markersize=4, color='red')
    ax.set_xlabel('Diffusion Step')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Guided: Gradient Magnitude')
    ax.set_xlim(0, max_step)
    ax.grid(True, alpha=0.3)
    if np.any(traj_guided['grad_norms'] > 0):
        ax.set_yscale('log')
    
    # Plot 3: Score vs Progress
    ax = axes[1, 0]
    if traj_unguided and len(traj_unguided.get('progress', [])) > 0:
        ax.plot(traj_unguided['progress'], traj_unguided['scores'], 'o-', 
               label='Unguided', linewidth=2, markersize=4, alpha=0.7)
    ax.plot(traj_guided['progress'], traj_guided['scores'], 'o-', 
           label='Guided', linewidth=2, markersize=4, color='red')
    ax.set_xlabel('Progress (0=start, 1=end)')
    ax.set_ylabel('Hydrophobicity Score')
    ax.set_title('Score vs Progress')
    ax.set_xlim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 4: Sigma (noise level) over steps
    ax = axes[1, 1]
    if traj_unguided and len(traj_unguided.get('steps', [])) > 0:
        ax.plot(traj_unguided['steps'], traj_unguided['sigmas'], 'o-', 
               label='Unguided', linewidth=2, markersize=4, alpha=0.7)
    ax.plot(traj_guided['steps'], traj_guided['sigmas'], 'o-', 
           label='Guided', linewidth=2, markersize=4, color='red')
    ax.set_xlabel('Diffusion Step')
    ax.set_ylabel('Sigma (Noise Level)')
    ax.set_title('Noise Schedule')
    ax.set_xlim(0, max_step)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def diagnose_guidance_effectiveness(guidance_info: List[Dict],
                                   initial_score: Optional[float] = None,
                                   final_score: Optional[float] = None) -> Dict:
    """Diagnose whether guidance is working correctly."""
    if not guidance_info:
        return {'error': 'No guidance info available'}
    
    traj = analyze_guidance_trajectory(guidance_info)
    
    diagnostics = {
        'guidance_applied': len(guidance_info) > 0,
        'num_steps_with_guidance': len(guidance_info),
        'avg_gradient_norm': float(np.mean(traj['grad_norms'])) if len(traj['grad_norms']) > 0 else 0.0,
        'max_gradient_norm': float(np.max(traj['grad_norms'])) if len(traj['grad_norms']) > 0 else 0.0,
        'min_gradient_norm': float(np.min(traj['grad_norms'])) if len(traj['grad_norms']) > 0 else 0.0,
        'score_trend': 'unknown',
        'score_change': None,
    }
    
    # Analyze score trend
    if len(traj['scores']) > 0:
        scores = traj['scores']
        if initial_score is not None and final_score is not None:
            diagnostics['score_change'] = float(final_score - initial_score)
            diagnostics['score_trend'] = 'increasing' if final_score > initial_score else 'decreasing'
        elif len(scores) > 1:
            # Use first and last scores
            diagnostics['score_change'] = float(scores[-1] - scores[0])
            diagnostics['score_trend'] = 'increasing' if scores[-1] > scores[0] else 'decreasing'
    
    # Check if gradients are non-zero
    if diagnostics['max_gradient_norm'] < 1e-6:
        diagnostics['warning'] = 'Gradients are very small - guidance may not be effective'
    
    return diagnostics


def print_guidance_diagnostics(guidance_info: List[Dict], label: str = "Guided"):
    """Print diagnostic information about guidance."""
    if not guidance_info:
        print(f"\n{label}: No guidance info available")
        return
    
    traj = analyze_guidance_trajectory(guidance_info)
    diag = diagnose_guidance_effectiveness(guidance_info)
    
    print(f"\n{'='*60}")
    print(f"{label.upper()} GUIDANCE DIAGNOSTICS")
    print(f"{'='*60}")
    print(f"Steps with guidance: {diag['num_steps_with_guidance']}")
    print(f"Gradient norms: avg={diag['avg_gradient_norm']:.2e}, "
          f"min={diag['min_gradient_norm']:.2e}, max={diag['max_gradient_norm']:.2e}")
    
    if len(traj['scores']) > 0:
        print(f"Score range: [{traj['scores'].min():.3f}, {traj['scores'].max():.3f}]")
        print(f"Initial score: {traj['scores'][0]:.3f}")
        print(f"Final score: {traj['scores'][-1]:.3f}")
        if diag['score_change'] is not None:
            print(f"Score change: {diag['score_change']:+.3f} ({diag['score_trend']})")
    
    if 'warning' in diag:
        print(f"⚠️  WARNING: {diag['warning']}")
