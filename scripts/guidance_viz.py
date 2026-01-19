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
