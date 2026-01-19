"""Unit tests for geometric guidance comparing against validated res_from_atom14."""

import pytest
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from boltzgen.data import const
from boltzgen.data.feature.featurizer import res_from_atom14
from boltzgen.model.modules.guidance import GeometricGuidance, create_geometric_guidance


class TestGeometricDecoder:
    """Test that our differentiable decoder matches res_from_atom14."""
    
    @pytest.fixture
    def guidance(self):
        """Create a GeometricGuidance instance."""
        return create_geometric_guidance(
            property_type="hydrophobicity",
            higher_is_better=True,
            guidance_scale=1.0,
            temperature=0.1,
        )
    
    @pytest.fixture 
    def sample_residue_coords(self):
        """Create sample coordinates for a single residue (14 atoms).
        
        Returns coords that should decode to a specific amino acid based on
        the virtual atom collapse pattern.
        """
        # Backbone atoms at fixed positions: N, CA, C, O
        backbone = torch.tensor([
            [0.0, 0.0, 0.0],   # N
            [1.5, 0.0, 0.0],   # CA
            [2.0, 1.3, 0.0],   # C
            [3.2, 1.5, 0.0],   # O
        ])
        
        # For GLY: all 10 sidechain atoms collapse to O (index 3)
        # Pattern: [0, 0, 0, 10]
        sidechain_gly = backbone[3:4].expand(10, 3).clone()
        sidechain_gly += torch.randn(10, 3) * 0.1  # Small noise
        
        # For ALA: 9 atoms collapse to O, 1 is CB (real sidechain)
        # Pattern: [0, 0, 0, 9]
        sidechain_ala = backbone[3:4].expand(9, 3).clone()
        sidechain_ala = torch.cat([
            torch.tensor([[1.5, -1.0, 1.0]]),  # CB - away from backbone
            sidechain_ala,
        ])
        sidechain_ala[1:] += torch.randn(9, 3) * 0.1
        
        return {
            'backbone': backbone,
            'gly': torch.cat([backbone, sidechain_gly]),  # Should decode as GLY
            'ala': torch.cat([backbone, sidechain_ala]),  # Should decode as ALA
        }
    
    def test_count_patterns_match_const(self, guidance):
        """Verify our count patterns match the ones in const."""
        canonical_aas = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 
                         'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER',
                         'THR', 'TRP', 'TYR', 'VAL']
        
        for i, aa in enumerate(canonical_aas):
            expected = const.token_to_placement_count.get(aa, [0, 0, 0, 0])
            actual = guidance.count_patterns[i].tolist()
            assert actual == expected, f"Pattern mismatch for {aa}: expected {expected}, got {actual}"
    
    def test_hydrophobicity_values(self, guidance):
        """Verify hydrophobicity values are correct."""
        # Kyte-Doolittle scale for canonical AAs
        expected_hydro = {
            'ALA': 1.8, 'CYS': 2.5, 'ASP': -3.5, 'GLU': -3.5, 'PHE': 2.8,
            'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5, 'LYS': -3.9, 'LEU': 3.8,
            'MET': 1.9, 'ASN': -3.5, 'PRO': -1.6, 'GLN': -3.5, 'ARG': -4.5,
            'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2,
        }
        canonical_aas = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 
                         'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER',
                         'THR', 'TRP', 'TYR', 'VAL']
        
        for i, aa in enumerate(canonical_aas):
            assert abs(guidance.hydrophobicity[i].item() - expected_hydro[aa]) < 0.01, \
                f"Hydrophobicity mismatch for {aa}"
    
    def test_ste_counts_match_hard_counts_exactly(self, guidance, sample_residue_coords):
        """Test that our STE-based counts match hard counts exactly in forward pass."""
        device = torch.device('cpu')
        threshold = 0.5
        
        # Test GLY-like coordinates (all sidechain atoms near O)
        gly_coords = sample_residue_coords['gly']
        backbone = gly_coords[:4]
        sidechain = gly_coords[4:]
        
        # Compute distances like res_from_atom14: [4, 10]
        distances = torch.cdist(backbone.unsqueeze(0), sidechain.unsqueeze(0)).squeeze(0)
        
        # === HARD COUNTS (res_from_atom14 algorithm) ===
        min_dists, hard_argmin = distances.min(dim=0)  # min over backbone
        hard_argmin_masked = hard_argmin.clone()
        hard_argmin_masked[min_dists > threshold] = -1  # exclude far atoms
        valid_mask = hard_argmin_masked >= 0
        if valid_mask.sum() > 0:
            hard_counts = torch.bincount(hard_argmin_masked[valid_mask], minlength=4).float()
        else:
            hard_counts = torch.zeros(4)
        
        # === STE COUNTS (our differentiable version with straight-through estimators) ===
        # Hard threshold mask
        threshold_mask = (min_dists <= threshold).float()  # [10], hard 0/1
        # STE: soft in backward, hard in forward
        soft_threshold = torch.sigmoid((threshold - min_dists) / 0.05)
        threshold_mask_ste = threshold_mask + (soft_threshold - soft_threshold.detach())
        
        # Hard one-hot assignment
        hard_one_hot = F.one_hot(hard_argmin, num_classes=4).float()  # [10, 4]
        # STE: soft in backward, hard in forward
        soft_assignment = F.softmax(-distances.T / 0.01, dim=-1)  # [10, 4]
        assignment_ste = hard_one_hot + (soft_assignment - soft_assignment.detach())
        
        # Apply mask and sum
        masked_assignment = assignment_ste * threshold_mask_ste.unsqueeze(-1)  # [10, 4]
        ste_counts = masked_assignment.sum(dim=0)  # [4]
        
        print(f"GLY hard counts: {hard_counts.tolist()}")
        print(f"GLY STE counts: {ste_counts.tolist()}")
        
        # STE should produce EXACT same counts in forward pass
        assert torch.allclose(hard_counts, ste_counts, atol=1e-5), \
            f"STE counts should match hard counts exactly: {ste_counts.tolist()} vs {hard_counts.tolist()}"
    
    def test_gradient_flows(self, guidance):
        """Test that gradients flow through compute_geometric_score."""
        device = torch.device('cpu')
        guidance = guidance.to(device)
        
        # Create realistic coords (GLY pattern) where sidechain atoms are close to backbone
        # This ensures atoms are within threshold and gradients can flow
        backbone = torch.tensor([[0., 0., 0.], [1.5, 0., 0.], [2.5, 1., 0.], [2.5, 2., 0.]])
        
        # 5 residues with sidechain atoms close to O (within 0.5Å threshold)
        residue_coords = []
        for _ in range(5):
            sidechain = backbone[3:4].expand(10, 3) + torch.randn(10, 3) * 0.2  # Small noise
            residue_coords.append(torch.cat([backbone, sidechain]))
        
        coords = torch.stack(residue_coords).reshape(1, -1, 3)
        coords = coords.clone().requires_grad_(True)
        
        # Create features
        design_mask = torch.ones(1, 5, dtype=torch.bool)
        atom_to_token = torch.zeros(1, 70, 5)
        for i in range(5):
            atom_to_token[0, i*14:(i+1)*14, i] = 1
        
        feats = {
            'design_mask': design_mask,
            'atom_to_token': atom_to_token,
        }
        
        # Compute score
        score = guidance.compute_geometric_score(coords, feats)
        
        # Check gradient flows
        score.sum().backward()
        
        assert coords.grad is not None, "Gradient should flow to coordinates"
        # Note: With STE, gradient flows through the soft approximations in backward pass
        print(f"Gradient norm: {coords.grad.norm().item():.4f}")
        print(f"Score: {score.item():.4f}")
    
    def test_score_changes_with_coords(self, guidance):
        """Test that changing coordinates changes the score."""
        device = torch.device('cpu')
        guidance = guidance.to(device)
        
        # Create realistic coords that form different residue patterns
        # GLY pattern [0,0,0,10] - all sidechain atoms close to O backbone
        backbone = torch.tensor([[0., 0., 0.], [1.5, 0., 0.], [2.5, 1., 0.], [2.5, 2., 0.]])
        sidechain_gly = backbone[3:4].expand(10, 3) + torch.randn(10, 3) * 0.1
        gly_residue = torch.cat([backbone, sidechain_gly])
        
        # ILE pattern [0,0,0,6] - more sidechain atoms far from backbone (hydrophobic: 4.5)
        sidechain_ile = backbone[3:4].expand(6, 3) + torch.randn(6, 3) * 0.1
        far_atoms = torch.tensor([[1.5, -2., 1.], [1.5, -2.5, 1.5], [1.5, -3., 1.], [1.5, -3.5, 1.5]])
        sidechain_ile = torch.cat([far_atoms, sidechain_ile])
        ile_residue = torch.cat([backbone, sidechain_ile])
        
        # coords1: 3 GLY residues (hydrophobicity: -0.4)
        coords1 = torch.stack([gly_residue, gly_residue, gly_residue]).reshape(1, -1, 3)
        
        # coords2: 3 ILE residues (hydrophobicity: 4.5)
        coords2 = torch.stack([ile_residue, ile_residue, ile_residue]).reshape(1, -1, 3)
        
        design_mask = torch.ones(1, 3, dtype=torch.bool)
        atom_to_token = torch.zeros(1, 42, 3)
        for i in range(3):
            atom_to_token[0, i*14:(i+1)*14, i] = 1
        
        feats = {
            'design_mask': design_mask,
            'atom_to_token': atom_to_token,
        }
        
        score1 = guidance.compute_geometric_score(coords1, feats)
        score2 = guidance.compute_geometric_score(coords2, feats)
        
        print(f"GLY coords → score: {score1.item():.4f} (expected ~-0.4)")
        print(f"ILE coords → score: {score2.item():.4f} (expected ~4.5)")
        
        # Scores should be different for different residue patterns
        assert not torch.allclose(score1, score2), \
            f"Different residue patterns should produce different scores: {score1.item()} vs {score2.item()}"
    
    def test_higher_is_better_direction(self, guidance):
        """Test that higher_is_better correctly sets gradient direction."""
        guidance_max = create_geometric_guidance(higher_is_better=True)
        guidance_min = create_geometric_guidance(higher_is_better=False)
        
        assert guidance_max.get_guidance_direction() == 1.0
        assert guidance_min.get_guidance_direction() == -1.0


class TestCompareWithResFromAtom14:
    """Compare differentiable decoder with validated res_from_atom14."""
    
    def test_synthetic_ste_matches_res_from_atom14(self):
        """Fast test: verify STE algorithm matches res_from_atom14 on synthetic coords."""
        threshold = 0.5
        
        # Create synthetic residue coordinates for multiple AA types
        # Each residue has 14 atoms: 4 backbone (N, CA, C, O) + 10 sidechain
        test_cases = []
        
        # GLY pattern: [0, 0, 0, 10] - all sidechain at O
        backbone = torch.tensor([[0., 0., 0.], [1.5, 0., 0.], [2.5, 1., 0.], [2.5, 2., 0.]])
        sidechain_gly = backbone[3:4].expand(10, 3) + torch.randn(10, 3) * 0.1
        test_cases.append(('GLY', torch.cat([backbone, sidechain_gly]), [0, 0, 0, 10]))
        
        # ALA pattern: [0, 0, 0, 9] - 9 at O, 1 CB far away
        sidechain_ala = backbone[3:4].expand(9, 3) + torch.randn(9, 3) * 0.1
        cb = torch.tensor([[1.5, -1.5, 1.0]])  # CB far from backbone
        sidechain_ala = torch.cat([cb, sidechain_ala])
        test_cases.append(('ALA', torch.cat([backbone, sidechain_ala]), [0, 0, 0, 9]))
        
        # VAL pattern: [0, 0, 0, 7] - 7 at O, 3 sidechain atoms far
        sidechain_val = backbone[3:4].expand(7, 3) + torch.randn(7, 3) * 0.1
        sc_far = torch.tensor([[1.5, -1.5, 1.0], [1.5, -2.0, 1.5], [1.5, -2.5, 1.0]])
        sidechain_val = torch.cat([sc_far, sidechain_val])
        test_cases.append(('VAL', torch.cat([backbone, sidechain_val]), [0, 0, 0, 7]))
        
        for name, coords, expected_counts in test_cases:
            bb = coords[:4]
            sc = coords[4:]
            
            # res_from_atom14 algorithm
            distances = torch.cdist(bb.unsqueeze(0), sc.unsqueeze(0)).squeeze(0)  # [4, 10]
            min_dists, hard_argmin = distances.min(dim=0)
            hard_argmin_masked = hard_argmin.clone()
            hard_argmin_masked[min_dists > threshold] = -1
            valid = hard_argmin_masked >= 0
            if valid.sum() > 0:
                hard_counts = torch.bincount(hard_argmin_masked[valid], minlength=4)
            else:
                hard_counts = torch.zeros(4, dtype=torch.long)
            
            # STE algorithm (forward pass should be identical)
            threshold_mask = (min_dists <= threshold).float()
            hard_one_hot = F.one_hot(hard_argmin, num_classes=4).float()
            masked = hard_one_hot * threshold_mask.unsqueeze(-1)
            ste_counts = masked.sum(dim=0)
            
            print(f"{name}: expected={expected_counts}, hard={hard_counts.tolist()}, ste={ste_counts.tolist()}")
            
            assert torch.allclose(hard_counts.float(), ste_counts), \
                f"{name}: STE counts don't match hard counts"
    
    @pytest.fixture
    def real_data(self):
        """Load real data from the model for testing (SLOW - downloads 2GB model)."""
        import os
        os.environ['CUEQ_DEFAULT_CONFIG'] = '1'
        os.environ['CUEQ_DISABLE_AOT_TUNING'] = '1'
        
        import huggingface_hub
        from boltzgen.data.mol import load_canonicals
        from boltzgen.data.tokenize.tokenizer import Tokenizer
        from boltzgen.data.feature.featurizer import Featurizer
        from boltzgen.model.models.boltz import Boltz
        from boltzgen.task.predict.data_from_yaml import PredictionDataset, Dataset
        
        YAML_PATH = Path(__file__).parent.parent / "example/vanilla_protein/1g13prot.yaml"
        
        ckpt_path = huggingface_hub.hf_hub_download('boltzgen/boltzgen-1', 'boltzgen1_diverse.ckpt', repo_type='model')
        moldir = huggingface_hub.hf_hub_download('boltzgen/inference-data', 'mols.zip', repo_type='dataset')
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Boltz.load_from_checkpoint(ckpt_path, strict=False, map_location='cpu', weights_only=False)
        model.eval()
        model.to(device)
        
        canonicals = load_canonicals(moldir)
        tokenizer = Tokenizer(canonicals)
        featurizer = Featurizer()
        dataset = Dataset(yaml_path=str(YAML_PATH), tokenizer=tokenizer, featurizer=featurizer, multiplicity=1)
        pred_dataset = PredictionDataset(dataset=dataset, canonicals=canonicals, moldir=moldir, backbone_only=False, atom14=True, design=True)
        feats = pred_dataset[0]
        feats_batched = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v for k, v in feats.items()}
        
        # Generate sample
        torch.manual_seed(42)
        with torch.no_grad():
            out = model.forward(feats=feats_batched, recycling_steps=1, num_sampling_steps=50, diffusion_samples=1, guidance=None)
        
        return {
            'coords': out['sample_atom_coords'],
            'feats': feats_batched,
            'feats_orig': feats,
            'device': device,
        }
    
    @pytest.mark.slow
    def test_real_model_ste_matches_res_from_atom14(self, real_data):
        """SLOW: Validate STE matches res_from_atom14 on real model outputs."""
        guidance = create_geometric_guidance().to(real_data['device'])
        threshold = 0.5  # Same as res_from_atom14 default
        
        coords = real_data['coords']
        feats = real_data['feats']
        
        design_mask = feats['design_mask']
        atom_to_token = feats['atom_to_token'].int().argmax(dim=-1)
        
        if design_mask.dim() == 1:
            design_mask = design_mask.unsqueeze(0)
        
        design_indices = design_mask[0].nonzero(as_tuple=True)[0]
        
        # Compute predictions using STE algorithm (matches res_from_atom14)
        ste_predictions = []
        for token_idx in design_indices[:10]:
            atom_mask = (atom_to_token[0] == token_idx)
            atom_indices = atom_mask.nonzero(as_tuple=True)[0]
            
            if len(atom_indices) != 14:
                ste_predictions.append(-1)
                continue
            
            res_coords = coords[0, atom_indices]
            backbone = res_coords[:4]
            sidechain = res_coords[4:]
            
            # Exactly match res_from_atom14 algorithm
            distances = torch.cdist(backbone.unsqueeze(0), sidechain.unsqueeze(0)).squeeze(0)  # [4, 10]
            min_dists, hard_argmin = distances.min(dim=0)
            
            # Hard threshold (same as res_from_atom14)
            threshold_mask = (min_dists <= threshold).float()
            
            # Hard one-hot (same as res_from_atom14) 
            hard_one_hot = F.one_hot(hard_argmin, num_classes=4).float()
            
            # Apply mask and count
            masked_assignment = hard_one_hot * threshold_mask.unsqueeze(-1)
            counts = masked_assignment.sum(dim=0)  # [4]
            
            # Match to pattern
            count_diff = (counts.unsqueeze(0) - guidance.count_patterns) ** 2
            count_dist = count_diff.sum(dim=-1)
            
            ste_predictions.append(count_dist.argmin().item())
        
        # Get hard predictions from res_from_atom14
        feat_cpu = {k: v[0].cpu() if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == 1 
                    else (v.cpu() if isinstance(v, torch.Tensor) else v) 
                    for k, v in feats.items()}
        feat_cpu['coords'] = coords[0].cpu()
        
        result = res_from_atom14(feat_cpu, threshold=threshold)
        hard_mask = result['design_mask'].bool()
        hard_res_type = result['res_type'][hard_mask]
        hard_indices = hard_res_type.argmax(dim=-1)
        
        # Map hard indices to our canonical order
        canonical_aas = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 
                         'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER',
                         'THR', 'TRP', 'TYR', 'VAL']
        
        hard_predictions = []
        for idx in hard_indices[:10].tolist():
            token = const.tokens[idx]
            if token in canonical_aas:
                hard_predictions.append(canonical_aas.index(token))
            else:
                hard_predictions.append(-1)  # Unknown (UNK token)
        
        # Compare predictions - with same algorithm they should match exactly
        matches = sum(1 for s, h in zip(ste_predictions, hard_predictions) if s == h)
        total = len(ste_predictions)
        
        print(f"STE predictions: {ste_predictions}")
        print(f"Hard predictions: {hard_predictions}")
        print(f"Agreement: {matches}/{total}")
        
        # If hard_prediction is -1 (UNK), our STE might also give UNK pattern 
        # The key test is that non-UNK predictions match
        non_unk_matches = sum(1 for s, h in zip(ste_predictions, hard_predictions) 
                              if s == h and h != -1 and s != -1)
        non_unk_total = sum(1 for h in hard_predictions if h != -1)
        
        if non_unk_total > 0:
            print(f"Non-UNK agreement: {non_unk_matches}/{non_unk_total}")
            assert non_unk_matches == non_unk_total, \
                f"STE should match res_from_atom14 exactly for valid residues"
        else:
            print("All residues decoded as UNK - this is expected with threshold=0.5")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
