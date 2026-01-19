#!/usr/bin/env python3
"""Validate designed sequences by folding them with Boltz2 and comparing confidence scores.

Reads sequences from guidance_results.csv and runs Boltz2 structure prediction on each,
then compares pLDDT (predicted local distance difference test) scores between guided
and unguided designs.

Usage:
    python scripts/validate_with_boltz2.py [--csv guidance_results.csv]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import os
os.environ["CUEQ_DEFAULT_CONFIG"] = "1"
os.environ["CUEQ_DISABLE_AOT_TUNING"] = "1"

import argparse
import csv
import torch
import numpy as np
import huggingface_hub
from scipy import stats
import time
import tempfile

from boltzgen.data.mol import load_canonicals
from boltzgen.data.tokenize.tokenizer import Tokenizer
from boltzgen.data.feature.featurizer import Featurizer
from boltzgen.model.models.boltz import Boltz

def create_yaml_for_sequence(sequence: str, output_path: Path, name: str = "designed"):
    """Create a YAML design spec for a single protein sequence."""
    yaml_content = f"""version: 1
name: {name}
sequences:
  - protein:
      id: A
      sequence: {sequence}
"""
    output_path.write_text(yaml_content)
    return output_path

def fold_sequence(model, sequence: str, tokenizer, featurizer, canonicals, moldir, device):
    """Fold a sequence using Boltz2 and return pLDDT score."""
    from boltzgen.task.predict.data_from_yaml import PredictionDataset, Dataset
    import uuid
    
    # Create temporary YAML for this sequence using BoltzGen format
    # Use UUID to avoid BoltzGen's reserved pattern _\d+\.yaml
    yaml_path = Path(tempfile.gettempdir()) / f"fold_{uuid.uuid4().hex}.yaml"
    yaml_path.write_text(f"""entities:
  - protein:
      id: A
      sequence: {sequence}
""")
    
    try:
        # Create dataset - structure prediction mode (no special atom formats)
        dataset = Dataset(yaml_path=str(yaml_path), tokenizer=tokenizer, featurizer=featurizer, multiplicity=1)
        pred_dataset = PredictionDataset(
            dataset=dataset,
            canonicals=canonicals,
            moldir=str(moldir),
            backbone_only=False,
            atom14=False,
            design=False,  # Structure prediction, not design
        )
        
        feats = pred_dataset[0]
        feats_batched = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v for k, v in feats.items()}
        
        # Run Boltz2 structure prediction
        with torch.no_grad():
            out = model.forward(
                feats=feats_batched,
                recycling_steps=3,  # More recycling for better structure prediction
                num_sampling_steps=50,
                diffusion_samples=1,
            )
        
        # Extract confidence metrics
        result = {
            'plddt': None,
            'ptm': None,
        }
        
        # Use complex_plddt for single-value summary (0-1 scale, multiply by 100 for percentage)
        if 'complex_plddt' in out:
            result['plddt'] = out['complex_plddt'].item() * 100  # Convert to 0-100 scale
        elif 'plddt' in out:
            result['plddt'] = out['plddt'].mean().item() * 100
        
        if 'ptm' in out:
            result['ptm'] = out['ptm'].item()
        
        return result
        
    finally:
        # Cleanup temp file
        yaml_path.unlink(missing_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Validate designed sequences with Boltz2")
    parser.add_argument("--csv", type=str, default="scripts/guidance_results.csv", help="Input CSV from guidance test")
    parser.add_argument("--output", type=str, default="scripts/validation_results.csv", help="Output CSV with confidence scores")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to process per condition")
    args = parser.parse_args()
    
    csv_path = Path(args.csv)
    if not csv_path.exists():
        # Try relative to script
        csv_path = Path(__file__).parent / Path(args.csv).name
    
    if not csv_path.exists():
        print(f"Error: CSV file not found: {args.csv}")
        print("Run `python scripts/guidance_comparison_test.py` first to generate sequences")
        sys.exit(1)
    
    print(f"\n=== Boltz2 Validation ===")
    print(f"  Input: {csv_path}")
    
    # Load sequences
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"  Loaded {len(rows)} sequences")
    
    # Setup Boltz2
    print(f"\n  Loading Boltz2 model...")
    t0 = time.perf_counter()
    
    ckpt_path = huggingface_hub.hf_hub_download("boltzgen/boltzgen-1", "boltz2_conf_final.ckpt", repo_type="model")
    moldir = huggingface_hub.hf_hub_download("boltzgen/inference-data", "mols.zip", repo_type="dataset")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Boltz.load_from_checkpoint(ckpt_path, strict=False, map_location=device, weights_only=False)
    model.eval()
    
    canonicals = load_canonicals(moldir)
    tokenizer = Tokenizer(canonicals)
    featurizer = Featurizer()
    
    print(f"  Model loaded: {time.perf_counter() - t0:.1f}s")
    
    # Process sequences
    results = []
    unguided_rows = [r for r in rows if r['condition'] == 'unguided']
    guided_rows = [r for r in rows if r['condition'] == 'guided']
    
    if args.max_samples:
        unguided_rows = unguided_rows[:args.max_samples]
        guided_rows = guided_rows[:args.max_samples]
    
    all_rows = unguided_rows + guided_rows
    
    print(f"\n  Processing {len(all_rows)} sequences...")
    
    for i, row in enumerate(all_rows):
        seq = row['designed_sequence']
        if not seq or len(seq) < 10:
            print(f"    [{i+1}/{len(all_rows)}] Skipping empty/short sequence")
            continue
        
        t0 = time.perf_counter()
        try:
            conf = fold_sequence(model, seq, tokenizer, featurizer, canonicals, moldir, device)
            elapsed = time.perf_counter() - t0
            
            results.append({
                'condition': row['condition'],
                'sample_id': row['sample_id'],
                'sequence': seq,
                'hydrophobicity': row['hydrophobicity'],
                'plddt': conf['plddt'],
                'ptm': conf['ptm'],
            })
            
            plddt_str = f"{conf['plddt']:.1f}" if conf['plddt'] else "N/A"
            print(f"    [{i+1}/{len(all_rows)}] {row['condition'][:7]} #{row['sample_id']}: pLDDT={plddt_str} ({elapsed:.1f}s)")
            
        except Exception as e:
            print(f"    [{i+1}/{len(all_rows)}] Error: {e}")
            results.append({
                'condition': row['condition'],
                'sample_id': row['sample_id'],
                'sequence': seq,
                'hydrophobicity': row['hydrophobicity'],
                'plddt': None,
                'ptm': None,
            })
    
    # Save results
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent / output_path.name
    
    print(f"\n  Saving results to {output_path}...")
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['condition', 'sample_id', 'sequence', 'hydrophobicity', 'plddt', 'ptm'])
        writer.writeheader()
        writer.writerows(results)
    
    # Summary
    unguided_results = [r for r in results if r['condition'] == 'unguided' and r['plddt'] is not None]
    guided_results = [r for r in results if r['condition'] == 'guided' and r['plddt'] is not None]
    
    if unguided_results and guided_results:
        print(f"\n{'='*60}")
        print(f"  VALIDATION RESULTS")
        print(f"{'='*60}")
        
        # Hydrophobicity (from design)
        u_hydro = [float(r['hydrophobicity']) for r in unguided_results]
        g_hydro = [float(r['hydrophobicity']) for r in guided_results]
        print(f"\n  HYDROPHOBICITY (from design):")
        print(f"    Unguided: {np.mean(u_hydro):.3f} ± {np.std(u_hydro):.3f}")
        print(f"    Guided:   {np.mean(g_hydro):.3f} ± {np.std(g_hydro):.3f}")
        
        # pLDDT (from Boltz2 folding)
        u_plddt = [r['plddt'] for r in unguided_results]
        g_plddt = [r['plddt'] for r in guided_results]
        print(f"\n  pLDDT (Boltz2 confidence, 0-100, higher=better):")
        print(f"    Unguided: {np.mean(u_plddt):.1f} ± {np.std(u_plddt):.1f}")
        print(f"    Guided:   {np.mean(g_plddt):.1f} ± {np.std(g_plddt):.1f}")
        plddt_diff = np.mean(g_plddt) - np.mean(u_plddt)
        print(f"    Difference: {plddt_diff:+.1f} {'(better)' if plddt_diff > 0 else '(worse)' if plddt_diff < 0 else ''}")
        
        # Statistical tests
        if len(u_plddt) == len(g_plddt) and len(u_plddt) > 1:
            t_stat, p_value = stats.ttest_rel(g_plddt, u_plddt)
            print(f"\n  pLDDT paired t-test:")
            print(f"    t-statistic: {t_stat:.3f}")
            print(f"    p-value: {p_value:.4f}")
            
            if p_value < 0.05:
                if plddt_diff > 0:
                    print(f"\n  ✓ Guided sequences fold with HIGHER confidence!")
                else:
                    print(f"\n  ⚠ Guided sequences fold with LOWER confidence")
            else:
                print(f"\n  ~ No significant difference in folding confidence")
        
        # Correlation: does higher hydrophobicity hurt confidence?
        all_hydro = u_hydro + g_hydro
        all_plddt = u_plddt + g_plddt
        if len(all_hydro) > 5:
            corr, p_corr = stats.pearsonr(all_hydro, all_plddt)
            print(f"\n  Hydrophobicity vs pLDDT correlation:")
            print(f"    r = {corr:.3f}, p = {p_corr:.4f}")
            if p_corr < 0.05 and corr < -0.3:
                print(f"    ⚠ Higher hydrophobicity associated with lower confidence")
            elif p_corr < 0.05 and corr > 0.3:
                print(f"    ✓ Higher hydrophobicity associated with higher confidence")
            else:
                print(f"    ~ No strong correlation")
    
    print(f"\n  Results saved to: {output_path}")

if __name__ == "__main__":
    main()
