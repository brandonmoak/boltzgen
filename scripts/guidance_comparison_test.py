#!/usr/bin/env python3
"""Test geometric guidance: 10 unguided vs 10 guided designs, compare hydrophobicity.

This uses the virtual atom representation to decode residue type directly from
coordinates, avoiding the untrained res_type_predictor head.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import os
os.environ["CUEQ_DEFAULT_CONFIG"] = "1"
os.environ["CUEQ_DISABLE_AOT_TUNING"] = "1"

import torch
import torch.nn.functional as F
import numpy as np
import huggingface_hub
from scipy import stats
import time

from boltzgen.data import const
from boltzgen.data.mol import load_canonicals
from boltzgen.data.tokenize.tokenizer import Tokenizer
from boltzgen.data.feature.featurizer import Featurizer, res_from_atom14
from boltzgen.model.models.boltz import Boltz
from boltzgen.model.modules.guidance import (
    create_geometric_guidance,
    enable_profiling,
    get_profiling_stats,
    reset_profiling,
)

# Config
YAML_PATH = Path(__file__).parent.parent / "example/vanilla_protein/1g13prot.yaml"
NUM_SAMPLES = 20  # Full statistical test
BATCH_SIZE = 10  # Number of samples to generate in parallel
SAMPLING_STEPS = 50
GUIDANCE_SCALE = 50.0  # Increased for stronger effect

# Masking toggle: Set to True to apply feature masking (matches notebook behavior)
USE_MASKING = True  # Set to True to enable masking

print("\n=== STARTUP PROFILING ===")
print(f"Masking: {'ENABLED' if USE_MASKING else 'DISABLED'}")
startup_start = time.perf_counter()

# Get checkpoint (cached after first download)
t0 = time.perf_counter()
ckpt_path = huggingface_hub.hf_hub_download("boltzgen/boltzgen-1", "boltzgen1_diverse.ckpt", repo_type="model")
print(f"  HF cache lookup (ckpt): {time.perf_counter() - t0:.2f}s")

# Get moldir (cached after first download)
t0 = time.perf_counter()
moldir = huggingface_hub.hf_hub_download("boltzgen/inference-data", "mols.zip", repo_type="dataset")
print(f"  HF cache lookup (moldir): {time.perf_counter() - t0:.2f}s")

# Load model directly to GPU (faster than CPU→GPU transfer)
t0 = time.perf_counter()
device = torch.device("cuda")
model = Boltz.load_from_checkpoint(
    ckpt_path, 
    strict=False, 
    map_location=device,  # Load directly to GPU
    weights_only=False,
)
model.eval()

# Override masker mask setting if USE_MASKING is set
model.masker.mask = USE_MASKING

torch.cuda.synchronize()
print(f"  Load checkpoint to GPU: {time.perf_counter() - t0:.2f}s")

# Setup data pipeline
t0 = time.perf_counter()
canonicals = load_canonicals(moldir)
print(f"  Load canonicals: {time.perf_counter() - t0:.2f}s")

t0 = time.perf_counter()
tokenizer = Tokenizer(canonicals)
featurizer = Featurizer()
print(f"  Init tokenizer/featurizer: {time.perf_counter() - t0:.2f}s")

# Create dataset
t0 = time.perf_counter()
from boltzgen.task.predict.data_from_yaml import PredictionDataset, Dataset
dataset = Dataset(yaml_path=str(YAML_PATH), tokenizer=tokenizer, featurizer=featurizer, multiplicity=1)
pred_dataset = PredictionDataset(
    dataset=dataset,
    canonicals=canonicals,
    moldir=str(moldir),
    backbone_only=False,
    atom14=True,
    design=True,
)
print(f"  Create dataset: {time.perf_counter() - t0:.2f}s")

# Load features
t0 = time.perf_counter()
feats = pred_dataset[0]
print(f"  Featurize input: {time.perf_counter() - t0:.2f}s")

print(f"  TOTAL STARTUP: {time.perf_counter() - startup_start:.2f}s")
print("=========================\n")

# Move to device and add batch dimension
feats_batched = {}
for k, v in feats.items():
    if isinstance(v, torch.Tensor):
        feats_batched[k] = v.unsqueeze(0).to(device)
    else:
        feats_batched[k] = v

# Create geometric guidance (uses virtual atom representation)
guidance = create_geometric_guidance(
    property_type="hydrophobicity",
    higher_is_better=True,
    guidance_scale=GUIDANCE_SCALE,
    temperature=0.1,  # For geometric decoding
    schedule="linear",
    schedule_start=0.2,  # Start guidance early
    enabled=True,
)
guidance.to(device)

def decode_sequence_from_coords(coords, feats, threshold=1.0):
    """Decode amino acid sequence from coordinates using res_from_atom14."""
    import copy
    
    # Build a feature dict for res_from_atom14 (needs CPU tensors without batch dim)
    feat_cpu = {}
    for k, v in feats.items():
        if isinstance(v, torch.Tensor):
            # Remove batch dimension and move to CPU
            if v.dim() > 0 and v.shape[0] == 1:
                feat_cpu[k] = v[0].cpu()
            else:
                feat_cpu[k] = v.cpu()
        else:
            feat_cpu[k] = v
    
    # Update coords with the output coordinates
    feat_cpu["coords"] = coords[0].cpu()
    
    # Use the existing res_from_atom14 function
    try:
        result = res_from_atom14(feat_cpu, threshold=threshold)
        
        # Extract designed residues
        design_mask = result["design_mask"].bool()
        res_type = result["res_type"][design_mask]
        
        # Convert to sequence string
        aa_indices = res_type.argmax(dim=-1)
        seq = "".join([const.tokens[idx][0] if len(const.tokens[idx]) == 3 else "?" 
                       for idx in aa_indices.tolist()])
        return [seq]
    except Exception as e:
        return [f"Error: {e}"]

def compute_hydrophobicity_from_coords(coords, feats):
    """Compute hydrophobicity from final coordinates using geometric decoding."""
    return guidance.compute_geometric_score(coords, feats)

# Enable profiling for geometric guidance
enable_profiling(True)

# Profiling storage
profile_times = {'unguided_forward': [], 'guided_forward': []}

# Calculate number of batches
num_batches = (NUM_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE

def compute_mean_bfactor(out, batch_size):
    """Compute mean B-factor per sample (lower = more confident)."""
    pbfactor = out.get("pbfactor")
    if pbfactor is None:
        return [None] * batch_size
    
    # pbfactor shape: [batch_size, n_tokens, n_bins] - softmax over bins
    # Convert to expected B-factor value
    n_bins = pbfactor.shape[-1]
    # Bins typically represent B-factor ranges, compute expected value
    bin_centers = torch.linspace(0, 100, n_bins, device=pbfactor.device)
    expected_bfactor = (pbfactor.softmax(dim=-1) * bin_centers).sum(dim=-1)  # [batch, n_tokens]
    
    # Mean across tokens for each sample
    mean_bfactors = expected_bfactor.mean(dim=-1).tolist()  # [batch_size]
    return mean_bfactors

# Run unguided (batched)
print(f"\n{'='*60}")
print(f"  Running {NUM_SAMPLES} UNGUIDED designs (batch_size={BATCH_SIZE})")
print(f"{'='*60}")
unguided_scores = []
unguided_seqs = []
unguided_bfactors = []

for batch_idx in range(num_batches):
    batch_start = batch_idx * BATCH_SIZE
    batch_end = min(batch_start + BATCH_SIZE, NUM_SAMPLES)
    current_batch_size = batch_end - batch_start
    
    torch.manual_seed(100 + batch_idx)  # Seed per batch
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    # Apply masking if enabled
    feats_to_use = model.masker(feats_batched) if USE_MASKING else feats_batched
    
    with torch.no_grad():
        out = model.forward(
            feats=feats_to_use,
            recycling_steps=1,
            num_sampling_steps=SAMPLING_STEPS,
            diffusion_samples=current_batch_size,
            guidance=None,
        )
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    profile_times['unguided_forward'].append(elapsed)
    
    if batch_idx == 0:
        print(f"  Output keys: {list(out.keys())}")
        if "pbfactor" in out:
            print(f"  pbfactor shape: {out['pbfactor'].shape}")
    
    coords = out.get("sample_atom_coords")
    if coords is None:
        coords = out.get("coords")
    
    # Get B-factors for this batch
    batch_bfactors = compute_mean_bfactor(out, current_batch_size)
    unguided_bfactors.extend(batch_bfactors)
    
    if coords is not None:
        # coords shape is [batch_size, n_atoms, 3] when diffusion_samples > 1
        for i in range(current_batch_size):
            sample_coords = coords[i:i+1]  # Keep batch dim
            score = compute_hydrophobicity_from_coords(sample_coords, feats_batched).item()
            seq = decode_sequence_from_coords(sample_coords, feats_batched)[0]
            unguided_scores.append(score)
            unguided_seqs.append(seq)
        
        bf_str = f", B={np.mean(batch_bfactors):.1f}" if batch_bfactors[0] is not None else ""
        print(f"  Batch {batch_idx+1}/{num_batches}: {current_batch_size} samples in {elapsed:.1f}s ({elapsed/current_batch_size:.1f}s/sample){bf_str}")
        print(f"    Hydro: {[f'{s:+.2f}' for s in unguided_scores[batch_start:batch_end]]}")
    else:
        print(f"  Batch {batch_idx+1}: No coords in output!")
        for _ in range(current_batch_size):
            unguided_scores.append(0.0)
            unguided_seqs.append("")

# Run guided (batched)
print(f"\n{'='*60}")
print(f"  Running {NUM_SAMPLES} GUIDED designs (batch_size={BATCH_SIZE})")
print(f"{'='*60}")
guided_scores = []
guided_seqs = []
guided_bfactors = []

for batch_idx in range(num_batches):
    batch_start = batch_idx * BATCH_SIZE
    batch_end = min(batch_start + BATCH_SIZE, NUM_SAMPLES)
    current_batch_size = batch_end - batch_start
    
    torch.manual_seed(100 + batch_idx)  # Same seed as unguided for pairing
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    # Apply masking if enabled
    feats_to_use = model.masker(feats_batched) if USE_MASKING else feats_batched
    
    out = model.forward(
        feats=feats_to_use,
        recycling_steps=1,
        num_sampling_steps=SAMPLING_STEPS,
        diffusion_samples=current_batch_size,
        guidance=guidance,
    )
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    profile_times['guided_forward'].append(elapsed)
    
    coords = out.get("sample_atom_coords")
    if coords is None:
        coords = out.get("coords")
    
    # Get B-factors for this batch
    batch_bfactors = compute_mean_bfactor(out, current_batch_size)
    guided_bfactors.extend(batch_bfactors)
    
    if coords is not None:
        for i in range(current_batch_size):
            sample_coords = coords[i:i+1]
            score = compute_hydrophobicity_from_coords(sample_coords, feats_batched).item()
            seq = decode_sequence_from_coords(sample_coords, feats_batched)[0]
            guided_scores.append(score)
            guided_seqs.append(seq)
        
        avg_grad_norm = 0.0
        if hasattr(model.structure_module, '_guidance_info') and model.structure_module._guidance_info:
            avg_grad_norm = np.mean([g['coord_grad_norm'] for g in model.structure_module._guidance_info])
            model.structure_module._guidance_info = []
        
        bf_str = f", B={np.mean(batch_bfactors):.1f}" if batch_bfactors[0] is not None else ""
        print(f"  Batch {batch_idx+1}/{num_batches}: {current_batch_size} samples in {elapsed:.1f}s ({elapsed/current_batch_size:.1f}s/sample), grad={avg_grad_norm:.2e}{bf_str}")
        print(f"    Hydro: {[f'{s:+.2f}' for s in guided_scores[batch_start:batch_end]]}")
        
        # Print detailed profile for first batch
        if batch_idx == 0 and 'diffusion_profile' in out:
            dp = out['diffusion_profile']
            print(f"    Profile: net_fwd={dp['network_forward_ms']:.0f}ms, score={dp['score_compute_ms']:.0f}ms, bwd={dp['backward_ms']:.0f}ms")
    else:
        print(f"  Batch {batch_idx+1}: No coords in output!")
        for _ in range(current_batch_size):
            guided_scores.append(0.0)
            guided_seqs.append("")

# Results
print(f"\n{'='*60}")
print(f"  RESULTS (n={NUM_SAMPLES} per condition)")
print(f"{'='*60}")
print(f"  HYDROPHOBICITY:")
print(f"    Unguided: mean={np.mean(unguided_scores):.3f}, std={np.std(unguided_scores):.3f}")
print(f"    Guided:   mean={np.mean(guided_scores):.3f}, std={np.std(guided_scores):.3f}")

mean_improvement = np.mean(guided_scores) - np.mean(unguided_scores)
print(f"    Improvement: {mean_improvement:+.3f}")

# B-factor comparison (structural confidence)
if unguided_bfactors[0] is not None:
    print(f"\n  B-FACTOR (lower = more confident structure):")
    print(f"    Unguided: mean={np.mean(unguided_bfactors):.1f}, std={np.std(unguided_bfactors):.1f}")
    print(f"    Guided:   mean={np.mean(guided_bfactors):.1f}, std={np.std(guided_bfactors):.1f}")
    bf_diff = np.mean(guided_bfactors) - np.mean(unguided_bfactors)
    print(f"    Difference: {bf_diff:+.1f} {'(worse)' if bf_diff > 0 else '(better)' if bf_diff < 0 else ''}")
    
    # Statistical test for B-factor
    if len(set(guided_bfactors)) > 1 and len(set(unguided_bfactors)) > 1:
        bf_t_stat, bf_p_value = stats.ttest_rel(guided_bfactors, unguided_bfactors)
        print(f"    B-factor change p-value: {bf_p_value:.4f}")

# Per-sample analysis
differences = np.array(guided_scores) - np.array(unguided_scores)
n_improved = np.sum(differences > 0.001)  # Small threshold to avoid float noise
n_same = np.sum(np.abs(differences) <= 0.001)
n_worse = np.sum(differences < -0.001)
print(f"\n  Per-sample: {n_improved} improved, {n_same} unchanged, {n_worse} worse")
print(f"  Mean difference: {np.mean(differences):+.3f} ± {np.std(differences):.3f}")

# Statistical test
if len(set(guided_scores)) > 1 and len(set(unguided_scores)) > 1:
    t_stat, p_value = stats.ttest_rel(guided_scores, unguided_scores, alternative='greater')
    print(f"\n  t-statistic: {t_stat:.3f}")
    print(f"  p-value (one-tailed, paired): {p_value:.4f}")
    
    if p_value < 0.05 and mean_improvement > 0:
        print(f"\n  ✓ SIGNIFICANT: Geometric guidance improves hydrophobicity!")
    elif mean_improvement > 0:
        print(f"\n  ~ Guidance improved scores but not statistically significant")
    else:
        print(f"\n  ✗ Guidance did not improve scores")
else:
    print(f"\n  (Cannot compute t-test: no variance in scores)")
    if mean_improvement > 0:
        print(f"  ~ Guidance appears to have improved scores")

# Profiling summary
print(f"\n{'='*60}")
print(f"  PROFILING")
print(f"{'='*60}")
total_unguided = sum(profile_times['unguided_forward'])
total_guided = sum(profile_times['guided_forward'])
print(f"  Unguided: {total_unguided:.1f}s total, {total_unguided/NUM_SAMPLES:.2f}s/sample")
print(f"  Guided:   {total_guided:.1f}s total, {total_guided/NUM_SAMPLES:.2f}s/sample")
print(f"  Slowdown: {total_guided / total_unguided:.2f}x")
print(f"  Batch efficiency: {BATCH_SIZE}x parallelism")

# Geometric guidance profiling
guidance_stats = get_profiling_stats()
print(f"\n  Geometric Guidance Breakdown:")
print(f"    Total calls: {guidance_stats['compute_score_calls']}")
print(f"    Residues processed: {guidance_stats['residues_processed']}")
print(f"    Total time: {guidance_stats['compute_score_total_ms']:.1f}ms")
if guidance_stats['compute_score_calls'] > 0:
    print(f"    Avg per call: {guidance_stats.get('avg_ms_per_call', 0):.2f}ms")
if guidance_stats['residues_processed'] > 0:
    print(f"    Avg per residue: {guidance_stats.get('avg_ms_per_residue', 0):.3f}ms")
    print(f"    Breakdown:")
    print(f"      cdist: {guidance_stats['cdist_ms']:.1f}ms ({100*guidance_stats['cdist_ms']/guidance_stats['compute_score_total_ms']:.1f}%)")
    print(f"      STE ops: {guidance_stats['ste_ms']:.1f}ms ({100*guidance_stats['ste_ms']/guidance_stats['compute_score_total_ms']:.1f}%)")
    print(f"      Pattern match: {guidance_stats['pattern_match_ms']:.1f}ms ({100*guidance_stats['pattern_match_ms']/guidance_stats['compute_score_total_ms']:.1f}%)")

# Sequence comparison
print(f"\n{'='*60}")
print(f"  SEQUENCE COMPARISON (designed region)")
print(f"{'='*60}")
print(f"  Sample  Unguided                           Guided")
print(f"  ------  ---------------------------------  ---------------------------------")
for i in range(NUM_SAMPLES):
    u_seq = unguided_seqs[i] if i < len(unguided_seqs) else ""
    g_seq = guided_seqs[i] if i < len(guided_seqs) else ""
    diff_marker = "←" if g_seq != u_seq else ""
    print(f"  {i+1:2d}.     {u_seq:33s}  {g_seq:33s} {diff_marker}")
