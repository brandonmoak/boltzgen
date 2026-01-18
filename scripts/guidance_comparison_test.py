#!/usr/bin/env python3
"""Simple test: 10 unguided vs 10 guided designs, compare hydrophobicity."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import os
os.environ["CUEQ_DEFAULT_CONFIG"] = "1"
os.environ["CUEQ_DISABLE_AOT_TUNING"] = "1"

import torch
import numpy as np
import huggingface_hub

from boltzgen.data import const
from boltzgen.data.mol import load_canonicals
from boltzgen.data.tokenize.tokenizer import Tokenizer
from boltzgen.data.feature.featurizer import Featurizer
from boltzgen.model.models.boltz import Boltz
from boltzgen.model.modules.guidance import DiffusionGuidance, HydrophobicityPredictor

# Config
YAML_PATH = Path(__file__).parent.parent / "example/vanilla_protein/1g13prot.yaml"
NUM_SAMPLES = 50
SAMPLING_STEPS = 50
GUIDANCE_SCALE = 50.0

# Download checkpoint - use inverse folding model which has trained res_type prediction
print("Downloading checkpoint...")
ckpt_path = huggingface_hub.hf_hub_download("boltzgen/boltzgen-1", "boltzgen1_ifold.ckpt", repo_type="model")

# Download moldir (use the zip directly)
print("Downloading moldir...")
moldir = huggingface_hub.hf_hub_download("boltzgen/inference-data", "mols.zip", repo_type="dataset")

# Load model with predict_res_type enabled
print("Loading model...")
device = torch.device("cuda")
model = Boltz.load_from_checkpoint(
    ckpt_path, 
    strict=False, 
    map_location="cpu", 
    weights_only=False,
    predict_res_type=True,  # Enable res_type prediction for guidance
)
model.eval()
model.to(device)
print(f"Model predict_res_type: {model.predict_res_type}")

# Setup data pipeline
print("Setting up data pipeline...")
canonicals = load_canonicals(moldir)
tokenizer = Tokenizer(canonicals)
featurizer = Featurizer()

# Create dataset to load features
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

# Load one sample of features
print(f"Loading features from {YAML_PATH}...")
feats = pred_dataset[0]

# Move to device and add batch dimension
feats_batched = {}
for k, v in feats.items():
    if isinstance(v, torch.Tensor):
        feats_batched[k] = v.unsqueeze(0).to(device)
    else:
        feats_batched[k] = v

# Create guidance
predictor = HydrophobicityPredictor(higher_is_better=True)
guidance = DiffusionGuidance(
    predictor=predictor,
    guidance_scale=GUIDANCE_SCALE,
    temperature=0.5,
    schedule="linear",
    schedule_start=0.3,
    enabled=True,
)

def get_hydrophobicity(res_type_logits, design_mask):
    """Compute mean hydrophobicity of designed residues."""
    kd_scale = torch.tensor([1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8, 
                             1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3], device=res_type_logits.device)
    canonical_logits = res_type_logits[..., const.canonicals_offset:const.canonicals_offset+20]
    pred_indices = canonical_logits.argmax(dim=-1)
    hydro = kd_scale[pred_indices]
    if design_mask is not None:
        return (hydro * design_mask.float()).sum() / design_mask.sum()
    return hydro.mean()

def get_sequence(res_type_logits, design_mask=None):
    """Get AA sequence string."""
    aa = "ACDEFGHIKLMNPQRSTVWY"
    canonical_logits = res_type_logits[..., const.canonicals_offset:const.canonicals_offset+20]
    indices = canonical_logits.argmax(dim=-1)
    if indices.dim() > 1:
        indices = indices[0]
    if design_mask is not None and design_mask.dim() > 1:
        design_mask = design_mask[0]
    if design_mask is not None:
        indices = indices[design_mask.bool()]
    return "".join([aa[i] for i in indices.cpu().numpy()])

# Run unguided
print(f"\n{'='*60}")
print(f"  Running {NUM_SAMPLES} UNGUIDED designs")
print(f"{'='*60}")
unguided_scores = []
unguided_seqs = []

for i in range(NUM_SAMPLES):
    torch.manual_seed(100 + i)
    with torch.no_grad():
        out = model.forward(
            feats=feats_batched,
            recycling_steps=1,
            num_sampling_steps=SAMPLING_STEPS,
            diffusion_samples=1,
            guidance=None,
        )
    # Debug: print keys on first run
    if i == 0:
        print(f"  Output keys: {list(out.keys())}")
    res_type = out.get("res_type")
    if res_type is None:
        res_type = out.get("res_type_logits")
    design_mask = feats_batched["design_mask"]
    score = get_hydrophobicity(res_type, design_mask).item()
    seq = get_sequence(res_type, design_mask)
    unguided_scores.append(score)
    unguided_seqs.append(seq)
    print(f"  {i+1:2d}. H={score:+.2f}  {seq[:30]}...")

# Run guided  
print(f"\n{'='*60}")
print(f"  Running {NUM_SAMPLES} GUIDED designs (maximize hydrophobicity)")
print(f"{'='*60}")
guided_scores = []
guided_seqs = []

for i in range(NUM_SAMPLES):
    torch.manual_seed(100 + i)  # Same seeds as unguided for paired comparison
    # Note: NOT using torch.no_grad() because guidance needs gradients for backprop
    out = model.forward(
        feats=feats_batched,
        recycling_steps=1,
        num_sampling_steps=SAMPLING_STEPS,
        diffusion_samples=1,
        guidance=guidance,
    )
    res_type = out.get("res_type")
    design_mask = feats_batched["design_mask"]
    score = get_hydrophobicity(res_type, design_mask).item()
    seq = get_sequence(res_type, design_mask)
    guided_scores.append(score)
    guided_seqs.append(seq)
    print(f"  {i+1:2d}. H={score:+.2f}  {seq[:30]}...")

# Results
from scipy import stats

print(f"\n{'='*60}")
print(f"  RESULTS (n={NUM_SAMPLES} per condition)")
print(f"{'='*60}")
print(f"  Unguided: mean={np.mean(unguided_scores):.3f}, std={np.std(unguided_scores):.3f}")
print(f"  Guided:   mean={np.mean(guided_scores):.3f}, std={np.std(guided_scores):.3f}")
print(f"  Improvement: {np.mean(guided_scores) - np.mean(unguided_scores):+.3f}")

# Per-sample differences (paired analysis)
differences = np.array(guided_scores) - np.array(unguided_scores)
n_improved = np.sum(differences > 0)
n_same = np.sum(differences == 0)
n_worse = np.sum(differences < 0)
print(f"\n  Per-sample: {n_improved} improved, {n_same} unchanged, {n_worse} worse")
print(f"  Mean difference: {np.mean(differences):+.3f} ± {np.std(differences):.3f}")

# Statistical significance test (paired one-tailed t-test: guided > unguided)
# Paired test is appropriate because we use the same seeds
t_stat, p_value = stats.ttest_rel(guided_scores, unguided_scores, alternative='greater')
print(f"\n  t-statistic: {t_stat:.3f}")
print(f"  p-value (one-tailed, paired): {p_value:.4f}")

if p_value < 0.05:
    print(f"\n  ✓ SIGNIFICANT (p < 0.05): Guidance reliably improves hydrophobicity!")
elif np.mean(guided_scores) > np.mean(unguided_scores):
    print(f"\n  ~ Guidance improved scores but not statistically significant")
else:
    print(f"\n  ✗ Guidance did not improve scores")
