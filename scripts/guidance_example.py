"""Minimal example of using guidance with BoltzGen's standard pipeline.

This uses BoltzGen's existing infrastructure with zero reimplementation.
The only custom code is creating the guidance object.
"""

from pathlib import Path
import torch
import huggingface_hub

# Standard BoltzGen imports
from boltzgen.model.models.boltz import Boltz
from boltzgen.task.predict.data_from_yaml import PredictionDataset, Dataset
from boltzgen.task.predict.writer import DesignWriter
from boltzgen.data.mol import load_canonicals
from boltzgen.data.tokenize.tokenizer import Tokenizer
from boltzgen.data.feature.featurizer import Featurizer

# Guidance imports
from boltzgen.model.modules.guidance import create_geometric_guidance

from pytorch_lightning import Trainer


def run_with_guidance(
    yaml_path: str,
    output_dir: str,
    num_designs: int = 5,
    guidance=None,
    sampling_steps: int = 50,
):
    """Run BoltzGen design with optional guidance using standard pipeline.
    
    Args:
        yaml_path: Path to design spec YAML
        output_dir: Where to save outputs  
        num_designs: Number of designs to generate
        guidance: Optional guidance object (from create_geometric_guidance)
        sampling_steps: Number of diffusion steps
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model checkpoint from HuggingFace
    ckpt_path = Path(huggingface_hub.hf_hub_download(
        'boltzgen/boltzgen-1', 'boltzgen1_diverse.ckpt', repo_type='model'))
    
    # Key: pass guidance via predict_args
    predict_args = {
        "recycling_steps": 1,
        "sampling_steps": sampling_steps,
        "diffusion_samples": 1,
        "guidance": guidance,  # This now works!
    }
    
    model = Boltz.load_from_checkpoint(
        ckpt_path,
        strict=True,
        map_location="cpu",
        predict_args=predict_args,
    )
    model.eval()
    
    # Load dataset
    moldir = Path(huggingface_hub.hf_hub_download(
        'boltzgen/inference-data', 'mols.zip', repo_type='dataset'))
    canonicals = load_canonicals(moldir)
    tokenizer = Tokenizer(canonicals)
    featurizer = Featurizer(canonicals)
    
    dataset = Dataset(yaml_path=yaml_path, tokenizer=tokenizer, featurizer=featurizer)
    pred_dataset = PredictionDataset(dataset=dataset, mode="predict", multiplicity=num_designs)
    
    # Use standard writer
    writer = DesignWriter(
        output_dir=str(output_path),
        res_atoms_only=True,
        atom14=True,
    )
    
    # Use standard trainer
    trainer = Trainer(
        default_root_dir=str(output_path),
        callbacks=[writer],
        accelerator="auto",
        devices=1,
    )
    
    # Run using standard predict
    trainer.predict(model, dataloaders=torch.utils.data.DataLoader(
        pred_dataset, batch_size=1, shuffle=False
    ))
    
    print(f"Outputs saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    yaml_path = "example/inverse_folding/1brs.yaml"
    
    print("=" * 60)
    print("Running UNGUIDED designs...")
    print("=" * 60)
    run_with_guidance(
        yaml_path=yaml_path,
        output_dir="workbench/unguided_std",
        num_designs=3,
        guidance=None,
    )
    
    print("\n" + "=" * 60)
    print("Running GUIDED designs (hydrophobicity)...")
    print("=" * 60)
    guidance = create_geometric_guidance(
        property_type="hydrophobicity",
        higher_is_better=True,
        guidance_scale=5.0,
    )
    run_with_guidance(
        yaml_path=yaml_path,
        output_dir="workbench/guided_std",
        num_designs=3,
        guidance=guidance,
    )
    
    print("\nDone! Compare outputs in workbench/unguided_std and workbench/guided_std")
