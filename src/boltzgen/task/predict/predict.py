from boltzgen.utils.quiet import quiet_startup

quiet_startup()

import os

# Disable Triton auto-tuning during inference
os.environ.setdefault("CUEQ_DEFAULT_CONFIG", "1")
os.environ.setdefault("CUEQ_DISABLE_AOT_TUNING", "1")

from typing import List, Optional, Union

import torch
from omegaconf import OmegaConf, listconfig
from pytorch_lightning import LightningModule, Trainer

from pytorch_lightning.strategies import DDPStrategy

from boltzgen.task.predict.data_from_generated import FromGeneratedDataModule
from boltzgen.task.predict.writer import (
    DesignWriter,
    FoldingWriter,
)
from boltzgen.task.task import Task
from boltzgen.utils.pipeline_progress_bar import PipelineProgressBar
from boltzgen.model.models.boltz import Boltz


class Predict(Task):
    """A task to run model inference."""

    def __init__(
        self,
        data: Union[FromGeneratedDataModule],
        writer: Union[DesignWriter, FoldingWriter],
        checkpoint: str,
        output: str,
        name: str,
        recycling_steps: int,
        sampling_steps: int,
        diffusion_samples: int = 1,
        keys_dict_out: Optional[List] = None,
        keys_dict_batch: Optional[List] = None,
        slurm: bool = False,
        matmul_precision: Optional[str] = None,
        trainer: Optional[dict] = None,
        override: Optional[dict] = None,
        debug: bool = False,
        use_ema: bool = False,
        write_manifest: bool = False,
        compile_pairformer: bool = False,
        compile_structure: bool = False,
        checkpoint_diffusion_conditioning: bool = False,
    ) -> None:
        """Initialize the task.

        Parameters
        ----------
        checkpoint : str
            The path to the model checkpoint.
        output : str
            The path to save the inference results.
        slurm : bool, optional
            Whether to run on SLURM, by default False
        matmul_precision : Optional[str], optional
            The matmul precision, by default None
        trainer : Optional[dict], optional
            The configuration for the trainer, by default None
        override : Optional[dict], optional
            The override configuration for the model, by default None

        """
        self.data = data
        self.checkpoint = checkpoint
        self.output = output
        self.slurm = slurm
        self.matmul_precision = matmul_precision
        self.trainer = trainer
        self.override = override if override is not None else {}
        self.predict_args = {
            "recycling_steps": recycling_steps,
            "sampling_steps": sampling_steps,
            "diffusion_samples": diffusion_samples,
        }
        if keys_dict_batch is not None:
            self.predict_args["keys_dict_batch"] = keys_dict_batch
        if keys_dict_out is not None:
            self.predict_args["keys_dict_out"] = keys_dict_out
        self.debug = debug
        self.use_ema = use_ema
        self.write_manifest = write_manifest
        self.writer = writer
        self.compile_pairformer = compile_pairformer
        self.compile_structure = compile_structure
        self.checkpoint_diffusion_conditioning = checkpoint_diffusion_conditioning

    def run(self, config: OmegaConf = None, run_prediction=True) -> None:  # noqa: ARG002
        # Silence warnings and pytorch lightning tips
        quiet_startup()

        # Exit quickly if no predictions are needed
        if len(self.data.predict_set) == 0:
            print("No predictions required")
            return

        # Set no grad
        torch.set_grad_enabled(False)

        # Experiment with this during training (high or medium)
        if self.matmul_precision is not None:
            torch.set_float32_matmul_precision(self.matmul_precision)

        # Create trainer dict
        if self.trainer is None:
            self.trainer = {}

        # Flip some arguments in debug mode
        devices = self.trainer.get("devices", 1)

        if self.debug:
            if isinstance(devices, int):
                devices = 1
            elif isinstance(devices, (list, listconfig.ListConfig)):
                devices = [devices[0]]
            self.trainer["devices"] = devices
            self.data.num_workers = 0

        # slurm
        if self.slurm:
            self.trainer["devices"] = int(
                os.environ.get("SLURM_NTASKS_PER_NODE", "auto")
            )
            self.trainer["num_nodes"] = int(os.environ.get("SLURM_NNODES", 1))

        # When property steering is enabled, new modules (stability_predictor, res_type_predictor)
        # won't be in the checkpoint. Use strict=False to allow missing keys.
        # Only use strict=False if steering is actually enabled (not just if diffusion_process_args exists)
        use_strict = True
        
        # Debug: Check what's in override and what the checkpoint has
        if self.override and "diffusion_process_args" in self.override:
            dp_args = self.override.get("diffusion_process_args", {})
            print("\n" + "="*60)
            print("DEBUG: diffusion_process_args override detected")
            print("="*60)
            print(f"Override diffusion_process_args keys: {list(dp_args.keys())}")
            print(f"Override diffusion_process_args values: {dp_args}")
            
            # Load checkpoint to inspect hyperparameters
            try:
                import torch
                checkpoint_data = torch.load(self.checkpoint, map_location="cpu")
                if "hyper_parameters" in checkpoint_data:
                    ckpt_hp = checkpoint_data["hyper_parameters"]
                    if "diffusion_process_args" in ckpt_hp:
                        ckpt_dp_args = ckpt_hp["diffusion_process_args"]
                        print(f"\nCheckpoint diffusion_process_args keys: {list(ckpt_dp_args.keys())}")
                        print(f"Checkpoint has {len(ckpt_dp_args)} parameters")
                        
                        # Check for missing keys
                        missing_keys = set(ckpt_dp_args.keys()) - set(dp_args.keys())
                        if missing_keys:
                            print(f"\n⚠️  WARNING: Override is missing {len(missing_keys)} parameters from checkpoint:")
                            for key in sorted(missing_keys)[:10]:  # Show first 10
                                print(f"  - {key}: {ckpt_dp_args[key]}")
                            if len(missing_keys) > 10:
                                print(f"  ... and {len(missing_keys) - 10} more")
                        
                        # Check for conflicting values
                        conflicting = []
                        for key in dp_args.keys():
                            if key in ckpt_dp_args and ckpt_dp_args[key] != dp_args[key]:
                                conflicting.append((key, ckpt_dp_args[key], dp_args[key]))
                        if conflicting:
                            print(f"\n⚠️  WARNING: Override conflicts with checkpoint for {len(conflicting)} parameters:")
                            for key, ckpt_val, override_val in conflicting[:5]:  # Show first 5
                                print(f"  - {key}: checkpoint={ckpt_val}, override={override_val}")
                            if len(conflicting) > 5:
                                print(f"  ... and {len(conflicting) - 5} more conflicts")
                    else:
                        print("\n⚠️  WARNING: Checkpoint does not contain diffusion_process_args in hyper_parameters")
                else:
                    print("\n⚠️  WARNING: Checkpoint does not contain hyper_parameters")
            except Exception as e:
                print(f"\n⚠️  Could not inspect checkpoint: {e}")
            
            if dp_args.get("enable_property_steering", False):
                use_strict = False
                print("\nUsing strict=False for checkpoint loading (property steering enabled)")
            else:
                print("\nUsing strict=True (steering disabled, checking for parameter mismatches)")
            print("="*60 + "\n")
        
        # Enable debug mode if we're overriding diffusion_process_args
        enable_debug = (self.override and "diffusion_process_args" in self.override)
        
        # Load model
        self.model_module: LightningModule = Boltz.load_from_checkpoint(
            self.checkpoint,
            strict=use_strict,
            use_ema=self.use_ema,
            checkpoint_diffusion_conditioning=self.checkpoint_diffusion_conditioning,
            map_location="cpu",
            predict_args=self.predict_args,
            **self.override,
        )
        
        # Enable debug on the loaded model
        if enable_debug:
            self.model_module._debug_init = True
            # Re-instantiate structure_module with debug (this won't actually recreate it, but triggers debug prints)
            # Actually, the debug prints should have already happened during __init__
            # Let's check what was actually set
            if hasattr(self.model_module, 'structure_module'):
                print("\n" + "="*60)
                print("DEBUG: After model load - checking what was actually set")
                print("="*60)
                sm = self.model_module.structure_module
                print(f"structure_module.enable_property_steering: {sm.enable_property_steering}")
                print(f"structure_module.sigma_min: {sm.sigma_min}, sigma_max: {sm.sigma_max}")
                print(f"structure_module.sampling_schedule: {sm.sampling_schedule}")
                print(f"structure_module.time_dilation: {sm.time_dilation}")
                print(f"structure_module.num_sampling_steps: {sm.num_sampling_steps}")
                print("="*60 + "\n")
        
        self.model_module.eval()

        if self.compile_pairformer:
            self.model_module.is_pairformer_compiled = True
            self.model_module.pairformer_module = torch.compile(
                self.model_module.pairformer_module, dynamic=True, fullgraph=False
            )
        if self.compile_structure:
            self.model_module.structure_module.score_model.is_token_transformer_compiled = True
            self.model_module.structure_module.score_model.token_transformer = (
                torch.compile(
                    self.model_module.structure_module.score_model.token_transformer,
                    dynamic=True,
                    fullgraph=False,
                )
            )

        # Set up trainer
        strategy = "auto"
        num_devices = (
            len(devices)
            if isinstance(devices, (list, listconfig.ListConfig))
            else devices
        )
        if num_devices > 1:
            strategy = DDPStrategy()
            if num_devices > len(self.data.predict_set):
                devices = max(1, len(self.data.predict_set))
                msg = f"Fewer designs than devices. Setting devices to {devices}."
                print(msg)
                self.trainer["devices"] = devices

        self.lightning_trainer = Trainer(
            default_root_dir=self.output,
            strategy=strategy,
            callbacks=[self.writer]
            + (
                [PipelineProgressBar()]
                if os.environ.get("BOLTZGEN_PIPELINE_STEP")
                else []
            ),
            **self.trainer,
        )
        if run_prediction:
            # Run training
            self.lightning_trainer.predict(
                self.model_module, datamodule=self.data, return_predictions=False
            )
            del self.model_module
