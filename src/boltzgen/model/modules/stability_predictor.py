"""TAPE stability prediction model wrapper."""

import time
from typing import Optional

import numpy as np
import torch
from torch import nn

import tape  # noqa: F401
from tape import ProteinBertModel, ProteinBertConfig, TAPETokenizer


class TAPEStabilityPredictor(nn.Module):
    """Wrapper for TAPE's stability prediction model.
    
    This module loads TAPE's pre-trained Transformer model and uses it
    to predict protein stability from amino acid sequences.
    
    Parameters
    ----------
    model_name : str, optional
        Name of the TAPE model to use. Default is 'bert-base'.
    checkpoint_path : str, optional
        Path to a specific checkpoint file. If None, uses default TAPE checkpoint.
    device : str, optional
        Device to load the model on. Default is 'cuda' if available, else 'cpu'.
        
    Notes
    -----
    - Requires TAPE package to be installed: `pip install tape-proteins`
    - The stability prediction is a regression task that outputs a scalar value
    - Higher values indicate higher stability
    """
    
    def __init__(
        self,
        model_name: str = "bert-base",
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        super().__init__()
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        initial_device = torch.device(device)
        
        # Load TAPE model
        if checkpoint_path is not None:
            # Load from specific checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=initial_device)
            config = ProteinBertConfig(**checkpoint["model_config"])
            self.model = ProteinBertModel(config)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Use ProteinBertModel.from_pretrained to load pre-trained model
            # model_name can be a pretrained model name or path
            self.model = ProteinBertModel.from_pretrained(model_name)
        
        self.model = self.model.to(initial_device)
        self.model.eval()
        
        # Add a regression head for stability prediction
        # TAPE's transformer outputs embeddings of size config.hidden_size
        hidden_size = self.model.config.hidden_size
        self.stability_head = nn.Linear(hidden_size, 1).to(initial_device)
        
        # Initialize stability head weights
        nn.init.xavier_uniform_(self.stability_head.weight)
        nn.init.zeros_(self.stability_head.bias)
        
        # Initialize tokenizer once (reused for all forward passes)
        self.tokenizer = TAPETokenizer()
        
        # Note: In practice, you may need to load pre-trained weights for the
        # stability head if TAPE provides them separately
        
    @property
    def device(self):
        """Get the device of the model parameters."""
        return next(self.model.parameters()).device
    
    def forward(self, sequence: str) -> float:
        """Predict stability for a given amino acid sequence.
        
        Parameters
        ----------
        sequence : str
            Amino acid sequence as a string of single-letter codes.
            
        Returns
        -------
        float
            Predicted stability score (scalar value).
        """
        # Get current device from model (handles device changes)
        device = self.device
        
        # Profile tokenization
        tok_start = time.time()
        # Convert sequence to token IDs using pre-initialized tokenizer
        token_ids = self.tokenizer.encode(sequence)
        # Handle numpy array or list from tokenizer
        if isinstance(token_ids, np.ndarray):
            token_ids = torch.tensor(
                token_ids, device=device, dtype=torch.long
            ).unsqueeze(0)
        else:
            token_ids = torch.tensor(
                [token_ids], device=device, dtype=torch.long
            )
        tok_time = time.time() - tok_start
        
        # Get model output
        model_start = time.time()
        with torch.no_grad():
            output = self.model(token_ids)
            # TAPE returns a tuple: (hidden_states, pooled_output)
            if isinstance(output, tuple):
                # Use pooled output if available (second element),
                # otherwise mean pool hidden states
                if len(output) >= 2 and output[1] is not None:
                    # Pooled output (batch, hidden_size)
                    sequence_repr = output[1]
                else:
                    # Mean pool hidden states
                    sequence_repr = output[0].mean(dim=1)
            elif hasattr(output, 'pooler_output') and output.pooler_output is not None:
                sequence_repr = output.pooler_output
            elif hasattr(output, 'last_hidden_state'):
                sequence_repr = output.last_hidden_state.mean(dim=1)
            else:
                # Fallback: assume output is the representation directly
                if len(output.shape) > 2:
                    sequence_repr = output.mean(dim=1)
                else:
                    sequence_repr = output
            
            # Predict stability
            head_start = time.time()
            stability = self.stability_head(sequence_repr)
            stability_score = stability.item()
            head_time = time.time() - head_start
        model_time = time.time() - model_start
        
        # Print detailed timing (only first few calls to avoid spam)
        if not hasattr(self, '_call_count'):
            self._call_count = 0
        self._call_count += 1
        if self._call_count <= 3:
            total_time = (time.time() - tok_start) * 1000
            print(
                f"    [TAPE Profile] Call #{self._call_count}: "
                f"total={total_time:.2f}ms "
                f"(tokenize={tok_time*1000:.2f}ms, "
                f"model={model_time*1000:.2f}ms, "
                f"head={head_time*1000:.2f}ms)"
            )
        
        return stability_score
    
    @torch.no_grad()
    def predict_batch(self, sequences: list[str]) -> list[float]:
        """Predict stability for multiple sequences.
        
        Parameters
        ----------
        sequences : list[str]
            List of amino acid sequences.
            
        Returns
        -------
        list[float]
            List of predicted stability scores.
        """
        return [self.forward(seq) for seq in sequences]



