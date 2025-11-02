"""Property-based steering module for denoising."""

from typing import Optional

import torch
from torch import nn
from torch.nn import Module


class PropertySteering(Module):
    """Generate attention bias terms from property predictions with target value.
    
    This module computes bias terms that steer the denoising process towards
    a target property value. The bias is computed as:
    
    bias = -(predicted_property - target_property) * weight / temperature
    
    The negative sign ensures negative feedback: when predicted > target,
    bias is negative (pushes down), when predicted < target, bias is positive (pushes up).
    
    Parameters
    ----------
    target_property : float
        Target property value to steer towards.
    bias_weight : float, default=1.0
        Strength of the steering bias. Higher values = stronger steering.
    temperature : float, default=1.0
        Temperature for smoothing the bias. Higher values = smoother control.
    atom_decoder_depth : int, default=3
        Number of layers in atom decoder (determines bias shape).
    atom_decoder_heads : int, default=4
        Number of attention heads (determines bias shape).
    token_transformer_depth : int, default=6
        Number of layers in token transformer (for token_trans_bias).
    token_transformer_heads : int, default=8
        Number of attention heads in token transformer.
    bias_type : str, default="atom_dec_bias"
        Type of bias to generate: "atom_dec_bias" or "token_trans_bias".
        
    Notes
    -----
    The bias tensors must match the expected shapes in the attention mechanisms:
    - atom_dec_bias: (batch, num_atoms, num_atoms, num_heads_per_layer)
    - token_trans_bias: (batch, num_tokens, num_tokens, num_heads_per_layer)
    """
    
    def __init__(
        self,
        target_property: float,
        bias_weight: float = 1.0,
        temperature: float = 1.0,
        atom_decoder_depth: int = 3,
        atom_decoder_heads: int = 4,
        token_transformer_depth: int = 6,
        token_transformer_heads: int = 8,
        bias_type: str = "atom_dec_bias",
    ):
        super().__init__()
        
        self.target_property = target_property
        self.bias_weight = bias_weight
        self.temperature = temperature
        self.bias_type = bias_type
        
        if bias_type == "atom_dec_bias":
            self.num_layers = atom_decoder_depth
            self.num_heads = atom_decoder_heads
        elif bias_type == "token_trans_bias":
            self.num_layers = token_transformer_depth
            self.num_heads = token_transformer_heads
        else:
            raise ValueError(
                f"Unknown bias_type: {bias_type}. "
                "Must be 'atom_dec_bias' or 'token_trans_bias'"
            )
    
    def compute_bias_from_property(
        self,
        predicted_property: float,
        shape: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Compute bias tensor from property prediction.
        
        Parameters
        ----------
        predicted_property : float
            Predicted property value.
        shape : tuple[int, ...]
            Shape of the bias tensor. Should be (batch, dim1, dim2, num_heads_per_layer).
        device : torch.device
            Device for the bias tensor.
        dtype : torch.dtype, default=torch.float32
            Data type for the bias tensor.
            
        Returns
        -------
        torch.Tensor
            Bias tensor with the specified shape.
        """
        # Compute deviation from target
        deviation = predicted_property - self.target_property
        
        # Compute base bias: negative feedback
        base_bias = -deviation * self.bias_weight / self.temperature
        
        # Expand to match the expected shape
        # The bias needs to be broadcastable to (batch, dim1, dim2, num_heads_per_layer)
        bias = torch.full(shape, base_bias, device=device, dtype=dtype)
        
        return bias
    
    def forward(
        self,
        predicted_property: float,
        num_atoms: Optional[int] = None,
        num_tokens: Optional[int] = None,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        target_shape: Optional[tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """Generate bias tensor for given property prediction.
        
        Parameters
        ----------
        predicted_property : float
            Predicted property value (e.g., stability score).
        num_atoms : int, optional
            Number of atoms (required if bias_type == "atom_dec_bias").
        num_tokens : int, optional
            Number of tokens (required if bias_type == "token_trans_bias").
        batch_size : int, default=1
            Batch size for the bias tensor.
        device : torch.device, optional
            Device for the bias tensor. If None, uses CPU.
        dtype : torch.dtype, default=torch.float32
            Data type for the bias tensor.
        target_shape : tuple[int, ...], optional
            Target shape for the bias tensor. If provided, this shape is used
            directly instead of calculating from num_layers and num_heads.
            Should be (batch, dim1, dim2, num_heads).
            
        Returns
        -------
        torch.Tensor
            Bias tensor with shape appropriate for the bias type:
            - atom_dec_bias: (batch, num_atoms, num_atoms, num_heads_per_layer)
            - token_trans_bias: (batch, num_tokens, num_tokens, num_heads_per_layer)
            
            Where num_heads_per_layer is flattened across all layers.
        """
        if device is None:
            device = torch.device("cpu")
        
        if target_shape is not None:
            # Use the provided target shape directly
            shape = target_shape
        elif self.bias_type == "atom_dec_bias":
            if num_atoms is None:
                raise ValueError("num_atoms required for atom_dec_bias")
            
            # Shape: (batch, num_atoms, num_atoms, num_heads_per_layer)
            # num_heads_per_layer = sum of heads across all layers
            num_heads_per_layer = self.num_layers * self.num_heads
            shape = (batch_size, num_atoms, num_atoms, num_heads_per_layer)
            
        elif self.bias_type == "token_trans_bias":
            if num_tokens is None:
                raise ValueError("num_tokens required for token_trans_bias")
            
            # Shape: (batch, num_tokens, num_tokens, num_heads_per_layer)
            num_heads_per_layer = self.num_layers * self.num_heads
            shape = (batch_size, num_tokens, num_tokens, num_heads_per_layer)
        else:
            raise ValueError(f"Invalid bias_type: {self.bias_type}")
        
        bias = self.compute_bias_from_property(
            predicted_property=predicted_property,
            shape=shape,
            device=device,
            dtype=dtype,
        )
        
        return bias
    
    def compute_property_error(self, predicted_property: float) -> float:
        """Compute error (deviation from target).
        
        Parameters
        ----------
        predicted_property : float
            Predicted property value.
            
        Returns
        -------
        float
            Error (predicted - target).
        """
        return predicted_property - self.target_property


class CombinedPropertySteering(Module):
    """Steering module that can generate both atom_dec_bias and token_trans_bias.
    
    This is a convenience class that wraps two PropertySteering modules
    for generating both types of biases simultaneously.
    """
    
    def __init__(
        self,
        target_property: float,
        bias_weight: float = 1.0,
        temperature: float = 1.0,
        atom_decoder_depth: int = 3,
        atom_decoder_heads: int = 4,
        token_transformer_depth: int = 6,
        token_transformer_heads: int = 8,
        use_atom_bias: bool = True,
        use_token_bias: bool = False,
    ):
        super().__init__()
        
        self.use_atom_bias = use_atom_bias
        self.use_token_bias = use_token_bias
        
        if use_atom_bias:
            self.atom_steering = PropertySteering(
                target_property=target_property,
                bias_weight=bias_weight,
                temperature=temperature,
                atom_decoder_depth=atom_decoder_depth,
                atom_decoder_heads=atom_decoder_heads,
                bias_type="atom_dec_bias",
            )
        
        if use_token_bias:
            self.token_steering = PropertySteering(
                target_property=target_property,
                bias_weight=bias_weight,
                temperature=temperature,
                token_transformer_depth=token_transformer_depth,
                token_transformer_heads=token_transformer_heads,
                bias_type="token_trans_bias",
            )
    
    def forward(
        self,
        predicted_property: float,
        num_atoms: Optional[int] = None,
        num_tokens: Optional[int] = None,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        atom_dec_bias_shape: Optional[tuple[int, ...]] = None,
        token_trans_bias_shape: Optional[tuple[int, ...]] = None,
    ) -> dict[str, torch.Tensor]:
        """Generate both bias types.
        
        Parameters
        ----------
        atom_dec_bias_shape : tuple[int, ...], optional
            Target shape for atom_dec_bias. If provided, overrides calculated shape.
        token_trans_bias_shape : tuple[int, ...], optional
            Target shape for token_trans_bias. If provided, overrides calculated shape.
        
        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with keys:
            - "atom_dec_bias": bias tensor (if use_atom_bias=True)
            - "token_trans_bias": bias tensor (if use_token_bias=True)
        """
        biases = {}
        
        if self.use_atom_bias:
            biases["atom_dec_bias"] = self.atom_steering(
                predicted_property=predicted_property,
                num_atoms=num_atoms,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
                target_shape=atom_dec_bias_shape,
            )
        
        if self.use_token_bias:
            biases["token_trans_bias"] = self.token_steering(
                predicted_property=predicted_property,
                num_tokens=num_tokens,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
                target_shape=token_trans_bias_shape,
            )
        
        return biases

