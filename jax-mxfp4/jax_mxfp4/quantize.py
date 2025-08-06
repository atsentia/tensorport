"""
Core quantization functions for MXFP4 format.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Dict, Any
import numpy as np
from functools import partial


@jax.jit
def quantize_to_mxfp4(
    x: jnp.ndarray,
    block_size: int = 32,
    stochastic: bool = False,
    key: Optional[jax.random.PRNGKey] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Quantize floating point tensor to MXFP4 format.
    
    MXFP4 format:
    - 4 bits per value: 1 sign bit + 3 mantissa bits
    - Shared exponent per block of values
    - Packed 2 values per byte
    
    Args:
        x: Input tensor to quantize
        block_size: Number of values sharing an exponent
        stochastic: Use stochastic rounding (better accuracy)
        key: Random key for stochastic rounding
    
    Returns:
        packed: Packed 4-bit values (uint8)
        scales: Shared exponents per block (int8)
    """
    original_shape = x.shape
    x_flat = x.flatten()
    n_elements = x_flat.shape[0]
    
    # Pad to multiple of block_size
    pad_size = (block_size - n_elements % block_size) % block_size
    if pad_size > 0:
        x_flat = jnp.pad(x_flat, (0, pad_size))
    
    # Reshape into blocks
    x_blocks = x_flat.reshape(-1, block_size)
    n_blocks = x_blocks.shape[0]
    
    # Compute shared exponent per block
    max_abs = jnp.max(jnp.abs(x_blocks), axis=1, keepdims=True)
    scales = jnp.floor(jnp.log2(max_abs + 1e-10)).astype(jnp.int8)
    
    # Normalize by shared exponent
    x_normalized = x_blocks / (2.0 ** scales)
    
    # Quantize to 3-bit mantissa with optional stochastic rounding
    if stochastic and key is not None:
        noise = jax.random.uniform(key, x_normalized.shape, minval=-0.5, maxval=0.5)
        x_normalized = x_normalized + noise / 8.0
    
    # Extract sign and mantissa
    signs = jnp.where(x_normalized < 0, 1, 0).astype(jnp.uint8)
    mantissas = jnp.clip(jnp.abs(x_normalized) * 8, 0, 7).astype(jnp.uint8)
    
    # Combine sign and mantissa (4 bits total)
    values_4bit = (signs << 3) | mantissas
    
    # Pack 2 values per byte
    values_4bit = values_4bit.reshape(-1, 2)
    packed = (values_4bit[:, 0] & 0xF) | ((values_4bit[:, 1] & 0xF) << 4)
    packed = packed.astype(jnp.uint8)
    
    # Remove padding from scales if needed
    if pad_size > 0:
        n_original_blocks = (n_elements + block_size - 1) // block_size
        scales = scales[:n_original_blocks]
    
    return packed, scales.squeeze()


@jax.jit
def dequantize_mxfp4(
    packed: jnp.ndarray,
    scales: jnp.ndarray,
    shape: Tuple[int, ...],
    block_size: int = 32
) -> jnp.ndarray:
    """
    Dequantize MXFP4 packed values back to floating point.
    
    Args:
        packed: Packed 4-bit values (uint8)
        scales: Shared exponents per block (int8)
        shape: Original tensor shape
        block_size: Number of values per block
    
    Returns:
        Dequantized tensor in original shape
    """
    n_elements = np.prod(shape)
    n_packed = len(packed)
    
    # Unpack 2 values per byte
    low_nibbles = packed & 0xF
    high_nibbles = (packed >> 4) & 0xF
    
    # Interleave to get original order
    values_4bit = jnp.empty((n_packed * 2,), dtype=jnp.uint8)
    values_4bit = values_4bit.at[0::2].set(low_nibbles)
    values_4bit = values_4bit.at[1::2].set(high_nibbles)
    
    # Extract sign and mantissa
    signs = jnp.where((values_4bit & 0x8) != 0, -1.0, 1.0)
    mantissas = (values_4bit & 0x7).astype(jnp.float32) / 8.0
    
    # Apply shared exponents
    scale_indices = jnp.arange(len(values_4bit)) // block_size
    scale_indices = jnp.minimum(scale_indices, len(scales) - 1)
    block_scales = 2.0 ** scales[scale_indices].astype(jnp.float32)
    
    # Reconstruct values
    values = signs * mantissas * block_scales
    
    # Trim padding and reshape
    values = values[:n_elements]
    return values.reshape(shape)


def quantize_model_weights(
    weights: Dict[str, jnp.ndarray],
    block_size: int = 32,
    excluded_layers: Optional[list] = None
) -> Dict[str, Any]:
    """
    Quantize all model weights to MXFP4 format.
    
    Args:
        weights: Dictionary of weight tensors
        block_size: Block size for shared exponents
        excluded_layers: List of layer names to skip (e.g., embeddings)
    
    Returns:
        Dictionary with quantized weights and metadata
    """
    excluded_layers = excluded_layers or []
    quantized = {}
    metadata = {}
    
    total_params = 0
    quantized_params = 0
    original_bytes = 0
    quantized_bytes = 0
    
    for name, weight in weights.items():
        total_params += weight.size
        original_bytes += weight.nbytes
        
        # Check if should be excluded
        should_exclude = any(excl in name for excl in excluded_layers)
        
        if should_exclude or weight.ndim == 1:  # Skip 1D tensors (biases)
            quantized[name] = weight
            quantized_bytes += weight.nbytes
            metadata[name] = {"quantized": False, "dtype": str(weight.dtype)}
        else:
            # Quantize to MXFP4
            packed, scales = quantize_to_mxfp4(weight, block_size)
            quantized[f"{name}.packed"] = packed
            quantized[f"{name}.scales"] = scales
            quantized_bytes += packed.nbytes + scales.nbytes
            
            metadata[name] = {
                "quantized": True,
                "original_shape": weight.shape,
                "block_size": block_size,
                "original_dtype": str(weight.dtype),
                "compression_ratio": weight.nbytes / (packed.nbytes + scales.nbytes)
            }
            quantized_params += weight.size
    
    # Add summary statistics
    metadata["_summary"] = {
        "total_parameters": total_params,
        "quantized_parameters": quantized_params,
        "quantization_ratio": quantized_params / total_params if total_params > 0 else 0,
        "block_size": block_size,
        "original_size_mb": original_bytes / (1024 * 1024),
        "quantized_size_mb": quantized_bytes / (1024 * 1024),
        "overall_compression": original_bytes / quantized_bytes if quantized_bytes > 0 else 0,
    }
    
    return {"weights": quantized, "metadata": metadata}


@partial(jax.jit, static_argnums=(1, 2))
def dynamic_quantize_activations(
    x: jnp.ndarray,
    block_size: int = 32,
    momentum: float = 0.95,
    running_scale: Optional[jnp.ndarray] = None
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Dynamically quantize activations during inference.
    
    Args:
        x: Input activations
        block_size: Block size for quantization
        momentum: Momentum for running scale estimation
        running_scale: Previous running scale (for smoothing)
    
    Returns:
        Quantized activations, scales, updated running scale
    """
    packed, scales = quantize_to_mxfp4(x, block_size)
    
    # Update running scale with momentum
    if running_scale is not None:
        avg_scale = jnp.mean(scales)
        new_running_scale = momentum * running_scale + (1 - momentum) * avg_scale
    else:
        new_running_scale = jnp.mean(scales)
    
    return packed, scales, new_running_scale


def analyze_quantization_error(
    original: jnp.ndarray,
    block_size: int = 32
) -> Dict[str, Any]:
    """
    Analyze quantization error for a given tensor.
    
    Args:
        original: Original floating point tensor
        block_size: Block size for quantization
    
    Returns:
        Dictionary with error statistics
    """
    # Quantize and dequantize
    packed, scales = quantize_to_mxfp4(original, block_size)
    reconstructed = dequantize_mxfp4(packed, scales, original.shape, block_size)
    
    # Compute errors
    abs_error = jnp.abs(original - reconstructed)
    rel_error = abs_error / (jnp.abs(original) + 1e-10)
    
    # Compute signal-to-noise ratio
    signal_power = jnp.mean(original ** 2)
    noise_power = jnp.mean((original - reconstructed) ** 2)
    snr_db = 10 * jnp.log10(signal_power / (noise_power + 1e-10))
    
    return {
        "mse": float(jnp.mean((original - reconstructed) ** 2)),
        "mae": float(jnp.mean(abs_error)),
        "max_abs_error": float(jnp.max(abs_error)),
        "mean_rel_error": float(jnp.mean(rel_error)),
        "max_rel_error": float(jnp.max(rel_error)),
        "snr_db": float(snr_db),
        "compression_ratio": original.nbytes / (packed.nbytes + scales.nbytes),
        "original_range": [float(jnp.min(original)), float(jnp.max(original))],
        "reconstructed_range": [float(jnp.min(reconstructed)), float(jnp.max(reconstructed))],
    }