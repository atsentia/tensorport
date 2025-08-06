"""
Optimized kernels for MXFP4 operations.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from functools import partial
from typing import Optional, Tuple
import numpy as np

from .quantize import dequantize_mxfp4


@partial(jax.jit, static_argnums=(3,))
def mxfp4_matmul(
    x: jnp.ndarray,
    w_packed: jnp.ndarray,
    w_scales: jnp.ndarray,
    block_size: int = 32
) -> jnp.ndarray:
    """
    Matrix multiplication with MXFP4 quantized weights.
    
    Args:
        x: Input tensor (batch_size, seq_len, in_features)
        w_packed: Packed MXFP4 weights
        w_scales: Weight scales
        block_size: MXFP4 block size
    
    Returns:
        Output tensor (batch_size, seq_len, out_features)
    """
    # Get weight shape from packed size
    # Each byte contains 2 4-bit values
    n_values = w_packed.shape[0] * 2
    
    # Infer matrix dimensions
    in_features = x.shape[-1]
    out_features = n_values // in_features
    
    # Dequantize weights
    w = dequantize_mxfp4(
        w_packed, 
        w_scales, 
        (in_features, out_features),
        block_size
    )
    
    # Perform matmul
    return jnp.matmul(x, w)


@jax.jit
def mxfp4_matmul_batched(
    x: jnp.ndarray,
    w_packed: jnp.ndarray,
    w_scales: jnp.ndarray,
    block_size: int = 32
) -> jnp.ndarray:
    """
    Batched matrix multiplication with MXFP4 weights.
    Optimized for transformer layers.
    
    Args:
        x: Input (batch, seq, heads, dim) or (batch, seq, dim)
        w_packed: Packed weights
        w_scales: Weight scales
        block_size: Block size
    
    Returns:
        Output tensor
    """
    # Handle different input shapes
    original_shape = x.shape
    if len(original_shape) == 4:
        batch, seq, heads, dim = original_shape
        x = x.reshape(batch * seq, heads * dim)
    else:
        x = x.reshape(-1, x.shape[-1])
    
    # Perform MXFP4 matmul
    output = mxfp4_matmul(x, w_packed, w_scales, block_size)
    
    # Reshape back
    if len(original_shape) == 4:
        output = output.reshape(batch, seq, -1)
    else:
        output = output.reshape(*original_shape[:-1], -1)
    
    return output


@jax.jit
def fused_mxfp4_gelu(
    x: jnp.ndarray,
    w_packed: jnp.ndarray,
    w_scales: jnp.ndarray,
    block_size: int = 32
) -> jnp.ndarray:
    """
    Fused MXFP4 matmul + GELU activation.
    
    Args:
        x: Input tensor
        w_packed: Packed MXFP4 weights
        w_scales: Weight scales
        block_size: Block size
    
    Returns:
        Output after matmul and GELU
    """
    # MXFP4 matmul
    output = mxfp4_matmul(x, w_packed, w_scales, block_size)
    
    # GELU activation
    return jax.nn.gelu(output)


@jax.jit
def fused_mxfp4_layernorm(
    x: jnp.ndarray,
    w_packed: jnp.ndarray,
    w_scales: jnp.ndarray,
    gamma: jnp.ndarray,
    beta: Optional[jnp.ndarray] = None,
    block_size: int = 32,
    epsilon: float = 1e-5
) -> jnp.ndarray:
    """
    Fused MXFP4 matmul + LayerNorm.
    
    Args:
        x: Input tensor
        w_packed: Packed MXFP4 weights
        w_scales: Weight scales
        gamma: LayerNorm scale parameter
        beta: LayerNorm shift parameter
        block_size: Block size
        epsilon: LayerNorm epsilon
    
    Returns:
        Output after matmul and LayerNorm
    """
    # MXFP4 matmul
    output = mxfp4_matmul(x, w_packed, w_scales, block_size)
    
    # LayerNorm
    mean = jnp.mean(output, axis=-1, keepdims=True)
    var = jnp.var(output, axis=-1, keepdims=True)
    normalized = (output - mean) / jnp.sqrt(var + epsilon)
    
    if beta is not None:
        return gamma * normalized + beta
    else:
        return gamma * normalized


def create_pallas_kernel(block_size: int = 32):
    """
    Create optimized Pallas kernel for MXFP4 operations.
    This compiles to efficient GPU code.
    """
    
    def mxfp4_pallas_kernel(
        x_ref,
        w_packed_ref,
        w_scales_ref,
        out_ref,
        block_size: int
    ):
        """Pallas kernel for MXFP4 matmul."""
        # Get program ID
        pid = pl.program_id(axis=0)
        
        # Load packed weights for this block
        packed_block = pl.load(w_packed_ref, (pid,))
        scale = pl.load(w_scales_ref, (pid // block_size,))
        
        # Unpack 4-bit values
        low = packed_block & 0xF
        high = (packed_block >> 4) & 0xF
        
        # Dequantize
        signs_low = pl.where((low & 0x8) != 0, -1.0, 1.0)
        mantissa_low = (low & 0x7).astype(jnp.float32) / 8.0
        
        signs_high = pl.where((high & 0x8) != 0, -1.0, 1.0)
        mantissa_high = (high & 0x7).astype(jnp.float32) / 8.0
        
        # Apply scale
        val_low = signs_low * mantissa_low * (2.0 ** scale)
        val_high = signs_high * mantissa_high * (2.0 ** scale)
        
        # Store results
        pl.store(out_ref, (pid * 2,), val_low)
        pl.store(out_ref, (pid * 2 + 1,), val_high)
    
    return mxfp4_pallas_kernel


def get_optimized_kernel(device_type: str = "gpu"):
    """
    Get optimized kernel based on device type.
    
    Args:
        device_type: "gpu", "tpu", or "cpu"
    
    Returns:
        Optimized kernel function
    """
    if device_type.lower() == "tpu":
        # TPU-optimized kernel
        return partial(mxfp4_matmul_batched, block_size=128)  # Larger blocks for TPU
    elif device_type.lower() == "gpu":
        # GPU-optimized kernel
        if "nvidia" in device_type.lower():
            # NVIDIA GPU - use smaller blocks
            return partial(mxfp4_matmul_batched, block_size=32)
        else:
            # Other GPUs
            return partial(mxfp4_matmul_batched, block_size=64)
    else:
        # CPU fallback
        return partial(mxfp4_matmul, block_size=32)


@jax.jit
def mxfp4_attention(
    query: jnp.ndarray,
    key_packed: jnp.ndarray,
    key_scales: jnp.ndarray,
    value_packed: jnp.ndarray,
    value_scales: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    block_size: int = 32
) -> jnp.ndarray:
    """
    Attention with MXFP4 quantized K/V matrices.
    
    Args:
        query: Query tensor (batch, seq, heads, dim)
        key_packed: Packed key weights
        key_scales: Key scales
        value_packed: Packed value weights
        value_scales: Value scales
        mask: Optional attention mask
        block_size: Block size
    
    Returns:
        Attention output
    """
    batch, seq_len, num_heads, head_dim = query.shape
    
    # Dequantize K and V
    key = dequantize_mxfp4(
        key_packed,
        key_scales,
        (batch, seq_len, num_heads, head_dim),
        block_size
    )
    
    value = dequantize_mxfp4(
        value_packed,
        value_scales,
        (batch, seq_len, num_heads, head_dim),
        block_size
    )
    
    # Compute attention scores
    scores = jnp.einsum("bqhd,bkhd->bhqk", query, key) / jnp.sqrt(head_dim)
    
    # Apply mask if provided
    if mask is not None:
        scores = scores + mask
    
    # Softmax
    attn_weights = jax.nn.softmax(scores, axis=-1)
    
    # Apply attention to values
    output = jnp.einsum("bhqk,bkhd->bqhd", attn_weights, value)
    
    return output