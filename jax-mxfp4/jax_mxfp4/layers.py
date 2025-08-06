"""
MXFP4 quantized layers for JAX/Flax.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple, Any
from functools import partial

from .quantize import quantize_to_mxfp4, dequantize_mxfp4
from .kernels import mxfp4_matmul, mxfp4_attention


class MXFP4Linear(nn.Module):
    """
    Linear layer with MXFP4 quantized weights.
    
    Attributes:
        features: Number of output features
        use_bias: Whether to use bias
        block_size: MXFP4 block size
        dtype: Computation dtype
    """
    features: int
    use_bias: bool = True
    block_size: int = 32
    dtype: jnp.dtype = jnp.float32
    kernel_init: Any = nn.initializers.lecun_normal()
    bias_init: Any = nn.initializers.zeros
    
    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Apply MXFP4 linear transformation.
        
        Args:
            inputs: Input tensor
        
        Returns:
            Output tensor
        """
        kernel = self.param(
            'kernel',
            self.kernel_init,
            (inputs.shape[-1], self.features),
            self.dtype
        )
        
        # Quantize kernel to MXFP4
        kernel_packed, kernel_scales = quantize_to_mxfp4(kernel, self.block_size)
        
        # Store quantized weights as variables for inspection
        self.variable('mxfp4', 'kernel_packed', lambda: kernel_packed)
        self.variable('mxfp4', 'kernel_scales', lambda: kernel_scales)
        
        # Perform MXFP4 matmul
        y = mxfp4_matmul(inputs, kernel_packed, kernel_scales, self.block_size)
        
        if self.use_bias:
            bias = self.param('bias', self.bias_init, (self.features,), self.dtype)
            y = y + bias
        
        return y


class MXFP4Embedding(nn.Module):
    """
    Embedding layer with MXFP4 quantized weights.
    
    Attributes:
        num_embeddings: Size of vocabulary
        features: Embedding dimension
        block_size: MXFP4 block size
    """
    num_embeddings: int
    features: int
    block_size: int = 32
    dtype: jnp.dtype = jnp.float32
    embedding_init: Any = nn.initializers.normal(stddev=0.02)
    
    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Embed input tokens.
        
        Args:
            inputs: Token indices
        
        Returns:
            Embedded tokens
        """
        embedding = self.param(
            'embedding',
            self.embedding_init,
            (self.num_embeddings, self.features),
            self.dtype
        )
        
        # For embeddings, we might want to keep them in full precision
        # or quantize them differently
        if self.num_embeddings * self.features > 1e6:  # Only quantize large embeddings
            emb_packed, emb_scales = quantize_to_mxfp4(embedding, self.block_size)
            embedding = dequantize_mxfp4(
                emb_packed, 
                emb_scales,
                (self.num_embeddings, self.features),
                self.block_size
            )
        
        return embedding[inputs]


class MXFP4Attention(nn.Module):
    """
    Multi-head attention with MXFP4 quantized weights.
    
    Attributes:
        num_heads: Number of attention heads
        head_dim: Dimension per head
        block_size: MXFP4 block size
        dropout_rate: Dropout rate
    """
    num_heads: int
    head_dim: int
    block_size: int = 32
    dropout_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32
    kernel_init: Any = nn.initializers.xavier_uniform()
    
    @nn.compact
    def __call__(
        self,
        inputs: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True
    ) -> jnp.ndarray:
        """
        Apply multi-head attention.
        
        Args:
            inputs: Input tensor (batch, seq, dim)
            mask: Attention mask
            deterministic: Whether to use dropout
        
        Returns:
            Attention output
        """
        batch, seq_len, dim = inputs.shape
        features = self.num_heads * self.head_dim
        
        # Q, K, V projections with MXFP4
        query = MXFP4Linear(
            features=features,
            block_size=self.block_size,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            name='query'
        )(inputs)
        
        key = MXFP4Linear(
            features=features,
            block_size=self.block_size,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            name='key'
        )(inputs)
        
        value = MXFP4Linear(
            features=features,
            block_size=self.block_size,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            name='value'
        )(inputs)
        
        # Reshape for multi-head attention
        query = query.reshape(batch, seq_len, self.num_heads, self.head_dim)
        key = key.reshape(batch, seq_len, self.num_heads, self.head_dim)
        value = value.reshape(batch, seq_len, self.num_heads, self.head_dim)
        
        # Compute attention
        scores = jnp.einsum('bqhd,bkhd->bhqk', query, key) / jnp.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores + mask
        
        attn_weights = jax.nn.softmax(scores, axis=-1)
        
        # Apply dropout
        if not deterministic and self.dropout_rate > 0:
            dropout_rng = self.make_rng('dropout')
            attn_weights = nn.Dropout(rate=self.dropout_rate)(
                attn_weights, deterministic=False, rng=dropout_rng
            )
        
        # Apply attention to values
        attn_output = jnp.einsum('bhqk,bkhd->bqhd', attn_weights, value)
        attn_output = attn_output.reshape(batch, seq_len, features)
        
        # Output projection with MXFP4
        output = MXFP4Linear(
            features=dim,
            block_size=self.block_size,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            name='output'
        )(attn_output)
        
        return output


class MXFP4MLP(nn.Module):
    """
    MLP block with MXFP4 quantized weights.
    
    Attributes:
        intermediate_size: Hidden layer size
        activation: Activation function
        block_size: MXFP4 block size
        dropout_rate: Dropout rate
    """
    intermediate_size: int
    activation: str = 'gelu'
    block_size: int = 32
    dropout_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32
    kernel_init: Any = nn.initializers.xavier_uniform()
    
    @nn.compact
    def __call__(
        self,
        inputs: jnp.ndarray,
        deterministic: bool = True
    ) -> jnp.ndarray:
        """
        Apply MLP transformation.
        
        Args:
            inputs: Input tensor
            deterministic: Whether to use dropout
        
        Returns:
            MLP output
        """
        dim = inputs.shape[-1]
        
        # First linear layer (up-projection)
        x = MXFP4Linear(
            features=self.intermediate_size,
            block_size=self.block_size,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            name='up_proj'
        )(inputs)
        
        # Activation
        if self.activation == 'gelu':
            x = jax.nn.gelu(x)
        elif self.activation == 'relu':
            x = jax.nn.relu(x)
        elif self.activation == 'silu':
            x = jax.nn.silu(x)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
        
        # Dropout
        if not deterministic and self.dropout_rate > 0:
            dropout_rng = self.make_rng('dropout')
            x = nn.Dropout(rate=self.dropout_rate)(
                x, deterministic=False, rng=dropout_rng
            )
        
        # Second linear layer (down-projection)
        x = MXFP4Linear(
            features=dim,
            block_size=self.block_size,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            name='down_proj'
        )(x)
        
        return x


class MXFP4TransformerBlock(nn.Module):
    """
    Transformer block with MXFP4 quantized weights.
    
    Attributes:
        num_heads: Number of attention heads
        head_dim: Dimension per head
        mlp_ratio: MLP hidden size ratio
        block_size: MXFP4 block size
        dropout_rate: Dropout rate
    """
    num_heads: int
    head_dim: int
    mlp_ratio: float = 4.0
    block_size: int = 32
    dropout_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(
        self,
        inputs: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True
    ) -> jnp.ndarray:
        """
        Apply transformer block.
        
        Args:
            inputs: Input tensor
            mask: Attention mask
            deterministic: Whether to use dropout
        
        Returns:
            Block output
        """
        dim = inputs.shape[-1]
        
        # Self-attention with residual
        attn_output = MXFP4Attention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            block_size=self.block_size,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype,
            name='attention'
        )(inputs, mask, deterministic)
        
        x = nn.LayerNorm(dtype=self.dtype, name='ln1')(inputs + attn_output)
        
        # MLP with residual
        mlp_output = MXFP4MLP(
            intermediate_size=int(dim * self.mlp_ratio),
            block_size=self.block_size,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype,
            name='mlp'
        )(x, deterministic)
        
        output = nn.LayerNorm(dtype=self.dtype, name='ln2')(x + mlp_output)
        
        return output