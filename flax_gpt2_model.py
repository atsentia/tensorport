#!/usr/bin/env python3
"""
Production-grade Flax implementation of GPT-2 model architecture.
Faithful recreation of Hugging Face GPT-2 in JAX/Flax with Linen modules.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from dataclasses import dataclass
from functools import partial


@struct.dataclass
class FlaxGPT2Config:
    """Configuration for Flax GPT-2 model."""
    vocab_size: int = 50257
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    head_dim: Optional[int] = None
    intermediate_size: Optional[int] = None
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 1024
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.head_dim is None:
            object.__setattr__(self, 'head_dim', self.hidden_size // self.num_attention_heads)
        if self.intermediate_size is None:
            object.__setattr__(self, 'intermediate_size', 4 * self.hidden_size)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create config from Hugging Face configuration dict."""
        return cls(
            vocab_size=config_dict.get('vocab_size', 50257),
            hidden_size=config_dict.get('hidden_size', 768),
            num_hidden_layers=config_dict.get('num_hidden_layers', 12),
            num_attention_heads=config_dict.get('num_attention_heads', 12),
            head_dim=config_dict.get('head_dim'),
            intermediate_size=config_dict.get('intermediate_size'),
            hidden_act=config_dict.get('hidden_act', 'gelu'),
            hidden_dropout_prob=config_dict.get('hidden_dropout_prob', 0.1),
            attention_probs_dropout_prob=config_dict.get('attention_probs_dropout_prob', 0.1),
            max_position_embeddings=config_dict.get('max_position_embeddings', 1024),
            layer_norm_epsilon=config_dict.get('layer_norm_epsilon', 1e-5),
            initializer_range=config_dict.get('initializer_range', 0.02),
            use_cache=config_dict.get('use_cache', True),
            tie_word_embeddings=config_dict.get('tie_word_embeddings', True)
        )


class FlaxAttention(nn.Module):
    """Multi-head causal self-attention module."""
    config: FlaxGPT2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.config.head_dim
        self.scale = 1.0 / jnp.sqrt(self.head_dim)

        # Combined QKV projection (as in original GPT-2)
        self.c_attn = nn.Dense(
            3 * self.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            use_bias=True
        )
        
        # Output projection
        self.c_proj = nn.Dense(
            self.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            use_bias=True
        )
        
        self.dropout = nn.Dropout(rate=self.config.attention_probs_dropout_prob)

    def __call__(self, 
                 hidden_states: jnp.ndarray,
                 attention_mask: Optional[jnp.ndarray] = None,
                 deterministic: bool = True) -> jnp.ndarray:
        batch_size, seq_length, _ = hidden_states.shape

        # Project to Q, K, V
        qkv = self.c_attn(hidden_states)
        query, key, value = jnp.split(qkv, 3, axis=-1)

        # Reshape for multi-head attention
        query = query.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        key = key.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        value = value.reshape(batch_size, seq_length, self.num_heads, self.head_dim)

        # Transpose to (batch, num_heads, seq_length, head_dim)
        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)

        # Compute attention scores
        attn_weights = jnp.einsum('bhqd,bhkd->bhqk', query, key) * self.scale

        # Apply causal mask
        causal_mask = jnp.tril(jnp.ones((seq_length, seq_length)))
        attn_weights = jnp.where(causal_mask, attn_weights, -jnp.inf)

        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask shape: (batch, seq_length)
            # Expand to (batch, 1, 1, seq_length) for broadcasting
            attention_mask = attention_mask[:, None, None, :]
            attn_weights = jnp.where(attention_mask, attn_weights, -jnp.inf)

        # Softmax
        attn_probs = jax.nn.softmax(attn_weights, axis=-1)
        attn_probs = self.dropout(attn_probs, deterministic=deterministic)

        # Apply attention to values
        attn_output = jnp.einsum('bhqk,bhkd->bhqd', attn_probs, value)
        
        # Reshape back to (batch, seq_length, hidden_size)
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)

        # Output projection
        output = self.c_proj(attn_output)
        return output


class FlaxMLP(nn.Module):
    """Feed-forward network module."""
    config: FlaxGPT2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.c_fc = nn.Dense(
            self.config.intermediate_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            use_bias=True
        )
        self.c_proj = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            use_bias=True
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

        # Activation function
        if self.config.hidden_act == "gelu":
            self.activation = nn.gelu
        elif self.config.hidden_act == "relu":
            self.activation = nn.relu
        elif self.config.hidden_act == "silu":
            self.activation = nn.silu
        else:
            raise ValueError(f"Unsupported activation: {self.config.hidden_act}")

    def __call__(self, hidden_states: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


class FlaxTransformerBlock(nn.Module):
    """Single transformer block."""
    config: FlaxGPT2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.ln_1 = nn.LayerNorm(
            epsilon=self.config.layer_norm_epsilon,
            dtype=self.dtype
        )
        self.attn = FlaxAttention(self.config, dtype=self.dtype)
        self.ln_2 = nn.LayerNorm(
            epsilon=self.config.layer_norm_epsilon,
            dtype=self.dtype
        )
        self.mlp = FlaxMLP(self.config, dtype=self.dtype)

    def __call__(self,
                 hidden_states: jnp.ndarray,
                 attention_mask: Optional[jnp.ndarray] = None,
                 deterministic: bool = True) -> jnp.ndarray:
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states, attention_mask, deterministic)
        hidden_states = residual + attn_output

        # Feed-forward with residual connection
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        mlp_output = self.mlp(hidden_states, deterministic)
        hidden_states = residual + mlp_output

        return hidden_states


class FlaxGPT2Model(nn.Module):
    """Core GPT-2 model (without language modeling head)."""
    config: FlaxGPT2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # Token embeddings
        self.wte = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype
        )
        
        # Position embeddings
        self.wpe = nn.Embed(
            num_embeddings=self.config.max_position_embeddings,
            features=self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype
        )
        
        # Transformer blocks
        self.h = [
            FlaxTransformerBlock(self.config, dtype=self.dtype, name=f'h_{i}')
            for i in range(self.config.num_hidden_layers)
        ]
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(
            epsilon=self.config.layer_norm_epsilon,
            dtype=self.dtype
        )

    def __call__(self,
                 input_ids: jnp.ndarray,
                 attention_mask: Optional[jnp.ndarray] = None,
                 position_ids: Optional[jnp.ndarray] = None,
                 deterministic: bool = True) -> jnp.ndarray:
        batch_size, seq_length = input_ids.shape

        # Default position IDs
        if position_ids is None:
            position_ids = jnp.arange(seq_length)[None, :].repeat(batch_size, axis=0)

        # Token and position embeddings
        input_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = input_embeds + position_embeds

        # Apply transformer blocks
        for layer in self.h:
            hidden_states = layer(hidden_states, attention_mask, deterministic)

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class FlaxGPT2LMHeadModel(nn.Module):
    """GPT-2 model with language modeling head."""
    config: FlaxGPT2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.transformer = FlaxGPT2Model(self.config, dtype=self.dtype)
        
        if not self.config.tie_word_embeddings:
            self.lm_head = nn.Dense(
                self.config.vocab_size,
                dtype=self.dtype,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
                use_bias=False
            )

    def __call__(self,
                 input_ids: jnp.ndarray,
                 attention_mask: Optional[jnp.ndarray] = None,
                 position_ids: Optional[jnp.ndarray] = None,
                 deterministic: bool = True) -> jnp.ndarray:
        
        # Get hidden states from transformer
        hidden_states = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic
        )

        # Language modeling head
        if self.config.tie_word_embeddings:
            # Use transposed token embeddings as output weights
            lm_logits = hidden_states @ self.transformer.wte.embedding.T
        else:
            lm_logits = self.lm_head(hidden_states)

        return lm_logits


# Utility functions for model creation and initialization

def create_model(config: FlaxGPT2Config) -> FlaxGPT2LMHeadModel:
    """Create a Flax GPT-2 model with the given configuration."""
    return FlaxGPT2LMHeadModel(config)


def init_model_params(model: FlaxGPT2LMHeadModel, 
                     rng: jax.random.PRNGKey,
                     input_shape: Tuple[int, int]) -> Dict[str, Any]:
    """Initialize model parameters."""
    dummy_input = jnp.ones(input_shape, dtype=jnp.int32)
    return model.init(rng, dummy_input, deterministic=True)


@jax.jit
def model_forward(params: Dict[str, Any],
                 model: FlaxGPT2LMHeadModel,
                 input_ids: jnp.ndarray,
                 attention_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """JIT-compiled forward pass."""
    return model.apply(params, input_ids, attention_mask, deterministic=True)


def print_model_info(config: FlaxGPT2Config):
    """Print model configuration and parameter count."""
    model = create_model(config)
    rng = jax.random.PRNGKey(0)
    params = init_model_params(model, rng, (1, config.max_position_embeddings))
    
    # Count parameters
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    param_size_mb = param_count * 4 / (1024 * 1024)  # Assuming float32
    
    print(f"Model Configuration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Attention heads: {config.num_attention_heads}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Max seq length: {config.max_position_embeddings}")
    print(f"  Parameters: {param_count:,}")
    print(f"  Model size: {param_size_mb:.1f} MB")


if __name__ == "__main__":
    # Example usage
    config = FlaxGPT2Config(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        vocab_size=50257
    )
    
    print_model_info(config)
    
    # Test model creation and forward pass
    model = create_model(config)
    rng = jax.random.PRNGKey(0)
    params = init_model_params(model, rng, (2, 10))  # batch=2, seq_len=10
    
    # Test forward pass
    test_input = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    
    logits = model_forward(params, model, test_input)
    print(f"\nTest forward pass:")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Output dtype: {logits.dtype}")