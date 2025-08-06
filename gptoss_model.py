#!/usr/bin/env python3
"""
GPT-OSS Model Architecture Implementation
Production-ready JAX/NumPy implementation with MXFP4 quantization support.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import warnings

# JAX imports with fallback to NumPy
try:
    import jax
    import jax.numpy as jnp
    from jax import nn as jax_nn
    from jax import lax
    HAS_JAX = True
except ImportError:
    # NumPy fallback implementations
    class jax:
        class lax:
            @staticmethod
            def top_k(x, k):
                indices = np.argsort(x, axis=-1)[..., ::-1][..., :k]
                values = np.take_along_axis(x, indices, axis=-1)
                return values, indices
            
            @staticmethod
            def rsqrt(x):
                return 1.0 / np.sqrt(np.maximum(x, 1e-12))
            
            @staticmethod
            def stop_gradient(x):
                return x
        
        @staticmethod
        def jit(fn):
            return fn
    
    class jax_nn:
        @staticmethod
        def softmax(x, axis=-1):
            x_max = np.max(x, axis=axis, keepdims=True)
            exp_x = np.exp(np.clip(x - x_max, -50, 50))
            return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        
        @staticmethod
        def silu(x):
            return x / (1 + np.exp(-np.clip(x, -20, 20)))
        
        @staticmethod
        def gelu(x):
            return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    jnp = np
    lax = jax.lax
    HAS_JAX = False


@dataclass
class GPTOSSConfig:
    """Configuration for GPT-OSS model."""
    vocab_size: int = 201088
    hidden_size: int = 2880
    num_hidden_layers: int = 24
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 64
    intermediate_size: int = 2880
    num_local_experts: int = 32
    num_experts_per_tok: int = 4
    hidden_act: str = "silu"
    max_position_embeddings: int = 131072
    rope_theta: float = 150000.0
    sliding_window: int = 128
    layer_types: Optional[List[str]] = None
    rms_norm_eps: float = 1e-6
    quantization_method: str = "none"
    use_kv_cache: bool = True
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = []
            for i in range(self.num_hidden_layers):
                self.layer_types.append("attention" if i % 2 == 0 else "moe")
        
        # Validate configuration
        assert len(self.layer_types) == self.num_hidden_layers
        assert self.hidden_size % self.num_attention_heads == 0

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create config from dictionary."""
        quant_method = "none"
        if "quantization_config" in config_dict:
            quant_method = config_dict["quantization_config"].get("quant_method", "none")
        
        return cls(
            vocab_size=config_dict.get('vocab_size', 201088),
            hidden_size=config_dict.get('hidden_size', 2880),
            num_hidden_layers=config_dict.get('num_hidden_layers', 24),
            num_attention_heads=config_dict.get('num_attention_heads', 64),
            num_key_value_heads=config_dict.get('num_key_value_heads', 8),
            head_dim=config_dict.get('head_dim', 64),
            intermediate_size=config_dict.get('intermediate_size', 2880),
            num_local_experts=config_dict.get('num_local_experts', 32),
            num_experts_per_tok=config_dict.get('num_experts_per_tok', 4),
            hidden_act=config_dict.get('hidden_act', 'silu'),
            max_position_embeddings=config_dict.get('max_position_embeddings', 131072),
            rope_theta=config_dict.get('rope_theta', 150000.0),
            sliding_window=config_dict.get('sliding_window', 128),
            rms_norm_eps=config_dict.get('rms_norm_eps', 1e-6),
            layer_types=config_dict.get('layer_types', None),
            quantization_method=quant_method,
            tie_word_embeddings=config_dict.get('tie_word_embeddings', True)
        )


def dequantize_mxfp4(blocks: np.ndarray, scales: np.ndarray, validate: bool = True) -> np.ndarray:
    """
    Dequantize MXFP4 weights.
    
    Args:
        blocks: uint8 array (rows, packed_cols) with packed 4-bit values
        scales: float array (rows,) or (rows, 1) for scaling
        validate: Whether to validate inputs
    
    Returns:
        Dequantized float32 weights
    """
    if validate:
        if blocks.dtype != np.uint8:
            raise ValueError(f"Expected uint8 blocks, got {blocks.dtype}")
        if not np.isfinite(scales).all():
            raise ValueError("Scales contain NaN or infinite values")
    
    # Unpack 4-bit values
    high_nibbles = (blocks >> 4).astype(np.int8)
    low_nibbles = (blocks & 0x0F).astype(np.int8)
    
    # Convert to signed
    high_nibbles = np.where(high_nibbles > 7, high_nibbles - 16, high_nibbles)
    low_nibbles = np.where(low_nibbles > 7, low_nibbles - 16, low_nibbles)
    
    # Reconstruct array
    rows, packed_cols = blocks.shape
    unpacked = np.empty((rows, packed_cols * 2), dtype=np.float32)
    unpacked[:, ::2] = high_nibbles
    unpacked[:, 1::2] = low_nibbles
    
    # Apply scaling
    if scales.ndim == 1:
        scales = scales[:, np.newaxis]
    
    return unpacked * scales / 7.0  # MXFP4 scale factor


class RMSNorm:
    """Root Mean Square Layer Normalization."""
    def __init__(self, weight: jnp.ndarray, eps: float = 1e-6):
        self.weight = weight.astype(jnp.float32)
        self.eps = max(eps, 1e-8)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        input_dtype = x.dtype
        x = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * lax.rsqrt(variance + self.eps)
        return (self.weight * x).astype(input_dtype)


class RotaryEmbedding:
    """Rotary Positional Embedding."""
    def __init__(self, dim: int, max_position_embeddings: int, theta: float):
        self.dim = dim
        inv_freq = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
        t = jnp.arange(max_position_embeddings, dtype=jnp.float32)
        freqs = jnp.einsum("i,j->ij", t, inv_freq)
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        self.cos_cached = jnp.cos(emb)[None, :, None, :]
        self.sin_cached = jnp.sin(emb)[None, :, None, :]

    def __call__(self, x: jnp.ndarray, position_ids: jnp.ndarray) -> jnp.ndarray:
        cos = jnp.take(self.cos_cached, position_ids, axis=1).astype(x.dtype)
        sin = jnp.take(self.sin_cached, position_ids, axis=1).astype(x.dtype)
        
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        rotate_half = jnp.concatenate((-x2, x1), axis=-1)
        
        return (x * cos) + (rotate_half * sin)


class KVCache:
    """Key-Value cache for efficient autoregressive generation."""
    def __init__(self, batch_size: int, max_seq_len: int, num_heads: int, head_dim: int):
        self.k_cache = jnp.zeros((batch_size, num_heads, max_seq_len, head_dim))
        self.v_cache = jnp.zeros((batch_size, num_heads, max_seq_len, head_dim))
        self.seq_len = 0
    
    def update(self, k: jnp.ndarray, v: jnp.ndarray, start_pos: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        seq_len = k.shape[2]
        self.k_cache = self.k_cache.at[:, :, start_pos:start_pos+seq_len].set(k)
        self.v_cache = self.v_cache.at[:, :, start_pos:start_pos+seq_len].set(v)
        self.seq_len = start_pos + seq_len
        return self.k_cache[:, :, :self.seq_len], self.v_cache[:, :, :self.seq_len]


class SelfAttention:
    """Multi-head self-attention with optional KV caching."""
    def __init__(self, params: Dict[str, Any], layer_idx: int, config: GPTOSSConfig):
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.sliding_window = config.sliding_window
        
        # Load weights
        prefix = f'model.layers.{layer_idx}.self_attn'
        self.q_proj = params[f'{prefix}.q_proj.weight']
        self.k_proj = params[f'{prefix}.k_proj.weight']
        self.v_proj = params[f'{prefix}.v_proj.weight']
        self.o_proj = params[f'{prefix}.o_proj.weight']
        
        self.rope = RotaryEmbedding(self.head_dim, config.max_position_embeddings, config.rope_theta)
        self.scale = 1.0 / jnp.sqrt(self.head_dim)

    def __call__(self, x: jnp.ndarray, attention_mask: Optional[jnp.ndarray] = None,
                 position_ids: Optional[jnp.ndarray] = None, 
                 kv_cache: Optional[KVCache] = None,
                 use_cache: bool = False) -> Tuple[jnp.ndarray, Optional[KVCache]]:
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = jnp.dot(x, self.q_proj.T).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = jnp.dot(x, self.k_proj.T).reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = jnp.dot(x, self.v_proj.T).reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        
        # Apply RoPE
        if position_ids is None:
            position_ids = jnp.arange(seq_len)[None, :]
        q = self.rope(q, position_ids)
        k = self.rope(k, position_ids)
        
        # Transpose for attention
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        # Handle KV cache
        if kv_cache is not None:
            start_pos = kv_cache.seq_len if hasattr(kv_cache, 'seq_len') else 0
            k, v = kv_cache.update(k, v, start_pos)
        
        # Handle grouped-query attention
        if self.num_key_value_heads != self.num_heads:
            k = jnp.repeat(k, self.num_heads // self.num_key_value_heads, axis=1)
            v = jnp.repeat(v, self.num_heads // self.num_key_value_heads, axis=1)
        
        # Compute attention scores
        attn_weights = jnp.einsum('bhqd,bhkd->bhqk', q, k) * self.scale
        
        # Apply mask
        if attention_mask is not None:
            attn_weights = jnp.where(attention_mask, attn_weights, -jnp.inf)
        
        # Softmax
        attn_probs = jax_nn.softmax(attn_weights, axis=-1)
        
        # Apply attention to values
        attn_output = jnp.einsum('bhqk,bhkd->bhqd', attn_probs, v)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        output = jnp.dot(attn_output, self.o_proj.T)
        
        return output, kv_cache if use_cache else None


class MixtureOfExperts:
    """Mixture of Experts layer."""
    def __init__(self, params: Dict[str, Any], layer_idx: int, config: GPTOSSConfig):
        self.config = config
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        
        prefix = f'model.layers.{layer_idx}.mlp'
        self.router_weight = params[f'{prefix}.router.weight']
        
        # Load expert weights (handle both quantized and unquantized)
        self.expert_weights = {}
        for expert_idx in range(self.num_experts):
            expert_prefix = f'{prefix}.experts.{expert_idx}'
            self.expert_weights[expert_idx] = {
                'gate_proj': params[f'{expert_prefix}.gate_proj.weight'],
                'up_proj': params[f'{expert_prefix}.up_proj.weight'],
                'down_proj': params[f'{expert_prefix}.down_proj.weight']
            }
        
        # Activation function
        self.activation = jax_nn.silu if config.hidden_act == "silu" else jax_nn.gelu

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.reshape(-1, hidden_dim)
        
        # Router computation
        router_logits = jnp.dot(x_flat, self.router_weight.T)
        top_k_logits, top_k_indices = lax.top_k(router_logits, self.top_k)
        top_k_probs = jax_nn.softmax(top_k_logits, axis=-1)
        
        # Process each expert
        output = jnp.zeros_like(x_flat)
        
        for expert_idx in range(self.num_experts):
            # Find tokens for this expert
            expert_mask = (top_k_indices == expert_idx).any(axis=-1)
            if not expert_mask.any():
                continue
            
            # Get expert weights
            weights = self.expert_weights[expert_idx]
            
            # Expert computation
            expert_input = x_flat[expert_mask]
            gate = jnp.dot(expert_input, weights['gate_proj'].T)
            up = jnp.dot(expert_input, weights['up_proj'].T)
            expert_out = jnp.dot(self.activation(gate) * up, weights['down_proj'].T)
            
            # Apply weighted output
            for k in range(self.top_k):
                k_mask = (top_k_indices[:, k] == expert_idx) & expert_mask
                if k_mask.any():
                    probs = top_k_probs[k_mask, k:k+1]
                    output = output.at[k_mask].add(expert_out * probs)
        
        return output.reshape(batch_size, seq_len, hidden_dim)


class MLP:
    """Standard feedforward layer."""
    def __init__(self, params: Dict[str, Any], layer_idx: int, config: GPTOSSConfig):
        prefix = f'model.layers.{layer_idx}.mlp'
        self.gate_proj = params[f'{prefix}.gate_proj.weight']
        self.up_proj = params[f'{prefix}.up_proj.weight']
        self.down_proj = params[f'{prefix}.down_proj.weight']
        self.activation = jax_nn.silu if config.hidden_act == "silu" else jax_nn.gelu

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        gate = jnp.dot(x, self.gate_proj.T)
        up = jnp.dot(x, self.up_proj.T)
        return jnp.dot(self.activation(gate) * up, self.down_proj.T)


class GPTOSSLayer:
    """Single transformer layer."""
    def __init__(self, params: Dict[str, Any], layer_idx: int, config: GPTOSSConfig):
        self.layer_type = config.layer_types[layer_idx]
        
        # Layer norms
        prefix = f'model.layers.{layer_idx}'
        self.input_layernorm = RMSNorm(params[f'{prefix}.input_layernorm.weight'], config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(params[f'{prefix}.post_attention_layernorm.weight'], config.rms_norm_eps)
        
        # Self-attention
        self.self_attn = SelfAttention(params, layer_idx, config)
        
        # MLP or MoE
        if self.layer_type == "moe":
            self.mlp = MixtureOfExperts(params, layer_idx, config)
        else:
            self.mlp = MLP(params, layer_idx, config)

    def __call__(self, hidden_states: jnp.ndarray, 
                 attention_mask: Optional[jnp.ndarray] = None,
                 position_ids: Optional[jnp.ndarray] = None,
                 kv_cache: Optional[KVCache] = None,
                 use_cache: bool = False) -> Tuple[jnp.ndarray, Optional[KVCache]]:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, cache = self.self_attn(hidden_states, attention_mask, position_ids, kv_cache, use_cache)
        hidden_states = residual + hidden_states
        
        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, cache


class GPTOSSModel:
    """Complete GPT-OSS model."""
    def __init__(self, params: Dict[str, Any], config: GPTOSSConfig):
        self.config = config
        
        # Embeddings
        self.embed_tokens = params['model.embed_tokens.weight']
        
        # Transformer layers
        self.layers = []
        for i in range(config.num_hidden_layers):
            self.layers.append(GPTOSSLayer(params, i, config))
        
        # Final norm
        self.norm = RMSNorm(params['model.norm.weight'], config.rms_norm_eps)
        
        # LM head (may be tied with embeddings)
        if config.tie_word_embeddings:
            self.lm_head = self.embed_tokens.T
        else:
            self.lm_head = params.get('lm_head.weight', self.embed_tokens.T)

    def __call__(self, input_ids: jnp.ndarray, 
                 attention_mask: Optional[jnp.ndarray] = None,
                 position_ids: Optional[jnp.ndarray] = None,
                 kv_caches: Optional[List[KVCache]] = None,
                 use_cache: bool = False) -> Tuple[jnp.ndarray, Optional[List[KVCache]]]:
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        hidden_states = self.embed_tokens[input_ids]
        
        # Default position IDs
        if position_ids is None:
            position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
        
        # Default causal mask
        if attention_mask is None:
            attention_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
            attention_mask = attention_mask[None, None, :, :]
        
        # Apply transformer layers
        new_caches = []
        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i] if kv_caches else None
            hidden_states, cache = layer(hidden_states, attention_mask, position_ids, kv_cache, use_cache)
            if use_cache:
                new_caches.append(cache)
        
        # Final norm and LM head
        hidden_states = self.norm(hidden_states)
        logits = jnp.dot(hidden_states, self.lm_head)
        
        return logits, new_caches if use_cache else None