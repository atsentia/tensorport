#!/usr/bin/env python3
"""
Test JAX inference with converted GPT-OSS model
This is a minimal test to verify the converted weights can be loaded and used.
"""

import json
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import time
from typing import Dict, Tuple

def load_tensor(base_path: Path, tensor_name: str) -> jnp.ndarray:
    """Load a single tensor from the sharded numpy files."""
    # Convert tensor name to file format
    file_name = tensor_name.replace('.', '_') + '.npy'
    
    # Search through shards
    for shard_dir in sorted(base_path.glob('shard_*')):
        tensor_path = shard_dir / file_name
        if tensor_path.exists():
            # Load numpy array and convert to JAX
            np_array = np.load(tensor_path)
            return jnp.array(np_array)
    
    return None

def load_model_subset(model_path: Path) -> Dict[str, jnp.ndarray]:
    """Load a subset of model parameters for testing."""
    print("Loading model parameters...")
    
    # Load manifest to understand structure
    with open(model_path / 'manifest.json') as f:
        manifest = json.load(f)
    
    config = manifest['config']
    print(f"Model config: {config['num_hidden_layers']} layers, hidden_size={config['hidden_size']}")
    
    # Load essential parameters for a simple forward pass test
    params = {}
    
    # Load embeddings
    print("Loading embeddings...")
    params['embed_tokens'] = load_tensor(model_path, 'model.embed_tokens.weight')
    params['norm'] = load_tensor(model_path, 'model.norm.weight')
    
    # Load first layer parameters as a test
    print("Loading layer 0 parameters...")
    layer_params = {}
    
    # Attention parameters
    layer_params['q_proj'] = load_tensor(model_path, 'model.layers.0.self_attn.q_proj.weight')
    layer_params['k_proj'] = load_tensor(model_path, 'model.layers.0.self_attn.k_proj.weight')
    layer_params['v_proj'] = load_tensor(model_path, 'model.layers.0.self_attn.v_proj.weight')
    layer_params['o_proj'] = load_tensor(model_path, 'model.layers.0.self_attn.o_proj.weight')
    
    # Layer norms
    layer_params['input_layernorm'] = load_tensor(model_path, 'model.layers.0.input_layernorm.weight')
    layer_params['post_attention_layernorm'] = load_tensor(model_path, 'model.layers.0.post_attention_layernorm.weight')
    
    params['layer_0'] = layer_params
    
    # Check what we loaded
    loaded_params = sum(1 for v in layer_params.values() if v is not None)
    print(f"Loaded {loaded_params} parameters for layer 0")
    
    return params, config

def simple_attention(query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray) -> jnp.ndarray:
    """Simple scaled dot-product attention."""
    # query, key, value shape: (batch, heads, seq_len, head_dim)
    d_k = query.shape[-1]
    # Transpose key: (batch, heads, seq_len, head_dim) -> (batch, heads, head_dim, seq_len)
    key_transposed = jnp.transpose(key, (0, 1, 3, 2))
    scores = jnp.matmul(query, key_transposed) / jnp.sqrt(d_k)
    attention_weights = jax.nn.softmax(scores, axis=-1)
    return jnp.matmul(attention_weights, value)

def test_forward_pass(params: Dict, config: Dict, input_ids: jnp.ndarray) -> jnp.ndarray:
    """Run a simple forward pass through one layer."""
    print("\nRunning forward pass test...")
    
    batch_size, seq_len = input_ids.shape
    hidden_size = config['hidden_size']
    num_heads = config['num_attention_heads']
    head_dim = config['head_dim']
    
    # Embedding lookup
    embeddings = params['embed_tokens'][input_ids]
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Layer normalization
    hidden_states = embeddings
    layer_params = params['layer_0']
    
    # Input layer norm
    if layer_params['input_layernorm'] is not None:
        # Simple layer norm (just multiply by weight for testing)
        hidden_states = hidden_states * layer_params['input_layernorm']
    
    # Self-attention (simplified)
    if all(layer_params[k] is not None for k in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
        # Project to Q, K, V
        query = jnp.matmul(hidden_states, layer_params['q_proj'].T)
        key = jnp.matmul(hidden_states, layer_params['k_proj'].T)
        value = jnp.matmul(hidden_states, layer_params['v_proj'].T)
        
        # Reshape for multi-head attention
        query = query.reshape(batch_size, seq_len, num_heads, head_dim)
        key = key.reshape(batch_size, seq_len, num_heads // 8, head_dim)  # num_kv_heads = 8
        value = value.reshape(batch_size, seq_len, num_heads // 8, head_dim)
        
        # Repeat KV heads to match Q heads
        key = jnp.repeat(key, 8, axis=2)
        value = jnp.repeat(value, 8, axis=2)
        
        # Transpose for attention
        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)
        
        # Compute attention
        attn_output = simple_attention(query, key, value)
        
        # Reshape back
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_len, num_heads * head_dim)  # Should be 4096
        
        # Output projection
        attn_output = jnp.matmul(attn_output, layer_params['o_proj'].T)
        
        # Residual connection
        hidden_states = embeddings + attn_output
        
        print(f"Attention output shape: {hidden_states.shape}")
    
    # Final layer norm
    if params['norm'] is not None:
        hidden_states = hidden_states * params['norm']
    
    return hidden_states

def main():
    print("="*60)
    print("JAX Inference Test with Converted GPT-OSS Model")
    print("="*60)
    
    # Check JAX installation
    print(f"\nJAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    
    model_path = Path('jax-numpy-model')
    if not model_path.exists():
        print(f"❌ Model directory not found: {model_path}")
        return
    
    # Load model parameters
    try:
        params, config = load_model_subset(model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Check what we loaded
    print("\n📊 Loaded parameters:")
    for key, value in params.items():
        if key == 'layer_0':
            print(f"  • {key}:")
            for k, v in value.items():
                if v is not None:
                    print(f"      - {k}: shape {v.shape}")
        elif value is not None:
            print(f"  • {key}: shape {value.shape}")
    
    # Create dummy input
    batch_size = 1
    seq_length = 10
    vocab_size = params['embed_tokens'].shape[0] if params['embed_tokens'] is not None else 200064
    
    # Random input IDs (make sure they're within vocab range)
    input_ids = jax.random.randint(
        jax.random.PRNGKey(42),
        shape=(batch_size, seq_length),
        minval=0,
        maxval=min(1000, vocab_size)  # Use small IDs to avoid index errors
    )
    
    print(f"\n🎯 Test input shape: {input_ids.shape}")
    print(f"   Input IDs: {input_ids[0, :5]}...")  # Show first 5 tokens
    
    # Run inference
    try:
        start_time = time.time()
        output = test_forward_pass(params, config, input_ids)
        inference_time = time.time() - start_time
        
        print(f"\n✅ Inference successful!")
        print(f"   Output shape: {output.shape}")
        print(f"   Output dtype: {output.dtype}")
        print(f"   Time taken: {inference_time:.3f} seconds")
        
        # Check for NaN or Inf
        if jnp.isnan(output).any():
            print("   ⚠️ Warning: Output contains NaN values")
        elif jnp.isinf(output).any():
            print("   ⚠️ Warning: Output contains Inf values")
        else:
            print("   ✓ Output values are finite")
            print(f"   Output mean: {jnp.mean(output):.6f}")
            print(f"   Output std: {jnp.std(output):.6f}")
        
    except Exception as e:
        print(f"\n❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)

if __name__ == "__main__":
    main()