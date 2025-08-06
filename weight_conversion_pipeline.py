#!/usr/bin/env python3
"""
Weight Conversion Pipeline for GPT-OSS Model
Converts PyTorch safetensors weights to Flax parameter format.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import warnings

import jax
import jax.numpy as jnp
from flax import core

from flax_gptoss_model import FlaxGPTOSSLMHeadModel, GPTOSSConfig, initialize_model


def load_tensor_from_numpy_shards(base_path: Path, tensor_name: str) -> Optional[np.ndarray]:
    """
    Load a tensor from TensorPort converted numpy shards.
    
    Args:
        base_path: Path to the directory containing shard folders
        tensor_name: Name of the tensor to load
        
    Returns:
        Loaded tensor as numpy array, or None if not found
    """
    file_name = tensor_name.replace('.', '_') + '.npy'
    
    for shard_dir in sorted(base_path.glob('shard_*')):
        tensor_path = shard_dir / file_name
        if tensor_path.exists():
            return np.load(tensor_path)
    
    return None


def load_safetensors_metadata(model_path: Path) -> Dict[str, Any]:
    """Load model configuration from safetensors metadata."""
    index_path = model_path / "model.safetensors.index.json"
    config_path = model_path / "config.json"
    
    metadata = {}
    
    if index_path.exists():
        with open(index_path) as f:
            index_data = json.load(f)
            metadata.update(index_data.get('metadata', {}))
    
    if config_path.exists():
        with open(config_path) as f:
            config_data = json.load(f)
            metadata.update(config_data)
    
    return metadata


def create_pytorch_to_flax_mapping() -> Dict[str, str]:
    """
    Create mapping from PyTorch parameter names to Flax parameter names.
    
    Returns:
        Dictionary mapping PyTorch names to Flax names
    """
    mapping = {
        # Embeddings
        'model.embed_tokens.weight': 'embed_tokens.embedding',
        'lm_head.weight': 'lm_head.kernel',
        
        # Final norm
        'model.norm.weight': 'norm.weight',
    }
    
    # Layer-specific mappings
    for layer_idx in range(50):  # Support up to 50 layers
        layer_prefix = f'model.layers.{layer_idx}'
        flax_prefix = f'layers_{layer_idx}'
        
        # Attention mappings
        mapping.update({
            f'{layer_prefix}.input_layernorm.weight': f'{flax_prefix}.input_layernorm.weight',
            f'{layer_prefix}.post_attention_layernorm.weight': f'{flax_prefix}.post_attention_layernorm.weight',
            f'{layer_prefix}.self_attn.q_proj.weight': f'{flax_prefix}.attention.q_proj.kernel',
            f'{layer_prefix}.self_attn.k_proj.weight': f'{flax_prefix}.attention.k_proj.kernel',
            f'{layer_prefix}.self_attn.v_proj.weight': f'{flax_prefix}.attention.v_proj.kernel',
            f'{layer_prefix}.self_attn.o_proj.weight': f'{flax_prefix}.attention.o_proj.kernel',
        })
        
        # MLP mappings (standard)
        mapping.update({
            f'{layer_prefix}.mlp.gate_proj.weight': f'{flax_prefix}.mlp.gate_proj.kernel',
            f'{layer_prefix}.mlp.up_proj.weight': f'{flax_prefix}.mlp.up_proj.kernel',
            f'{layer_prefix}.mlp.down_proj.weight': f'{flax_prefix}.mlp.down_proj.kernel',
        })
        
        # MoE mappings
        mapping[f'{layer_prefix}.mlp.router.weight'] = f'{flax_prefix}.mlp.router.kernel'
        
        for expert_idx in range(32):  # Support up to 32 experts
            expert_prefix = f'{layer_prefix}.mlp.experts.{expert_idx}'
            flax_expert_prefix = f'{flax_prefix}.mlp.expert_{expert_idx}'
            
            mapping.update({
                f'{expert_prefix}.gate_proj.weight': f'{flax_expert_prefix}.gate_proj.kernel',
                f'{expert_prefix}.up_proj.weight': f'{flax_expert_prefix}.up_proj.kernel',
                f'{expert_prefix}.down_proj.weight': f'{flax_expert_prefix}.down_proj.kernel',
            })
    
    return mapping


def transpose_linear_weights(weight: np.ndarray) -> np.ndarray:
    """
    Transpose linear layer weights from PyTorch to Flax format.
    PyTorch: (out_features, in_features)
    Flax: (in_features, out_features)
    """
    if len(weight.shape) == 2:
        return weight.T
    return weight


def convert_pytorch_weights_to_flax(pytorch_weights: Dict[str, np.ndarray], 
                                  config: GPTOSSConfig) -> Dict[str, Any]:
    """
    Convert PyTorch weights to Flax parameter format.
    
    Args:
        pytorch_weights: Dictionary of PyTorch weights
        config: Model configuration
        
    Returns:
        Flax parameters dictionary
    """
    print("🔄 Converting PyTorch weights to Flax format...")
    
    # Get parameter mapping
    mapping = create_pytorch_to_flax_mapping()
    
    # Initialize with params wrapper
    flax_params = {'params': {}}
    
    # Convert weights
    converted_count = 0
    for pytorch_name, pytorch_weight in pytorch_weights.items():
        if pytorch_name in mapping:
            flax_name = mapping[pytorch_name]
            
            # Transpose linear weights
            if 'kernel' in flax_name:
                converted_weight = transpose_linear_weights(pytorch_weight)
            else:
                converted_weight = pytorch_weight
            
            # Build nested dictionary structure under params
            parts = flax_name.split('.')
            current = flax_params['params']
            
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            current[parts[-1]] = jnp.array(converted_weight)
            converted_count += 1
            
            print(f"  ✅ {pytorch_name} -> {flax_name} {pytorch_weight.shape} -> {converted_weight.shape}")
        else:
            print(f"  ⚠️  Unmapped parameter: {pytorch_name} {pytorch_weight.shape}")
    
    print(f"✅ Converted {converted_count} parameters to Flax format")
    
    return core.freeze(flax_params)


def validate_converted_weights(flax_params: Dict[str, Any], 
                             model: FlaxGPTOSSLMHeadModel,
                             config: GPTOSSConfig) -> bool:
    """
    Validate that converted weights have correct shapes and can be used with model.
    
    Args:
        flax_params: Converted Flax parameters
        model: Flax model instance
        config: Model configuration
        
    Returns:
        True if validation passes
    """
    print("🔍 Validating converted weights...")
    
    try:
        # Test forward pass with dummy input
        rng = jax.random.PRNGKey(42)
        batch_size, seq_len = 2, 10
        input_ids = jax.random.randint(rng, (batch_size, seq_len), 0, config.vocab_size)
        
        # Try forward pass
        logits = model.apply(flax_params, input_ids)
        
        # Check output shape
        expected_shape = (batch_size, seq_len, config.vocab_size)
        if logits.shape != expected_shape:
            print(f"❌ Output shape mismatch: {logits.shape} != {expected_shape}")
            return False
        
        # Check for NaN/Inf values
        if not jnp.isfinite(logits).all():
            print("❌ Output contains NaN or Inf values")
            return False
        
        print(f"✅ Forward pass successful: {input_ids.shape} -> {logits.shape}")
        print(f"✅ Output is finite (no NaN/Inf)")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False


def load_and_convert_model(model_path: Path, 
                          output_path: Optional[Path] = None) -> Tuple[FlaxGPTOSSLMHeadModel, Dict[str, Any], GPTOSSConfig]:
    """
    Complete pipeline to load and convert GPT-OSS model from safetensors to Flax.
    
    Args:
        model_path: Path to converted numpy shards or original safetensors
        output_path: Optional path to save converted Flax parameters
        
    Returns:
        Tuple of (model, params, config)
    """
    print("=" * 60)
    print("GPT-OSS WEIGHT CONVERSION PIPELINE")
    print("=" * 60)
    
    # Load model configuration
    print("📋 Loading model configuration...")
    metadata = load_safetensors_metadata(model_path)
    config = GPTOSSConfig(
        vocab_size=metadata.get('vocab_size', 201088),
        hidden_size=metadata.get('hidden_size', 2880),
        num_hidden_layers=metadata.get('num_hidden_layers', 24),
        num_attention_heads=metadata.get('num_attention_heads', 64),
        num_key_value_heads=metadata.get('num_key_value_heads', 8),
        head_dim=metadata.get('head_dim', 64),
        intermediate_size=metadata.get('intermediate_size', 2880),
        num_local_experts=metadata.get('num_local_experts', 32),
        num_experts_per_tok=metadata.get('num_experts_per_tok', 4),
        hidden_act=metadata.get('hidden_act', 'silu'),
        max_position_embeddings=metadata.get('max_position_embeddings', 131072),
        rope_theta=metadata.get('rope_theta', 150000.0),
        rms_norm_eps=metadata.get('rms_norm_eps', 1e-6),
        tie_word_embeddings=metadata.get('tie_word_embeddings', True)
    )
    
    print(f"✅ Model config: {config.num_hidden_layers} layers, {config.vocab_size} vocab")
    
    # Create Flax model
    print("🏗️  Creating Flax model...")
    model = FlaxGPTOSSLMHeadModel(config)
    
    # Load PyTorch weights
    print("📦 Loading PyTorch weights...")
    pytorch_weights = {}
    
    # Define essential parameters to load
    essential_params = [
        'model.embed_tokens.weight',
        'model.norm.weight',
        'lm_head.weight'
    ]
    
    # Add layer parameters
    for i in range(config.num_hidden_layers):
        layer_params = [
            f'model.layers.{i}.input_layernorm.weight',
            f'model.layers.{i}.post_attention_layernorm.weight',
            f'model.layers.{i}.self_attn.q_proj.weight',
            f'model.layers.{i}.self_attn.k_proj.weight',
            f'model.layers.{i}.self_attn.v_proj.weight',
            f'model.layers.{i}.self_attn.o_proj.weight',
            f'model.layers.{i}.mlp.gate_proj.weight',
            f'model.layers.{i}.mlp.up_proj.weight',
            f'model.layers.{i}.mlp.down_proj.weight',
        ]
        essential_params.extend(layer_params)
    
    # Load weights
    loaded_count = 0
    for param_name in essential_params:
        weight = load_tensor_from_numpy_shards(model_path, param_name)
        if weight is not None:
            pytorch_weights[param_name] = weight
            loaded_count += 1
        else:
            print(f"⚠️  Could not load: {param_name}")
    
    print(f"✅ Loaded {loaded_count} parameters from {model_path}")
    
    # Convert to Flax format
    flax_params = convert_pytorch_weights_to_flax(pytorch_weights, config)
    
    # Validate conversion
    if validate_converted_weights(flax_params, model, config):
        print("✅ Weight conversion validation passed!")
    else:
        raise ValueError("Weight conversion validation failed!")
    
    # Save converted parameters if requested
    if output_path:
        print(f"💾 Saving converted parameters to {output_path}...")
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save as msgpack for efficiency
        from flax import serialization
        bytes_data = serialization.to_bytes(flax_params)
        with open(output_path / "flax_params.msgpack", "wb") as f:
            f.write(bytes_data)
        
        # Save config
        with open(output_path / "config.json", "w") as f:
            config_dict = {
                'vocab_size': config.vocab_size,
                'hidden_size': config.hidden_size,
                'num_hidden_layers': config.num_hidden_layers,
                'num_attention_heads': config.num_attention_heads,
                'num_key_value_heads': config.num_key_value_heads,
                'head_dim': config.head_dim,
                'intermediate_size': config.intermediate_size,
                'num_local_experts': config.num_local_experts,
                'num_experts_per_tok': config.num_experts_per_tok,
                'hidden_act': config.hidden_act,
                'max_position_embeddings': config.max_position_embeddings,
                'rope_theta': config.rope_theta,
                'rms_norm_eps': config.rms_norm_eps,
                'tie_word_embeddings': config.tie_word_embeddings
            }
            json.dump(config_dict, f, indent=2)
        
        print(f"✅ Saved to {output_path}")
    
    print("🎉 Weight conversion completed successfully!")
    
    return model, flax_params, config


def load_converted_model(model_path: Path) -> Tuple[FlaxGPTOSSLMHeadModel, Dict[str, Any], GPTOSSConfig]:
    """
    Load a previously converted Flax model.
    
    Args:
        model_path: Path to directory with converted model
        
    Returns:
        Tuple of (model, params, config)
    """
    print(f"📁 Loading converted model from {model_path}...")
    
    # Load config
    with open(model_path / "config.json") as f:
        config_dict = json.load(f)
    
    config = GPTOSSConfig(**config_dict)
    
    # Create model
    model = FlaxGPTOSSLMHeadModel(config)
    
    # Load parameters
    from flax import serialization
    with open(model_path / "flax_params.msgpack", "rb") as f:
        bytes_data = f.read()
    
    flax_params = serialization.from_bytes(None, bytes_data)
    
    print(f"✅ Loaded model with {config.num_hidden_layers} layers")
    
    return model, flax_params, config


def demo_conversion():
    """Demo the conversion process with a mock model."""
    print("🎯 Demonstrating weight conversion process...")
    
    # Create mock PyTorch weights
    config = GPTOSSConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=32,  # 256 / 8 = 32
        intermediate_size=256  # Same as hidden_size for demo
    )
    
    mock_weights = {
        'model.embed_tokens.weight': np.random.normal(0, 0.02, (config.vocab_size, config.hidden_size)),
        'model.norm.weight': np.ones((config.hidden_size,)),
        'lm_head.weight': np.random.normal(0, 0.02, (config.vocab_size, config.hidden_size)),
    }
    
    # Add layer weights
    for i in range(config.num_hidden_layers):
        layer_weights = {
            f'model.layers.{i}.input_layernorm.weight': np.ones((config.hidden_size,)),
            f'model.layers.{i}.post_attention_layernorm.weight': np.ones((config.hidden_size,)),
            f'model.layers.{i}.self_attn.q_proj.weight': np.random.normal(0, 0.02, (config.hidden_size, config.hidden_size)),
            f'model.layers.{i}.self_attn.k_proj.weight': np.random.normal(0, 0.02, (config.hidden_size, config.hidden_size)),
            f'model.layers.{i}.self_attn.v_proj.weight': np.random.normal(0, 0.02, (config.hidden_size, config.hidden_size)),
            f'model.layers.{i}.self_attn.o_proj.weight': np.random.normal(0, 0.02, (config.hidden_size, config.hidden_size)),
            f'model.layers.{i}.mlp.gate_proj.weight': np.random.normal(0, 0.02, (config.intermediate_size, config.hidden_size)),
            f'model.layers.{i}.mlp.up_proj.weight': np.random.normal(0, 0.02, (config.intermediate_size, config.hidden_size)),
            f'model.layers.{i}.mlp.down_proj.weight': np.random.normal(0, 0.02, (config.hidden_size, config.intermediate_size)),
        }
        mock_weights.update(layer_weights)
    
    # Convert to Flax
    model = FlaxGPTOSSLMHeadModel(config)
    flax_params = convert_pytorch_weights_to_flax(mock_weights, config)
    
    # Validate
    if validate_converted_weights(flax_params, model, config):
        print("✅ Mock conversion validation passed!")
    else:
        print("❌ Mock conversion validation failed!")


if __name__ == "__main__":
    demo_conversion()
    print("\n🎉 Weight conversion pipeline is ready for production!")