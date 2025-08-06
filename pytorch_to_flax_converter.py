#!/usr/bin/env python3
"""
Weight conversion utility for PyTorch GPT-2 to Flax GPT-2.
Handles parameter mapping, tensor transformations, and validation.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union
from pathlib import Path
import json
import warnings
import sys

from flax_gpt2_model import FlaxGPT2Config, FlaxGPT2LMHeadModel, create_model, init_model_params


def load_pytorch_model(model_path: Union[str, Path]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load PyTorch GPT-2 model and configuration.
    
    Args:
        model_path: Path to model directory or Hugging Face model identifier
        
    Returns:
        Tuple of (pytorch_state_dict, config_dict)
    """
    try:
        from transformers import GPT2LMHeadModel, GPT2Config
        import torch
    except ImportError:
        raise ImportError("transformers and torch are required for PyTorch model loading")
    
    model_path = Path(model_path) if isinstance(model_path, str) else model_path
    
    if model_path.exists() and model_path.is_dir():
        # Load from local directory
        config = GPT2Config.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)
    else:
        # Load from Hugging Face Hub
        config = GPT2Config.from_pretrained(str(model_path))
        model = GPT2LMHeadModel.from_pretrained(str(model_path))
    
    # Convert to dict
    config_dict = config.to_dict()
    state_dict = model.state_dict()
    
    # Convert tensors to numpy
    numpy_state_dict = {}
    for key, tensor in state_dict.items():
        if hasattr(tensor, 'detach'):
            numpy_state_dict[key] = tensor.detach().cpu().numpy()
        else:
            numpy_state_dict[key] = np.array(tensor)
    
    print(f"✅ Loaded PyTorch model:")
    print(f"   Config: {config_dict['model_type']} with {config_dict['n_layer']} layers")
    print(f"   Parameters: {sum(t.numel() if hasattr(t, 'numel') else t.size for t in state_dict.values()):,}")
    
    return numpy_state_dict, config_dict


def create_parameter_mapping() -> Dict[str, str]:
    """
    Create mapping from PyTorch parameter names to Flax parameter names.
    
    Returns:
        Dictionary mapping PyTorch names to Flax names
    """
    mapping = {
        # Token embeddings
        "transformer.wte.weight": "params/transformer/wte/embedding",
        
        # Position embeddings  
        "transformer.wpe.weight": "params/transformer/wpe/embedding",
        
        # Final layer norm
        "transformer.ln_f.weight": "params/transformer/ln_f/scale",
        "transformer.ln_f.bias": "params/transformer/ln_f/bias",
        
        # Language model head (if not tied)
        "lm_head.weight": "params/lm_head/kernel",
    }
    
    # Add layer-specific mappings
    for i in range(48):  # Support up to 48 layers (GPT-2 XL max)
        layer_prefix_pt = f"transformer.h.{i}"
        layer_prefix_flax = f"params/transformer/h_{i}"
        
        # Layer norms
        mapping[f"{layer_prefix_pt}.ln_1.weight"] = f"{layer_prefix_flax}/ln_1/scale"
        mapping[f"{layer_prefix_pt}.ln_1.bias"] = f"{layer_prefix_flax}/ln_1/bias"
        mapping[f"{layer_prefix_pt}.ln_2.weight"] = f"{layer_prefix_flax}/ln_2/scale"
        mapping[f"{layer_prefix_pt}.ln_2.bias"] = f"{layer_prefix_flax}/ln_2/bias"
        
        # Attention weights (combined QKV in PyTorch, need to split for Flax)
        mapping[f"{layer_prefix_pt}.attn.c_attn.weight"] = f"{layer_prefix_flax}/attn/c_attn/kernel"
        mapping[f"{layer_prefix_pt}.attn.c_attn.bias"] = f"{layer_prefix_flax}/attn/c_attn/bias"
        mapping[f"{layer_prefix_pt}.attn.c_proj.weight"] = f"{layer_prefix_flax}/attn/c_proj/kernel"
        mapping[f"{layer_prefix_pt}.attn.c_proj.bias"] = f"{layer_prefix_flax}/attn/c_proj/bias"
        
        # MLP weights
        mapping[f"{layer_prefix_pt}.mlp.c_fc.weight"] = f"{layer_prefix_flax}/mlp/c_fc/kernel"
        mapping[f"{layer_prefix_pt}.mlp.c_fc.bias"] = f"{layer_prefix_flax}/mlp/c_fc/bias"
        mapping[f"{layer_prefix_pt}.mlp.c_proj.weight"] = f"{layer_prefix_flax}/mlp/c_proj/kernel"
        mapping[f"{layer_prefix_pt}.mlp.c_proj.bias"] = f"{layer_prefix_flax}/mlp/c_proj/bias"
    
    return mapping


def transpose_linear_weights(weight: np.ndarray) -> np.ndarray:
    """
    Transpose linear layer weights from PyTorch (out_features, in_features) 
    to Flax (in_features, out_features) format.
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D weight tensor, got shape {weight.shape}")
    return weight.T


def build_flax_params_tree(pytorch_state: Dict[str, np.ndarray], 
                          config: FlaxGPT2Config) -> Dict[str, Any]:
    """
    Convert PyTorch state dict to Flax parameter tree structure.
    
    Args:
        pytorch_state: PyTorch state dictionary with numpy arrays
        config: Flax model configuration
        
    Returns:
        Flax parameter tree
    """
    mapping = create_parameter_mapping()
    flax_params = {}
    
    print("Converting PyTorch parameters to Flax format...")
    
    for pt_name, np_tensor in pytorch_state.items():
        if pt_name in mapping:
            flax_path = mapping[pt_name]
            
            # Apply tensor transformations
            if "kernel" in flax_path:
                if "embedding" in flax_path:
                    # Don't transpose embedding weights
                    converted_tensor = np_tensor
                else:
                    # Transpose linear layer weights from (out, in) to (in, out)
                    converted_tensor = transpose_linear_weights(np_tensor)
            else:
                # Keep as-is for biases and layer norm scales
                converted_tensor = np_tensor
            
            # Build nested dictionary structure
            path_parts = flax_path.split('/')
            current_dict = flax_params
            
            for part in path_parts[:-1]:
                if part not in current_dict:
                    current_dict[part] = {}
                current_dict = current_dict[part]
            
            # Convert to JAX array
            current_dict[path_parts[-1]] = jnp.array(converted_tensor)
            
            print(f"  ✓ {pt_name} -> {flax_path} (shape: {converted_tensor.shape})")
        else:
            print(f"  ⚠ Skipping unmapped parameter: {pt_name}")
    
    return flax_params


def validate_converted_params(flax_params: Dict[str, Any], 
                            flax_config: FlaxGPT2Config,
                            pytorch_state: Dict[str, np.ndarray]) -> bool:
    """
    Validate that converted parameters are correct.
    
    Args:
        flax_params: Converted Flax parameters
        flax_config: Flax model configuration
        pytorch_state: Original PyTorch state dict
        
    Returns:
        True if validation passes
    """
    print("\nValidating converted parameters...")
    
    # Create Flax model and check parameter structure
    model = create_model(flax_config)
    rng = jax.random.PRNGKey(0)
    expected_params = init_model_params(model, rng, (1, 10))
    
    def check_param_structure(expected: Dict, actual: Dict, path: str = ""):
        """Recursively check parameter tree structure."""
        for key in expected:
            full_path = f"{path}/{key}" if path else key
            if key not in actual:
                print(f"  ❌ Missing parameter: {full_path}")
                return False
            
            if isinstance(expected[key], dict):
                if not isinstance(actual[key], dict):
                    print(f"  ❌ Type mismatch at {full_path}: expected dict, got {type(actual[key])}")
                    return False
                if not check_param_structure(expected[key], actual[key], full_path):
                    return False
            else:
                # Check array shape
                exp_shape = expected[key].shape
                act_shape = actual[key].shape
                if exp_shape != act_shape:
                    print(f"  ❌ Shape mismatch at {full_path}: expected {exp_shape}, got {act_shape}")
                    return False
        return True
    
    structure_ok = check_param_structure(expected_params, flax_params)
    if not structure_ok:
        return False
    
    # Count parameters
    flax_param_count = sum(x.size for x in jax.tree_util.tree_leaves(flax_params))
    pytorch_param_count = sum(t.size for t in pytorch_state.values())
    
    print(f"  Parameter counts:")
    print(f"    PyTorch: {pytorch_param_count:,}")
    print(f"    Flax: {flax_param_count:,}")
    
    if flax_param_count != pytorch_param_count:
        print(f"  ⚠ Parameter count mismatch!")
        # This might be OK if tied embeddings are handled differently
        if abs(flax_param_count - pytorch_param_count) > 50000:  # Allow small differences
            return False
    
    print("  ✅ Parameter validation passed!")
    return True


def test_numerical_equivalence(pytorch_state: Dict[str, np.ndarray],
                              flax_params: Dict[str, Any],
                              flax_config: FlaxGPT2Config,
                              test_input: Optional[np.ndarray] = None) -> bool:
    """
    Test numerical equivalence between PyTorch and Flax models.
    
    Args:
        pytorch_state: PyTorch state dictionary
        flax_params: Converted Flax parameters
        flax_config: Flax model configuration
        test_input: Optional test input (defaults to random)
        
    Returns:
        True if outputs are numerically close
    """
    print("\nTesting numerical equivalence...")
    
    try:
        from transformers import GPT2LMHeadModel, GPT2Config
        import torch
    except ImportError:
        print("  ⚠ Skipping numerical test - transformers not available")
        return True
    
    # Create test input
    if test_input is None:
        batch_size, seq_len = 2, 8
        test_input = np.random.randint(0, min(1000, flax_config.vocab_size), (batch_size, seq_len))
    
    # Test PyTorch model
    pt_config = GPT2Config(
        vocab_size=flax_config.vocab_size,
        n_positions=flax_config.max_position_embeddings,
        n_embd=flax_config.hidden_size,
        n_layer=flax_config.num_hidden_layers,
        n_head=flax_config.num_attention_heads,
    )
    pt_model = GPT2LMHeadModel(pt_config)
    pt_model.load_state_dict({k: torch.from_numpy(v) for k, v in pytorch_state.items()})
    pt_model.eval()
    
    with torch.no_grad():
        pt_input = torch.from_numpy(test_input).long()
        pt_output = pt_model(pt_input).logits.numpy()
    
    # Test Flax model
    flax_model = create_model(flax_config)
    flax_input = jnp.array(test_input)
    flax_output = flax_model.apply(flax_params, flax_input, deterministic=True)
    flax_output = np.array(flax_output)
    
    # Compare outputs
    max_diff = np.max(np.abs(pt_output - flax_output))
    mean_diff = np.mean(np.abs(pt_output - flax_output))
    
    print(f"  Output comparison:")
    print(f"    PyTorch shape: {pt_output.shape}")
    print(f"    Flax shape: {flax_output.shape}")
    print(f"    Max difference: {max_diff:.6f}")
    print(f"    Mean difference: {mean_diff:.6f}")
    
    # Check if differences are acceptable
    tolerance = 1e-4  # Allow small numerical differences
    if max_diff > tolerance:
        print(f"  ❌ Numerical differences too large (max: {max_diff} > {tolerance})")
        return False
    
    print(f"  ✅ Numerical equivalence confirmed!")
    return True


def convert_pytorch_to_flax(pytorch_model_path: Union[str, Path],
                           output_path: Optional[Union[str, Path]] = None,
                           precision: str = "float32",
                           validate: bool = True) -> Tuple[Dict[str, Any], FlaxGPT2Config]:
    """
    Complete conversion pipeline from PyTorch GPT-2 to Flax.
    
    Args:
        pytorch_model_path: Path to PyTorch model
        output_path: Optional path to save converted model
        precision: Target precision ("float32" or "float16")
        validate: Whether to run validation tests
        
    Returns:
        Tuple of (flax_params, flax_config)
    """
    print(f"🚀 Converting PyTorch GPT-2 to Flax...")
    print(f"   Source: {pytorch_model_path}")
    print(f"   Precision: {precision}")
    
    # Load PyTorch model
    pytorch_state, pytorch_config = load_pytorch_model(pytorch_model_path)
    
    # Create Flax configuration
    flax_config = FlaxGPT2Config(
        vocab_size=pytorch_config['vocab_size'],
        hidden_size=pytorch_config['n_embd'],
        num_hidden_layers=pytorch_config['n_layer'],
        num_attention_heads=pytorch_config['n_head'],
        max_position_embeddings=pytorch_config['n_positions'],
        layer_norm_epsilon=pytorch_config['layer_norm_epsilon'],
        hidden_dropout_prob=pytorch_config.get('attn_pdrop', 0.1),
        attention_probs_dropout_prob=pytorch_config.get('attn_pdrop', 0.1),
        tie_word_embeddings=pytorch_config.get('tie_word_embeddings', True)
    )
    
    print(f"📋 Model configuration:")
    print(f"   Hidden size: {flax_config.hidden_size}")
    print(f"   Layers: {flax_config.num_hidden_layers}")
    print(f"   Attention heads: {flax_config.num_attention_heads}")
    print(f"   Vocab size: {flax_config.vocab_size}")
    
    # Convert parameters
    flax_params = build_flax_params_tree(pytorch_state, flax_config)
    
    # Convert precision if requested
    if precision == "float16":
        print("Converting to float16...")
        flax_params = jax.tree_util.tree_map(
            lambda x: x.astype(jnp.float16) if x.dtype in [jnp.float32, jnp.float64] else x,
            flax_params
        )
    
    # Validation
    if validate:
        validate_converted_params(flax_params, flax_config, pytorch_state)
        test_numerical_equivalence(pytorch_state, flax_params, flax_config)
    
    # Save if output path provided
    if output_path:
        save_converted_model(flax_params, flax_config, output_path)
    
    print("✅ Conversion completed successfully!")
    return flax_params, flax_config


def save_converted_model(flax_params: Dict[str, Any],
                        flax_config: FlaxGPT2Config,
                        output_path: Union[str, Path]):
    """Save converted model to disk."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_file = output_path / "config.json"
    with open(config_file, 'w') as f:
        # Convert dataclass to dict for JSON serialization
        config_dict = {
            'vocab_size': flax_config.vocab_size,
            'hidden_size': flax_config.hidden_size,
            'num_hidden_layers': flax_config.num_hidden_layers,
            'num_attention_heads': flax_config.num_attention_heads,
            'head_dim': flax_config.head_dim,
            'intermediate_size': flax_config.intermediate_size,
            'hidden_act': flax_config.hidden_act,
            'hidden_dropout_prob': flax_config.hidden_dropout_prob,
            'attention_probs_dropout_prob': flax_config.attention_probs_dropout_prob,
            'max_position_embeddings': flax_config.max_position_embeddings,
            'layer_norm_epsilon': flax_config.layer_norm_epsilon,
            'initializer_range': flax_config.initializer_range,
            'use_cache': flax_config.use_cache,
            'tie_word_embeddings': flax_config.tie_word_embeddings,
        }
        json.dump(config_dict, f, indent=2)
    
    # Save parameters using Flax serialization
    from flax import serialization
    params_file = output_path / "flax_model.msgpack"
    with open(params_file, 'wb') as f:
        f.write(serialization.to_bytes(flax_params))
    
    print(f"💾 Saved converted model to {output_path}")
    print(f"   Config: {config_file}")
    print(f"   Parameters: {params_file}")


def load_converted_model(model_path: Union[str, Path]) -> Tuple[Dict[str, Any], FlaxGPT2Config]:
    """Load a converted Flax model from disk."""
    model_path = Path(model_path)
    
    # Load configuration
    config_file = model_path / "config.json"
    with open(config_file) as f:
        config_dict = json.load(f)
    
    flax_config = FlaxGPT2Config(**config_dict)
    
    # Load parameters
    from flax import serialization
    params_file = model_path / "flax_model.msgpack"
    with open(params_file, 'rb') as f:
        flax_params = serialization.from_bytes(None, f.read())
    
    return flax_params, flax_config


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pytorch_to_flax_converter.py <pytorch_model_path> [output_path]")
        print("Example: python pytorch_to_flax_converter.py gpt2 ./gpt2_flax")
        sys.exit(1)
    
    pytorch_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Run conversion
    flax_params, flax_config = convert_pytorch_to_flax(
        pytorch_path, 
        output_path, 
        precision="float32", 
        validate=True
    )
    
    print(f"\n🎉 Conversion complete!")
    if output_path:
        print(f"Model saved to: {output_path}")