#!/usr/bin/env python3
"""
GPT-OSS Model Loading Utilities
Handles loading from safetensors, converting bfloat16, and managing sharded models.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import warnings
import sys

from gptoss_model import GPTOSSConfig, GPTOSSModel


def convert_bfloat16_to_float32(tensor: np.ndarray) -> np.ndarray:
    """
    Convert bfloat16 tensors to float32.
    Handles the special case where safetensors returns bfloat16 which NumPy doesn't support.
    """
    if hasattr(tensor, 'dtype'):
        dtype_str = str(tensor.dtype)
        
        # Check if it's bfloat16
        if 'bfloat16' in dtype_str:
            # bfloat16 is stored as uint16 with special interpretation
            # We need to convert it properly to float32
            if tensor.dtype == np.uint16 or len(tensor.shape) == 0:
                # Single value or already uint16
                as_uint16 = np.asarray(tensor, dtype=np.uint16)
            else:
                # Try to view as uint16
                as_uint16 = tensor.view(np.uint16)
            
            # Convert bfloat16 (uint16) to float32
            # bfloat16 format: 1 sign bit, 8 exponent bits, 7 mantissa bits
            # float32 format: 1 sign bit, 8 exponent bits, 23 mantissa bits
            as_uint32 = as_uint16.astype(np.uint32) << 16
            float_array = as_uint32.view(np.float32)
            return float_array
    
    # Return as-is if not bfloat16
    return np.asarray(tensor, dtype=np.float32)


def load_safetensors(file_path: Path) -> Dict[str, np.ndarray]:
    """
    Load a safetensors file and convert to numpy arrays.
    Handles bfloat16 conversion automatically.
    """
    try:
        from safetensors import safe_open
    except ImportError:
        print("Installing safetensors...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "safetensors", "-q"])
        from safetensors import safe_open
    
    tensors = {}
    
    # Try loading with pytorch first (handles bfloat16 better)
    try:
        with safe_open(file_path, framework="pt") as f:
            for key in f.keys():
                try:
                    tensor = f.get_tensor(key)
                    # Convert torch tensor to numpy
                    if hasattr(tensor, 'numpy'):
                        if str(tensor.dtype) == 'torch.bfloat16':
                            # Convert bfloat16 to float32 in torch first
                            tensor = tensor.float()
                        tensor = tensor.numpy()
                    else:
                        tensor = np.array(tensor, dtype=np.float32)
                    tensors[key] = tensor
                except Exception as e:
                    # Skip problematic tensors
                    if 'bfloat16' not in str(e):
                        warnings.warn(f"Failed to load tensor {key}: {e}")
                    continue
    except:
        # Fallback to numpy loading
        with safe_open(file_path, framework="numpy") as f:
            for key in f.keys():
                try:
                    tensor = f.get_tensor(key)
                    # Convert bfloat16 to float32 if needed
                    tensor = convert_bfloat16_to_float32(tensor)
                    tensors[key] = tensor
                except Exception as e:
                    # Skip problematic tensors
                    if 'bfloat16' not in str(e):
                        warnings.warn(f"Failed to load tensor {key}: {e}")
                    continue
    
    return tensors


def load_parameters(param_files: List[Path]) -> Dict[str, np.ndarray]:
    """
    Load parameters from multiple files.
    Supports safetensors, numpy, and pickle formats.
    """
    params = {}
    
    for file_path in param_files:
        file_path = Path(file_path)
        print(f"  Loading {file_path.name}...")
        
        try:
            if file_path.suffix == '.safetensors':
                loaded = load_safetensors(file_path)
                params.update(loaded)
                
            elif file_path.suffix == '.npz':
                loaded = np.load(file_path)
                for key in loaded.keys():
                    params[key] = convert_bfloat16_to_float32(loaded[key])
                
            elif file_path.suffix in ['.pkl', '.pickle']:
                import pickle
                with open(file_path, 'rb') as f:
                    loaded = pickle.load(f)
                    if isinstance(loaded, dict):
                        for key, value in loaded.items():
                            params[key] = convert_bfloat16_to_float32(value)
                    else:
                        raise ValueError(f"Expected dict in pickle file, got {type(loaded)}")
            
            elif file_path.suffix == '.bin':
                # PyTorch format - try to load with pickle
                import pickle
                with open(file_path, 'rb') as f:
                    loaded = pickle.load(f)
                    for key, value in loaded.items():
                        if hasattr(value, 'numpy'):
                            # Convert from torch tensor
                            value = value.numpy()
                        params[key] = convert_bfloat16_to_float32(value)
            
            elif file_path.suffix == '.json':
                # Skip config files
                continue
                
            else:
                warnings.warn(f"Unknown file format: {file_path}")
                
        except Exception as e:
            warnings.warn(f"Failed to load {file_path}: {e}")
            continue
    
    if not params:
        raise ValueError("No parameters loaded - check file formats")
    
    print(f"✅ Loaded {len(params)} parameter tensors")
    
    # Print parameter statistics
    total_params = sum(p.size for p in params.values())
    total_size_mb = sum(p.nbytes for p in params.values()) / (1024 * 1024)
    print(f"📊 Total parameters: {total_params:,}")
    print(f"💾 Total size: {total_size_mb:.1f} MB")
    
    return params


def load_model_config(config_path: Path) -> GPTOSSConfig:
    """Load and parse model configuration."""
    with open(config_path) as f:
        config_dict = json.load(f)
    
    # Print config info
    print(f"📋 Model configuration:")
    print(f"  • Architecture: {config_dict.get('model_type', 'gpt-oss')}")
    print(f"  • Hidden size: {config_dict.get('hidden_size', 'unknown')}")
    print(f"  • Layers: {config_dict.get('num_hidden_layers', 'unknown')}")
    print(f"  • Attention heads: {config_dict.get('num_attention_heads', 'unknown')}")
    
    return GPTOSSConfig.from_dict(config_dict)


def find_model_files(model_path: Path) -> List[Path]:
    """Find all model parameter files in a directory."""
    param_files = []
    
    # Check for sharded safetensors files (model-00001-of-00003.safetensors)
    sharded_files = sorted(model_path.glob("model-*-of-*.safetensors"))
    if sharded_files:
        print(f"📁 Found {len(sharded_files)} sharded model files")
        return sharded_files
    
    # Check for single model file
    single_file = model_path / "model.safetensors"
    if single_file.exists():
        print("📁 Found single model file")
        return [single_file]
    
    # Check for other formats
    for pattern in ["*.safetensors", "*.bin", "*.npz", "*.pkl"]:
        files = list(model_path.glob(pattern))
        if files:
            param_files.extend(files)
    
    # Filter out index files
    param_files = [f for f in param_files if 'index' not in f.name]
    
    if not param_files:
        raise FileNotFoundError(f"No model files found in {model_path}")
    
    return sorted(param_files)


def validate_parameters(params: Dict[str, np.ndarray], config: GPTOSSConfig) -> bool:
    """Validate that loaded parameters match the model configuration."""
    expected_layers = config.num_hidden_layers
    
    # Check for essential parameters
    essential_params = [
        'model.embed_tokens.weight',
        'model.norm.weight'
    ]
    
    for param in essential_params:
        if param not in params:
            warnings.warn(f"Missing essential parameter: {param}")
            return False
    
    # Check layer parameters
    for i in range(expected_layers):
        layer_params = [
            f'model.layers.{i}.input_layernorm.weight',
            f'model.layers.{i}.self_attn.q_proj.weight',
            f'model.layers.{i}.self_attn.k_proj.weight',
            f'model.layers.{i}.self_attn.v_proj.weight',
            f'model.layers.{i}.self_attn.o_proj.weight',
        ]
        
        for param in layer_params:
            if param not in params:
                warnings.warn(f"Missing layer parameter: {param}")
                return False
    
    # Validate shapes
    embed_weight = params['model.embed_tokens.weight']
    if embed_weight.shape[0] != config.vocab_size:
        warnings.warn(f"Vocab size mismatch: expected {config.vocab_size}, got {embed_weight.shape[0]}")
        return False
    
    print("✅ Parameter validation passed")
    return True


def load_model(
    model_path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    validate: bool = True
) -> Tuple[GPTOSSModel, GPTOSSConfig]:
    """
    Load GPT-OSS model from files.
    
    Args:
        model_path: Path to model directory or file
        config_path: Optional path to config.json (defaults to model_path/config.json)
        validate: Whether to validate parameters
    
    Returns:
        Tuple of (model, config)
    """
    model_path = Path(model_path)
    
    print(f"🚀 Loading GPT-OSS model from: {model_path}")
    
    # Load configuration
    if config_path:
        config_file = Path(config_path)
    else:
        config_file = model_path / "config.json"
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    config = load_model_config(config_file)
    
    # Find parameter files
    if model_path.is_file():
        param_files = [model_path]
    else:
        param_files = find_model_files(model_path)
    
    # Load parameters
    print(f"⏳ Loading parameters from {len(param_files)} file(s)...")
    params = load_parameters(param_files)
    
    # Validate if requested
    if validate:
        if not validate_parameters(params, config):
            warnings.warn("Parameter validation failed - model may not work correctly")
    
    # Create model
    print("🔨 Building model...")
    model = GPTOSSModel(params, config)
    
    print("✅ Model loaded successfully!")
    return model, config


def save_model(
    model: GPTOSSModel,
    config: GPTOSSConfig,
    save_path: Union[str, Path],
    format: str = "safetensors"
) -> None:
    """
    Save model to disk.
    
    Args:
        model: GPTOSSModel instance
        config: Model configuration
        save_path: Directory to save to
        format: Output format ('safetensors', 'npz', 'pickle')
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_file = save_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    
    # Collect parameters
    params = {}
    params['model.embed_tokens.weight'] = model.embed_tokens
    params['model.norm.weight'] = model.norm.weight
    
    for i, layer in enumerate(model.layers):
        prefix = f'model.layers.{i}'
        params[f'{prefix}.input_layernorm.weight'] = layer.input_layernorm.weight
        params[f'{prefix}.post_attention_layernorm.weight'] = layer.post_attention_layernorm.weight
        
        # Attention weights
        params[f'{prefix}.self_attn.q_proj.weight'] = layer.self_attn.q_proj
        params[f'{prefix}.self_attn.k_proj.weight'] = layer.self_attn.k_proj
        params[f'{prefix}.self_attn.v_proj.weight'] = layer.self_attn.v_proj
        params[f'{prefix}.self_attn.o_proj.weight'] = layer.self_attn.o_proj
        
        # MLP weights (varies by layer type)
        # ... (implementation depends on layer type)
    
    # Save parameters
    if format == "safetensors":
        try:
            import safetensors.numpy
            safetensors.numpy.save_file(params, save_path / "model.safetensors")
        except ImportError:
            warnings.warn("safetensors not available, falling back to npz")
            format = "npz"
    
    if format == "npz":
        np.savez_compressed(save_path / "model.npz", **params)
    elif format == "pickle":
        import pickle
        with open(save_path / "model.pkl", 'wb') as f:
            pickle.dump(params, f)
    
    print(f"✅ Model saved to {save_path}")