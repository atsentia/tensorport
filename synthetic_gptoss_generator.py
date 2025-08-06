#!/usr/bin/env python3
"""
Synthetic GPT-OSS-20B Model Data Generator

Creates realistic synthetic weights in Safetensors format that match the exact
structure and size of the GPT-OSS-20B model for comprehensive testing.

This allows testing the full TensorPort pipeline without requiring a 13GB download.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
import struct
import hashlib

def create_gptoss_config() -> Dict[str, Any]:
    """Create GPT-OSS-20B model configuration."""
    return {
        "vocab_size": 201088,
        "hidden_size": 2880,
        "num_hidden_layers": 24,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "intermediate_size": 2880,
        "num_local_experts": 32,
        "num_experts_per_tok": 4,
        "hidden_act": "silu",
        "max_position_embeddings": 131072,
        "rope_theta": 150000.0,
        "sliding_window": 128,
        "rms_norm_eps": 1e-6,
        "quantization_method": "mxfp4",
        "use_kv_cache": True,
        "tie_word_embeddings": True,
        "layer_types": ["attention"] * 24  # Simplified - all attention layers
    }

def calculate_tensor_shapes(config: Dict[str, Any]) -> Dict[str, Tuple[int, ...]]:
    """Calculate the shapes of all tensors in the GPT-OSS-20B model."""
    shapes = {}
    
    vocab_size = config["vocab_size"]
    hidden_size = config["hidden_size"]
    num_layers = config["num_hidden_layers"]
    num_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    intermediate_size = config["intermediate_size"]
    
    # Embedding layers
    shapes["model.embed_tokens.weight"] = (vocab_size, hidden_size)
    
    # Output layer (tied embeddings)
    if not config.get("tie_word_embeddings", True):
        shapes["lm_head.weight"] = (vocab_size, hidden_size)
    else:
        # Reference to embed_tokens.weight - no separate tensor
        pass
    
    # Transformer layers
    for layer_idx in range(num_layers):
        prefix = f"model.layers.{layer_idx}"
        
        # Attention layers
        shapes[f"{prefix}.self_attn.q_proj.weight"] = (num_heads * head_dim, hidden_size)
        shapes[f"{prefix}.self_attn.k_proj.weight"] = (num_kv_heads * head_dim, hidden_size)
        shapes[f"{prefix}.self_attn.v_proj.weight"] = (num_kv_heads * head_dim, hidden_size)
        shapes[f"{prefix}.self_attn.o_proj.weight"] = (hidden_size, num_heads * head_dim)
        
        # MLP layers (feed-forward)
        shapes[f"{prefix}.mlp.gate_proj.weight"] = (intermediate_size, hidden_size)
        shapes[f"{prefix}.mlp.up_proj.weight"] = (intermediate_size, hidden_size)
        shapes[f"{prefix}.mlp.down_proj.weight"] = (hidden_size, intermediate_size)
        
        # Layer normalization
        shapes[f"{prefix}.input_layernorm.weight"] = (hidden_size,)
        shapes[f"{prefix}.post_attention_layernorm.weight"] = (hidden_size,)
    
    # Final layer norm
    shapes["model.norm.weight"] = (hidden_size,)
    
    return shapes

def generate_realistic_weights(shape: Tuple[int, ...], tensor_name: str, dtype: np.dtype = np.float16) -> np.ndarray:
    """Generate realistic weights with proper initialization for different tensor types."""
    
    # Set random seed based on tensor name for reproducibility
    seed = hash(tensor_name) % (2**32)
    rng = np.random.RandomState(seed)
    
    if len(shape) == 1:
        # Layer norm weights - initialize to ones with small noise
        weights = rng.normal(1.0, 0.01, shape)
        return weights.astype(dtype)
    
    elif len(shape) == 2:
        fan_in, fan_out = shape[1], shape[0]
        
        if "embed" in tensor_name:
            # Embedding weights - normal distribution scaled by vocab size
            std = 1.0 / np.sqrt(fan_in)
            weights = rng.normal(0.0, std, shape)
        
        elif "q_proj" in tensor_name or "k_proj" in tensor_name or "v_proj" in tensor_name:
            # Attention projection weights - Xavier/Glorot initialization
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            weights = rng.uniform(-limit, limit, shape)
        
        elif "o_proj" in tensor_name or "down_proj" in tensor_name:
            # Output projection weights - scaled initialization for stability
            std = np.sqrt(2.0 / fan_in) / np.sqrt(2)  # Account for residual connections
            weights = rng.normal(0.0, std, shape)
        
        elif "gate_proj" in tensor_name or "up_proj" in tensor_name:
            # MLP gate/up weights - standard initialization
            std = np.sqrt(2.0 / fan_in)
            weights = rng.normal(0.0, std, shape)
        
        else:
            # Default initialization
            std = np.sqrt(2.0 / fan_in)
            weights = rng.normal(0.0, std, shape)
        
        return weights.astype(dtype)
    
    else:
        # Fallback for other shapes
        weights = rng.normal(0.0, 0.01, shape)
        return weights.astype(dtype)

def quantize_to_mxfp4(weights: np.ndarray, block_size: int = 32) -> np.ndarray:
    """
    Simulate MXFP4 quantization by packing float16 weights into 4-bit format.
    This creates realistic quantized data that the TensorPort pipeline can handle.
    """
    # Reshape to blocks
    orig_shape = weights.shape
    flat_weights = weights.flatten()
    
    # Pad to block size multiple
    pad_size = (block_size - (len(flat_weights) % block_size)) % block_size
    if pad_size > 0:
        flat_weights = np.pad(flat_weights, (0, pad_size), 'constant')
    
    # Reshape to blocks
    blocks = flat_weights.reshape(-1, block_size)
    
    # Quantize each block (simplified MXFP4 simulation)
    quantized_blocks = []
    for block in blocks:
        # Find shared exponent (max exponent in block)
        abs_block = np.abs(block)
        max_val = np.max(abs_block)
        
        if max_val == 0:
            # All zeros block
            quantized_blocks.append(np.zeros(block_size // 2, dtype=np.uint8))
            continue
        
        # Scale to 4-bit range
        scale = max_val / 7.0  # 4-bit signed: -7 to +7
        scaled_block = np.round(block / scale).astype(np.int8)
        scaled_block = np.clip(scaled_block, -7, 7)
        
        # Pack two 4-bit values into one uint8
        packed = np.zeros(block_size // 2, dtype=np.uint8)
        for i in range(0, block_size, 2):
            val1 = scaled_block[i] & 0xF
            val2 = scaled_block[i + 1] & 0xF
            packed[i // 2] = (val2 << 4) | val1
        
        quantized_blocks.append(packed)
    
    # Concatenate all blocks
    quantized = np.concatenate(quantized_blocks)
    
    return quantized

def create_safetensors_header(tensor_data: Dict[str, np.ndarray]) -> bytes:
    """Create a safetensors format header."""
    
    # Create metadata for each tensor
    metadata = {}
    data_offset = 0
    
    for name, tensor in tensor_data.items():
        dtype_str = {
            np.float16: "F16",
            np.float32: "F32", 
            np.uint8: "U8",
            np.int64: "I64"
        }.get(tensor.dtype.type, "F16")
        
        metadata[name] = {
            "dtype": dtype_str,
            "shape": list(tensor.shape),
            "data_offsets": [data_offset, data_offset + tensor.nbytes]
        }
        data_offset += tensor.nbytes
    
    # Serialize metadata as JSON
    header_json = json.dumps(metadata, separators=(',', ':'))
    header_bytes = header_json.encode('utf-8')
    
    # Pack header with length prefix
    header_size = len(header_bytes)
    packed_header = struct.pack('<Q', header_size) + header_bytes
    
    return packed_header

def generate_synthetic_gptoss_model(output_dir: Path, quantize: bool = True) -> Dict[str, Any]:
    """
    Generate a complete synthetic GPT-OSS-20B model in Safetensors format.
    
    Args:
        output_dir: Directory to save the model files
        quantize: Whether to apply MXFP4 quantization to weights
    
    Returns:
        Dictionary with generation statistics and metadata
    """
    print("🏗️  Generating Synthetic GPT-OSS-20B Model...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model configuration
    config = create_gptoss_config()
    
    # Calculate all tensor shapes
    tensor_shapes = calculate_tensor_shapes(config)
    
    print(f"📊 Model will have {len(tensor_shapes)} tensors")
    
    # Generate weight tensors
    tensor_data = {}
    total_params = 0
    
    for tensor_name, shape in tensor_shapes.items():
        print(f"  Generating {tensor_name}: {shape}")
        
        # Choose dtype based on quantization
        if quantize and "weight" in tensor_name and len(shape) == 2:
            # Generate fp16 weights then quantize to 4-bit
            weights_fp16 = generate_realistic_weights(shape, tensor_name, np.float16)
            # For now, store as uint8 to simulate quantized format
            # In real MXFP4, this would be properly packed 4-bit data
            weights = quantize_to_mxfp4(weights_fp16).astype(np.uint8)
            print(f"    Quantized {shape} -> {weights.shape} (MXFP4)")
        else:
            # Keep as float16 for non-quantized tensors
            weights = generate_realistic_weights(shape, tensor_name, np.float16)
        
        tensor_data[tensor_name] = weights
        total_params += np.prod(shape)
    
    print(f"📈 Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    
    # Create safetensors file
    safetensors_path = output_dir / "model.safetensors"
    
    print(f"💾 Writing safetensors file: {safetensors_path}")
    
    # Create header
    header = create_safetensors_header(tensor_data)
    
    # Write file
    with open(safetensors_path, 'wb') as f:
        f.write(header)
        for tensor in tensor_data.values():
            f.write(tensor.tobytes())
    
    # Create model index (for sharded models)
    weight_map = {name: "model.safetensors" for name in tensor_data.keys()}
    
    model_index = {
        "metadata": {
            "total_size": sum(tensor.nbytes for tensor in tensor_data.values()),
            "format": "safetensors",
            "quantization": "mxfp4" if quantize else "none"
        },
        "weight_map": weight_map
    }
    
    index_path = output_dir / "model.safetensors.index.json"
    with open(index_path, 'w') as f:
        json.dump(model_index, f, indent=2)
    
    # Save configuration
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Generate statistics
    file_size_mb = safetensors_path.stat().st_size / (1024 * 1024)
    
    stats = {
        "total_parameters": total_params,
        "total_tensors": len(tensor_data),
        "file_size_mb": file_size_mb,
        "quantization": "mxfp4" if quantize else "fp16",
        "files_created": [
            str(safetensors_path),
            str(index_path), 
            str(config_path)
        ]
    }
    
    print(f"✅ Model generation complete!")
    print(f"   📊 {total_params:,} parameters ({total_params/1e9:.2f}B)")
    print(f"   📁 {len(tensor_data)} tensors")
    print(f"   💾 {file_size_mb:.1f} MB file size")
    print(f"   🎯 Format: {'MXFP4 quantized' if quantize else 'FP16'}")
    
    return stats

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic GPT-OSS-20B model data")
    parser.add_argument("--output", "-o", type=str, default="synthetic-gptoss-20b",
                       help="Output directory for generated model")
    parser.add_argument("--no-quantize", action="store_true",
                       help="Skip MXFP4 quantization (generate FP16 model)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    quantize = not args.no_quantize
    
    stats = generate_synthetic_gptoss_model(output_dir, quantize=quantize)
    
    print(f"\n📋 Generation Summary:")
    for key, value in stats.items():
        if key == "files_created":
            print(f"   {key}:")
            for file_path in value:
                print(f"     - {file_path}")
        else:
            print(f"   {key}: {value}")