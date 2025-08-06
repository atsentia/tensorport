#!/usr/bin/env python3
"""
JAX model loader for NumPy array format.
Loads .npy files directly into JAX arrays.
"""

import json
import jax
import jax.numpy as jnp
from pathlib import Path

def load_numpy_jax_model(model_dir):
    """Load JAX model from NumPy arrays."""
    model_dir = Path(model_dir)
    
    # Load manifest
    with open(model_dir / 'manifest.json') as f:
        manifest = json.load(f)
    
    print(f"Loading JAX model with {manifest['total_shards']} shards...")
    print(f"Total parameters: {manifest['total_parameters']:,}")
    
    # Load all tensors
    params = {}
    for tensor_info in manifest['tensors']:
        tensor_path = model_dir / tensor_info['file']
        print(f"  Loading {tensor_info['name']} from {tensor_info['file']}...")
        
        # Load numpy array directly into JAX
        array = jnp.load(str(tensor_path))
        
        # Build nested dict structure
        parts = tensor_info['name'].split('.')
        current = params
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = array
    
    print("✅ Model loaded successfully!")
    return params, manifest['config']

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python load_numpy_jax.py <model_dir>")
        sys.exit(1)
    
    params, config = load_numpy_jax_model(sys.argv[1])
    print(f"\nModel type: {config.get('model_type')}")
    print(f"Ready for JAX inference!")
