#!/usr/bin/env python3
"""
Test script to verify JAX can load the NumPy arrays created by tensorport.
"""

import json
import numpy as np
from pathlib import Path
import sys

def test_numpy_loading(model_dir):
    """Test loading NumPy arrays (without JAX for systems that don't have it)."""
    model_dir = Path(model_dir)
    
    # Load manifest
    with open(model_dir / 'manifest.json') as f:
        manifest = json.load(f)
    
    print(f"📊 Model Statistics:")
    print(f"   Total shards: {manifest['total_shards']}")
    print(f"   Total parameters: {manifest['total_parameters']:,}")
    print(f"   Total tensors: {len(manifest['tensors'])}")
    
    # Test loading several types of tensors
    test_cases = [
        # Regular weights
        ("regular", lambda t: "weight" in t["name"] and "blocks" not in t["name"]),
        # Quantized blocks
        ("quantized_blocks", lambda t: "_blocks" in t["name"]),
        # Scales
        ("scales", lambda t: "_scales" in t["name"]),
        # Biases
        ("bias", lambda t: "bias" in t["name"]),
    ]
    
    for case_name, filter_fn in test_cases:
        matching = [t for t in manifest['tensors'] if filter_fn(t)]
        if matching:
            tensor = matching[0]
            array_path = model_dir / tensor['file']
            array = np.load(str(array_path))
            
            print(f"\n✅ {case_name.upper()} tensor loaded successfully:")
            print(f"   Name: {tensor['name']}")
            print(f"   Shape: {array.shape}")
            print(f"   Dtype: {array.dtype}")
            print(f"   Size: {tensor['size_mb']:.2f}MB")
            
            # Verify shape matches
            assert list(array.shape) == tensor['shape'], f"Shape mismatch!"
            
            # Check data validity
            if array.dtype in [np.float16, np.float32]:
                print(f"   Range: [{np.min(array):.4f}, {np.max(array):.4f}]")
                print(f"   Mean: {np.mean(array):.4f}, Std: {np.std(array):.4f}")
                
                # Check for NaN or Inf
                assert not np.any(np.isnan(array)), "Found NaN values!"
                assert not np.any(np.isinf(array)), "Found Inf values!"
    
    return manifest

def test_jax_loading(model_dir):
    """Test loading with JAX if available."""
    try:
        import jax
        import jax.numpy as jnp
        print("\n🎯 Testing JAX loading...")
        
        model_dir = Path(model_dir)
        manifest = json.load(open(model_dir / 'manifest.json'))
        
        # Load a few tensors with JAX
        test_tensor = manifest['tensors'][0]
        array_path = model_dir / test_tensor['file']
        
        # Test JAX loading
        jax_array = jnp.load(str(array_path))
        
        print(f"✅ JAX loaded successfully!")
        print(f"   Array type: {type(jax_array)}")
        print(f"   Device: {jax_array.device()}")
        print(f"   Shape: {jax_array.shape}")
        print(f"   Dtype: {jax_array.dtype}")
        
        # Test a simple JAX operation
        result = jnp.sum(jax_array)
        print(f"   Sum: {result:.4f}")
        
        # Test building nested structure
        params = {}
        for i, tensor_info in enumerate(manifest['tensors'][:5]):
            tensor_path = model_dir / tensor_info['file']
            array = jnp.load(str(tensor_path))
            
            # Build nested dict structure
            parts = tensor_info['name'].split('.')
            current = params
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = array
            
            print(f"   Loaded {i+1}/5: {tensor_info['name']}")
        
        # Verify we can access nested structure
        if 'model' in params and 'layers' in params['model']:
            print(f"\n✅ Nested structure works!")
            print(f"   Found model.layers with {len(params['model']['layers'])} layer indices")
        
        return True
        
    except ImportError:
        print("\n⚠️  JAX not installed - skipping JAX-specific tests")
        print("   Install with: pip install jax jaxlib")
        return False

def test_mxfp4_handling(model_dir):
    """Test MXFP4 quantized tensor handling."""
    model_dir = Path(model_dir)
    manifest = json.load(open(model_dir / 'manifest.json'))
    
    print("\n🔍 Testing MXFP4 Quantized Tensors:")
    
    # Find quantized tensors
    blocks_tensors = [t for t in manifest['tensors'] if '_blocks' in t['name']]
    scales_tensors = [t for t in manifest['tensors'] if '_scales' in t['name']]
    
    if blocks_tensors and scales_tensors:
        # Load a blocks tensor
        blocks_tensor = blocks_tensors[0]
        blocks_path = model_dir / blocks_tensor['file']
        blocks = np.load(str(blocks_path))
        
        # Find corresponding scales
        base_name = blocks_tensor['name'].replace('_blocks', '')
        scales_tensor = next((t for t in scales_tensors if base_name in t['name']), None)
        
        if scales_tensor:
            scales_path = model_dir / scales_tensor['file']
            scales = np.load(str(scales_path))
            
            print(f"✅ MXFP4 Quantized weight pair found:")
            print(f"   Blocks: {blocks_tensor['name']}")
            print(f"      Shape: {blocks.shape}, Dtype: {blocks.dtype}")
            print(f"   Scales: {scales_tensor['name']}")
            print(f"      Shape: {scales.shape}, Dtype: {scales.dtype}")
            
            # Verify uint8 for blocks
            assert blocks.dtype == np.uint8, f"Blocks should be uint8, got {blocks.dtype}"
            
            # Calculate actual parameters
            block_params = np.prod(blocks.shape) * 2  # Each uint8 packs 2 4-bit values
            print(f"   Actual parameters in blocks: {block_params:,}")
            
            return True
    
    print("   No MXFP4 quantized tensors found")
    return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_jax_loading.py <model_dir>")
        print("Example: python test_jax_loading.py /tmp/gpt-oss-20b-numpy-v2")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    
    print(f"🧪 Testing NumPy/JAX loading for: {model_dir}\n")
    
    # Test basic NumPy loading
    manifest = test_numpy_loading(model_dir)
    
    # Test MXFP4 handling
    test_mxfp4_handling(model_dir)
    
    # Test JAX loading if available
    jax_success = test_jax_loading(model_dir)
    
    print("\n" + "="*50)
    print("✅ All tests passed!")
    
    # Print loading instructions
    print("\n📝 To use in your code:")
    print("```python")
    print("import jax.numpy as jnp")
    print("from pathlib import Path")
    print("import json")
    print("")
    print(f"model_dir = Path('{model_dir}')")
    print("manifest = json.load(open(model_dir / 'manifest.json'))")
    print("")
    print("# Load all tensors")
    print("params = {}")
    print("for tensor_info in manifest['tensors']:")
    print("    array = jnp.load(str(model_dir / tensor_info['file']))")
    print("    # Build nested structure from tensor name")
    print("    # ... (see load_numpy_jax.py for full code)")
    print("```")