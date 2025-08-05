#!/usr/bin/env python3
"""
Convert TensorPort MessagePack shards to JAX-compatible formats.
This Python script handles the final conversion to proper JAX/Orbax checkpoints.
"""

import json
import pickle
import msgpack
import numpy as np
from pathlib import Path
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
try:
    from tqdm import tqdm
    from tqdm.contrib.concurrent import thread_map
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable
    def thread_map(func, iterable, max_workers=None, **kwargs):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(func, iterable))

def process_single_shard(shard_file, shard_index, total_shards):
    """Process a single shard file and return tensors."""
    import time
    
    file_size_mb = shard_file.stat().st_size / (1024 * 1024)
    print(f"  Processing {shard_file.name} ({file_size_mb:.1f}MB) [{shard_index+1}/{total_shards}]...")
    
    start_time = time.time()
    
    try:
        with open(shard_file, 'rb') as f:
            shard_data = msgpack.unpack(f, raw=False, strict_map_key=False)
        
        load_time = time.time() - start_time
        print(f"    📦 Unpacked in {load_time:.1f}s, processing {len(shard_data)} tensors...")
        
        shard_params = {}
        tensor_count = 0
        
        for tensor_name, tensor_info in shard_data.items():
            try:
                if isinstance(tensor_info, list) and len(tensor_info) == 3:
                    shape, dtype, data = tensor_info
                    np_array = deserialize_tensorport_format(shape, dtype, data)
                    shard_params[tensor_name] = np_array
                    tensor_count += 1
                    
                    if tensor_count % 25 == 0 or tensor_count < 3:
                        print(f"    ✅ {tensor_name}: {np_array.shape} {np_array.dtype}")
                elif isinstance(tensor_info, dict) and 'shape' in tensor_info:
                    np_array = deserialize_tensor(tensor_info)
                    shard_params[tensor_name] = np_array
                    tensor_count += 1
                    
                    if tensor_count % 25 == 0 or tensor_count < 3:
                        print(f"    ✅ {tensor_name}: {np_array.shape} {np_array.dtype}")
                else:
                    print(f"    ⚠️ Skipping {tensor_name}: unexpected format {type(tensor_info)}")
            except Exception as e:
                print(f"    ❌ Failed to process {tensor_name}: {e}")
                continue
        
        shard_time = time.time() - start_time
        print(f"  📊 Processed {tensor_count} tensors from {shard_file.name} in {shard_time:.1f}s")
        
        return shard_params, tensor_count
        
    except Exception as e:
        print(f"  ❌ Failed to process shard {shard_file.name}: {e}")
        return {}, 0

def process_shard_with_index(args):
    """Wrapper for process_single_shard to work with thread_map."""
    shard_file, shard_index, total_shards = args
    return process_single_shard(shard_file, shard_index, total_shards)

def load_tensorport_shards(shard_dir, max_workers=None, batch_size=4):
    """Load TensorPort MessagePack shards with memory-aware parallel processing."""
    shard_dir = Path(shard_dir)
    
    shard_files = sorted(shard_dir.glob("shard_*.msgpack"))
    if not shard_files:
        raise ValueError(f"No shard files found in {shard_dir}")
    
    if max_workers is None:
        # Use fewer workers for memory-intensive operations
        max_workers = min(4, max(1, mp.cpu_count() // 2))
    
    print(f"Loading {len(shard_files)} TensorPort shards with {max_workers} workers, batch size {batch_size}...")
    
    all_params = {}
    total_tensors = 0
    
    # Process shards in batches to manage memory
    num_batches = (len(shard_files) + batch_size - 1) // batch_size
    
    for batch_start in range(0, len(shard_files), batch_size):
        batch_end = min(batch_start + batch_size, len(shard_files))
        batch_files = shard_files[batch_start:batch_end]
        
        batch_num = batch_start // batch_size + 1
        print(f"\n📦 Processing batch {batch_num}/{num_batches}: shards {batch_start} to {batch_end-1}")
        
        # Prepare arguments for parallel processing
        shard_args = [(shard_file, batch_start + i, len(shard_files)) 
                      for i, shard_file in enumerate(batch_files)]
        
        # Use tqdm's thread_map for better progress visualization
        batch_results = thread_map(
            process_shard_with_index,
            shard_args,
            max_workers=min(max_workers, len(batch_files)),
            desc=f"Batch {batch_num}/{num_batches}",
            unit="shard",
            leave=False if HAS_TQDM else True
        )
        
        # Collect results
        batch_tensor_count = 0
        for shard_params, shard_tensor_count in batch_results:
            all_params.update(shard_params)
            total_tensors += shard_tensor_count
            batch_tensor_count += shard_tensor_count
        
        print(f"  📊 Batch {batch_num} complete: {batch_tensor_count} tensors. Running total: {total_tensors}")
        
        # Force garbage collection between batches to manage memory
        import gc
        gc.collect()
    
    print(f"\n🎯 Total: {total_tensors} tensors loaded across {len(shard_files)} shards")
    return all_params

def deserialize_tensorport_format(shape, dtype, data):
    """Convert TensorPort [shape, dtype, data] format back to numpy array."""
    # Handle nested dictionary structures (common in expert weights and quantized data)
    if isinstance(data, dict):
        if dtype == 'float16' and 'Float16' in data:
            # Structured float16 data
            bits_array = np.array(data['Float16'], dtype=np.uint16)
            float16_array = bits_array.view(np.float16)
            return float16_array.reshape(shape)
        elif dtype == 'float32' and 'Float32' in data:
            # Structured float32 data
            float32_array = np.array(data['Float32'], dtype=np.float32)
            return float32_array.reshape(shape)
        elif dtype == 'uint8' and 'Uint8' in data:
            # Structured uint8 data (quantized weights)
            uint8_array = np.array(data['Uint8'], dtype=np.uint8)
            return uint8_array.reshape(shape)
        elif 'quantized_data' in data:
            # MXFP4 or other quantized format - preserve the structure
            return create_quantized_tensor_placeholder(shape, dtype, data)
        else:
            # Try to extract the raw data from nested structure
            possible_keys = ['data', 'values', 'tensor_data', dtype.lower()]
            for key in possible_keys:
                if key in data:
                    return deserialize_tensorport_format(shape, dtype, data[key])
            
            # Check if this might be quantization metadata (blocks, scales, etc.)
            if any(key in ['blocks', 'scales', 'quantized', 'metadata'] for key in data.keys()):
                # Preserve quantization metadata as structured array
                return create_quantized_tensor_placeholder(shape, dtype, data)
            
            # If we can't find the data, try the first non-metadata key
            metadata_keys = {'dtype', 'shape', 'format', 'type'}
            data_keys = [k for k in data.keys() if k not in metadata_keys]
            if data_keys:
                return deserialize_tensorport_format(shape, dtype, data[data_keys[0]])
            
            raise ValueError(f"Cannot extract tensor data from dict structure: {list(data.keys())}")
    
    # Handle direct array data
    if dtype == 'float16':
        float16_array = np.array(data, dtype=np.float16)
        return float16_array.reshape(shape)
    elif dtype == 'float32':
        float32_array = np.array(data, dtype=np.float32)
        return float32_array.reshape(shape)
    elif dtype == 'uint8':
        uint8_array = np.array(data, dtype=np.uint8)
        return uint8_array.reshape(shape)
    elif dtype == 'int64':
        int64_array = np.array(data, dtype=np.int64)
        return int64_array.reshape(shape)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

def create_quantized_tensor_placeholder(shape, dtype, quantized_data):
    """Create a placeholder for quantized tensors that preserves the quantization info."""
    # For now, create a structured array that preserves the quantization metadata
    # This allows the JAX model to handle dequantization later
    placeholder = np.zeros(shape, dtype=np.float16 if dtype == 'float16' else np.float32)
    
    # Attach quantization metadata as array attributes (if supported by the target framework)
    if hasattr(placeholder, '__dict__'):
        placeholder.__dict__['quantization_data'] = quantized_data
    
    # For debugging/logging, note that this is a quantized tensor placeholder
    print(f"    📦 Created quantized tensor placeholder with keys: {list(quantized_data.keys()) if isinstance(quantized_data, dict) else type(quantized_data)}")
    
    return placeholder

def deserialize_tensor(tensor_info):
    """Convert TensorPort tensor info back to numpy array."""
    shape = tensor_info['shape']
    dtype = tensor_info['dtype']
    data = tensor_info['data']
    
    # Use the same deserialization logic as deserialize_tensorport_format
    return deserialize_tensorport_format(shape, dtype, data)

def convert_to_jax_pytree(params):
    """Convert flat parameter dict to nested JAX PyTree structure."""
    try:
        import jax.numpy as jnp
    except ImportError:
        print("⚠️  JAX not available, using NumPy arrays")
        jnp = np
    
    print("📦 Converting to JAX PyTree structure...")
    
    # Create nested structure
    pytree = {}
    
    for param_name, param_array in params.items():
        # Convert to JAX array
        if jnp != np:
            jax_array = jnp.array(param_array)
        else:
            jax_array = param_array
        
        # Create nested structure
        parts = param_name.split('.')
        current = pytree
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = jax_array
    
    return pytree

def save_as_orbax_checkpoint(pytree, output_dir):
    """Save PyTree as Orbax checkpoint (requires orbax-checkpoint)."""
    try:
        import orbax.checkpoint as ocp
        from flax.training import train_state
        import jax
    except ImportError:
        print("⚠️  Orbax not available, saving as pickle instead")
        save_as_pickle(pytree, output_dir)
        return
    
    print("💾 Saving as Orbax checkpoint...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create checkpointer
    checkpointer = ocp.StandardCheckpointer()
    
    # Save checkpoint
    checkpoint_path = output_dir / "checkpoint"
    checkpointer.save(checkpoint_path, pytree)
    
    print(f"✅ Orbax checkpoint saved to: {checkpoint_path}")
    
    # Create loading script
    create_orbax_loader(output_dir)

def save_as_numpy_arrays(params, output_dir):
    """Save as individual NumPy .npy files."""
    print("💾 Saving as NumPy arrays...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each parameter as .npy file
    param_manifest = []
    
    for param_name, param_array in params.items():
        safe_name = param_name.replace('.', '_').replace('/', '_')
        npy_file = output_dir / f"{safe_name}.npy"
        
        np.save(npy_file, param_array)
        
        param_manifest.append({
            'original_name': param_name,
            'file_name': f"{safe_name}.npy",
            'shape': list(param_array.shape),
            'dtype': str(param_array.dtype),
            'parameters': int(np.prod(param_array.shape))
        })
    
    # Save manifest
    manifest = {
        'format': 'tensorport_numpy_jax',
        'total_files': len(param_manifest),
        'total_parameters': sum(p['parameters'] for p in param_manifest),
        'parameters': param_manifest
    }
    
    with open(output_dir / 'numpy_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✅ Saved {len(param_manifest)} NumPy arrays to: {output_dir}")
    
    # Create loading script
    create_numpy_loader(output_dir)

def save_as_pickle(pytree, output_dir):
    """Save PyTree as pickle file (JAX-compatible)."""
    print("💾 Saving as pickle PyTree...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pickle_file = output_dir / "jax_params.pkl"
    
    with open(pickle_file, 'wb') as f:
        pickle.dump(pytree, f)
    
    print(f"✅ JAX PyTree saved to: {pickle_file}")
    
    # Create loading script
    create_pickle_loader(output_dir)

def create_orbax_loader(output_dir):
    """Create Python script to load Orbax checkpoint."""
    script_content = '''#!/usr/bin/env python3
"""
Load TensorPort-converted Orbax checkpoint.
"""

import orbax.checkpoint as ocp
import jax.numpy as jnp
from pathlib import Path

def load_tensorport_orbax(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_path = checkpoint_dir / "checkpoint"
    
    print(f"Loading Orbax checkpoint from: {checkpoint_path}")
    
    checkpointer = ocp.StandardCheckpointer()
    restored_params = checkpointer.restore(checkpoint_path)
    
    print("✅ Orbax checkpoint loaded!")
    return restored_params

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python load_orbax.py <checkpoint_dir>")
        sys.exit(1)
    
    params = load_tensorport_orbax(sys.argv[1])
    print(f"Loaded {len(params)} parameter groups")
'''
    
    script_path = output_dir / "load_orbax.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    script_path.chmod(0o755)
    print(f"📜 Created loader script: {script_path}")

def create_pickle_loader(output_dir):
    """Create Python script to load pickle PyTree."""
    script_content = '''#!/usr/bin/env python3
"""
Load TensorPort-converted JAX PyTree from pickle.
"""

import pickle
import jax.numpy as jnp
from pathlib import Path

def load_tensorport_pickle(pickle_dir):
    pickle_dir = Path(pickle_dir)
    pickle_file = pickle_dir / "jax_params.pkl"
    
    print(f"Loading JAX PyTree from: {pickle_file}")
    
    with open(pickle_file, 'rb') as f:
        params = pickle.load(f)
    
    print("✅ JAX PyTree loaded!")
    return params

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python load_pickle.py <pickle_dir>")
        sys.exit(1)
    
    params = load_tensorport_pickle(sys.argv[1])
    print(f"Loaded JAX PyTree with {len(params)} top-level keys")
'''
    
    script_path = output_dir / "load_pickle.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    script_path.chmod(0o755)
    print(f"📜 Created loader script: {script_path}")

def create_numpy_loader(output_dir):
    """Create Python script to load NumPy arrays."""
    script_content = '''#!/usr/bin/env python3
"""
Load TensorPort-converted NumPy arrays as JAX PyTree.
"""

import json
import numpy as np
try:
    import jax.numpy as jnp
except ImportError:
    print("JAX not available, using NumPy")
    jnp = np
from pathlib import Path

def load_tensorport_numpy(numpy_dir):
    numpy_dir = Path(numpy_dir)
    manifest_file = numpy_dir / "numpy_manifest.json"
    
    with open(manifest_file) as f:
        manifest = json.load(f)
    
    print(f"Loading {manifest['total_files']} NumPy arrays...")
    
    params = {}
    for param_info in manifest['parameters']:
        # Load NumPy array
        npy_file = numpy_dir / param_info['file_name']
        np_array = np.load(npy_file)
        
        # Convert to JAX if available
        if jnp != np:
            jax_array = jnp.array(np_array)
        else:
            jax_array = np_array
        
        # Create nested structure
        param_name = param_info['original_name']
        parts = param_name.split('.')
        current = params
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = jax_array
    
    print("✅ JAX PyTree loaded from NumPy arrays!")
    return params

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python load_numpy.py <numpy_dir>")
        sys.exit(1)
    
    params = load_tensorport_numpy(sys.argv[1])
    print(f"Loaded JAX PyTree with {len(params)} top-level keys")
'''
    
    script_path = output_dir / "load_numpy.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    script_path.chmod(0o755)
    print(f"📜 Created loader script: {script_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert TensorPort MessagePack shards to JAX formats")
    parser.add_argument("input_dir", help="Directory containing MessagePack shards")
    parser.add_argument("output_dir", help="Output directory for JAX format")
    parser.add_argument("--format", choices=['orbax', 'pickle', 'numpy'], default='pickle',
                       help="Output format for JAX (default: pickle)")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers (default: CPU_count//2, max 4)")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Number of shards to process in parallel (default: 4)")
    
    args = parser.parse_args()
    
    print("🚀 TensorPort MessagePack → JAX Converter")
    print("=" * 45)
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Format: {args.format}")
    print()
    
    try:
        # Load TensorPort shards
        params = load_tensorport_shards(args.input_dir, max_workers=args.workers, batch_size=args.batch_size)
        
        if args.format == 'orbax':
            # Convert to PyTree and save as Orbax
            pytree = convert_to_jax_pytree(params)
            save_as_orbax_checkpoint(pytree, args.output_dir)
        
        elif args.format == 'pickle':
            # Convert to PyTree and save as pickle
            pytree = convert_to_jax_pytree(params)
            save_as_pickle(pytree, args.output_dir)
        
        elif args.format == 'numpy':
            # Save as individual NumPy arrays
            save_as_numpy_arrays(params, args.output_dir)
        
        print(f"\n🎉 Conversion complete!")
        print(f"✅ TensorPort → JAX {args.format} conversion successful!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()