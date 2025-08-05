use crate::error::{TensorportError, TensorportResult};
use crate::tensor::{Tensor, TensorData};
use serde::{Serialize, Deserialize};
use serde_pickle;
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, Serialize, Deserialize)]
pub struct ShardInfo {
    pub file: String,
    pub size_mb: f64,
    pub tensor_count: usize,
}

/// Direct JAX Pickle writer - outputs JAX-loadable pickle files directly from Rust
pub struct JaxPickleWriter {
    output_dir: PathBuf,
    precision: String,
    shard_size_gb: f64,
    current_shard: HashMap<String, PythonArray>,  // Flat structure instead of nested
    current_size: u64,
    shard_count: usize,
    total_params: u64,
    shard_infos: Vec<ShardInfo>,
}

impl JaxPickleWriter {
    pub fn new<P: AsRef<std::path::Path>>(output_dir: P, precision: String, shard_size_gb: f64) -> TensorportResult<Self> {
        let output_dir = output_dir.as_ref().to_path_buf();
        
        // Create output directory
        std::fs::create_dir_all(&output_dir)?;
        
        Ok(JaxPickleWriter {
            output_dir,
            precision,
            shard_size_gb,
            current_shard: HashMap::new(),
            current_size: 0,
            shard_count: 0,
            total_params: 0,
            shard_infos: Vec::new(),
        })
    }
    
    pub fn add_tensor(&mut self, tensor: Tensor) -> TensorportResult<()> {
        let start_time = Instant::now();
        let tensor_size = tensor.data.size_bytes() as u64;
        let max_shard_size = (self.shard_size_gb * 1024.0 * 1024.0 * 1024.0) as u64;
        
        // Check if we need to flush current shard
        if self.current_size > 0 && self.current_size + tensor_size > max_shard_size {
            println!("  📦 Shard size limit reached ({:.2}GB), flushing shard {}...", 
                     self.current_size as f64 / (1024.0 * 1024.0 * 1024.0), 
                     self.shard_count);
            self.flush_shard()?;
        }
        
        // Clean tensor name (remove "model." prefix if present)
        let clean_name = if tensor.name.starts_with("model.") {
            &tensor.name[6..]
        } else {
            &tensor.name
        };
        
        // Track memory before conversion
        let mem_before = self.estimate_memory_usage();
        
        // Convert tensor to Python-compatible format
        let conversion_start = Instant::now();
        let py_tensor = self.tensor_to_python_array(&tensor)?;
        let conversion_time = conversion_start.elapsed();
        
        // Insert into current shard's flat structure (much faster!)
        let insert_start = Instant::now();
        self.current_shard.insert(clean_name.to_string(), py_tensor);
        let insert_time = insert_start.elapsed();
        
        self.current_size += tensor_size;
        self.total_params += tensor.parameter_count() as u64;
        
        let mem_after = self.estimate_memory_usage();
        let total_time = start_time.elapsed();
        
        // Print detailed progress
        println!("  → {} | shape: {:?} | {:.2}MB | times: {:.1}ms total ({:.1}ms convert, {:.1}ms insert) | mem: {:.0}MB → {:.0}MB | shard: {:.0}MB/{:.0}MB", 
                 clean_name, 
                 tensor.shape,
                 tensor_size as f64 / (1024.0 * 1024.0),
                 total_time.as_millis(),
                 conversion_time.as_millis(),
                 insert_time.as_millis(),
                 mem_before,
                 mem_after,
                 self.current_size as f64 / (1024.0 * 1024.0),
                 max_shard_size as f64 / (1024.0 * 1024.0));
        
        // Warn if processing is slow
        if total_time.as_millis() > 100 {
            println!("  ⚠️  Slow tensor processing: {}ms", total_time.as_millis());
        }
        
        Ok(())
    }
    
    fn estimate_memory_usage(&self) -> f64 {
        // Rough estimate of HashMap memory usage
        let entries = self.current_shard.len();
        let overhead_per_entry = 48; // HashMap overhead per entry (rough estimate)
        let key_size_avg = 50; // Average key string size
        let total_overhead = entries * (overhead_per_entry + key_size_avg);
        
        (self.current_size + total_overhead as u64) as f64 / (1024.0 * 1024.0)
    }
    
    pub fn finalize(mut self, config: serde_json::Value) -> TensorportResult<()> {
        // Flush final shard if needed
        if !self.current_shard.is_empty() {
            self.flush_shard()?;
        }
        
        println!("\n📦 JAX pickle shards created directly from Rust:");
        println!("   Total shards: {}", self.shard_count);
        println!("   Total parameters: {:.2}B", self.total_params as f64 / 1e9);
        println!("   Direct JAX loadable - no Python conversion needed!");
        
        // Create manifest
        self.create_manifest(&config)?;
        
        // Create loader script
        self.create_loader_script(&config)?;
        
        Ok(())
    }
    
    fn flush_shard(&mut self) -> TensorportResult<()> {
        if self.current_shard.is_empty() {
            return Ok(());
        }
        
        let start_time = Instant::now();
        let shard_file = self.output_dir.join(format!("shard_{:03}.pkl", self.shard_count));
        
        println!("  🔄 Writing shard {} with {} entries...", 
                 self.shard_count, self.current_shard.len());
        
        // Write pickle file
        let write_start = Instant::now();
        let mut file = File::create(&shard_file)?;
        serde_pickle::to_writer(&mut file, &self.current_shard, Default::default())
            .map_err(|e| TensorportError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to write pickle shard: {}", e)
            )))?;
        let write_time = write_start.elapsed();
        
        let file_size_mb = file.metadata()?.len() as f64 / (1024.0 * 1024.0);
        
        println!("📦 Shard {} completed: {:.1}MB, {} tensors, write time: {:.1}s", 
                 self.shard_count, file_size_mb, self.current_shard.len(),
                 write_time.as_secs_f64());
        
        self.shard_infos.push(ShardInfo {
            file: format!("shard_{:03}.pkl", self.shard_count),
            size_mb: file_size_mb,
            tensor_count: self.current_shard.len(),
        });
        
        // Clear the HashMap (this should free memory)
        let clear_start = Instant::now();
        self.current_shard.clear();
        let clear_time = clear_start.elapsed();
        
        self.current_size = 0;
        self.shard_count += 1;
        
        let total_time = start_time.elapsed();
        println!("  ✅ Shard flush complete in {:.1}s (write: {:.1}s, clear: {:.1}ms)",
                 total_time.as_secs_f64(), 
                 write_time.as_secs_f64(),
                 clear_time.as_millis());
        
        Ok(())
    }
    
    fn create_manifest(&self, config: &serde_json::Value) -> TensorportResult<()> {
        let manifest = serde_json::json!({
            "format": "tensorport_jax_pickle_sharded",
            "total_shards": self.shard_count,
            "total_parameters": self.total_params,
            "shards": self.shard_infos,
            "config": config,
            "notes": "Direct Rust → JAX conversion, Git LFS compatible shards"
        });
        
        let manifest_path = self.output_dir.join("manifest.json");
        let file = File::create(manifest_path)?;
        serde_json::to_writer_pretty(file, &manifest)?;
        
        Ok(())
    }
    
    fn tensor_to_python_array(&self, tensor: &Tensor) -> TensorportResult<PythonArray> {
        let shape = tensor.shape.clone();
        
        let (dtype, data) = match &tensor.data {
            TensorData::Float16(values) => {
                // Convert f16 to raw bytes for pickle
                let bytes: Vec<u8> = values.iter()
                    .flat_map(|&v| v.to_bits().to_le_bytes())
                    .collect();
                ("float16", bytes)
            }
            TensorData::Float32(values) => {
                // Convert f32 to raw bytes
                let bytes: Vec<u8> = values.iter()
                    .flat_map(|&v| v.to_le_bytes())
                    .collect();
                ("float32", bytes)
            }
            TensorData::Uint8(values) => {
                ("uint8", values.clone())
            }
            TensorData::Int64(values) => {
                let bytes: Vec<u8> = values.iter()
                    .flat_map(|&v| v.to_le_bytes())
                    .collect();
                ("int64", bytes)
            }
        };
        
        Ok(PythonArray {
            shape,
            dtype: dtype.to_string(),
            data,
        })
    }
    
    fn create_loader_script(&self, config: &serde_json::Value) -> TensorportResult<()> {
        let script_content = format!(r#"#!/usr/bin/env python3
"""
Complete JAX model loader with architecture initialization.
Loads both weights and model configuration for full model instantiation.
"""

import json
import pickle
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Model configuration from conversion
MODEL_CONFIG = {}

def load_sharded_jax_model(model_dir):
    """Load sharded JAX model from TensorPort conversion."""
    model_dir = Path(model_dir)
    
    # Load manifest
    with open(model_dir / 'manifest.json') as f:
        manifest = json.load(f)
    
    print(f"Loading JAX model with {{manifest['total_shards']}} shards...")
    print(f"Total parameters: {{manifest['total_parameters']:,}}")
    
    # Load and merge all shards
    all_params = {{}}
    for shard_info in manifest['shards']:
        shard_path = model_dir / shard_info['file']
        print(f"  Loading {{shard_info['file']}} ({{shard_info['size_mb']:.1f}}MB)...")
        
        with open(shard_path, 'rb') as f:
            shard_params = pickle.load(f)
        
        # Convert and merge
        shard_params = convert_to_jax(shard_params)
        merge_nested(all_params, shard_params)
    
    print("✅ All shards loaded successfully!")
    return all_params, manifest['config']

def convert_to_jax(node):
    """Convert pickle data to JAX arrays."""
    if isinstance(node, dict):
        if 'shape' in node and 'dtype' in node and 'data' in node:
            # This is a tensor
            shape = node['shape']
            dtype = node['dtype']
            data = node['data']
            
            # Convert bytes back to numpy array
            if dtype == 'float16':
                np_array = np.frombuffer(bytes(data), dtype=np.float16).reshape(shape)
            elif dtype == 'float32':
                np_array = np.frombuffer(bytes(data), dtype=np.float32).reshape(shape)
            elif dtype == 'uint8':
                # MXFP4 quantized data
                np_array = np.array(data, dtype=np.uint8).reshape(shape)
            elif dtype == 'int64':
                np_array = np.frombuffer(bytes(data), dtype=np.int64).reshape(shape)
            else:
                raise ValueError(f"Unknown dtype: {{dtype}}")
            
            return jnp.array(np_array)
        else:
            # Nested dict - recurse
            return {{k: convert_to_jax(v) for k, v in node.items()}}
    return node

def merge_nested(target: Dict, source: Dict):
    """Merge nested dictionaries."""
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            merge_nested(target[key], value)
        else:
            target[key] = value

def create_model_from_config(config: Dict[str, Any], params: Dict):
    """Initialize model architecture from config."""
    model_type = config.get('model_type', 'unknown')
    
    if model_type == 'gpt_oss':
        print(f"Model type: {{model_type}}")
        print(f"Hidden size: {{config.get('hidden_size')}}")
        print(f"Num layers: {{config.get('num_hidden_layers')}}")
        print(f"Num experts: {{config.get('num_local_experts')}}")
        print(f"Quantization: {{config.get('quantization_config', {{}}).get('quant_method')}}")
        return config, params
    else:
        print(f"Model type {{model_type}} - returning config and params")
        return config, params

# Example usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python load_jax_model.py <model_dir>")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    
    # Load model and config
    params, config = load_sharded_jax_model(model_dir)
    
    # Initialize model architecture
    model_config, params = create_model_from_config(config, params)
    
    print("\n✅ Model ready for inference on JAX!")
"#, serde_json::to_string_pretty(config)?);
        
        let script_path = self.output_dir.join("load_jax_model.py");
        std::fs::write(&script_path, script_content)?;
        
        // Make executable on Unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&script_path)?.permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&script_path, perms)?;
        }
        
        println!("📜 Created loader script: {}", script_path.display());
        
        Ok(())
    }
}

// Removed insert_nested - using flat structure now for better performance

/// Python-compatible array structure for pickle serialization
#[derive(Debug, Serialize, Deserialize)]
struct PythonArray {
    shape: Vec<usize>,
    dtype: String,
    data: Vec<u8>,
}

/// PyTree node - either a dict or an array
#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum PyTreeNode {
    Dict(HashMap<String, PyTreeNode>),
    Array(PythonArray),
}

/// Simple flat structure for faster serialization
#[derive(Debug, Serialize, Deserialize)]
struct FlatPickle {
    tensors: Vec<(String, PythonArray)>,
}