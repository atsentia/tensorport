use crate::error::TensorportResult;
use crate::tensor::{Tensor, TensorData};
use std::fs::{File, create_dir_all};
use std::io::{Write, BufWriter};
use std::path::PathBuf;
use byteorder::{LittleEndian, WriteBytesExt};

/// Direct NumPy array writer - writes .npy files that JAX can load directly
pub struct NumpyDirectWriter {
    output_dir: PathBuf,
    shard_size_gb: f64,
    current_shard_dir: PathBuf,
    current_size: u64,
    shard_count: usize,
    total_params: u64,
    tensor_manifest: Vec<TensorEntry>,
}

#[derive(Debug, serde::Serialize)]
struct TensorEntry {
    name: String,
    file: String,
    shape: Vec<usize>,
    dtype: String,
    size_mb: f64,
}

impl NumpyDirectWriter {
    pub fn new<P: AsRef<std::path::Path>>(output_dir: P, shard_size_gb: f64) -> TensorportResult<Self> {
        let output_dir = output_dir.as_ref().to_path_buf();
        create_dir_all(&output_dir)?;
        
        let current_shard_dir = output_dir.join(format!("shard_{:03}", 0));
        create_dir_all(&current_shard_dir)?;
        
        Ok(NumpyDirectWriter {
            output_dir,
            shard_size_gb,
            current_shard_dir,
            current_size: 0,
            shard_count: 0,
            total_params: 0,
            tensor_manifest: Vec::new(),
        })
    }
    
    pub fn add_tensor(&mut self, tensor: Tensor) -> TensorportResult<()> {
        let tensor_size = tensor.data.size_bytes() as u64;
        let max_shard_size = (self.shard_size_gb * 1024.0 * 1024.0 * 1024.0) as u64;
        
        // Check if we need a new shard
        if self.current_size > 0 && self.current_size + tensor_size > max_shard_size {
            self.new_shard()?;
        }
        
        // Clean tensor name for filename
        let clean_name = tensor.name.replace(".", "_").replace("/", "_");
        let npy_file = self.current_shard_dir.join(format!("{}.npy", clean_name));
        
        // Write NumPy array
        self.write_numpy_array(&npy_file, &tensor)?;
        
        // Track in manifest
        self.tensor_manifest.push(TensorEntry {
            name: tensor.name.clone(),
            file: format!("shard_{:03}/{}.npy", self.shard_count, clean_name),
            shape: tensor.shape.clone(),
            dtype: self.get_numpy_dtype(&tensor.data),
            size_mb: tensor_size as f64 / (1024.0 * 1024.0),
        });
        
        self.current_size += tensor_size;
        self.total_params += tensor.parameter_count() as u64;
        
        println!("  → {} | shape: {:?} | {:.2}MB | saved to shard_{:03}/{}.npy",
                 tensor.name, tensor.shape, 
                 tensor_size as f64 / (1024.0 * 1024.0),
                 self.shard_count, clean_name);
        
        Ok(())
    }
    
    fn new_shard(&mut self) -> TensorportResult<()> {
        self.shard_count += 1;
        self.current_shard_dir = self.output_dir.join(format!("shard_{:03}", self.shard_count));
        create_dir_all(&self.current_shard_dir)?;
        self.current_size = 0;
        
        println!("📦 Starting new shard: shard_{:03}", self.shard_count);
        Ok(())
    }
    
    fn write_numpy_array(&self, path: &PathBuf, tensor: &Tensor) -> TensorportResult<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        
        // Write NumPy header
        // Magic string
        writer.write_all(&[0x93u8])?;
        writer.write_all(b"NUMPY")?;
        
        // Version (1.0)
        writer.write_u8(0x01)?;
        writer.write_u8(0x00)?;
        
        // Build header dict
        let dtype = self.get_numpy_dtype(&tensor.data);
        let fortran_order = false;
        let shape_str = tensor.shape.iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        
        let header = format!(
            "{{'descr': '{}', 'fortran_order': {}, 'shape': ({}{}), }}",
            dtype,
            if fortran_order { "True" } else { "False" },
            shape_str,
            if tensor.shape.len() == 1 { "," } else { "" }
        );
        
        // Pad header to 64-byte alignment
        let header_bytes = header.as_bytes();
        let header_len = header_bytes.len();
        let padding = (64 - (10 + header_len) % 64) % 64;
        let total_header_len = header_len + padding + 1; // +1 for newline
        
        // Write header length (little-endian)
        writer.write_u16::<LittleEndian>(total_header_len as u16)?;
        
        // Write header
        writer.write_all(header_bytes)?;
        for _ in 0..padding {
            writer.write_u8(b' ')?;
        }
        writer.write_u8(b'\n')?;
        
        // Write data
        match &tensor.data {
            TensorData::Float16(values) => {
                for val in values {
                    writer.write_u16::<LittleEndian>(val.to_bits())?;
                }
            }
            TensorData::Float32(values) => {
                for val in values {
                    writer.write_f32::<LittleEndian>(*val)?;
                }
            }
            TensorData::Uint8(values) => {
                writer.write_all(values)?;
            }
            TensorData::Int64(values) => {
                for val in values {
                    writer.write_i64::<LittleEndian>(*val)?;
                }
            }
        }
        
        writer.flush()?;
        Ok(())
    }
    
    fn get_numpy_dtype(&self, data: &TensorData) -> String {
        match data {
            TensorData::Float16(_) => "<f2".to_string(),
            TensorData::Float32(_) => "<f4".to_string(),
            TensorData::Uint8(_) => "|u1".to_string(),
            TensorData::Int64(_) => "<i8".to_string(),
        }
    }
    
    pub fn finalize(self, config: serde_json::Value) -> TensorportResult<()> {
        // Write manifest
        let manifest = serde_json::json!({
            "format": "numpy_direct",
            "total_shards": self.shard_count + 1,
            "total_parameters": self.total_params,
            "tensors": self.tensor_manifest,
            "config": config,
        });
        
        let manifest_path = self.output_dir.join("manifest.json");
        let file = File::create(manifest_path)?;
        serde_json::to_writer_pretty(file, &manifest)?;
        
        // Create JAX loader script
        self.create_loader_script()?;
        
        println!("\n✅ NumPy array conversion complete!");
        println!("   Total shards: {}", self.shard_count + 1);
        println!("   Total parameters: {:.2}B", self.total_params as f64 / 1e9);
        println!("   Output: {}", self.output_dir.display());
        
        Ok(())
    }
    
    fn create_loader_script(&self) -> TensorportResult<()> {
        let script = r#"#!/usr/bin/env python3
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
"#;
        
        let script_path = self.output_dir.join("load_numpy_jax.py");
        std::fs::write(&script_path, script)?;
        
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