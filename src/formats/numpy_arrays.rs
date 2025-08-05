use crate::error::TensorportResult;
use crate::tensor::{Tensor, TensorData};
use serde_json::{json, Value};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

/// NumPy arrays format writer - creates individual .npy files
pub struct NumpyArraysWriter {
    output_dir: PathBuf,
    precision: String,
}

impl NumpyArraysWriter {
    pub fn new<P: AsRef<Path>>(output_dir: P, precision: String) -> TensorportResult<Self> {
        let output_dir = output_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&output_dir)?;
        
        Ok(NumpyArraysWriter {
            output_dir,
            precision,
        })
    }
    
    pub fn write_model(&self, tensors: Vec<Tensor>, config: Value) -> TensorportResult<()> {
        // Save each tensor as individual .npy file
        let mut tensor_manifest = Vec::new();
        
        for tensor in tensors {
            let safe_name = tensor.name.replace(".", "_").replace("/", "_");
            let npy_path = self.output_dir.join(format!("{}.npy", safe_name));
            
            self.save_as_numpy(&npy_path, &tensor)?;
            
            tensor_manifest.push(json!({
                "original_name": tensor.name,
                "file_name": format!("{}.npy", safe_name),
                "shape": tensor.shape,
                "dtype": tensor.data.dtype_name(),
                "parameter_count": tensor.parameter_count()
            }));
        }
        
        // Save manifest
        self.save_manifest(tensor_manifest, config)?;
        
        // Create JAX loading script
        self.create_jax_loading_script()?;
        
        Ok(())
    }
    
    fn save_as_numpy(&self, path: &Path, tensor: &Tensor) -> TensorportResult<()> {
        // Simple .npy format implementation
        // Magic string: \x93NUMPY
        // Version: \x01\x00 (version 1.0)
        // Header length: 2 bytes (little endian)
        // Header: dict string with shape, dtype, fortran_order
        // Data: raw tensor data
        
        let mut file = File::create(path)?;
        
        // Magic number
        file.write_all(b"\x93NUMPY")?;
        
        // Version
        file.write_all(&[0x01, 0x00])?;
        
        // Create header
        let dtype_str = match tensor.data.dtype_name() {
            "float16" => "<f2",  // little-endian float16
            "float32" => "<f4",  // little-endian float32
            "uint8" => "|u1",   // uint8
            "int64" => "<i8",   // little-endian int64
            _ => "<f4",         // default to float32
        };
        
        let shape_str = if tensor.shape.is_empty() {
            "()".to_string()
        } else {
            format!("({}{})",
                tensor.shape.iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
                if tensor.shape.len() == 1 { "," } else { "" }
            )
        };
        
        let header = format!(
            "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
            dtype_str, shape_str
        );
        
        // Pad header to 64-byte boundary
        let header_len = header.len();
        let padding_needed = (64 - (header_len + 10) % 64) % 64;  // 10 = magic(6) + version(2) + len(2)
        let padded_header = format!("{}{}\n", header, " ".repeat(padding_needed));
        
        // Write header length
        let header_len = padded_header.len() as u16;
        file.write_all(&header_len.to_le_bytes())?;
        
        // Write header
        file.write_all(padded_header.as_bytes())?;
        
        // Write tensor data
        match &tensor.data {
            TensorData::Float16(data) => {
                for &value in data {
                    file.write_all(&value.to_bits().to_le_bytes())?;
                }
            }
            TensorData::Float32(data) => {
                for &value in data {
                    file.write_all(&value.to_le_bytes())?;
                }
            }
            TensorData::Uint8(data) => {
                file.write_all(data)?;
            }
            TensorData::Int64(data) => {
                for &value in data {
                    file.write_all(&value.to_le_bytes())?;
                }
            }
        }
        
        Ok(())
    }
    
    fn save_manifest(&self, tensor_manifest: Vec<Value>, config: Value) -> TensorportResult<()> {
        let manifest = json!({
            "format": "tensorport_numpy_arrays",
            "version": env!("CARGO_PKG_VERSION"),
            "precision": self.precision,
            "tensor_count": tensor_manifest.len(),
            "total_parameters": tensor_manifest.iter()
                .map(|t| t["parameter_count"].as_u64().unwrap_or(0))
                .sum::<u64>(),
            "tensors": tensor_manifest,
            "config": config
        });
        
        let manifest_path = self.output_dir.join("numpy_manifest.json");
        let mut file = File::create(manifest_path)?;
        serde_json::to_writer_pretty(&mut file, &manifest)?;
        
        Ok(())
    }
    
    fn create_jax_loading_script(&self) -> TensorportResult<()> {
        let script_content = r#"#!/usr/bin/env python3
"""
JAX loader for TensorPort NumPy arrays format.
"""

import json
import numpy as np
import jax.numpy as jnp
from pathlib import Path

def load_tensorport_numpy_model(model_dir):
    """Load TensorPort NumPy arrays as JAX PyTree."""
    model_dir = Path(model_dir)
    
    # Load manifest
    manifest_path = model_dir / "numpy_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    print(f"Loading TensorPort NumPy model:")
    print(f"  Format: {manifest['format']}")
    print(f"  Precision: {manifest['precision']}")
    print(f"  Tensors: {manifest['tensor_count']}")
    print(f"  Parameters: {manifest['total_parameters']:,}")
    
    # Load tensors into nested structure
    params = {}
    
    for tensor_info in manifest['tensors']:
        original_name = tensor_info['original_name']
        file_name = tensor_info['file_name']
        
        # Load NumPy array
        npy_path = model_dir / file_name
        np_array = np.load(npy_path)
        
        # Convert to JAX array
        jax_array = jnp.array(np_array)
        
        # Create nested structure
        clean_name = original_name.replace("model.", "") if original_name.startswith("model.") else original_name
        parts = clean_name.split('.')
        
        current = params
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = jax_array
    
    print("✅ JAX PyTree loaded successfully!")
    return params, manifest

def print_model_structure(params, max_depth=3, current_depth=0):
    """Print the structure of the loaded model."""
    if current_depth >= max_depth:
        return
        
    for key in sorted(list(params.keys())[:10]):  # Show first 10 keys
        value = params[key]
        indent = "  " * current_depth
        
        if isinstance(value, dict):
            print(f"{indent}{key}:")
            print_model_structure(value, max_depth, current_depth + 1)
        else:
            # JAX/NumPy array
            print(f"{indent}{key}: {value.shape} {value.dtype}")
    
    if len(params) > 10:
        indent = "  " * current_depth
        print(f"{indent}... and {len(params) - 10} more")

# Example usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python load_numpy_arrays.py <model_dir>")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    params, manifest = load_tensorport_numpy_model(model_dir)
    
    print(f"\nModel structure:")
    print_model_structure(params)
    
    # Example: Access a specific parameter
    print(f"\nExample parameter access:")
    first_key = next(iter(params.keys()))
    print(f"  params['{first_key}'] = {type(params[first_key])}")
    
    if isinstance(params[first_key], dict):
        sub_key = next(iter(params[first_key].keys()))
        example_param = params[first_key][sub_key]
        print(f"  params['{first_key}']['{sub_key}'] = {example_param.shape} {example_param.dtype}")
"#;
        
        let script_path = self.output_dir.join("load_numpy_arrays.py");
        let mut file = File::create(&script_path)?;
        file.write_all(script_content.as_bytes())?;
        
        // Make script executable
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = file.metadata()?.permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&script_path, perms)?;
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{TensorData, Tensor};
    use half::f16;
    use tempfile::TempDir;
    
    #[test]
    fn test_numpy_arrays_writer() {
        let temp_dir = TempDir::new().unwrap();
        let writer = NumpyArraysWriter::new(temp_dir.path(), "float16".to_string()).unwrap();
        
        let test_data = vec![f16::from_f32(1.0), f16::from_f32(2.0)];
        let tensor = Tensor {
            name: "layers.0.weight".to_string(),
            shape: vec![2, 1],
            data: TensorData::Float16(test_data),
        };
        
        let config = json!({"test": true});
        writer.write_model(vec![tensor], config).unwrap();
        
        // Check that files were created
        assert!(temp_dir.path().join("numpy_manifest.json").exists());
        assert!(temp_dir.path().join("load_numpy_arrays.py").exists());
        assert!(temp_dir.path().join("layers_0_weight.npy").exists());
    }
}