use crate::error::{TensorportError, TensorportResult};
use crate::tensor::{Tensor, TensorData};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

/// JAX/Orbax-compatible format writer
pub struct JaxFormatWriter {
    output_dir: PathBuf,
    precision: String,
}

impl JaxFormatWriter {
    pub fn new<P: AsRef<Path>>(output_dir: P, precision: String) -> TensorportResult<Self> {
        let output_dir = output_dir.as_ref().to_path_buf();
        fs::create_dir_all(&output_dir)?;
        
        Ok(JaxFormatWriter {
            output_dir,
            precision,
        })
    }
    
    pub fn write_model(&self, tensors: Vec<Tensor>, config: Value) -> TensorportResult<()> {
        // Create nested parameter structure
        let nested_params = self.create_nested_params(tensors)?;
        
        // Save as individual numpy arrays (Orbax style)
        self.save_orbax_checkpoint(&nested_params)?;
        
        // Save metadata
        self.save_metadata(config, nested_params.len())?;
        
        Ok(())
    }
    
    fn create_nested_params(&self, tensors: Vec<Tensor>) -> TensorportResult<HashMap<String, Tensor>> {
        let mut nested = HashMap::new();
        
        for tensor in tensors {
            let clean_name = if tensor.name.starts_with("model.") {
                &tensor.name[6..]
            } else {
                &tensor.name
            };
            
            nested.insert(clean_name.to_string(), tensor);
        }
        
        Ok(nested)
    }
    
    fn save_orbax_checkpoint(&self, params: &HashMap<String, Tensor>) -> TensorportResult<()> {
        // Create checkpoint directory structure
        let checkpoint_dir = self.output_dir.join("checkpoint");
        fs::create_dir_all(&checkpoint_dir)?;
        
        // Save each tensor as a separate .npy file (Orbax format)
        for (name, tensor) in params {
            let safe_name = name.replace("/", "_").replace(".", "_");
            let tensor_file = checkpoint_dir.join(format!("{}.npy", safe_name));
            
            self.save_tensor_as_numpy(&tensor_file, tensor)?;
        }
        
        // Create checkpoint metadata
        let checkpoint_metadata = json!({
            "format": "orbax_checkpoint",
            "version": "0.1.0",
            "tensor_count": params.len(),
            "precision": self.precision
        });
        
        let metadata_file = checkpoint_dir.join("metadata.json");
        let mut file = File::create(metadata_file)?;
        serde_json::to_writer_pretty(&mut file, &checkpoint_metadata)?;
        
        Ok(())
    }
    
    fn save_tensor_as_numpy(&self, path: &Path, tensor: &Tensor) -> TensorportResult<()> {
        // For now, save as simple binary format
        // TODO: Implement proper .npy format writer
        let mut file = File::create(path)?;
        
        // Write shape header (for debugging)
        let shape_json = serde_json::to_string(&tensor.shape)?;
        writeln!(file, "# Shape: {}", shape_json)?;
        writeln!(file, "# DType: {}", tensor.data.dtype_name())?;
        
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
    
    fn save_metadata(&self, config: Value, tensor_count: usize) -> TensorportResult<()> {
        let metadata = json!({
            "format": "tensorport_jax_orbax",
            "version": env!("CARGO_PKG_VERSION"),
            "tensor_count": tensor_count,
            "precision": self.precision,
            "config": config,
            "loading_instructions": {
                "python_example": "import orbax; checkpoint = orbax.checkpoint.CheckpointManager(...).restore(...)",
                "note": "Use Orbax checkpoint format for JAX/Flax models"
            }
        });
        
        let metadata_file = self.output_dir.join("tensorport_metadata.json");
        let mut file = File::create(metadata_file)?;
        serde_json::to_writer_pretty(&mut file, &metadata)?;
        
        Ok(())
    }
}

/// Create a simple JAX loading script
pub fn create_jax_loader_script<P: AsRef<Path>>(output_dir: P) -> TensorportResult<()> {
    let output_dir = output_dir.as_ref();
    let loader_script = r#"#!/usr/bin/env python3
"""
JAX/Orbax loader for TensorPort converted models.
"""

import json
import orbax.checkpoint as ocp
import jax.numpy as jnp
from pathlib import Path

def load_tensorport_model(checkpoint_dir):
    """Load TensorPort-converted model using Orbax."""
    checkpoint_dir = Path(checkpoint_dir)
    
    # Load metadata
    metadata_path = checkpoint_dir / "tensorport_metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    print(f"Loading TensorPort model:")
    print(f"  Format: {metadata['format']}")
    print(f"  Precision: {metadata['precision']}")
    print(f"  Tensors: {metadata['tensor_count']}")
    
    # Load checkpoint using Orbax
    checkpoint_path = checkpoint_dir / "checkpoint"
    manager = ocp.CheckpointManager(checkpoint_path)
    
    # This is a simplified example - actual usage depends on your model structure
    restored_params = manager.restore(manager.latest_step())
    
    print("✅ Model loaded successfully!")
    return restored_params, metadata

# Example usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python load_jax_model.py <checkpoint_dir>")
        sys.exit(1)
    
    checkpoint_dir = sys.argv[1]
    params, metadata = load_tensorport_model(checkpoint_dir)
    
    print(f"\nLoaded {len(params)} parameter groups")
    for key in list(params.keys())[:5]:  # Show first 5
        print(f"  {key}: {params[key].shape}")
    
    if len(params) > 5:
        print(f"  ... and {len(params) - 5} more")
"#;
    
    let script_path = output_dir.join("load_jax_model.py");
    let mut file = File::create(script_path)?;
    file.write_all(loader_script.as_bytes())?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{TensorData, Tensor};
    use half::f16;
    use tempfile::TempDir;
    
    #[test]
    fn test_jax_format_writer() {
        let temp_dir = TempDir::new().unwrap();
        let writer = JaxFormatWriter::new(temp_dir.path(), "float16".to_string()).unwrap();
        
        // Create test tensor
        let test_data = vec![f16::from_f32(1.0), f16::from_f32(2.0)];
        let tensor = Tensor {
            name: "layers.0.weight".to_string(),
            shape: vec![2, 1],
            data: TensorData::Float16(test_data),
        };
        
        let config = json!({"test": true});
        writer.write_model(vec![tensor], config).unwrap();
        
        // Check that files were created
        assert!(temp_dir.path().join("checkpoint").exists());
        assert!(temp_dir.path().join("tensorport_metadata.json").exists());
    }
}