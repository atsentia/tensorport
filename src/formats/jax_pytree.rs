use crate::error::TensorportResult;
use crate::tensor::{Tensor, TensorData};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

/// JAX PyTree format writer that creates pickle-loadable nested structures
pub struct JaxPyTreeWriter {
    output_dir: PathBuf,
    precision: String,
}

impl JaxPyTreeWriter {
    pub fn new<P: AsRef<Path>>(output_dir: P, precision: String) -> TensorportResult<Self> {
        let output_dir = output_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&output_dir)?;
        
        Ok(JaxPyTreeWriter {
            output_dir,
            precision,
        })
    }
    
    pub fn write_model(&self, tensors: Vec<Tensor>, config: Value) -> TensorportResult<()> {
        // Create nested PyTree structure
        let pytree = self.create_pytree_structure(tensors)?;
        
        // Save as pickle file (Python can load this directly)
        self.save_as_python_pickle(&pytree)?;
        
        // Save JAX loading script
        self.create_jax_loading_script(config)?;
        
        Ok(())
    }
    
    fn create_pytree_structure(&self, tensors: Vec<Tensor>) -> TensorportResult<HashMap<String, JaxTensor>> {
        let mut pytree = HashMap::new();
        
        for tensor in tensors {
            let clean_name = if tensor.name.starts_with("model.") {
                tensor.name[6..].to_string()
            } else {
                tensor.name.clone()
            };
            
            let jax_tensor = JaxTensor::from_tensor(tensor)?;
            pytree.insert(clean_name, jax_tensor);
        }
        
        Ok(pytree)
    }
    
    fn save_as_python_pickle(&self, pytree: &HashMap<String, JaxTensor>) -> TensorportResult<()> {
        // For now, save as JSON that Python can easily load and convert to JAX arrays
        // TODO: Implement proper Python pickle format writer
        let pickle_path = self.output_dir.join("model_weights.json");
        let mut file = File::create(pickle_path)?;
        
        // Convert to JSON-serializable format
        let json_data = json!({
            "format": "tensorport_jax_pytree",
            "precision": self.precision,
            "tensors": pytree.iter().map(|(k, v)| {
                json!({
                    "name": k,
                    "shape": v.shape,
                    "dtype": v.dtype,
                    "data_file": format!("{}.bin", k.replace(".", "_"))
                })
            }).collect::<Vec<_>>()
        });
        
        serde_json::to_writer_pretty(&mut file, &json_data)?;
        
        // Save individual tensor data as binary files
        for (name, tensor) in pytree {
            let safe_name = name.replace(".", "_");
            let data_path = self.output_dir.join(format!("{}.bin", safe_name));
            tensor.save_binary(&data_path)?;
        }
        
        Ok(())
    }
    
    fn create_jax_loading_script(&self, config: Value) -> TensorportResult<()> {
        let script_content = format!(r#"#!/usr/bin/env python3
"""
JAX PyTree loader for TensorPort converted models.
"""

import json
import numpy as np
import jax.numpy as jnp
from pathlib import Path

def load_tensorport_jax_model(model_dir):
    """Load TensorPort-converted model as JAX PyTree."""
    model_dir = Path(model_dir)
    
    # Load metadata
    metadata_path = model_dir / "model_weights.json"
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    print(f"Loading TensorPort JAX model:")
    print(f"  Format: {{metadata['format']}}")
    print(f"  Precision: {{metadata['precision']}}")
    print(f"  Tensors: {{len(metadata['tensors'])}}")
    
    # Load tensors
    params = {{}}
    
    for tensor_info in metadata['tensors']:
        name = tensor_info['name']
        shape = tensor_info['shape']
        dtype = tensor_info['dtype']
        data_file = tensor_info['data_file']
        
        # Load binary data
        data_path = model_dir / data_file
        raw_data = np.fromfile(data_path, dtype=np.{})
        
        # Reshape and convert to JAX array
        tensor_data = raw_data.reshape(shape)
        jax_array = jnp.array(tensor_data)
        
        # Create nested structure
        parts = name.split('.')
        current = params
        for part in parts[:-1]:
            if part not in current:
                current[part] = {{}}
            current = current[part]
        current[parts[-1]] = jax_array
    
    print("✅ JAX PyTree loaded successfully!")
    return params

# Configuration from original model
CONFIG = {}

# Example usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python load_jax_pytree.py <model_dir>")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    params = load_tensorport_jax_model(model_dir)
    
    print(f"\\nModel structure:")
    def print_structure(obj, prefix=""):
        if isinstance(obj, dict):
            for key in sorted(obj.keys())[:10]:  # Show first 10
                print(f"{{prefix}}{{key}}: ", end="")
                if isinstance(obj[key], dict):
                    print("{{...}}")
                    if len(prefix) < 10:  # Limit depth
                        print_structure(obj[key], prefix + "  ")
                else:
                    print(f"{{obj[key].shape}} {{obj[key].dtype}}")
            if len(obj) > 10:
                print(f"{{prefix}}... and {{len(obj) - 10}} more")
        
    print_structure(params)
"#, 
        match self.precision.as_str() {
            "float16" => "float16",
            "float32" => "float32", 
            _ => "float32"
        },
        serde_json::to_string_pretty(&config)?
    );
        
        let script_path = self.output_dir.join("load_jax_pytree.py");
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

/// Simplified tensor representation for JAX PyTree
#[derive(Debug)]
struct JaxTensor {
    shape: Vec<usize>,
    dtype: String,
    data: Vec<u8>,
}

impl JaxTensor {
    fn from_tensor(tensor: Tensor) -> TensorportResult<Self> {
        let shape = tensor.shape;
        let dtype = tensor.data.dtype_name().to_string();
        
        let data = match tensor.data {
            TensorData::Float16(values) => {
                let mut bytes = Vec::with_capacity(values.len() * 2);
                for value in values {
                    bytes.extend_from_slice(&value.to_bits().to_le_bytes());
                }
                bytes
            }
            TensorData::Float32(values) => {
                let mut bytes = Vec::with_capacity(values.len() * 4);
                for value in values {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                bytes
            }
            TensorData::Uint8(values) => values,
            TensorData::Int64(values) => {
                let mut bytes = Vec::with_capacity(values.len() * 8);
                for value in values {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                bytes
            }
        };
        
        Ok(JaxTensor { shape, dtype, data })
    }
    
    fn save_binary(&self, path: &Path) -> TensorportResult<()> {
        let mut file = File::create(path)?;
        file.write_all(&self.data)?;
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
    fn test_jax_pytree_writer() {
        let temp_dir = TempDir::new().unwrap();
        let writer = JaxPyTreeWriter::new(temp_dir.path(), "float16".to_string()).unwrap();
        
        let test_data = vec![f16::from_f32(1.0), f16::from_f32(2.0)];
        let tensor = Tensor {
            name: "layers.0.weight".to_string(),
            shape: vec![2, 1],
            data: TensorData::Float16(test_data),
        };
        
        let config = json!({"test": true});
        writer.write_model(vec![tensor], config).unwrap();
        
        // Check that files were created
        assert!(temp_dir.path().join("model_weights.json").exists());
        assert!(temp_dir.path().join("load_jax_pytree.py").exists());
    }
}