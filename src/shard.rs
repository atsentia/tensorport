use crate::error::{TensorportError, TensorportResult};
use crate::tensor::{Tensor, TensorData};
use rmp_serde;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardInfo {
    pub file: String,
    pub size_mb: f64,
    pub tensor_count: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ConversionManifest {
    pub format: String,
    pub total_shards: usize,
    pub total_size_gb: f64,
    pub total_parameters: u64,
    pub config: serde_json::Value,
    pub shards: Vec<ShardInfo>,
    pub conversion_notes: String,
}

pub struct ShardWriter {
    output_dir: PathBuf,
    max_shard_size: u64,
    precision: String,
    
    // Current shard state
    current_shard: HashMap<String, SerializableTensor>,
    current_size: u64,
    shard_count: usize,
    
    // Statistics
    total_parameters: u64,
    shard_infos: Vec<ShardInfo>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SerializableTensor {
    shape: Vec<usize>,
    dtype: String,
    data: SerializableData,
}

#[derive(Debug, Serialize, Deserialize)]
enum SerializableData {
    Float16(Vec<u16>), // Store as raw bits for efficiency
    Float32(Vec<f32>),
    Uint8(Vec<u8>),
    Int64(Vec<i64>),
}

impl ShardWriter {
    pub fn new<P: AsRef<Path>>(
        output_dir: P,
        max_shard_size_gb: f64,
        precision: String,
    ) -> TensorportResult<Self> {
        let output_dir = output_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&output_dir)?;
        
        let max_shard_size = (max_shard_size_gb * 1024.0 * 1024.0 * 1024.0) as u64;
        
        Ok(ShardWriter {
            output_dir,
            max_shard_size,
            precision,
            current_shard: HashMap::new(),
            current_size: 0,
            shard_count: 0,
            total_parameters: 0,
            shard_infos: Vec::new(),
        })
    }
    
    pub fn add_tensor(&mut self, tensor: Tensor) -> TensorportResult<()> {
        let tensor_size = tensor.data.size_bytes() as u64;
        self.total_parameters += tensor.parameter_count() as u64;
        
        // Check if we need to finalize current shard
        if self.current_size + tensor_size > self.max_shard_size && !self.current_shard.is_empty() {
            self.finalize_current_shard()?;
        }
        
        // Convert tensor to serializable format
        let serializable_tensor = self.tensor_to_serializable(tensor)?;
        let tensor_name = serializable_tensor.0;
        let serializable_data = serializable_tensor.1;
        
        // Add to nested structure
        self.add_to_nested_structure(tensor_name, serializable_data);
        self.current_size += tensor_size;
        
        Ok(())
    }
    
    fn tensor_to_serializable(&self, tensor: Tensor) -> TensorportResult<(String, SerializableTensor)> {
        let dtype_name = tensor.data.dtype_name().to_string();
        let serializable_data = match tensor.data {
            TensorData::Float16(data) => {
                let bits: Vec<u16> = data.iter().map(|f| f.to_bits()).collect();
                SerializableData::Float16(bits)
            }
            TensorData::Float32(data) => SerializableData::Float32(data),
            TensorData::Uint8(data) => SerializableData::Uint8(data),
            TensorData::Int64(data) => SerializableData::Int64(data),
        };
        
        let serializable_tensor = SerializableTensor {
            shape: tensor.shape,
            dtype: dtype_name,
            data: serializable_data,
        };
        
        Ok((tensor.name, serializable_tensor))
    }
    
    fn add_to_nested_structure(&mut self, tensor_name: String, tensor_data: SerializableTensor) {
        // Clean tensor name (remove "model." prefix if present)
        let clean_name = if tensor_name.starts_with("model.") {
            &tensor_name[6..]
        } else {
            &tensor_name
        };
        
        // For now, store flat - we can add nested structure later if needed
        self.current_shard.insert(clean_name.to_string(), tensor_data);
    }
    
    fn finalize_current_shard(&mut self) -> TensorportResult<()> {
        if self.current_shard.is_empty() {
            return Ok(());
        }
        
        let shard_filename = format!("shard_{:03}.msgpack", self.shard_count);
        let shard_path = self.output_dir.join(&shard_filename);
        
        // Serialize using MessagePack for efficiency
        let mut file = File::create(&shard_path)?;
        let serialized = rmp_serde::to_vec(&self.current_shard)
            .map_err(|e| TensorportError::MessagePack(e))?;
        file.write_all(&serialized)?;
        
        let size_bytes = serialized.len() as u64;
        let size_mb = size_bytes as f64 / (1024.0 * 1024.0);
        let tensor_count = self.current_shard.len();
        
        self.shard_infos.push(ShardInfo {
            file: shard_filename,
            size_mb,
            tensor_count,
        });
        
        println!("📦 Shard {} completed: {:.1}MB, {} tensors", 
                self.shard_count, size_mb, tensor_count);
        
        // Reset for next shard
        self.current_shard.clear();
        self.current_size = 0;
        self.shard_count += 1;
        
        Ok(())
    }
    
    pub fn finalize(mut self, config: serde_json::Value) -> TensorportResult<ConversionResult> {
        // Finalize any remaining shard
        if !self.current_shard.is_empty() {
            self.finalize_current_shard()?;
        }
        
        let total_size_mb: f64 = self.shard_infos.iter().map(|s| s.size_mb).sum();
        let total_size_gb = total_size_mb / 1024.0;
        
        // Create manifest
        let manifest = ConversionManifest {
            format: format!("tensorport_{}", self.precision),
            total_shards: self.shard_infos.len(),
            total_size_gb,
            total_parameters: self.total_parameters,
            config,
            shards: self.shard_infos.clone(),
            conversion_notes: format!(
                "Converted with TensorPort v{} - Fast Rust tensor conversion with custom bfloat16 support",
                env!("CARGO_PKG_VERSION")
            ),
        };
        
        // Save manifest
        let manifest_path = self.output_dir.join("manifest.json");
        let manifest_file = File::create(&manifest_path)?;
        serde_json::to_writer_pretty(manifest_file, &manifest)?;
        
        Ok(ConversionResult {
            output_path: self.output_dir,
            total_params: self.total_parameters,
            shard_count: self.shard_infos.len(),
            total_size_gb,
            manifest_path,
        })
    }
}

#[derive(Debug)]
pub struct ConversionResult {
    pub output_path: PathBuf,
    pub total_params: u64,
    pub shard_count: usize,
    pub total_size_gb: f64,
    pub manifest_path: PathBuf,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{TensorData, Tensor};
    use half::f16;
    use tempfile::TempDir;
    
    #[test]
    fn test_shard_writer() {
        let temp_dir = TempDir::new().unwrap();
        let mut writer = ShardWriter::new(temp_dir.path(), 0.001, "float16".to_string()).unwrap(); // 1MB max
        
        // Create test tensor
        let test_data = vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0)];
        let tensor = Tensor {
            name: "test.weight".to_string(),
            shape: vec![2, 2],
            data: TensorData::Float16(test_data),
        };
        
        writer.add_tensor(tensor).unwrap();
        
        let config = serde_json::json!({"test": true});
        let result = writer.finalize(config).unwrap();
        
        assert_eq!(result.total_params, 4);
        assert_eq!(result.shard_count, 1);
        assert!(result.manifest_path.exists());
    }
}