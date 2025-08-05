use crate::error::{TensorportError, TensorportResult};
use crate::tensor::{Tensor, TensorInfo};
use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::Cursor;
use std::path::Path;

pub struct SafetensorsReader {
    file_path: String,
    mmap: Mmap,
    header: HashMap<String, Value>,
    data_offset: usize,
}

impl SafetensorsReader {
    pub fn new<P: AsRef<Path>>(file_path: P) -> TensorportResult<Self> {
        let file_path = file_path.as_ref().to_string_lossy().to_string();
        let file = File::open(&file_path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        
        // Parse safetensors header
        let (header, data_offset) = Self::parse_header(&mmap)?;
        
        Ok(SafetensorsReader {
            file_path,
            mmap,
            header,
            data_offset,
        })
    }
    
    fn parse_header(mmap: &Mmap) -> TensorportResult<(HashMap<String, Value>, usize)> {
        if mmap.len() < 8 {
            return Err(TensorportError::FileFormat(
                "File too small to contain safetensors header".to_string()
            ));
        }
        
        // Read header length (first 8 bytes, little endian)
        let mut cursor = Cursor::new(&mmap[0..8]);
        let header_length = cursor.read_u64::<LittleEndian>()? as usize;
        
        if mmap.len() < 8 + header_length {
            return Err(TensorportError::FileFormat(
                "File truncated: header extends beyond file".to_string()
            ));
        }
        
        // Read and parse JSON header
        let header_bytes = &mmap[8..8 + header_length];
        let header_str = std::str::from_utf8(header_bytes)
            .map_err(|_| TensorportError::FileFormat("Invalid UTF-8 in header".to_string()))?;
        
        let header: HashMap<String, Value> = serde_json::from_str(header_str)?;
        let data_offset = 8 + header_length;
        
        Ok((header, data_offset))
    }
    
    pub fn get_tensor_info(&self, tensor_name: &str) -> TensorportResult<TensorInfo> {
        let tensor_value = self.header.get(tensor_name)
            .ok_or_else(|| TensorportError::InvalidTensorData(
                format!("Tensor '{}' not found in file '{}'", tensor_name, self.file_path)
            ))?;
        
        let tensor_obj = tensor_value.as_object()
            .ok_or_else(|| TensorportError::FileFormat(
                format!("Invalid tensor metadata for '{}'", tensor_name)
            ))?;
        
        let dtype = tensor_obj.get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| TensorportError::FileFormat("Missing or invalid dtype".to_string()))?
            .to_string();
        
        let shape: Vec<usize> = tensor_obj.get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| TensorportError::FileFormat("Missing or invalid shape".to_string()))?
            .iter()
            .map(|v| v.as_u64().map(|u| u as usize))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| TensorportError::FileFormat("Invalid shape values".to_string()))?;
        
        let data_offsets: [usize; 2] = tensor_obj.get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| TensorportError::FileFormat("Missing or invalid data_offsets".to_string()))?
            .iter()
            .map(|v| v.as_u64().map(|u| u as usize))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| TensorportError::FileFormat("Invalid data_offsets values".to_string()))?
            .try_into()
            .map_err(|_| TensorportError::FileFormat("data_offsets must have exactly 2 elements".to_string()))?;
        
        Ok(TensorInfo {
            dtype,
            shape,
            data_offsets,
        })
    }
    
    pub fn read_tensor(&self, tensor_name: &str, target_precision: &str) -> TensorportResult<Tensor> {
        let tensor_info = self.get_tensor_info(tensor_name)?;
        
        // Calculate tensor data bounds
        let tensor_start = self.data_offset + tensor_info.data_offsets[0];
        let tensor_end = self.data_offset + tensor_info.data_offsets[1];
        
        if tensor_end > self.mmap.len() {
            return Err(TensorportError::FileFormat(
                format!("Tensor data extends beyond file: {} > {}", tensor_end, self.mmap.len())
            ));
        }
        
        let raw_bytes = &self.mmap[tensor_start..tensor_end];
        
        Tensor::from_raw_bytes(
            tensor_name.to_string(),
            &tensor_info,
            raw_bytes,
            target_precision,
        )
    }
    
    pub fn list_tensors(&self) -> Vec<&String> {
        self.header.keys()
            .filter(|k| *k != "__metadata__") // Skip metadata key
            .collect()
    }
    
    pub fn file_path(&self) -> &str {
        &self.file_path
    }
}

pub struct ModelIndex {
    pub weight_map: HashMap<String, String>,
    pub metadata: HashMap<String, Value>,
}

impl ModelIndex {
    pub fn load<P: AsRef<Path>>(index_path: P) -> TensorportResult<Self> {
        let file = File::open(index_path)?;
        let index_data: Value = serde_json::from_reader(file)?;
        
        let index_obj = index_data.as_object()
            .ok_or_else(|| TensorportError::FileFormat("Invalid index format".to_string()))?;
        
        let weight_map = index_obj.get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| TensorportError::FileFormat("Missing or invalid weight_map".to_string()))?
            .iter()
            .map(|(k, v)| {
                let file_name = v.as_str()
                    .ok_or_else(|| TensorportError::FileFormat("Invalid file name in weight_map".to_string()))?;
                Ok((k.clone(), file_name.to_string()))
            })
            .collect::<TensorportResult<HashMap<_, _>>>()?;
        
        let metadata = index_obj.iter()
            .filter(|(k, _)| *k != "weight_map")
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        
        Ok(ModelIndex {
            weight_map,
            metadata,
        })
    }
    
    pub fn get_files(&self) -> Vec<String> {
        let mut files: Vec<String> = self.weight_map.values().cloned().collect();
        files.sort();
        files.dedup();
        files
    }
    
    pub fn get_tensors_for_file(&self, file_name: &str) -> Vec<&String> {
        self.weight_map.iter()
            .filter_map(|(tensor_name, file)| {
                if file == file_name {
                    Some(tensor_name)
                } else {
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_safetensors_header_parsing() {
        // Create a minimal safetensors file for testing
        let mut temp_file = NamedTempFile::new().unwrap();
        
        // Create header JSON
        let header = r#"{"test_tensor":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
        let header_bytes = header.as_bytes();
        let header_len = header_bytes.len() as u64;
        
        // Write header length (little endian)
        temp_file.write_all(&header_len.to_le_bytes()).unwrap();
        
        // Write header JSON
        temp_file.write_all(header_bytes).unwrap();
        
        // Write tensor data (4 f32 values)
        let tensor_data = [
            0x00, 0x00, 0x80, 0x3F, // 1.0
            0x00, 0x00, 0x00, 0x40, // 2.0
            0x00, 0x00, 0x40, 0x40, // 3.0
            0x00, 0x00, 0x80, 0x40, // 4.0
        ];
        temp_file.write_all(&tensor_data).unwrap();
        temp_file.flush().unwrap();
        
        // Test reading
        let reader = SafetensorsReader::new(temp_file.path()).unwrap();
        let tensors = reader.list_tensors();
        
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors[0], "test_tensor");
        
        let tensor_info = reader.get_tensor_info("test_tensor").unwrap();
        assert_eq!(tensor_info.dtype, "F32");
        assert_eq!(tensor_info.shape, vec![2, 2]);
        assert_eq!(tensor_info.data_offsets, [0, 16]);
        
        let tensor = reader.read_tensor("test_tensor", "float32").unwrap();
        assert_eq!(tensor.parameter_count(), 4);
    }
}