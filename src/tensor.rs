use crate::error::{TensorportError, TensorportResult};
use byteorder::{LittleEndian, ReadBytesExt};
use half::f16;
use serde::{Deserialize, Serialize};
use std::io::Cursor;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

#[derive(Debug, Clone)]
pub enum TensorData {
    Float16(Vec<f16>),
    Float32(Vec<f32>),
    Uint8(Vec<u8>),
    Int64(Vec<i64>),
}

impl TensorData {
    pub fn len(&self) -> usize {
        match self {
            TensorData::Float16(v) => v.len(),
            TensorData::Float32(v) => v.len(),
            TensorData::Uint8(v) => v.len(),
            TensorData::Int64(v) => v.len(),
        }
    }
    
    pub fn size_bytes(&self) -> usize {
        match self {
            TensorData::Float16(v) => v.len() * 2,
            TensorData::Float32(v) => v.len() * 4,
            TensorData::Uint8(v) => v.len(),
            TensorData::Int64(v) => v.len() * 8,
        }
    }
    
    pub fn dtype_name(&self) -> &'static str {
        match self {
            TensorData::Float16(_) => "float16",
            TensorData::Float32(_) => "float32",
            TensorData::Uint8(_) => "uint8",
            TensorData::Int64(_) => "int64",
        }
    }
}

#[derive(Debug)]
pub struct Tensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: TensorData,
}

impl Tensor {
    pub fn from_raw_bytes(
        name: String,
        tensor_info: &TensorInfo,
        raw_bytes: &[u8],
        target_precision: &str,
    ) -> TensorportResult<Self> {
        let shape = tensor_info.shape.clone();
        let expected_elements: usize = shape.iter().product();
        
        let data = match tensor_info.dtype.as_str() {
            "BF16" => {
                // Custom bfloat16 parsing - our key innovation!
                let bf16_data = parse_bfloat16_bytes(raw_bytes)?;
                
                match target_precision {
                    "float16" => {
                        let f16_data: Vec<f16> = bf16_data.into_iter()
                            .map(|f| f16::from_f32(f))
                            .collect();
                        TensorData::Float16(f16_data)
                    }
                    "float32" => TensorData::Float32(bf16_data),
                    _ => return Err(TensorportError::Config(
                        format!("Unsupported target precision: {}", target_precision)
                    )),
                }
            }
            
            "F32" => {
                if raw_bytes.len() != expected_elements * 4 {
                    return Err(TensorportError::InvalidTensorData(
                        format!("F32 tensor size mismatch: expected {} bytes, got {}", 
                                expected_elements * 4, raw_bytes.len())
                    ));
                }
                
                let mut cursor = Cursor::new(raw_bytes);
                let mut f32_data = Vec::with_capacity(expected_elements);
                
                for _ in 0..expected_elements {
                    f32_data.push(cursor.read_f32::<LittleEndian>()?);
                }
                
                match target_precision {
                    "float16" => {
                        let f16_data: Vec<f16> = f32_data.into_iter()
                            .map(|f| f16::from_f32(f.clamp(-65504.0, 65504.0)))
                            .collect();
                        TensorData::Float16(f16_data)
                    }
                    "float32" => TensorData::Float32(f32_data),
                    _ => return Err(TensorportError::Config(
                        format!("Unsupported target precision: {}", target_precision)
                    )),
                }
            }
            
            "F16" => {
                if raw_bytes.len() != expected_elements * 2 {
                    return Err(TensorportError::InvalidTensorData(
                        format!("F16 tensor size mismatch: expected {} bytes, got {}", 
                                expected_elements * 2, raw_bytes.len())
                    ));
                }
                
                let mut cursor = Cursor::new(raw_bytes);
                let mut f16_data = Vec::with_capacity(expected_elements);
                
                for _ in 0..expected_elements {
                    let raw_f16 = cursor.read_u16::<LittleEndian>()?;
                    f16_data.push(f16::from_bits(raw_f16));
                }
                
                TensorData::Float16(f16_data)
            }
            
            "U8" => {
                if raw_bytes.len() != expected_elements {
                    return Err(TensorportError::InvalidTensorData(
                        format!("U8 tensor size mismatch: expected {} bytes, got {}", 
                                expected_elements, raw_bytes.len())
                    ));
                }
                TensorData::Uint8(raw_bytes.to_vec())
            }
            
            "I64" => {
                if raw_bytes.len() != expected_elements * 8 {
                    return Err(TensorportError::InvalidTensorData(
                        format!("I64 tensor size mismatch: expected {} bytes, got {}", 
                                expected_elements * 8, raw_bytes.len())
                    ));
                }
                
                let mut cursor = Cursor::new(raw_bytes);
                let mut i64_data = Vec::with_capacity(expected_elements);
                
                for _ in 0..expected_elements {
                    i64_data.push(cursor.read_i64::<LittleEndian>()?);
                }
                
                TensorData::Int64(i64_data)
            }
            
            dtype => return Err(TensorportError::UnsupportedDataType(dtype.to_string())),
        };
        
        // Verify element count matches
        if data.len() != expected_elements {
            return Err(TensorportError::InvalidTensorData(
                format!("Element count mismatch: expected {}, got {}", expected_elements, data.len())
            ));
        }
        
        Ok(Tensor { name, shape, data })
    }
    
    pub fn parameter_count(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Parse bfloat16 bytes to f32 - this is our key innovation from the Python version!
/// bfloat16 format: 1 sign + 8 exponent + 7 mantissa bits
/// Conversion: shift bfloat16 to upper 16 bits of u32, interpret as f32
fn parse_bfloat16_bytes(raw_bytes: &[u8]) -> TensorportResult<Vec<f32>> {
    if raw_bytes.len() % 2 != 0 {
        return Err(TensorportError::InvalidTensorData(
            "bfloat16 data must have even number of bytes".to_string()
        ));
    }
    
    let mut cursor = Cursor::new(raw_bytes);
    let mut f32_data = Vec::with_capacity(raw_bytes.len() / 2);
    
    while cursor.position() < raw_bytes.len() as u64 {
        // Read bfloat16 as u16 (little endian)
        let bf16_bits = cursor.read_u16::<LittleEndian>()?;
        
        // Convert bfloat16 to f32 by shifting to upper 16 bits
        let f32_bits = (bf16_bits as u32) << 16;
        let f32_value = f32::from_bits(f32_bits);
        
        f32_data.push(f32_value);
    }
    
    Ok(f32_data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    
    #[test]
    fn test_bfloat16_conversion() {
        // Test known bfloat16 values
        // bfloat16 1.0 = 0x3F80 (same exponent as f32, but 7-bit mantissa)
        let bf16_bytes = [0x80, 0x3F]; // Little endian representation of 0x3F80
        let result = parse_bfloat16_bytes(&bf16_bytes).unwrap();
        
        assert_eq!(result.len(), 1);
        assert_approx_eq!(result[0], 1.0f32, 1e-6);
    }
    
    #[test]
    fn test_tensor_creation() {
        let tensor_info = TensorInfo {
            dtype: "F32".to_string(),
            shape: vec![2, 2],
            data_offsets: [0, 16],
        };
        
        // 4 f32 values: [1.0, 2.0, 3.0, 4.0]
        let raw_bytes = [
            0x00, 0x00, 0x80, 0x3F, // 1.0
            0x00, 0x00, 0x00, 0x40, // 2.0
            0x00, 0x00, 0x40, 0x40, // 3.0
            0x00, 0x00, 0x80, 0x40, // 4.0
        ];
        
        let tensor = Tensor::from_raw_bytes(
            "test_tensor".to_string(),
            &tensor_info,
            &raw_bytes,
            "float32"
        ).unwrap();
        
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.parameter_count(), 4);
        
        if let TensorData::Float32(data) = tensor.data {
            assert_approx_eq!(data[0], 1.0f32, 1e-6);
            assert_approx_eq!(data[1], 2.0f32, 1e-6);
            assert_approx_eq!(data[2], 3.0f32, 1e-6);
            assert_approx_eq!(data[3], 4.0f32, 1e-6);
        } else {
            panic!("Expected Float32 data");
        }
    }
}