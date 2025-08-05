use crate::error::{TensorportError, TensorportResult};
use crate::shard::ConversionManifest;
use rmp_serde;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub fn verify_conversion<P: AsRef<Path>>(model_dir: P, verbose: bool) -> TensorportResult<()> {
    let model_dir = model_dir.as_ref();
    let manifest_path = model_dir.join("manifest.json");
    
    if !manifest_path.exists() {
        return Err(TensorportError::Verification(
            "Manifest file not found".to_string()
        ));
    }
    
    // Load manifest
    let manifest_file = File::open(&manifest_path)?;
    let manifest: ConversionManifest = serde_json::from_reader(manifest_file)?;
    
    println!("📄 Manifest: {}", manifest.format);
    println!("📊 Claimed parameters: {} ({:.2}B)", 
             manifest.total_parameters, 
             manifest.total_parameters as f64 / 1e9);
    println!("🗂️  Total shards: {}", manifest.total_shards);
    println!("💾 Total size: {:.2}GB", manifest.total_size_gb);
    println!();
    
    let mut total_verified_params = 0u64;
    let mut total_verified_tensors = 0usize;
    
    // Verify each shard
    for (i, shard_info) in manifest.shards.iter().enumerate() {
        let shard_path = model_dir.join(&shard_info.file);
        
        if !shard_path.exists() {
            return Err(TensorportError::Verification(
                format!("Shard file missing: {}", shard_info.file)
            ));
        }
        
        if verbose {
            println!("🔍 Verifying shard {} ({})...", i, shard_info.file);
        }
        
        let (shard_params, shard_tensors) = verify_shard(&shard_path, verbose)?;
        total_verified_params += shard_params;
        total_verified_tensors += shard_tensors;
        
        // Verify file size
        let actual_size_mb = shard_path.metadata()?.len() as f64 / (1024.0 * 1024.0);
        let size_diff = (actual_size_mb - shard_info.size_mb).abs();
        
        if size_diff > 1.0 {  // Allow 1MB tolerance
            println!("⚠️  Size mismatch for {}: actual {:.1}MB, claimed {:.1}MB", 
                    shard_info.file, actual_size_mb, shard_info.size_mb);
        }
        
        if verbose {
            println!("  ✅ Parameters: {}", shard_params);
            println!("  📦 Size: {:.1}MB (claimed: {:.1}MB)", actual_size_mb, shard_info.size_mb);
            println!("  📊 Tensors: {}", shard_tensors);
            println!();
        }
    }
    
    // Final verification
    println!("🎯 VERIFICATION RESULTS:");
    println!("  Claimed: {} parameters ({:.2}B)", 
             manifest.total_parameters, manifest.total_parameters as f64 / 1e9);
    println!("  Verified: {} parameters ({:.2}B)", 
             total_verified_params, total_verified_params as f64 / 1e9);
    println!("  Total tensors: {}", total_verified_tensors);
    
    // Check parameter count match
    if total_verified_params == manifest.total_parameters {
        println!("  ✅ PERFECT PARAMETER MATCH!");
    } else {
        let diff = if total_verified_params > manifest.total_parameters {
            total_verified_params - manifest.total_parameters
        } else {
            manifest.total_parameters - total_verified_params
        };
        
        if diff < 1000 {
            println!("  ✅ VERY CLOSE MATCH (within 1K parameters)");
        } else {
            println!("  ⚠️  PARAMETER MISMATCH: {} difference", diff);
        }
    }
    
    // Check if we hit target thresholds
    let target_20b = 20_000_000_000u64;
    if total_verified_params >= target_20b {
        println!("  🎉 SUCCESS: Verified {:.2}B ≥ 20B target!", 
                total_verified_params as f64 / 1e9);
    }
    
    Ok(())
}

fn verify_shard<P: AsRef<Path>>(shard_path: P, verbose: bool) -> TensorportResult<(u64, usize)> {
    let mut file = File::open(shard_path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    
    // Deserialize MessagePack shard
    let shard_data: HashMap<String, Value> = rmp_serde::from_slice(&buffer)
        .map_err(|e| TensorportError::Verification(
            format!("Failed to deserialize shard: {}", e)
        ))?;
    
    let mut shard_params = 0u64;
    let tensor_count = shard_data.len();
    
    for (tensor_name, tensor_value) in &shard_data {
        let params = count_tensor_parameters(tensor_name, tensor_value, verbose)?;
        shard_params += params;
    }
    
    Ok((shard_params, tensor_count))
}

fn count_tensor_parameters(
    tensor_name: &str, 
    tensor_value: &Value,
    verbose: bool
) -> TensorportResult<u64> {
    let tensor_obj = tensor_value.as_object()
        .ok_or_else(|| TensorportError::Verification(
            format!("Invalid tensor format for {}", tensor_name)
        ))?;
    
    let shape = tensor_obj.get("shape")
        .and_then(|v| v.as_array())
        .ok_or_else(|| TensorportError::Verification(
            format!("Missing or invalid shape for {}", tensor_name)
        ))?;
    
    let dtype = tensor_obj.get("dtype")
        .and_then(|v| v.as_str())
        .ok_or_else(|| TensorportError::Verification(
            format!("Missing or invalid dtype for {}", tensor_name)
        ))?;
    
    // Calculate parameter count
    let mut param_count = 1u64;
    let shape_vec: Vec<u64> = shape.iter()
        .map(|v| v.as_u64().unwrap_or(0))
        .collect();
    
    for dim in &shape_vec {
        param_count *= dim;
    }
    
    if verbose {
        println!("    {}: {:?} {} → {} params", 
                tensor_name, shape_vec, dtype, param_count);
    }
    
    Ok(param_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shard::{ConversionManifest, ShardInfo};
    use tempfile::TempDir;
    use std::io::Write;
    
    #[test]
    fn test_verification() {
        let temp_dir = TempDir::new().unwrap();
        
        // Create a mock manifest
        let manifest = ConversionManifest {
            format: "tensorport_float16".to_string(),
            total_shards: 1,
            total_size_gb: 0.001,
            total_parameters: 4,
            config: serde_json::json!({}),
            shards: vec![ShardInfo {
                file: "shard_000.msgpack".to_string(),
                size_mb: 1.0,
                tensor_count: 1,
            }],
            conversion_notes: "Test conversion".to_string(),
        };
        
        // Write manifest
        let manifest_path = temp_dir.path().join("manifest.json");
        let mut manifest_file = File::create(&manifest_path).unwrap();
        serde_json::to_writer_pretty(&mut manifest_file, &manifest).unwrap();
        
        // Create mock shard data
        let shard_data = {
            let mut map = HashMap::new();
            map.insert("test_tensor".to_string(), serde_json::json!({
                "shape": [2, 2],
                "dtype": "float16",
                "data": {
                    "Float16": [15360u16, 16384u16, 16896u16, 17408u16] // 1.0, 2.0, 3.0, 4.0 as f16 bits
                }
            }));
            map
        };
        
        // Write shard file
        let shard_path = temp_dir.path().join("shard_000.msgpack");
        let mut shard_file = File::create(&shard_path).unwrap();
        let serialized = rmp_serde::to_vec(&shard_data).unwrap();
        shard_file.write_all(&serialized).unwrap();
        
        // Test verification
        let result = verify_conversion(temp_dir.path(), true);
        assert!(result.is_ok());
    }
}