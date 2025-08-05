use crate::error::{TensorportError, TensorportResult};
use crate::formats::{ModelIndex, SafetensorsReader, OutputFormat};
use crate::shard::{ConversionResult, ShardWriter};
use indicatif::{ProgressBar, ProgressStyle};
use std::fs::File;
use std::path::PathBuf;

pub struct TensorportConverter {
    input_path: PathBuf,
    output_path: PathBuf,
    shard_size_gb: f64,
    #[allow(dead_code)]
    workers: usize,
    precision: String,
    #[allow(dead_code)]
    output_format: OutputFormat,
}

impl TensorportConverter {
    pub fn new(
        input_path: PathBuf,
        output_path: PathBuf,
        shard_size_gb: f64,
        workers: Option<usize>,
        precision: String,
        format: String,
        resume: bool,
    ) -> TensorportResult<Self> {
        // Validate input
        if !input_path.exists() {
            return Err(TensorportError::Config(
                format!("Input path does not exist: {}", input_path.display())
            ));
        }
        
        // Validate precision
        match precision.as_str() {
            "float16" | "float32" => {}
            _ => return Err(TensorportError::Config(
                format!("Unsupported precision: {}. Use 'float16' or 'float32'", precision)
            )),
        }
        
        // Parse output format
        let output_format = OutputFormat::from_str(&format)?;
        
        let workers = workers.unwrap_or_else(|| num_cpus::get());
        
        // Handle resume logic
        if resume && output_path.exists() {
            let existing_shards: Vec<_> = std::fs::read_dir(&output_path)?
                .filter_map(|entry| {
                    let entry = entry.ok()?;
                    let path = entry.path();
                    if path.is_file() && 
                       path.file_name()?.to_str()?.starts_with("shard_") &&
                       path.extension()? == "msgpack" {
                        Some(path)
                    } else {
                        None
                    }
                })
                .collect();
            
            if !existing_shards.is_empty() {
                println!("🔄 Resume mode: Found {} existing shards", existing_shards.len());
                println!("   Continuing conversion from where it left off...");
            }
        } else if output_path.exists() && !resume {
            println!("⚠️  Output directory exists. Use --resume to continue or remove directory.");
        }
        
        println!("🔧 Configuration:");
        println!("   Input: {}", input_path.display());
        println!("   Output: {}", output_path.display());
        println!("   Format: {} ({})", format, output_format.description());
        println!("   Shard size: {:.1}GB", shard_size_gb);
        println!("   Precision: {}", precision);
        println!("   Workers: {}", workers);
        println!("   Resume: {}", resume);
        println!();
        
        Ok(TensorportConverter {
            input_path,
            output_path,
            shard_size_gb,
            workers,
            precision,
            output_format,
        })
    }
    
    pub fn convert(&self) -> TensorportResult<ConversionResult> {
        // Load model configuration
        let config = self.load_config()?;
        
        // Load model index
        let index_path = self.input_path.join("model.safetensors.index.json");
        let model_index = ModelIndex::load(&index_path)?;
        
        // Get list of safetensor files
        let safetensor_files = model_index.get_files();
        println!("📂 Found {} safetensor files", safetensor_files.len());
        
        // Initialize shard writer
        let mut shard_writer = ShardWriter::new(
            &self.output_path,
            self.shard_size_gb,
            self.precision.clone(),
        )?;
        
        // Count total tensors for progress bar
        let total_tensors = self.count_total_tensors(&model_index, &safetensor_files)?;
        
        // Setup progress bar
        let progress = ProgressBar::new(total_tensors as u64);
        progress.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} tensors ({msg})"
            ).map_err(|e| TensorportError::Template(e.to_string()))?
            .progress_chars("#>-")
        );
        
        let mut processed_tensors = 0usize;
        let mut bfloat16_count = 0usize;
        
        // Process each safetensor file
        for file_name in &safetensor_files {
            let file_path = self.input_path.join(file_name);
            let file_size_gb = file_path.metadata()?.len() as f64 / (1024.0 * 1024.0 * 1024.0);
            
            progress.set_message(format!("Processing {} ({:.1}GB)", file_name, file_size_gb));
            
            // Open safetensor file
            let reader = SafetensorsReader::new(&file_path)?;
            
            // Get tensors for this file
            let tensor_names = model_index.get_tensors_for_file(file_name);
            
            // Process each tensor
            for tensor_name in tensor_names {
                let tensor_info = reader.get_tensor_info(tensor_name)?;
                
                // Track bfloat16 usage
                if tensor_info.dtype == "BF16" {
                    bfloat16_count += 1;
                }
                
                // Read and convert tensor
                let tensor = reader.read_tensor(tensor_name, &self.precision)?;
                
                // Add to shard writer
                shard_writer.add_tensor(tensor)?;
                
                processed_tensors += 1;
                progress.inc(1);
            }
        }
        
        progress.finish_with_message("Finalizing shards...");
        
        // Finalize conversion
        let result = shard_writer.finalize(config)?;
        
        println!("\n📊 Conversion Statistics:");
        println!("   Total tensors: {}", processed_tensors);
        println!("   BFloat16 tensors: {}", bfloat16_count);
        println!("   Regular tensors: {}", processed_tensors - bfloat16_count);
        
        Ok(result)
    }
    
    fn load_config(&self) -> TensorportResult<serde_json::Value> {
        let config_path = self.input_path.join("config.json");
        
        if !config_path.exists() {
            return Ok(serde_json::json!({}));
        }
        
        let config_file = File::open(config_path)?;
        let config: serde_json::Value = serde_json::from_reader(config_file)?;
        
        Ok(config)
    }
    
    fn count_total_tensors(
        &self,
        model_index: &ModelIndex,
        safetensor_files: &[String],
    ) -> TensorportResult<usize> {
        let mut total = 0;
        
        for file_name in safetensor_files {
            let tensor_names = model_index.get_tensors_for_file(file_name);
            total += tensor_names.len();
        }
        
        Ok(total)
    }
}

// For future parallelization (backlog item)
#[allow(dead_code)]
struct ParallelTensorProcessor {
    workers: usize,
    precision: String,
}

#[allow(dead_code)]
impl ParallelTensorProcessor {
    fn new(workers: usize, precision: String) -> Self {
        Self { workers, precision }
    }
    
    // TODO: Implement parallel tensor processing with memory boundaries
    // This would process multiple tensors concurrently while respecting memory limits
    // Key considerations:
    // - Memory budget per worker
    // - Load balancing across workers  
    // - Coordination with shard writer
    // - Error handling across threads
    fn process_tensors_parallel(&self, _tensors: Vec<String>) -> TensorportResult<Vec<crate::tensor::Tensor>> {
        todo!("Implement parallel processing with memory boundaries")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_converter_validation() {
        let temp_dir = TempDir::new().unwrap();
        let non_existent = temp_dir.path().join("nonexistent");
        let output = temp_dir.path().join("output");
        
        // Test non-existent input
        let result = TensorportConverter::new(
            non_existent,
            output.clone(),
            1.8,
            None,
            "float16".to_string(),
            "jax".to_string(),
            false,
        );
        assert!(result.is_err());
        
        // Test invalid precision
        let result = TensorportConverter::new(
            temp_dir.path().to_path_buf(),
            output,
            1.8,
            None,
            "invalid".to_string(),
            "jax".to_string(),
            false,
        );
        assert!(result.is_err());
    }
}