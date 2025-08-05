pub mod safetensors;
pub mod jax_pytree;
pub mod numpy_arrays;
pub mod msgpack_shards;

pub use safetensors::{SafetensorsReader, ModelIndex};

use crate::error::TensorportResult;

/// Supported output formats for TensorPort
#[derive(Debug, Clone)]
pub enum OutputFormat {
    /// JAX PyTree format (pickle with proper structure)
    JaxPyTree,
    /// Individual NumPy arrays (compatible with JAX loading)
    NumpyArrays,
    /// MessagePack shards (custom format, fast loading)
    MessagePackShards,
    /// Safetensors format (for Candle/Rust inference)
    Safetensors,
}

impl OutputFormat {
    pub fn from_str(s: &str) -> TensorportResult<Self> {
        match s.to_lowercase().as_str() {
            "jax" | "jax-pytree" | "pytree" => Ok(OutputFormat::JaxPyTree),
            "numpy" | "npy" | "numpy-arrays" => Ok(OutputFormat::NumpyArrays),
            "msgpack" | "msgpack-shards" => Ok(OutputFormat::MessagePackShards),
            "safetensors" | "candle" => Ok(OutputFormat::Safetensors),
            _ => Err(crate::error::TensorportError::Config(
                format!("Unsupported output format: {}. Supported formats: jax, numpy, msgpack, safetensors", s)
            )),
        }
    }
    
    pub fn recommended_extension(&self) -> &'static str {
        match self {
            OutputFormat::JaxPyTree => "pkl",
            OutputFormat::NumpyArrays => "npy",
            OutputFormat::MessagePackShards => "msgpack",
            OutputFormat::Safetensors => "safetensors",
        }
    }
    
    pub fn description(&self) -> &'static str {
        match self {
            OutputFormat::JaxPyTree => "JAX-compatible PyTree structure (pickle format)",
            OutputFormat::NumpyArrays => "Individual NumPy arrays (JAX-loadable)",
            OutputFormat::MessagePackShards => "MessagePack shards (fast custom format)",
            OutputFormat::Safetensors => "Safetensors format (Candle/Rust compatible)",
        }
    }
}