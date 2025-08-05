pub mod safetensors;
pub mod jax_pytree;
pub mod numpy_arrays;
pub mod msgpack_shards;
pub mod jax_pickle;
pub mod numpy_direct;

pub use safetensors::{SafetensorsReader, ModelIndex};
pub use jax_pickle::JaxPickleWriter;
pub use numpy_direct::NumpyDirectWriter;

use crate::error::TensorportResult;

/// Supported output formats for TensorPort
#[derive(Debug, Clone)]
pub enum OutputFormat {
    /// JAX PyTree format (pickle with proper structure) - DIRECT FROM RUST
    JaxPyTree,
    /// Individual NumPy arrays (compatible with JAX loading) - DIRECT FROM RUST
    NumpyArrays,
    /// MessagePack shards (custom format, fast loading)
    MessagePackShards,
    /// Safetensors format (for Candle/Rust inference)
    Safetensors,
    /// JAX Pickle format (direct Rust → JAX, no Python needed)
    JaxPickle,
    /// NumPy Direct - writes .npy files directly for JAX
    NumpyDirect,
}

impl OutputFormat {
    pub fn from_str(s: &str) -> TensorportResult<Self> {
        match s.to_lowercase().as_str() {
            "jax" | "jax-pytree" | "pytree" => Ok(OutputFormat::JaxPyTree),
            "numpy" | "npy" | "numpy-arrays" => Ok(OutputFormat::NumpyArrays),
            "msgpack" | "msgpack-shards" => Ok(OutputFormat::MessagePackShards),
            "safetensors" | "candle" => Ok(OutputFormat::Safetensors),
            "jax-pickle" | "pickle" | "jax-direct" => Ok(OutputFormat::JaxPickle),
            "numpy-direct" | "npy-direct" => Ok(OutputFormat::NumpyDirect),
            _ => Err(crate::error::TensorportError::Config(
                format!("Unsupported output format: {}. Supported formats: jax, numpy, msgpack, safetensors, jax-pickle, numpy-direct", s)
            )),
        }
    }
    
    pub fn recommended_extension(&self) -> &'static str {
        match self {
            OutputFormat::JaxPyTree => "pkl",
            OutputFormat::NumpyArrays => "npy",
            OutputFormat::MessagePackShards => "msgpack",
            OutputFormat::Safetensors => "safetensors",
            OutputFormat::JaxPickle => "pkl",
            OutputFormat::NumpyDirect => "npy",
        }
    }
    
    pub fn description(&self) -> &'static str {
        match self {
            OutputFormat::JaxPyTree => "JAX-compatible PyTree structure (pickle format)",
            OutputFormat::NumpyArrays => "Individual NumPy arrays (JAX-loadable)",
            OutputFormat::MessagePackShards => "MessagePack shards (fast custom format)",
            OutputFormat::Safetensors => "Safetensors format (Candle/Rust compatible)",
            OutputFormat::JaxPickle => "Direct JAX pickle format (no Python conversion needed)",
            OutputFormat::NumpyDirect => "Direct NumPy arrays (.npy files) for JAX",
        }
    }
}