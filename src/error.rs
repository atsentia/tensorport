use thiserror::Error;

pub type TensorportResult<T> = Result<T, TensorportError>;

#[derive(Error, Debug)]
pub enum TensorportError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("JSON parsing error: {0}")]
    Json(#[from] serde_json::Error),
    
    #[error("MessagePack error: {0}")]
    MessagePack(#[from] rmp_serde::encode::Error),
    
    #[error("Invalid tensor data: {0}")]
    InvalidTensorData(String),
    
    #[error("Unsupported data type: {0}")]
    UnsupportedDataType(String),
    
    #[error("File format error: {0}")]
    FileFormat(String),
    
    #[error("Configuration error: {0}")]
    Config(String),
    
    #[error("Verification failed: {0}")]
    Verification(String),
    
    #[error("Memory allocation error: {0}")]
    Memory(String),
    
    #[error("Template error: {0}")]
    Template(String),
}

impl From<anyhow::Error> for TensorportError {
    fn from(err: anyhow::Error) -> Self {
        TensorportError::Config(err.to_string())
    }
}