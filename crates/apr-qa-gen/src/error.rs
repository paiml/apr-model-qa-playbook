//! Error types for apr-qa-gen

use thiserror::Error;

/// Result type alias for apr-qa-gen operations
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur during scenario generation
#[derive(Debug, Error)]
pub enum Error {
    /// Model not found in registry
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    /// Invalid scenario configuration
    #[error("Invalid scenario: {0}")]
    InvalidScenario(String),

    /// Oracle evaluation failed
    #[error("Oracle error: {0}")]
    OracleError(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// YAML serialization error
    #[error("YAML error: {0}")]
    YamlError(#[from] serde_yaml::Error),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = Error::ModelNotFound("test-model".to_string());
        assert_eq!(err.to_string(), "Model not found: test-model");
    }

    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: Error = io_err.into();
        assert!(matches!(err, Error::IoError(_)));
    }
}
