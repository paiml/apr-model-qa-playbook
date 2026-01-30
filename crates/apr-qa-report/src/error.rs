//! Error types for apr-qa-report

use thiserror::Error;

/// Result type alias
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur during report generation
#[derive(Debug, Error)]
pub enum Error {
    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// Invalid score calculation
    #[error("Invalid score calculation: {0}")]
    InvalidScore(String),

    /// Template rendering error
    #[error("Template error: {0}")]
    TemplateError(String),

    /// Gateway check failed
    #[error("Gateway {gate} failed: {reason}")]
    GatewayFailed {
        /// Gateway identifier
        gate: String,
        /// Failure reason
        reason: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = Error::GatewayFailed {
            gate: "G1".to_string(),
            reason: "Model failed to load".to_string(),
        };
        assert!(err.to_string().contains("G1"));
        assert!(err.to_string().contains("Model failed to load"));
    }

    #[test]
    fn test_invalid_score() {
        let err = Error::InvalidScore("Negative weight".to_string());
        assert!(err.to_string().contains("Negative weight"));
    }
}
