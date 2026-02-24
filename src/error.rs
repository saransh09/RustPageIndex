//! Error types for the page indexer.

use std::path::PathBuf;
use thiserror::Error;

/// Result type alias using our custom error.
pub type Result<T> = std::result::Result<T, PageIndexError>;

/// Errors that can occur in the page indexer.
#[derive(Error, Debug)]
pub enum PageIndexError {
    /// Error reading or writing files.
    #[error("I/O error for path '{path}': {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// Error during serialization/deserialization.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// The document path does not exist.
    #[error("Document not found at '{0}'")]
    DocumentNotFound(PathBuf),

    /// The corpus directory does not exist or is not a directory.
    #[error("Corpus path '{0}' does not exist or is not a directory")]
    InvalidCorpusPath(PathBuf),

    /// No documents found in the corpus.
    #[error("No documents found in corpus at '{0}'")]
    EmptyCorpus(PathBuf),

    /// The index file does not exist.
    #[error("Index file not found at '{0}'")]
    IndexNotFound(PathBuf),

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// LLM API error.
    #[error("LLM API error: {0}")]
    LlmApi(String),

    /// LLM response parsing error.
    #[error("Failed to parse LLM response: {0}")]
    LlmParse(String),

    /// HTTP request error.
    #[error("HTTP request failed: {0}")]
    Http(String),

    /// Configuration file error.
    #[error("Configuration error: {0}")]
    Config(String),

    /// Tree structure error.
    #[error("Tree structure error: {0}")]
    TreeError(String),
}

impl PageIndexError {
    /// Create an I/O error with path context.
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}

impl From<reqwest::Error> for PageIndexError {
    fn from(err: reqwest::Error) -> Self {
        PageIndexError::Http(err.to_string())
    }
}

impl From<serde_json::Error> for PageIndexError {
    fn from(err: serde_json::Error) -> Self {
        PageIndexError::LlmParse(err.to_string())
    }
}
