//! Configuration for the page indexer.
//!
//! Supports both environment variables and YAML config file.
//! Environment variables take precedence over config file values.

use crate::error::{PageIndexError, Result};
use serde::{Deserialize, Serialize};
use std::env;
use std::path::PathBuf;

/// LLM configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// Base URL for the LLM API (e.g., "https://api.openai.com")
    pub api_base: String,

    /// API key for authentication
    pub api_key: String,

    /// Model name (e.g., "gpt-4", "claude-3-opus")
    pub model: String,

    /// Maximum tokens for response (optional)
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,

    /// Temperature for generation (optional)
    #[serde(default = "default_temperature")]
    pub temperature: f32,
}

fn default_max_tokens() -> u32 {
    4096
}

fn default_temperature() -> f32 {
    0.0
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            api_base: String::new(),
            api_key: String::new(),
            model: "claude-latest".to_string(),
            max_tokens: default_max_tokens(),
            temperature: default_temperature(),
        }
    }
}

/// Full application configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    /// LLM settings
    pub llm: LlmConfig,
}

/// Configuration file structure (YAML format).
#[derive(Debug, Deserialize)]
struct ConfigFile {
    llm: Option<LlmFileSection>,
}

#[derive(Debug, Deserialize)]
struct LlmFileSection {
    api_base: Option<String>,
    api_key: Option<String>,
    model: Option<String>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
}

impl Config {
    /// Load configuration from environment variables and optional config file.
    ///
    /// Priority (highest to lowest):
    /// 1. Environment variables (LLM_API_BASE, LLM_API_KEY, LLM_MODEL)
    /// 2. Config file (~/.config/rust-page-indexer/config.yaml)
    /// 3. Default values
    pub fn load() -> Result<Self> {
        let mut config = Config::default();

        // Try to load from config file first
        if let Some(config_path) = Self::config_file_path() {
            if config_path.exists() {
                config = Self::load_from_file(&config_path)?;
            }
        }

        // Override with environment variables
        if let Ok(api_base) = env::var("LLM_API_BASE") {
            config.llm.api_base = api_base;
        }

        if let Ok(api_key) = env::var("LLM_API_KEY") {
            config.llm.api_key = api_key;
        }

        if let Ok(model) = env::var("LLM_MODEL") {
            config.llm.model = model;
        }

        if let Ok(max_tokens) = env::var("LLM_MAX_TOKENS") {
            if let Ok(tokens) = max_tokens.parse() {
                config.llm.max_tokens = tokens;
            }
        }

        if let Ok(temperature) = env::var("LLM_TEMPERATURE") {
            if let Ok(temp) = temperature.parse() {
                config.llm.temperature = temp;
            }
        }

        Ok(config)
    }

    /// Load configuration from a specific file path.
    pub fn load_from_file(path: &PathBuf) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| PageIndexError::io(path, e))?;

        let file_config: ConfigFile = serde_yaml::from_str(&content)
            .map_err(|e| PageIndexError::Config(format!("Failed to parse config file: {}", e)))?;

        let mut config = Config::default();

        if let Some(llm) = file_config.llm {
            if let Some(api_base) = llm.api_base {
                config.llm.api_base = api_base;
            }
            if let Some(api_key) = llm.api_key {
                config.llm.api_key = api_key;
            }
            if let Some(model) = llm.model {
                config.llm.model = model;
            }
            if let Some(max_tokens) = llm.max_tokens {
                config.llm.max_tokens = max_tokens;
            }
            if let Some(temperature) = llm.temperature {
                config.llm.temperature = temperature;
            }
        }

        Ok(config)
    }

    /// Get the default config file path.
    pub fn config_file_path() -> Option<PathBuf> {
        directories::ProjectDirs::from("", "", "rust-page-indexer")
            .map(|dirs| dirs.config_dir().join("config.yaml"))
    }

    /// Validate that required configuration is present.
    pub fn validate(&self) -> Result<()> {
        if self.llm.api_base.is_empty() {
            return Err(PageIndexError::Config(
                "LLM API base URL is required. Set LLM_API_BASE environment variable or add to config file.".to_string()
            ));
        }

        if self.llm.api_key.is_empty() {
            return Err(PageIndexError::Config(
                "LLM API key is required. Set LLM_API_KEY environment variable or add to config file.".to_string()
            ));
        }

        if self.llm.model.is_empty() {
            return Err(PageIndexError::Config(
                "LLM model is required. Set LLM_MODEL environment variable or add to config file."
                    .to_string(),
            ));
        }

        Ok(())
    }

    /// Create a config from explicit values (useful for testing).
    pub fn with_llm(
        api_base: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            llm: LlmConfig {
                api_base: api_base.into(),
                api_key: api_key.into(),
                model: model.into(),
                ..Default::default()
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert!(config.llm.api_base.is_empty());
        assert!(config.llm.api_key.is_empty());
        assert_eq!(config.llm.model, "claude-latest");
        assert_eq!(config.llm.max_tokens, 4096);
        assert_eq!(config.llm.temperature, 0.0);
    }

    #[test]
    fn test_validate_fails_without_required_fields() {
        let config = Config::default();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_with_llm() {
        let config = Config::with_llm("https://api.example.com", "test-key", "gpt-4");
        assert_eq!(config.llm.api_base, "https://api.example.com");
        assert_eq!(config.llm.api_key, "test-key");
        assert_eq!(config.llm.model, "gpt-4");
    }
}
