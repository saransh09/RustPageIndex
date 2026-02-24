//! OpenAI-compatible LLM client.
//!
//! This client works with any OpenAI-compatible API endpoint,
//! including proxies like the one at your-llm-proxy.example.com.

use crate::config::LlmConfig;
use crate::error::{PageIndexError, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// Message role in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

/// A message in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }
}

/// Request body for chat completion.
#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

/// Response from chat completion.
#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ResponseMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ResponseMessage {
    content: String,
}

#[derive(Debug, Deserialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

/// OpenAI API error response.
#[derive(Debug, Deserialize)]
struct ApiError {
    error: ApiErrorDetail,
}

#[derive(Debug, Deserialize)]
struct ApiErrorDetail {
    message: String,
    #[serde(rename = "type")]
    #[allow(dead_code)]
    error_type: Option<String>,
}

/// Response from an LLM call including metadata.
#[derive(Debug)]
pub struct LlmResponse {
    /// The generated content.
    pub content: String,
    /// Reason the model stopped generating.
    pub finish_reason: Option<String>,
    /// Token usage (if available).
    pub usage: Option<TokenUsage>,
}

#[derive(Debug)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// OpenAI-compatible LLM client.
#[derive(Clone)]
pub struct LlmClient {
    client: Client,
    config: LlmConfig,
}

impl LlmClient {
    /// Create a new LLM client with the given configuration.
    pub fn new(config: LlmConfig) -> Self {
        Self {
            client: Client::new(),
            config,
        }
    }

    /// Get the API endpoint URL.
    fn endpoint(&self) -> String {
        let base = self.config.api_base.trim_end_matches('/');
        format!("{}/v1/chat/completions", base)
    }

    /// Send a chat completion request.
    pub async fn chat(&self, messages: Vec<Message>) -> Result<LlmResponse> {
        let request = ChatCompletionRequest {
            model: self.config.model.clone(),
            messages,
            max_tokens: Some(self.config.max_tokens),
            temperature: Some(self.config.temperature),
        };

        let response = self
            .client
            .post(self.endpoint())
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        let body = response.text().await?;

        if !status.is_success() {
            // Try to parse as API error
            if let Ok(api_error) = serde_json::from_str::<ApiError>(&body) {
                return Err(PageIndexError::LlmApi(format!(
                    "API error ({}): {}",
                    status, api_error.error.message
                )));
            }
            return Err(PageIndexError::LlmApi(format!(
                "Request failed ({}): {}",
                status, body
            )));
        }

        let completion: ChatCompletionResponse = serde_json::from_str(&body)?;

        let choice = completion
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| PageIndexError::LlmApi("No choices in response".to_string()))?;

        Ok(LlmResponse {
            content: choice.message.content,
            finish_reason: choice.finish_reason,
            usage: completion.usage.map(|u| TokenUsage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
            }),
        })
    }

    /// Convenience method: single user message with optional system prompt.
    pub async fn complete(&self, system: Option<&str>, user: &str) -> Result<String> {
        let mut messages = Vec::new();

        if let Some(sys) = system {
            messages.push(Message::system(sys));
        }
        messages.push(Message::user(user));

        let response = self.chat(messages).await?;
        Ok(response.content)
    }

    /// Test connectivity to the API.
    pub async fn test_connection(&self) -> Result<()> {
        let messages = vec![Message::user("Say 'hello' and nothing else.")];

        let response = self.chat(messages).await?;

        if response.content.to_lowercase().contains("hello") {
            Ok(())
        } else {
            Err(PageIndexError::LlmApi(format!(
                "Unexpected response: {}",
                response.content
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let sys = Message::system("You are helpful.");
        let user = Message::user("Hello!");
        let assistant = Message::assistant("Hi there!");

        assert!(matches!(sys.role, Role::System));
        assert!(matches!(user.role, Role::User));
        assert!(matches!(assistant.role, Role::Assistant));
    }

    #[test]
    fn test_endpoint_construction() {
        let config = LlmConfig {
            api_base: "https://api.example.com/".to_string(),
            api_key: "test".to_string(),
            model: "gpt-4".to_string(),
            ..Default::default()
        };
        let client = LlmClient::new(config);
        assert_eq!(client.endpoint(), "https://api.example.com/v1/chat/completions");

        // Without trailing slash
        let config2 = LlmConfig {
            api_base: "https://api.example.com".to_string(),
            api_key: "test".to_string(),
            model: "gpt-4".to_string(),
            ..Default::default()
        };
        let client2 = LlmClient::new(config2);
        assert_eq!(client2.endpoint(), "https://api.example.com/v1/chat/completions");
    }
}
