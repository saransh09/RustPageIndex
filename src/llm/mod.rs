//! LLM integration module.
//!
//! Provides an OpenAI-compatible client for LLM API calls and
//! the prompts used for tree generation and search.

mod client;
mod prompts;

pub use client::LlmClient;
pub use prompts::Prompts;
