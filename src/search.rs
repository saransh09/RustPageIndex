//! Tree-based search using LLM reasoning.
//!
//! This module implements the search functionality of PageIndex:
//! given a query and a document tree, use LLM reasoning to find
//! the most relevant sections.

use crate::document::Document;
use crate::error::{PageIndexError, Result};
use crate::llm::{LlmClient, Prompts};
use crate::tree::DocumentTree;
use serde::{Deserialize, Serialize};

/// A search result from tree-based search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Section title.
    pub title: String,
    /// Starting page index.
    pub start_index: usize,
    /// Ending page index.
    pub end_index: usize,
    /// Relevance level.
    pub relevance: Relevance,
    /// Reason for relevance.
    pub reason: String,
    /// Extracted content from the section (if available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// Relevance level for search results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Relevance {
    High,
    Medium,
    Low,
}

impl Relevance {
    /// Convert from string (case-insensitive).
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "high" => Relevance::High,
            "medium" => Relevance::Medium,
            _ => Relevance::Low,
        }
    }

    /// Get numeric score for sorting.
    pub fn score(&self) -> u8 {
        match self {
            Relevance::High => 3,
            Relevance::Medium => 2,
            Relevance::Low => 1,
        }
    }
}

/// Search options.
#[derive(Debug, Clone)]
pub struct SearchOptions {
    /// Maximum number of results to return.
    pub top_k: usize,
    /// Minimum relevance level to include.
    pub min_relevance: Relevance,
    /// Whether to include section content in results.
    pub include_content: bool,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            top_k: 10,
            min_relevance: Relevance::Low,
            include_content: false,
        }
    }
}

/// Tree-based searcher using LLM reasoning.
pub struct TreeSearcher {
    client: LlmClient,
    options: SearchOptions,
}

impl TreeSearcher {
    /// Create a new tree searcher.
    pub fn new(client: LlmClient) -> Self {
        Self {
            client,
            options: SearchOptions::default(),
        }
    }

    /// Create with custom options.
    pub fn with_options(client: LlmClient, options: SearchOptions) -> Self {
        Self { client, options }
    }

    /// Search the document tree for relevant sections.
    pub async fn search(
        &self,
        tree: &DocumentTree,
        query: &str,
    ) -> Result<Vec<SearchResult>> {
        let tree_json = tree
            .to_json()
            .map_err(|e| PageIndexError::Serialization(e.to_string()))?;

        let prompt = Prompts::tree_search()
            .replace("{tree_structure}", &tree_json)
            .replace("{query}", query);

        let response = self
            .client
            .complete(Some(Prompts::system_document_analyzer()), &prompt)
            .await?;

        let mut results = self.parse_search_response(&response)?;

        // Filter by minimum relevance
        results.retain(|r| r.relevance.score() >= self.options.min_relevance.score());

        // Sort by relevance (high to low)
        results.sort_by(|a, b| b.relevance.score().cmp(&a.relevance.score()));

        // Limit to top_k
        results.truncate(self.options.top_k);

        Ok(results)
    }

    /// Search and include content from the document.
    pub async fn search_with_content(
        &self,
        tree: &DocumentTree,
        document: &Document,
        query: &str,
    ) -> Result<Vec<SearchResult>> {
        let mut results = self.search(tree, query).await?;

        // Add content for each result
        for result in &mut results {
            let content = document.content_range(result.start_index, result.end_index);
            // Strip the index tags for cleaner output
            let clean_content = content
                .lines()
                .filter(|line| !line.starts_with("<physical_index_"))
                .collect::<Vec<_>>()
                .join("\n")
                .trim()
                .to_string();

            result.content = Some(clean_content);
        }

        Ok(results)
    }

    /// Parse LLM search response into results.
    fn parse_search_response(&self, response: &str) -> Result<Vec<SearchResult>> {
        let json_str = Self::extract_json(response);

        #[derive(Deserialize)]
        struct SearchResponse {
            #[serde(default)]
            #[allow(dead_code)]
            thinking: Option<String>,
            relevant_sections: Vec<RawSearchResult>,
        }

        #[derive(Deserialize)]
        struct RawSearchResult {
            title: String,
            start_index: usize,
            end_index: usize,
            relevance: String,
            reason: String,
        }

        let parsed: SearchResponse = serde_json::from_str(&json_str).map_err(|e| {
            PageIndexError::LlmParse(format!(
                "Failed to parse search response: {}. Response: {}",
                e,
                &response[..response.len().min(200)]
            ))
        })?;

        let results = parsed
            .relevant_sections
            .into_iter()
            .map(|r| SearchResult {
                title: r.title,
                start_index: r.start_index,
                end_index: r.end_index,
                relevance: Relevance::from_str(&r.relevance),
                reason: r.reason,
                content: None,
            })
            .collect();

        Ok(results)
    }

    /// Extract JSON from response (same logic as indexer).
    fn extract_json(response: &str) -> String {
        let response = response.trim();

        if response.starts_with("```json") {
            if let Some(end) = response.rfind("```") {
                let start = "```json".len();
                if end > start {
                    return response[start..end].trim().to_string();
                }
            }
        }

        if response.starts_with("```") {
            if let Some(end) = response.rfind("```") {
                let start = response.find('\n').map(|n| n + 1).unwrap_or(3);
                if end > start {
                    return response[start..end].trim().to_string();
                }
            }
        }

        if let Some(start) = response.find('{') {
            if let Some(end) = response.rfind('}') {
                if end > start {
                    return response[start..=end].to_string();
                }
            }
        }

        response.to_string()
    }
}

/// Convenience function to search a document tree.
pub async fn search_tree(
    tree: &DocumentTree,
    query: &str,
    client: LlmClient,
) -> Result<Vec<SearchResult>> {
    let searcher = TreeSearcher::new(client);
    searcher.search(tree, query).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relevance_ordering() {
        assert!(Relevance::High.score() > Relevance::Medium.score());
        assert!(Relevance::Medium.score() > Relevance::Low.score());
    }

    #[test]
    fn test_relevance_from_str() {
        assert_eq!(Relevance::from_str("HIGH"), Relevance::High);
        assert_eq!(Relevance::from_str("medium"), Relevance::Medium);
        assert_eq!(Relevance::from_str("Low"), Relevance::Low);
        assert_eq!(Relevance::from_str("unknown"), Relevance::Low);
    }

    #[test]
    fn test_search_options_default() {
        let options = SearchOptions::default();
        assert_eq!(options.top_k, 10);
        assert_eq!(options.min_relevance, Relevance::Low);
        assert!(!options.include_content);
    }

    #[test]
    fn test_extract_json() {
        let response = r#"{"thinking": "...", "relevant_sections": []}"#;
        let extracted = TreeSearcher::extract_json(response);
        assert!(extracted.contains("relevant_sections"));
    }
}
