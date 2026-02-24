//! Tree indexer - generates hierarchical tree structure from documents.
//!
//! This module implements the core PageIndex algorithm:
//! 1. Optionally detect table of contents in the document
//! 2. Generate hierarchical tree structure using LLM
//! 3. Map sections to physical page indices
//! 4. Verify and correct page mappings

use crate::document::Document;
use crate::error::{PageIndexError, Result};
use crate::llm::{LlmClient, Prompts};
use crate::tree::{build_tree_from_toc, DocumentTree, RawTocItem};

/// Options for tree index generation.
#[derive(Debug, Clone)]
pub struct IndexerOptions {
    /// Maximum tokens per LLM request chunk.
    pub max_tokens_per_chunk: usize,
    /// Whether to verify page indices after generation.
    pub verify_indices: bool,
    /// Maximum attempts to fix incorrect indices.
    pub max_fix_attempts: usize,
}

impl Default for IndexerOptions {
    fn default() -> Self {
        Self {
            max_tokens_per_chunk: 20000,
            verify_indices: true,
            max_fix_attempts: 3,
        }
    }
}

/// Tree indexer that uses LLM to build document structure.
pub struct TreeIndexer {
    client: LlmClient,
    #[allow(dead_code)]
    options: IndexerOptions,
}

impl TreeIndexer {
    /// Create a new tree indexer.
    pub fn new(client: LlmClient) -> Self {
        Self {
            client,
            options: IndexerOptions::default(),
        }
    }

    /// Create with custom options.
    pub fn with_options(client: LlmClient, options: IndexerOptions) -> Self {
        Self { client, options }
    }

    /// Build a tree index for a document.
    pub async fn index(&self, document: &Document) -> Result<DocumentTree> {
        // For MVP: Generate tree structure directly (no TOC detection)
        // This is the "process_no_toc" path from the Python implementation

        let content = document.content_with_tags();
        let toc_items = self.generate_toc_init(&content).await?;

        // Build tree structure from flat TOC items
        let nodes = build_tree_from_toc(&toc_items, document.page_count());

        let tree = DocumentTree::new(&document.name, nodes, document.page_count());

        Ok(tree)
    }

    /// Generate initial TOC/structure from document content.
    async fn generate_toc_init(&self, content: &str) -> Result<Vec<RawTocItem>> {
        let prompt = format!(
            "{}\nGiven text\n:{}",
            Prompts::generate_toc_init(),
            content
        );

        let response = self
            .client
            .complete(Some(Prompts::system_document_analyzer()), &prompt)
            .await?;

        // Parse response as JSON array of TOC items
        let items = Self::parse_toc_response(&response)?;

        Ok(items)
    }

    /// Parse LLM response into TOC items.
    fn parse_toc_response(response: &str) -> Result<Vec<RawTocItem>> {
        // Try to extract JSON from response (may have markdown code blocks)
        let json_str = Self::extract_json(response);

        // Try parsing as array directly
        if let Ok(items) = serde_json::from_str::<Vec<RawTocItem>>(&json_str) {
            return Ok(items);
        }

        // Try parsing as object with table_of_contents field
        #[derive(serde::Deserialize)]
        struct TocWrapper {
            table_of_contents: Vec<RawTocItem>,
        }

        if let Ok(wrapper) = serde_json::from_str::<TocWrapper>(&json_str) {
            return Ok(wrapper.table_of_contents);
        }

        Err(PageIndexError::LlmParse(format!(
            "Failed to parse TOC response: {}",
            &response[..response.len().min(200)]
        )))
    }

    /// Extract JSON from potentially markdown-wrapped response.
    fn extract_json(response: &str) -> String {
        let response = response.trim();

        // Check for ```json code block
        if response.starts_with("```json") {
            if let Some(end) = response.rfind("```") {
                let start = "```json".len();
                if end > start {
                    return response[start..end].trim().to_string();
                }
            }
        }

        // Check for ``` code block
        if response.starts_with("```") {
            if let Some(end) = response.rfind("```") {
                let start = response.find('\n').map(|n| n + 1).unwrap_or(3);
                if end > start {
                    return response[start..end].trim().to_string();
                }
            }
        }

        // Find JSON array or object
        if let Some(start) = response.find('[') {
            if let Some(end) = response.rfind(']') {
                if end > start {
                    return response[start..=end].to_string();
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

    /// Continue generating TOC for additional document parts.
    #[allow(dead_code)]
    async fn generate_toc_continue(
        &self,
        previous_toc: &[RawTocItem],
        content: &str,
    ) -> Result<Vec<RawTocItem>> {
        let prompt = format!(
            "{}\nGiven text\n:{}\nPrevious tree structure\n:{}",
            Prompts::generate_toc_continue(),
            content,
            serde_json::to_string_pretty(previous_toc)
                .map_err(|e| PageIndexError::Serialization(e.to_string()))?
        );

        let response = self
            .client
            .complete(Some(Prompts::system_document_analyzer()), &prompt)
            .await?;

        Self::parse_toc_response(&response)
    }

    /// Verify that section titles appear on their claimed pages.
    #[allow(dead_code)]
    async fn verify_title_on_page(
        &self,
        title: &str,
        page_content: &str,
    ) -> Result<bool> {
        let prompt = Prompts::check_title_appearance()
            .replace("{title}", title)
            .replace("{page_text}", page_content);

        let response = self
            .client
            .complete(Some(Prompts::system_document_analyzer()), &prompt)
            .await?;

        // Parse response
        let json_str = Self::extract_json(&response);

        #[derive(serde::Deserialize)]
        struct VerifyResponse {
            answer: String,
        }

        if let Ok(parsed) = serde_json::from_str::<VerifyResponse>(&json_str) {
            Ok(parsed.answer.to_lowercase() == "yes")
        } else {
            // Default to false if parsing fails
            Ok(false)
        }
    }
}

/// Convenience function to index a document from a file path.
pub async fn index_document(
    path: &std::path::Path,
    client: LlmClient,
) -> Result<DocumentTree> {
    let document = Document::from_text_file(path)?;
    let indexer = TreeIndexer::new(client);
    indexer.index(&document).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_json_plain() {
        let response = r#"[{"title": "Test"}]"#;
        let extracted = TreeIndexer::extract_json(response);
        assert_eq!(extracted, r#"[{"title": "Test"}]"#);
    }

    #[test]
    fn test_extract_json_markdown() {
        let response = r#"```json
[{"title": "Test"}]
```"#;
        let extracted = TreeIndexer::extract_json(response);
        assert_eq!(extracted, r#"[{"title": "Test"}]"#);
    }

    #[test]
    fn test_extract_json_with_text() {
        let response = r#"Here's the structure:
[{"title": "Test"}]
That's the result."#;
        let extracted = TreeIndexer::extract_json(response);
        assert_eq!(extracted, r#"[{"title": "Test"}]"#);
    }

    #[test]
    fn test_parse_toc_response_array() {
        let response = r#"[
            {"structure": "1", "title": "Chapter 1", "physical_index": 1},
            {"structure": "2", "title": "Chapter 2", "physical_index": 10}
        ]"#;

        let items = TreeIndexer::parse_toc_response(response).unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].title, "Chapter 1");
    }

    #[test]
    fn test_parse_toc_response_wrapped() {
        let response = r#"{
            "table_of_contents": [
                {"structure": "1", "title": "Chapter 1", "page": 1}
            ]
        }"#;

        let items = TreeIndexer::parse_toc_response(response).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].title, "Chapter 1");
    }

    #[test]
    fn test_indexer_options_default() {
        let options = IndexerOptions::default();
        assert_eq!(options.max_tokens_per_chunk, 20000);
        assert!(options.verify_indices);
        assert_eq!(options.max_fix_attempts, 3);
    }
}
