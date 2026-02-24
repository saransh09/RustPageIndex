//! Document representation for PageIndex.
//!
//! Documents are represented as a collection of pages, where each page
//! has content and a page number. For text files, the entire content
//! is treated as a single page.

use crate::error::{PageIndexError, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// A single page in a document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Page {
    /// 1-indexed page number.
    pub number: usize,
    /// Text content of the page.
    pub content: String,
    /// Approximate token count (words / 0.75).
    pub token_count: usize,
}

impl Page {
    /// Create a new page.
    pub fn new(number: usize, content: String) -> Self {
        let token_count = estimate_tokens(&content);
        Self {
            number,
            content,
            token_count,
        }
    }

    /// Format page content with physical index tags for LLM processing.
    pub fn with_index_tags(&self) -> String {
        format!(
            "<physical_index_{}>\n{}\n<physical_index_{}>\n\n",
            self.number, self.content, self.number
        )
    }
}

/// A document consisting of one or more pages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Document name/title.
    pub name: String,
    /// Original file path (if loaded from file).
    pub path: Option<PathBuf>,
    /// Pages in the document.
    pub pages: Vec<Page>,
}

impl Document {
    /// Create a new document with given name and pages.
    pub fn new(name: impl Into<String>, pages: Vec<Page>) -> Self {
        Self {
            name: name.into(),
            path: None,
            pages,
        }
    }

    /// Load a text file as a single-page document.
    pub fn from_text_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| PageIndexError::io(path, e))?;

        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("untitled")
            .to_string();

        let pages = vec![Page::new(1, content)];

        Ok(Self {
            name,
            path: Some(path.to_path_buf()),
            pages,
        })
    }

    /// Load a text file with custom page delimiter.
    ///
    /// Splits content on the delimiter and treats each part as a page.
    pub fn from_text_file_with_delimiter(path: &Path, delimiter: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| PageIndexError::io(path, e))?;

        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("untitled")
            .to_string();

        let pages: Vec<Page> = content
            .split(delimiter)
            .enumerate()
            .filter(|(_, s)| !s.trim().is_empty())
            .map(|(i, s)| Page::new(i + 1, s.to_string()))
            .collect();

        if pages.is_empty() {
            return Err(PageIndexError::DocumentNotFound(path.to_path_buf()));
        }

        Ok(Self {
            name,
            path: Some(path.to_path_buf()),
            pages,
        })
    }

    /// Create a document from raw text content.
    pub fn from_text(name: impl Into<String>, content: String) -> Self {
        let pages = vec![Page::new(1, content)];
        Self {
            name: name.into(),
            path: None,
            pages,
        }
    }

    /// Get total number of pages.
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Get total token count across all pages.
    pub fn total_tokens(&self) -> usize {
        self.pages.iter().map(|p| p.token_count).sum()
    }

    /// Get a specific page by number (1-indexed).
    pub fn get_page(&self, number: usize) -> Option<&Page> {
        if number == 0 || number > self.pages.len() {
            None
        } else {
            Some(&self.pages[number - 1])
        }
    }

    /// Get all page content concatenated with index tags.
    pub fn content_with_tags(&self) -> String {
        self.pages
            .iter()
            .map(|p| p.with_index_tags())
            .collect::<Vec<_>>()
            .join("")
    }

    /// Get content for a range of pages (1-indexed, inclusive).
    pub fn content_range(&self, start: usize, end: usize) -> String {
        self.pages
            .iter()
            .filter(|p| p.number >= start && p.number <= end)
            .map(|p| p.with_index_tags())
            .collect::<Vec<_>>()
            .join("")
    }

    /// Get raw content without index tags.
    pub fn raw_content(&self) -> String {
        self.pages
            .iter()
            .map(|p| p.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

/// Estimate token count from text (rough approximation: words / 0.75).
fn estimate_tokens(text: &str) -> usize {
    let word_count = text.split_whitespace().count();
    (word_count as f64 / 0.75) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_creation() {
        let page = Page::new(1, "Hello world, this is a test.".to_string());
        assert_eq!(page.number, 1);
        assert!(!page.content.is_empty());
        assert!(page.token_count > 0);
    }

    #[test]
    fn test_page_with_index_tags() {
        let page = Page::new(5, "Test content".to_string());
        let tagged = page.with_index_tags();
        assert!(tagged.contains("<physical_index_5>"));
        assert!(tagged.contains("Test content"));
    }

    #[test]
    fn test_document_from_text() {
        let doc = Document::from_text("Test Doc", "This is the content.".to_string());
        assert_eq!(doc.name, "Test Doc");
        assert_eq!(doc.page_count(), 1);
        assert!(doc.path.is_none());
    }

    #[test]
    fn test_document_page_access() {
        let doc = Document::from_text("Test", "Content".to_string());

        assert!(doc.get_page(0).is_none()); // 0 is invalid
        assert!(doc.get_page(1).is_some()); // Valid
        assert!(doc.get_page(2).is_none()); // Out of range
    }

    #[test]
    fn test_estimate_tokens() {
        let text = "one two three four five six seven eight";
        let tokens = estimate_tokens(text);
        // 8 words / 0.75 ≈ 10-11 tokens
        assert!(tokens >= 10 && tokens <= 12);
    }
}
