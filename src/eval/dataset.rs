//! Dataset loading for evaluation benchmarks.
//!
//! Supports:
//! - QuALITY (Question Answering with Long Input Texts, Yes!)
//! - Custom JSON format for user-provided datasets

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// A single evaluation item with a document and question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetItem {
    /// Unique identifier for this item.
    pub id: String,
    /// The document/article text.
    pub document: String,
    /// The question to answer.
    pub question: String,
    /// Ground truth answer (if available).
    pub answer: Option<String>,
    /// Multiple choice options (if applicable).
    pub options: Option<Vec<String>>,
    /// Index of correct option (0-indexed, if applicable).
    pub correct_option: Option<usize>,
    /// Source dataset name.
    pub source: String,
}

/// A collection of evaluation items.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    /// Dataset name.
    pub name: String,
    /// Dataset items.
    pub items: Vec<DatasetItem>,
}

impl Dataset {
    /// Create a new empty dataset.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            items: Vec::new(),
        }
    }

    /// Add an item to the dataset.
    pub fn add_item(&mut self, item: DatasetItem) {
        self.items.push(item);
    }

    /// Number of items in the dataset.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Check if dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Get a subset of items (for quick testing).
    pub fn take(&self, n: usize) -> Self {
        Self {
            name: self.name.clone(),
            items: self.items.iter().take(n).cloned().collect(),
        }
    }

    /// Load from a JSON file.
    pub fn load_json(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read dataset file: {:?}", path))?;
        let dataset: Dataset =
            serde_json::from_str(&content).with_context(|| "Failed to parse dataset JSON")?;
        Ok(dataset)
    }

    /// Save to a JSON file.
    pub fn save_json(&self, path: &Path) -> Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }
}

/// QuALITY dataset format (from NYU official source).
/// Structure: https://github.com/nyu-mll/quality
#[derive(Debug, Deserialize)]
struct QualityRawArticle {
    /// Article text.
    article: String,
    /// Article ID.
    article_id: String,
    /// Set ID.
    set_unique_id: String,
    /// Questions for this article.
    questions: Vec<QualityRawQuestion>,
}

#[derive(Debug, Deserialize)]
struct QualityRawQuestion {
    /// Question text.
    question: String,
    /// Unique ID for this question.
    question_unique_id: String,
    /// Answer options (4 choices).
    options: Vec<String>,
    /// Gold label (1-4, 1-indexed).
    gold_label: i32,
    /// Writer label.
    #[serde(default)]
    writer_label: Option<i32>,
}

/// Load QuALITY dataset from a JSONL file.
///
/// The QuALITY dataset can be downloaded from NYU:
/// ```bash
/// # Download the dataset
/// wget https://raw.githubusercontent.com/nyu-mll/quality/main/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.dev
/// ```
pub fn load_quality_dataset(path: &Path) -> Result<Dataset> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read QuALITY file: {:?}", path))?;

    let mut dataset = Dataset::new("QuALITY");

    // QuALITY is in JSONL format (one JSON per line, each containing an article with multiple questions)
    for (line_num, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let raw: QualityRawArticle = serde_json::from_str(line)
            .with_context(|| format!("Failed to parse QuALITY article at line {}", line_num + 1))?;

        // Create one DatasetItem per question
        for q in raw.questions {
            // Convert 1-indexed gold_label to 0-indexed
            let correct_option = if q.gold_label >= 1 && q.gold_label <= 4 {
                Some((q.gold_label - 1) as usize)
            } else {
                None
            };

            // Get the answer text from options
            let answer = correct_option.and_then(|idx| q.options.get(idx).cloned());

            let item = DatasetItem {
                id: q.question_unique_id,
                document: raw.article.clone(),
                question: q.question,
                answer,
                options: Some(q.options),
                correct_option,
                source: "QuALITY".to_string(),
            };

            dataset.add_item(item);
        }
    }

    Ok(dataset)
}

/// Load a simple Q&A dataset from JSON.
///
/// Expected format:
/// ```json
/// {
///   "name": "my_dataset",
///   "items": [
///     {
///       "id": "1",
///       "document": "Document text...",
///       "question": "What is X?",
///       "answer": "X is Y"
///     }
///   ]
/// }
/// ```
pub fn load_simple_dataset(path: &Path) -> Result<Dataset> {
    Dataset::load_json(path)
}

/// Create a sample dataset for testing.
pub fn create_sample_dataset() -> Dataset {
    let mut dataset = Dataset::new("sample");

    dataset.add_item(DatasetItem {
        id: "sample_1".to_string(),
        document: r#"
Rust is a systems programming language focused on safety, speed, and concurrency.
It achieves memory safety without garbage collection through its ownership system.
The borrow checker ensures references are valid and prevents data races at compile time.
Rust was originally designed by Graydon Hoare at Mozilla Research.
The first stable release, Rust 1.0, was announced in May 2015.
"#.to_string(),
        question: "What mechanism does Rust use to achieve memory safety?".to_string(),
        answer: Some("Rust uses an ownership system and borrow checker to achieve memory safety without garbage collection.".to_string()),
        options: None,
        correct_option: None,
        source: "sample".to_string(),
    });

    dataset.add_item(DatasetItem {
        id: "sample_2".to_string(),
        document: r#"
Python is a high-level, interpreted programming language known for its clear syntax.
Created by Guido van Rossum, Python was first released in 1991.
Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.
The Python Package Index (PyPI) hosts thousands of third-party packages.
Python is widely used in data science, machine learning, and web development.
"#.to_string(),
        question: "Who created Python and when was it first released?".to_string(),
        answer: Some("Python was created by Guido van Rossum and first released in 1991.".to_string()),
        options: None,
        correct_option: None,
        source: "sample".to_string(),
    });

    dataset.add_item(DatasetItem {
        id: "sample_3".to_string(),
        document: r#"
Machine learning is a subset of artificial intelligence that enables systems to learn from data.
Supervised learning uses labeled data to train models, while unsupervised learning finds patterns in unlabeled data.
Neural networks are computing systems inspired by biological neural networks in animal brains.
Deep learning uses neural networks with many layers to model complex patterns.
Common applications include image recognition, natural language processing, and recommendation systems.
"#.to_string(),
        question: "What is the difference between supervised and unsupervised learning?".to_string(),
        answer: Some("Supervised learning uses labeled data to train models, while unsupervised learning finds patterns in unlabeled data.".to_string()),
        options: None,
        correct_option: None,
        source: "sample".to_string(),
    });

    dataset
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_operations() {
        let mut dataset = Dataset::new("test");
        assert!(dataset.is_empty());

        dataset.add_item(DatasetItem {
            id: "1".to_string(),
            document: "Test doc".to_string(),
            question: "Test question?".to_string(),
            answer: Some("Test answer".to_string()),
            options: None,
            correct_option: None,
            source: "test".to_string(),
        });

        assert_eq!(dataset.len(), 1);
        assert!(!dataset.is_empty());
    }

    #[test]
    fn test_dataset_take() {
        let dataset = create_sample_dataset();
        assert_eq!(dataset.len(), 3);

        let subset = dataset.take(2);
        assert_eq!(subset.len(), 2);
    }

    #[test]
    fn test_sample_dataset() {
        let dataset = create_sample_dataset();
        assert!(!dataset.is_empty());
        assert_eq!(dataset.name, "sample");

        for item in &dataset.items {
            assert!(!item.document.is_empty());
            assert!(!item.question.is_empty());
            assert!(item.answer.is_some());
        }
    }
}
