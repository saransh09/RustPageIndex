//! Rust Page Indexer - An LLM-powered hierarchical tree indexing system.
//!
//! This library is a Rust port of [PageIndex](https://github.com/VectifyAI/PageIndex),
//! which uses LLM reasoning to build hierarchical tree indexes from documents
//! and perform intelligent retrieval.
//!
//! # Overview
//!
//! Unlike traditional RAG systems that chunk documents and use vector embeddings,
//! PageIndex:
//! 1. Uses an LLM to extract the document's hierarchical structure
//! 2. Maps sections to physical page indices
//! 3. Uses LLM reasoning to navigate the tree during search
//!
//! # Quick Start
//!
//! ```no_run
//! use rust_page_indexer::{
//!     config::Config,
//!     document::Document,
//!     indexer::TreeIndexer,
//!     search::TreeSearcher,
//!     llm::LlmClient,
//!     persistence::{save_tree, load_tree},
//! };
//! use std::path::Path;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Load configuration
//!     let config = Config::load()?;
//!     config.validate()?;
//!
//!     // Create LLM client
//!     let client = LlmClient::new(config.llm.clone());
//!
//!     // Load a document
//!     let document = Document::from_text_file(Path::new("document.txt"))?;
//!
//!     // Build tree index
//!     let indexer = TreeIndexer::new(client.clone());
//!     let tree = indexer.index(&document).await?;
//!
//!     // Save the tree for later use
//!     save_tree(&tree, Path::new("tree_index.json"))?;
//!
//!     // Search the tree
//!     let searcher = TreeSearcher::new(client);
//!     let results = searcher.search(&tree, "your query here").await?;
//!
//!     for result in results {
//!         println!("{}: {:?}", result.title, result.relevance);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! # Architecture
//!
//! - **Document**: Page-based document representation
//! - **LlmClient**: OpenAI-compatible API client for LLM calls
//! - **TreeIndexer**: Builds hierarchical tree structure from documents
//! - **TreeSearcher**: Searches trees using LLM reasoning
//! - **DocumentTree**: The hierarchical tree structure

pub mod config;
pub mod document;
pub mod error;
pub mod indexer;
pub mod llm;
pub mod persistence;
pub mod search;
pub mod tree;

#[cfg(feature = "eval")]
pub mod eval;

// Re-export commonly used types
pub use config::Config;
pub use document::Document;
pub use error::{PageIndexError, Result};
pub use indexer::TreeIndexer;
pub use llm::LlmClient;
pub use persistence::{load_tree, save_tree};
pub use search::{SearchResult, TreeSearcher};
pub use tree::{DocumentTree, TreeNode};
