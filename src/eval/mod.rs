//! Evaluation module for comparing PageIndex vs Vector Search.
//!
//! This module provides:
//! - Vector search baseline using local embeddings (candle + sentence-transformers)
//! - LLM-as-judge evaluation framework
//! - Benchmark dataset loading (QuALITY, QASPER)

pub mod embeddings;
pub mod vector_search;
pub mod judge;
pub mod dataset;
pub mod benchmark;

pub use embeddings::EmbeddingModel;
pub use vector_search::{VectorIndex, VectorSearcher, ChunkConfig};
pub use judge::{LlmJudge, JudgeResult, ComparisonResult};
pub use dataset::{Dataset, DatasetItem, load_quality_dataset, load_simple_dataset, create_sample_dataset};
pub use benchmark::{Benchmark, BenchmarkConfig, BenchmarkResults};
