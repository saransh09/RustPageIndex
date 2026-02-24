//! Evaluation module for comparing PageIndex vs Vector Search.
//!
//! This module provides:
//! - Vector search baseline using local embeddings (candle + sentence-transformers)
//! - LLM-as-judge evaluation framework
//! - Benchmark dataset loading (QuALITY, QASPER)

pub mod benchmark;
pub mod dataset;
pub mod embeddings;
pub mod judge;
pub mod vector_search;

pub use benchmark::{Benchmark, BenchmarkConfig, BenchmarkResults};
pub use dataset::{
    Dataset, DatasetItem, create_sample_dataset, load_quality_dataset, load_simple_dataset,
};
pub use embeddings::EmbeddingModel;
pub use judge::{ComparisonResult, JudgeResult, LlmJudge};
pub use vector_search::{ChunkConfig, VectorIndex, VectorSearcher};
