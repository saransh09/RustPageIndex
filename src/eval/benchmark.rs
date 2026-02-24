//! Benchmark runner for comparing PageIndex vs Vector Search.

use super::dataset::{Dataset, DatasetItem};
use super::embeddings::EmbeddingModel;
use super::judge::{ComparisonResult, LlmJudge};
use super::vector_search::{ChunkConfig, VectorIndex, VectorSearcher};
use crate::config::LlmConfig;
use crate::document::Document;
use crate::indexer::TreeIndexer;
use crate::llm::{LlmClient, Prompts};
use crate::search::{Relevance, TreeSearcher};
use crate::tree::DocumentTree;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Configuration for the benchmark.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of top results to retrieve.
    pub top_k: usize,
    /// Chunk configuration for vector search.
    pub chunk_config: ChunkConfig,
    /// Whether to run PageIndex (requires LLM).
    pub run_pageindex: bool,
    /// Whether to run vector search (requires embedding model).
    pub run_vector: bool,
    /// Maximum items to evaluate (for quick testing).
    pub max_items: Option<usize>,
    /// Verbose output.
    pub verbose: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            top_k: 3,
            chunk_config: ChunkConfig::default(),
            run_pageindex: true,
            run_vector: true,
            max_items: None,
            verbose: false,
        }
    }
}

/// Results for a single item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemResult {
    /// Item ID.
    pub item_id: String,
    /// PageIndex retrieved content.
    pub pageindex_content: Option<String>,
    /// PageIndex generated answer.
    pub pageindex_answer: Option<String>,
    /// PageIndex retrieval time.
    pub pageindex_time_ms: Option<u64>,
    /// Vector search retrieved content.
    pub vector_content: Option<String>,
    /// Vector search generated answer (RAG).
    pub vector_answer: Option<String>,
    /// Vector search retrieval time.
    pub vector_time_ms: Option<u64>,
    /// Comparison result from judge.
    pub comparison: Option<ComparisonResult>,
    /// Error message if any.
    pub error: Option<String>,
}

/// Aggregated benchmark results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    /// Dataset name.
    pub dataset_name: String,
    /// Total items evaluated.
    pub total_items: usize,
    /// Items where PageIndex won.
    pub pageindex_wins: usize,
    /// Items where Vector search won.
    pub vector_wins: usize,
    /// Ties.
    pub ties: usize,
    /// Average PageIndex score.
    pub avg_pageindex_score: f64,
    /// Average Vector score.
    pub avg_vector_score: f64,
    /// Average PageIndex retrieval time (ms).
    pub avg_pageindex_time_ms: f64,
    /// Average Vector retrieval time (ms).
    pub avg_vector_time_ms: f64,
    /// Individual item results.
    pub item_results: Vec<ItemResult>,
    /// Total benchmark time (seconds).
    pub total_time_secs: f64,
}

impl BenchmarkResults {
    /// Create empty results.
    pub fn new(dataset_name: &str) -> Self {
        Self {
            dataset_name: dataset_name.to_string(),
            total_items: 0,
            pageindex_wins: 0,
            vector_wins: 0,
            ties: 0,
            avg_pageindex_score: 0.0,
            avg_vector_score: 0.0,
            avg_pageindex_time_ms: 0.0,
            avg_vector_time_ms: 0.0,
            item_results: Vec::new(),
            total_time_secs: 0.0,
        }
    }

    /// Calculate summary statistics from item results.
    pub fn calculate_summary(&mut self) {
        if self.item_results.is_empty() {
            return;
        }

        self.total_items = self.item_results.len();

        let mut pageindex_scores = Vec::new();
        let mut vector_scores = Vec::new();
        let mut pageindex_times = Vec::new();
        let mut vector_times = Vec::new();

        for result in &self.item_results {
            if let Some(ref comparison) = result.comparison {
                match comparison.winner {
                    1 => self.pageindex_wins += 1,
                    2 => self.vector_wins += 1,
                    _ => self.ties += 1,
                }
                pageindex_scores.push(comparison.score_system1 as f64);
                vector_scores.push(comparison.score_system2 as f64);
            }

            if let Some(time) = result.pageindex_time_ms {
                pageindex_times.push(time as f64);
            }
            if let Some(time) = result.vector_time_ms {
                vector_times.push(time as f64);
            }
        }

        if !pageindex_scores.is_empty() {
            self.avg_pageindex_score =
                pageindex_scores.iter().sum::<f64>() / pageindex_scores.len() as f64;
        }
        if !vector_scores.is_empty() {
            self.avg_vector_score = vector_scores.iter().sum::<f64>() / vector_scores.len() as f64;
        }
        if !pageindex_times.is_empty() {
            self.avg_pageindex_time_ms =
                pageindex_times.iter().sum::<f64>() / pageindex_times.len() as f64;
        }
        if !vector_times.is_empty() {
            self.avg_vector_time_ms = vector_times.iter().sum::<f64>() / vector_times.len() as f64;
        }
    }

    /// Print summary to stdout.
    pub fn print_summary(&self) {
        println!("\n========== Benchmark Results ==========");
        println!("Dataset: {}", self.dataset_name);
        println!("Total items: {}", self.total_items);
        println!("----------------------------------------");
        println!(
            "PageIndex wins: {} ({:.1}%)",
            self.pageindex_wins,
            if self.total_items > 0 {
                self.pageindex_wins as f64 / self.total_items as f64 * 100.0
            } else {
                0.0
            }
        );
        println!(
            "Vector wins:    {} ({:.1}%)",
            self.vector_wins,
            if self.total_items > 0 {
                self.vector_wins as f64 / self.total_items as f64 * 100.0
            } else {
                0.0
            }
        );
        println!(
            "Ties:           {} ({:.1}%)",
            self.ties,
            if self.total_items > 0 {
                self.ties as f64 / self.total_items as f64 * 100.0
            } else {
                0.0
            }
        );
        println!("----------------------------------------");
        println!("Avg PageIndex score: {:.2}/5", self.avg_pageindex_score);
        println!("Avg Vector score:    {:.2}/5", self.avg_vector_score);
        println!("----------------------------------------");
        println!("Avg PageIndex time: {:.0}ms", self.avg_pageindex_time_ms);
        println!("Avg Vector time:    {:.0}ms", self.avg_vector_time_ms);
        println!("----------------------------------------");
        println!("Total time: {:.1}s", self.total_time_secs);
        println!("========================================\n");
    }
}

/// Compute a simple hash for a document to use as cache key.
fn document_hash(doc: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut hasher = DefaultHasher::new();
    doc.hash(&mut hasher);
    hasher.finish()
}

/// Cache entry for a document tree with the original document.
struct CachedTree {
    tree: DocumentTree,
    document: Document,
}

/// Benchmark runner.
pub struct Benchmark {
    config: BenchmarkConfig,
    llm_config: LlmConfig,
    /// Cache of document trees keyed by document hash.
    tree_cache: Arc<RwLock<HashMap<u64, CachedTree>>>,
}

impl Benchmark {
    /// Create a new benchmark runner.
    pub fn new(llm_config: LlmConfig, config: BenchmarkConfig) -> Self {
        Self {
            config,
            llm_config,
            tree_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Run the benchmark on a dataset.
    pub async fn run(&self, dataset: &Dataset) -> Result<BenchmarkResults> {
        let start_time = Instant::now();
        let mut results = BenchmarkResults::new(&dataset.name);

        // Load embedding model for vector search
        let embedding_model = if self.config.run_vector {
            println!("Loading embedding model...");
            Some(EmbeddingModel::load_minilm()?)
        } else {
            None
        };

        // Create LLM client and judge
        let llm_client = LlmClient::new(self.llm_config.clone());
        let judge = LlmJudge::new(llm_client.clone());
        let indexer = TreeIndexer::new(llm_client.clone());
        let searcher = TreeSearcher::new(llm_client.clone());

        // Determine items to process
        let items: Vec<_> = if let Some(max) = self.config.max_items {
            dataset.items.iter().take(max).collect()
        } else {
            dataset.items.iter().collect()
        };

        println!("Running benchmark on {} items...", items.len());

        for (idx, item) in items.iter().enumerate() {
            if self.config.verbose {
                println!("\n[{}/{}] Processing: {}", idx + 1, items.len(), item.id);
            } else {
                print!(".");
                use std::io::Write;
                std::io::stdout().flush().ok();
            }

            let item_result = self
                .process_item(
                    item,
                    &indexer,
                    &searcher,
                    &judge,
                    &llm_client,
                    embedding_model.as_ref(),
                )
                .await;

            results.item_results.push(item_result);
        }

        if !self.config.verbose {
            println!(); // Newline after dots
        }

        results.total_time_secs = start_time.elapsed().as_secs_f64();
        results.calculate_summary();

        Ok(results)
    }

    /// Process a single dataset item.
    async fn process_item(
        &self,
        item: &DatasetItem,
        indexer: &TreeIndexer,
        searcher: &TreeSearcher,
        judge: &LlmJudge,
        llm_client: &LlmClient,
        embedding_model: Option<&EmbeddingModel>,
    ) -> ItemResult {
        let mut result = ItemResult {
            item_id: item.id.clone(),
            pageindex_content: None,
            pageindex_answer: None,
            pageindex_time_ms: None,
            vector_content: None,
            vector_answer: None,
            vector_time_ms: None,
            comparison: None,
            error: None,
        };

        // Run PageIndex retrieval
        if self.config.run_pageindex {
            match self.run_pageindex(item, indexer, searcher).await {
                Ok((content, duration)) => {
                    result.pageindex_content = Some(content.clone());
                    result.pageindex_time_ms = Some(duration.as_millis() as u64);

                    // Generate answer from retrieved content
                    if self.config.verbose {
                        println!("  [PageIndex] Generating answer from retrieved content...");
                    }
                    match self
                        .generate_answer(llm_client, &item.question, &content)
                        .await
                    {
                        Ok(answer) => {
                            result.pageindex_answer = Some(answer);
                        }
                        Err(e) => {
                            if self.config.verbose {
                                eprintln!("  PageIndex answer generation error: {}", e);
                            }
                        }
                    }
                }
                Err(e) => {
                    result.error = Some(format!("PageIndex error: {}", e));
                    if self.config.verbose {
                        eprintln!("  PageIndex error: {}", e);
                    }
                }
            }
        }

        // Run Vector search retrieval
        if self.config.run_vector {
            if let Some(model) = embedding_model {
                match self.run_vector_search(item, model).await {
                    Ok((content, duration)) => {
                        result.vector_content = Some(content.clone());
                        result.vector_time_ms = Some(duration.as_millis() as u64);

                        // Generate answer from retrieved content (standard RAG)
                        if self.config.verbose {
                            println!("  [Vector RAG] Generating answer from retrieved content...");
                        }
                        match self
                            .generate_answer(llm_client, &item.question, &content)
                            .await
                        {
                            Ok(answer) => {
                                result.vector_answer = Some(answer);
                            }
                            Err(e) => {
                                if self.config.verbose {
                                    eprintln!("  Vector RAG answer generation error: {}", e);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let err_msg = format!("Vector search error: {}", e);
                        if result.error.is_some() {
                            result.error = Some(format!("{}\n{}", result.error.unwrap(), err_msg));
                        } else {
                            result.error = Some(err_msg);
                        }
                        if self.config.verbose {
                            eprintln!("  Vector search error: {}", e);
                        }
                    }
                }
            }
        }

        // Run comparison if both methods produced ANSWERS (not just content)
        if let (Some(pi_answer), Some(vec_answer)) =
            (&result.pageindex_answer, &result.vector_answer)
        {
            if self.config.verbose {
                println!("  [Judge] Comparing answers...");
            }
            match judge
                .compare_answers(
                    &item.question,
                    "PageIndex",
                    pi_answer,
                    "VectorRAG",
                    vec_answer,
                    item.answer.as_deref(),
                )
                .await
            {
                Ok(comparison) => {
                    if self.config.verbose {
                        let winner_str = match comparison.winner {
                            1 => "PageIndex",
                            2 => "VectorRAG",
                            _ => "Tie",
                        };
                        println!(
                            "  Winner: {} (PI: {}/5, Vec: {}/5)",
                            winner_str, comparison.score_system1, comparison.score_system2
                        );
                    }
                    result.comparison = Some(comparison);
                }
                Err(e) => {
                    if self.config.verbose {
                        eprintln!("  Judge error: {}", e);
                    }
                }
            }
        }

        result
    }

    /// Generate an answer from retrieved content using LLM.
    async fn generate_answer(
        &self,
        client: &LlmClient,
        question: &str,
        context: &str,
    ) -> Result<String> {
        let prompt = Prompts::rag_answer()
            .replace("{question}", question)
            .replace("{context}", context);

        let answer = client.complete(None, &prompt).await?;
        Ok(answer.trim().to_string())
    }

    /// Run PageIndex on a single item.
    async fn run_pageindex(
        &self,
        item: &DatasetItem,
        indexer: &TreeIndexer,
        searcher: &TreeSearcher,
    ) -> Result<(String, Duration)> {
        let start = Instant::now();

        // Compute document hash for caching
        let doc_hash = document_hash(&item.document);

        // Check cache first
        let cache = self.tree_cache.read().await;
        let cached = cache
            .get(&doc_hash)
            .map(|c| (c.tree.clone(), c.document.clone()));
        drop(cache);

        let (tree, doc) = if let Some((tree, doc)) = cached {
            if self.config.verbose {
                println!("  [PageIndex] Using cached tree for document");
            }
            (tree, doc)
        } else {
            // Create document from item
            let doc = Document::from_text(&item.id, item.document.clone());

            // Build tree index
            if self.config.verbose {
                println!("  [PageIndex] Building tree index (not cached)...");
            }
            let tree = indexer.index(&doc).await?;

            // Cache the tree
            let mut cache = self.tree_cache.write().await;
            cache.insert(
                doc_hash,
                CachedTree {
                    tree: tree.clone(),
                    document: doc.clone(),
                },
            );

            (tree, doc)
        };

        // Search WITH CONTENT - this is critical for PageIndex to work!
        let search_results = searcher
            .search_with_content(&tree, &doc, &item.question)
            .await?;

        // Combine relevant content - now we actually have content!
        let content = search_results
            .iter()
            .filter(|r| matches!(r.relevance, Relevance::High | Relevance::Medium))
            .take(self.config.top_k)
            .map(|r| {
                format!(
                    "[Section: {}] (pages {}-{})\n{}",
                    r.title,
                    r.start_index,
                    r.end_index,
                    r.content.as_deref().unwrap_or("(no content)")
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n---\n\n");

        // If no high/medium results, fall back to any results
        let content = if content.is_empty() {
            search_results
                .iter()
                .take(self.config.top_k)
                .map(|r| {
                    format!(
                        "[Section: {}] (pages {}-{})\n{}",
                        r.title,
                        r.start_index,
                        r.end_index,
                        r.content.as_deref().unwrap_or("(no content)")
                    )
                })
                .collect::<Vec<_>>()
                .join("\n\n---\n\n")
        } else {
            content
        };

        let duration = start.elapsed();
        Ok((content, duration))
    }

    /// Run vector search on a single item.
    async fn run_vector_search(
        &self,
        item: &DatasetItem,
        model: &EmbeddingModel,
    ) -> Result<(String, Duration)> {
        let start = Instant::now();

        // Build vector index
        let index = VectorIndex::build(&item.document, model, self.config.chunk_config.clone())?;

        // Search
        let vec_searcher = VectorSearcher::new(&index, model);
        let content = vec_searcher.search_context(&item.question, self.config.top_k)?;

        let duration = start.elapsed();
        Ok((content, duration))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_config_default() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.top_k, 3);
        assert!(config.run_pageindex);
        assert!(config.run_vector);
    }

    #[test]
    fn test_benchmark_results_summary() {
        let mut results = BenchmarkResults::new("test");

        results.item_results.push(ItemResult {
            item_id: "1".to_string(),
            pageindex_content: Some("content".to_string()),
            pageindex_answer: Some("answer 1".to_string()),
            pageindex_time_ms: Some(100),
            vector_content: Some("content".to_string()),
            vector_answer: Some("answer 1".to_string()),
            vector_time_ms: Some(50),
            comparison: Some(ComparisonResult {
                winner: 1,
                score_system1: 4,
                score_system2: 3,
                explanation: "test".to_string(),
            }),
            error: None,
        });

        results.item_results.push(ItemResult {
            item_id: "2".to_string(),
            pageindex_content: Some("content".to_string()),
            pageindex_answer: Some("answer 2".to_string()),
            pageindex_time_ms: Some(150),
            vector_content: Some("content".to_string()),
            vector_answer: Some("answer 2".to_string()),
            vector_time_ms: Some(60),
            comparison: Some(ComparisonResult {
                winner: 2,
                score_system1: 3,
                score_system2: 5,
                explanation: "test".to_string(),
            }),
            error: None,
        });

        results.calculate_summary();

        assert_eq!(results.total_items, 2);
        assert_eq!(results.pageindex_wins, 1);
        assert_eq!(results.vector_wins, 1);
        assert_eq!(results.ties, 0);
        assert!((results.avg_pageindex_score - 3.5).abs() < 0.01);
        assert!((results.avg_vector_score - 4.0).abs() < 0.01);
    }
}
