//! Vector search implementation for baseline comparison.

use super::embeddings::{EmbeddingModel, cosine_similarity};
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Configuration for text chunking.
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Maximum characters per chunk.
    pub chunk_size: usize,
    /// Overlap between consecutive chunks.
    pub chunk_overlap: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            chunk_overlap: 50,
        }
    }
}

/// A chunk of text with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// Chunk text content.
    pub text: String,
    /// Start character position in original document.
    pub start_pos: usize,
    /// End character position in original document.
    pub end_pos: usize,
    /// Chunk index.
    pub index: usize,
}

/// A vector index entry.
#[derive(Debug, Clone)]
pub struct IndexEntry {
    pub chunk: Chunk,
    pub embedding: Vec<f32>,
}

/// Vector search index.
pub struct VectorIndex {
    entries: Vec<IndexEntry>,
    config: ChunkConfig,
}

impl VectorIndex {
    /// Create a new empty vector index.
    pub fn new(config: ChunkConfig) -> Self {
        Self {
            entries: Vec::new(),
            config,
        }
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(ChunkConfig::default())
    }

    /// Build index from document text.
    pub fn build(text: &str, model: &EmbeddingModel, config: ChunkConfig) -> Result<Self> {
        let chunks = Self::chunk_text(text, &config);

        // Generate embeddings in batches
        let batch_size = 32;
        let mut entries = Vec::new();

        for batch in chunks.chunks(batch_size) {
            let texts: Vec<&str> = batch.iter().map(|c| c.text.as_str()).collect();
            let embeddings = model.embed_batch(&texts)?;

            for (chunk, embedding) in batch.iter().zip(embeddings) {
                entries.push(IndexEntry {
                    chunk: chunk.clone(),
                    embedding,
                });
            }
        }

        Ok(Self { entries, config })
    }

    /// Chunk text into overlapping segments.
    fn chunk_text(text: &str, config: &ChunkConfig) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let text_len = chars.len();

        if text_len == 0 {
            return chunks;
        }

        let mut start = 0;
        let mut index = 0;

        while start < text_len {
            let end = (start + config.chunk_size).min(text_len);

            // Try to break at sentence boundary if possible
            let adjusted_end = if end < text_len && end > start {
                // Look for sentence ending within last 100 chars (but not before start)
                let search_start = end.saturating_sub(100).max(start);
                if search_start < end {
                    let search_text: String = chars[search_start..end].iter().collect();

                    if let Some(pos) = search_text.rfind(|c| c == '.' || c == '!' || c == '?') {
                        let candidate = search_start + pos + 1;
                        // Only use this if it's past start
                        if candidate > start { candidate } else { end }
                    } else {
                        end
                    }
                } else {
                    end
                }
            } else {
                end
            };

            // Ensure adjusted_end > start
            let final_end = adjusted_end.max(start + 1).min(text_len);

            let final_text: String = chars[start..final_end].iter().collect();

            if !final_text.trim().is_empty() {
                chunks.push(Chunk {
                    text: final_text.trim().to_string(),
                    start_pos: start,
                    end_pos: final_end,
                    index,
                });
                index += 1;
            }

            // Move start forward, ensuring progress
            if final_end >= text_len {
                break;
            }

            // Calculate next start with overlap
            let next_start = if config.chunk_overlap > 0 && final_end > config.chunk_overlap {
                final_end - config.chunk_overlap
            } else {
                final_end
            };

            // Ensure we always move forward by at least 1 character
            start = if next_start <= start {
                start + 1
            } else {
                next_start
            };
        }

        chunks
    }

    /// Number of chunks in the index.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if index is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get all entries.
    pub fn entries(&self) -> &[IndexEntry] {
        &self.entries
    }
}

/// Vector search result.
#[derive(Debug, Clone)]
pub struct VectorSearchResult {
    /// The matched chunk.
    pub chunk: Chunk,
    /// Similarity score (0-1).
    pub score: f32,
}

/// Vector searcher for querying the index.
pub struct VectorSearcher<'a> {
    index: &'a VectorIndex,
    model: &'a EmbeddingModel,
}

impl<'a> VectorSearcher<'a> {
    /// Create a new searcher.
    pub fn new(index: &'a VectorIndex, model: &'a EmbeddingModel) -> Self {
        Self { index, model }
    }

    /// Search for similar chunks.
    pub fn search(&self, query: &str, top_k: usize) -> Result<Vec<VectorSearchResult>> {
        let query_embedding = self.model.embed(query)?;

        let mut results: Vec<VectorSearchResult> = self
            .index
            .entries
            .iter()
            .map(|entry| {
                let score = cosine_similarity(&query_embedding, &entry.embedding);
                VectorSearchResult {
                    chunk: entry.chunk.clone(),
                    score,
                }
            })
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top_k
        results.truncate(top_k);

        Ok(results)
    }

    /// Search and return concatenated context.
    pub fn search_context(&self, query: &str, top_k: usize) -> Result<String> {
        let results = self.search(query, top_k)?;

        let context = results
            .iter()
            .map(|r| format!("[Score: {:.3}]\n{}", r.score, r.chunk.text))
            .collect::<Vec<_>>()
            .join("\n\n---\n\n");

        Ok(context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_text() {
        let text = "This is a test. Another sentence here. And one more.";
        let config = ChunkConfig {
            chunk_size: 20,
            chunk_overlap: 5,
        };

        let chunks = VectorIndex::chunk_text(text, &config);

        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(!chunk.text.is_empty());
        }
    }

    #[test]
    fn test_chunk_config_default() {
        let config = ChunkConfig::default();
        assert_eq!(config.chunk_size, 512);
        assert_eq!(config.chunk_overlap, 50);
    }
}
