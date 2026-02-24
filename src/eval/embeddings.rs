//! Local embedding model using candle + sentence-transformers.

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig, DTYPE};
use hf_hub::{Repo, RepoType, api::sync::Api};
use tokenizers::Tokenizer;

/// Embedding model for generating text embeddings.
pub struct EmbeddingModel {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    normalize: bool,
}

impl EmbeddingModel {
    /// Load the all-MiniLM-L6-v2 model from Hugging Face Hub.
    pub fn load_minilm() -> Result<Self> {
        Self::load("sentence-transformers/all-MiniLM-L6-v2")
    }

    /// Load a sentence-transformers model by name.
    pub fn load(model_id: &str) -> Result<Self> {
        let device = Device::Cpu; // Use CPU for portability

        let api = Api::new().context("Failed to create HF Hub API")?;
        let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

        // Download model files
        let config_path = repo
            .get("config.json")
            .context("Failed to get config.json")?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .context("Failed to get tokenizer.json")?;
        let weights_path = repo
            .get("model.safetensors")
            .or_else(|_| repo.get("pytorch_model.bin"))
            .context("Failed to get model weights")?;

        // Load config
        let config: BertConfig = serde_json::from_str(&std::fs::read_to_string(&config_path)?)
            .context("Failed to parse config")?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // Load model weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, &device)
                .context("Failed to load model weights")?
        };

        let model = BertModel::load(vb, &config).context("Failed to load BERT model")?;

        Ok(Self {
            model,
            tokenizer,
            device,
            normalize: true,
        })
    }

    /// Generate embedding for a single text.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.embed_batch(&[text])?;
        Ok(embeddings.into_iter().next().unwrap())
    }

    /// Generate embeddings for a batch of texts.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Tokenize
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);

        // Prepare tensors
        let mut input_ids_vec = Vec::new();
        let mut attention_mask_vec = Vec::new();
        let mut token_type_ids_vec = Vec::new();

        for encoding in &encodings {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();

            // Pad to max_len
            let mut padded_ids = ids.to_vec();
            let mut padded_mask = mask.to_vec();
            let mut padded_types = vec![0u32; ids.len()];

            padded_ids.resize(max_len, 0);
            padded_mask.resize(max_len, 0);
            padded_types.resize(max_len, 0);

            input_ids_vec.extend(padded_ids);
            attention_mask_vec.extend(padded_mask);
            token_type_ids_vec.extend(padded_types);
        }

        let batch_size = texts.len();

        let input_ids = Tensor::from_vec(input_ids_vec, (batch_size, max_len), &self.device)?;
        let attention_mask =
            Tensor::from_vec(attention_mask_vec, (batch_size, max_len), &self.device)?;
        let token_type_ids =
            Tensor::from_vec(token_type_ids_vec, (batch_size, max_len), &self.device)?;

        // Run model
        let output = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))?;

        // Mean pooling over sequence dimension (with attention mask)
        let attention_mask_expanded = attention_mask
            .unsqueeze(2)?
            .to_dtype(output.dtype())?
            .broadcast_as(output.shape())?;

        let sum_embeddings = (output * &attention_mask_expanded)?.sum(1)?;
        let sum_mask = attention_mask_expanded.sum(1)?.clamp(1e-9, f64::MAX)?;
        let mean_embeddings = (sum_embeddings / sum_mask)?;

        // Normalize if requested
        let final_embeddings = if self.normalize {
            let norms = mean_embeddings.sqr()?.sum_keepdim(1)?.sqrt()?;
            let shape = mean_embeddings.shape().clone();
            (mean_embeddings / norms.broadcast_as(&shape)?)?
        } else {
            mean_embeddings
        };

        // Convert to Vec<Vec<f32>>
        let embeddings_vec: Vec<f32> = final_embeddings
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();

        let embedding_dim = final_embeddings.dim(1)?;
        let result: Vec<Vec<f32>> = embeddings_vec
            .chunks(embedding_dim)
            .map(|c| c.to_vec())
            .collect();

        Ok(result)
    }

    /// Get embedding dimension.
    pub fn dimension(&self) -> usize {
        384 // all-MiniLM-L6-v2 has 384 dimensions
    }
}

/// Compute cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 1e-6);
    }
}
