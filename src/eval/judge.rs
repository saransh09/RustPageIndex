//! LLM-as-Judge evaluation framework.

use crate::config::LlmConfig;
use crate::error::Result;
use crate::llm::LlmClient;
use serde::{Deserialize, Serialize};

/// Result from judging a single retrieval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeResult {
    /// Relevance score (1-5).
    pub relevance: u8,
    /// Whether the retrieved content could answer the question.
    pub answerable: bool,
    /// Explanation from the judge.
    pub explanation: String,
}

/// Result comparing two systems.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    /// Which system won (1 = first, 2 = second, 0 = tie).
    pub winner: u8,
    /// Score for first system (1-5).
    pub score_system1: u8,
    /// Score for second system (1-5).
    pub score_system2: u8,
    /// Explanation of the comparison.
    pub explanation: String,
}

/// LLM-as-Judge for evaluating retrieval quality.
pub struct LlmJudge {
    client: LlmClient,
}

impl LlmJudge {
    /// Create a new judge with the given LLM client.
    pub fn new(client: LlmClient) -> Self {
        Self { client }
    }

    /// Create from LLM config.
    pub fn from_config(config: LlmConfig) -> Self {
        Self::new(LlmClient::new(config))
    }

    /// Judge the relevance of retrieved content for a query.
    pub async fn judge_relevance(
        &self,
        query: &str,
        retrieved_content: &str,
        ground_truth: Option<&str>,
    ) -> Result<JudgeResult> {
        let ground_truth_section = ground_truth
            .map(|gt| format!("\n\nGround Truth Answer: {}", gt))
            .unwrap_or_default();

        let prompt = format!(
            r#"You are an expert judge evaluating the quality of retrieved content for answering a question.

Question: {}

Retrieved Content:
{}
{}

Evaluate the retrieved content on these criteria:
1. Relevance: How relevant is the content to the question? (1=not relevant, 5=highly relevant)
2. Answerability: Could someone answer the question using ONLY this retrieved content?

Respond in JSON format:
{{
    "relevance": <1-5>,
    "answerable": <true/false>,
    "explanation": "<brief explanation>"
}}

Respond with only the JSON, no other text."#,
            query, retrieved_content, ground_truth_section
        );

        let response = self.client.complete(None, &prompt).await?;
        let result = Self::parse_judge_response(&response)?;
        Ok(result)
    }

    /// Compare two retrieval systems head-to-head.
    pub async fn compare_systems(
        &self,
        query: &str,
        system1_name: &str,
        system1_content: &str,
        system2_name: &str,
        system2_content: &str,
        ground_truth: Option<&str>,
    ) -> Result<ComparisonResult> {
        let ground_truth_section = ground_truth
            .map(|gt| format!("\n\nGround Truth Answer: {}", gt))
            .unwrap_or_default();

        let prompt = format!(
            r#"You are an expert judge comparing two retrieval systems.

Question: {}

System A ({}) Retrieved:
{}

---

System B ({}) Retrieved:
{}
{}

Compare the two systems:
1. Which system retrieved more relevant content for answering the question?
2. Rate each system's retrieval quality (1-5).

Respond in JSON format:
{{
    "winner": "<A, B, or TIE>",
    "score_system_a": <1-5>,
    "score_system_b": <1-5>,
    "explanation": "<brief explanation of why one system is better>"
}}

Respond with only the JSON, no other text."#,
            query,
            system1_name,
            system1_content,
            system2_name,
            system2_content,
            ground_truth_section
        );

        let response = self.client.complete(None, &prompt).await?;
        let result = Self::parse_comparison_response(&response)?;
        Ok(result)
    }

    /// Parse judge response JSON.
    fn parse_judge_response(response: &str) -> Result<JudgeResult> {
        let json_str = Self::extract_json(response);

        #[derive(Deserialize)]
        struct RawJudgeResult {
            relevance: u8,
            answerable: bool,
            explanation: String,
        }

        let raw: RawJudgeResult = serde_json::from_str(&json_str).map_err(|e| {
            crate::error::PageIndexError::LlmParse(format!(
                "Failed to parse judge response: {}. Response: {}",
                e, response
            ))
        })?;

        Ok(JudgeResult {
            relevance: raw.relevance.clamp(1, 5),
            answerable: raw.answerable,
            explanation: raw.explanation,
        })
    }

    /// Parse comparison response JSON.
    fn parse_comparison_response(response: &str) -> Result<ComparisonResult> {
        let json_str = Self::extract_json(response);

        #[derive(Deserialize)]
        struct RawComparisonResult {
            winner: String,
            score_system_a: u8,
            score_system_b: u8,
            explanation: String,
        }

        let raw: RawComparisonResult = serde_json::from_str(&json_str).map_err(|e| {
            crate::error::PageIndexError::LlmParse(format!(
                "Failed to parse comparison response: {}. Response: {}",
                e, response
            ))
        })?;

        let winner = match raw.winner.to_uppercase().as_str() {
            "A" => 1,
            "B" => 2,
            _ => 0, // TIE
        };

        Ok(ComparisonResult {
            winner,
            score_system1: raw.score_system_a.clamp(1, 5),
            score_system2: raw.score_system_b.clamp(1, 5),
            explanation: raw.explanation,
        })
    }

    /// Extract JSON from response.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_judge_response() {
        let response = r#"{"relevance": 4, "answerable": true, "explanation": "Good content"}"#;
        let result = LlmJudge::parse_judge_response(response).unwrap();

        assert_eq!(result.relevance, 4);
        assert!(result.answerable);
    }

    #[test]
    fn test_parse_comparison_response() {
        let response = r#"{"winner": "A", "score_system_a": 5, "score_system_b": 3, "explanation": "A is better"}"#;
        let result = LlmJudge::parse_comparison_response(response).unwrap();

        assert_eq!(result.winner, 1);
        assert_eq!(result.score_system1, 5);
        assert_eq!(result.score_system2, 3);
    }
}
