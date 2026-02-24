//! Tree structure for hierarchical document indexing.
//!
//! This module implements the core data structure used by PageIndex:
//! a hierarchical tree where each node represents a section of the document.

use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};

/// A node in the document tree structure.
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct TreeNode {
    /// Section title.
    pub title: String,

    /// Hierarchical structure index (e.g., "1", "1.1", "1.2.3").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub structure: Option<String>,

    /// Starting page index (1-indexed).
    pub start_index: usize,

    /// Ending page index (1-indexed, inclusive).
    pub end_index: usize,

    /// Child nodes (subsections).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nodes: Vec<TreeNode>,

    /// Optional summary of the section.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,

    /// Optional node ID for reference.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_id: Option<String>,
}

impl TreeNode {
    /// Create a new tree node.
    pub fn new(title: impl Into<String>, start_index: usize, end_index: usize) -> Self {
        Self {
            title: title.into(),
            structure: None,
            start_index,
            end_index,
            nodes: Vec::new(),
            summary: None,
            node_id: None,
        }
    }

    /// Set the structure index.
    pub fn with_structure(mut self, structure: impl Into<String>) -> Self {
        self.structure = Some(structure.into());
        self
    }

    /// Add a child node.
    pub fn add_child(&mut self, child: TreeNode) {
        self.nodes.push(child);
    }

    /// Check if this node has children.
    pub fn has_children(&self) -> bool {
        !self.nodes.is_empty()
    }

    /// Get the page span (number of pages covered).
    pub fn page_span(&self) -> usize {
        if self.end_index >= self.start_index {
            self.end_index - self.start_index + 1
        } else {
            0
        }
    }

    /// Recursively count all nodes in this subtree (including self).
    pub fn node_count(&self) -> usize {
        1 + self.nodes.iter().map(|n| n.node_count()).sum::<usize>()
    }

    /// Find all leaf nodes (nodes without children).
    pub fn leaves(&self) -> Vec<&TreeNode> {
        if self.nodes.is_empty() {
            vec![self]
        } else {
            self.nodes.iter().flat_map(|n| n.leaves()).collect()
        }
    }

    /// Find a node by title (case-insensitive).
    pub fn find_by_title(&self, title: &str) -> Option<&TreeNode> {
        let title_lower = title.to_lowercase();
        if self.title.to_lowercase() == title_lower {
            return Some(self);
        }
        for child in &self.nodes {
            if let Some(found) = child.find_by_title(title) {
                return Some(found);
            }
        }
        None
    }

    /// Get all page indices covered by this node and its children.
    pub fn all_page_indices(&self) -> Vec<usize> {
        (self.start_index..=self.end_index).collect()
    }

    /// Format the tree as a string for display.
    pub fn format_tree(&self, indent: usize) -> String {
        let prefix = "  ".repeat(indent);
        let structure_str = self
            .structure
            .as_ref()
            .map(|s| format!("{} ", s))
            .unwrap_or_default();

        let mut result = format!(
            "{}{}{} [pages {}-{}]\n",
            prefix, structure_str, self.title, self.start_index, self.end_index
        );

        for child in &self.nodes {
            result.push_str(&child.format_tree(indent + 1));
        }

        result
    }
}

/// A complete document tree (wrapper around root nodes).
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct DocumentTree {
    /// Document name.
    pub name: String,

    /// Root-level nodes.
    pub nodes: Vec<TreeNode>,

    /// Total page count.
    pub total_pages: usize,

    /// Optional document description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl DocumentTree {
    /// Create a new document tree.
    pub fn new(name: impl Into<String>, nodes: Vec<TreeNode>, total_pages: usize) -> Self {
        Self {
            name: name.into(),
            nodes,
            total_pages,
            description: None,
        }
    }

    /// Get total node count.
    pub fn node_count(&self) -> usize {
        self.nodes.iter().map(|n| n.node_count()).sum()
    }

    /// Get maximum depth of the tree.
    pub fn max_depth(&self) -> usize {
        fn depth(node: &TreeNode) -> usize {
            if node.nodes.is_empty() {
                1
            } else {
                1 + node.nodes.iter().map(depth).max().unwrap_or(0)
            }
        }

        self.nodes.iter().map(depth).max().unwrap_or(0)
    }

    /// Find a node by title.
    pub fn find_by_title(&self, title: &str) -> Option<&TreeNode> {
        for node in &self.nodes {
            if let Some(found) = node.find_by_title(title) {
                return Some(found);
            }
        }
        None
    }

    /// Format the entire tree for display.
    pub fn format(&self) -> String {
        let mut result = format!(
            "Document: {} ({} pages, {} sections)\n",
            self.name,
            self.total_pages,
            self.node_count()
        );
        result.push_str(&"─".repeat(50));
        result.push('\n');

        for node in &self.nodes {
            result.push_str(&node.format_tree(0));
        }

        result
    }

    /// Convert to JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Parse from JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

/// Raw TOC item from LLM response (before tree construction).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawTocItem {
    /// Section structure index (e.g., "1.2.3").
    pub structure: Option<String>,

    /// Section title.
    pub title: String,

    /// Physical page index (may be string like "<physical_index_5>" or integer).
    #[serde(alias = "page")]
    pub physical_index: Option<serde_json::Value>,
}

impl RawTocItem {
    /// Extract the page number from physical_index field.
    pub fn get_page_number(&self) -> Option<usize> {
        match &self.physical_index {
            Some(serde_json::Value::Number(n)) => n.as_u64().map(|n| n as usize),
            Some(serde_json::Value::String(s)) => {
                // Parse "<physical_index_X>" format
                if s.starts_with("<physical_index_") {
                    s.trim_start_matches("<physical_index_")
                        .trim_end_matches('>')
                        .parse()
                        .ok()
                } else {
                    s.parse().ok()
                }
            }
            _ => None,
        }
    }
}

/// Build a tree structure from flat TOC items.
///
/// This converts the flat list of TOC items (with structure indices like "1", "1.1", "1.2")
/// into a hierarchical tree structure.
pub fn build_tree_from_toc(items: &[RawTocItem], total_pages: usize) -> Vec<TreeNode> {
    if items.is_empty() {
        return Vec::new();
    }

    let mut nodes: Vec<TreeNode> = Vec::new();

    // Convert items to nodes with page indices
    let mut processed: Vec<(Vec<usize>, TreeNode)> = Vec::new();

    for (i, item) in items.iter().enumerate() {
        let start_index = item.get_page_number().unwrap_or(1);

        // Calculate end_index: either next item's start or total_pages
        let end_index = items
            .get(i + 1)
            .and_then(|next| next.get_page_number())
            .map(|n| if n > 1 { n - 1 } else { n })
            .unwrap_or(total_pages);

        let mut node = TreeNode::new(&item.title, start_index, end_index);

        if let Some(ref structure) = item.structure {
            node.structure = Some(structure.clone());

            // Parse structure into indices for hierarchy building
            let indices: Vec<usize> = structure
                .split('.')
                .filter_map(|s| s.parse().ok())
                .collect();

            processed.push((indices, node));
        } else {
            // No structure, treat as top-level
            processed.push((vec![processed.len() + 1], node));
        }
    }

    // Build hierarchy based on structure indices
    // For now, simple implementation: just use depth based on number of parts
    for (indices, node) in processed {
        if indices.len() == 1 {
            // Top-level node
            nodes.push(node);
        } else {
            // Find parent and add as child
            // This is a simplified version - the real PageIndex uses more sophisticated logic
            if let Some(parent) = nodes.last_mut() {
                add_to_tree(parent, &indices[1..], node);
            } else {
                nodes.push(node);
            }
        }
    }

    // Fix end_index for parent nodes to include children
    for node in &mut nodes {
        fix_end_indices(node);
    }

    nodes
}

/// Recursively add a node to the tree based on structure indices.
fn add_to_tree(parent: &mut TreeNode, remaining_indices: &[usize], node: TreeNode) {
    if remaining_indices.len() <= 1 {
        parent.add_child(node);
    } else if let Some(child) = parent.nodes.last_mut() {
        add_to_tree(child, &remaining_indices[1..], node);
    } else {
        parent.add_child(node);
    }
}

/// Fix end indices so parent nodes span their children.
fn fix_end_indices(node: &mut TreeNode) {
    for child in &mut node.nodes {
        fix_end_indices(child);
    }

    if !node.nodes.is_empty() {
        let max_end = node
            .nodes
            .iter()
            .map(|n| n.end_index)
            .max()
            .unwrap_or(node.end_index);
        if max_end > node.end_index {
            node.end_index = max_end;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_node_creation() {
        let node = TreeNode::new("Chapter 1", 1, 10).with_structure("1");

        assert_eq!(node.title, "Chapter 1");
        assert_eq!(node.start_index, 1);
        assert_eq!(node.end_index, 10);
        assert_eq!(node.structure, Some("1".to_string()));
        assert_eq!(node.page_span(), 10);
    }

    #[test]
    fn test_tree_node_children() {
        let mut parent = TreeNode::new("Chapter 1", 1, 20);
        parent.add_child(TreeNode::new("Section 1.1", 1, 10));
        parent.add_child(TreeNode::new("Section 1.2", 11, 20));

        assert!(parent.has_children());
        assert_eq!(parent.node_count(), 3);
        assert_eq!(parent.leaves().len(), 2);
    }

    #[test]
    fn test_document_tree() {
        let nodes = vec![
            TreeNode::new("Chapter 1", 1, 10),
            TreeNode::new("Chapter 2", 11, 20),
        ];
        let tree = DocumentTree::new("Test Doc", nodes, 20);

        assert_eq!(tree.node_count(), 2);
        assert_eq!(tree.max_depth(), 1);
    }

    #[test]
    fn test_find_by_title() {
        let mut ch1 = TreeNode::new("Chapter 1", 1, 10);
        ch1.add_child(TreeNode::new("Section 1.1", 1, 5));

        let tree = DocumentTree::new("Test", vec![ch1], 10);

        assert!(tree.find_by_title("Chapter 1").is_some());
        assert!(tree.find_by_title("Section 1.1").is_some());
        assert!(tree.find_by_title("Not Found").is_none());
    }

    #[test]
    fn test_raw_toc_item_page_number() {
        let item1 = RawTocItem {
            structure: Some("1".to_string()),
            title: "Test".to_string(),
            physical_index: Some(serde_json::Value::Number(5.into())),
        };
        assert_eq!(item1.get_page_number(), Some(5));

        let item2 = RawTocItem {
            structure: Some("2".to_string()),
            title: "Test 2".to_string(),
            physical_index: Some(serde_json::Value::String("<physical_index_10>".to_string())),
        };
        assert_eq!(item2.get_page_number(), Some(10));
    }

    #[test]
    fn test_tree_json_roundtrip() {
        let tree = DocumentTree::new("Test", vec![TreeNode::new("Chapter 1", 1, 10)], 10);

        let json = tree.to_json().unwrap();
        let parsed = DocumentTree::from_json(&json).unwrap();

        assert_eq!(parsed.name, tree.name);
        assert_eq!(parsed.total_pages, tree.total_pages);
    }
}
