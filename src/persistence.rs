//! Persistence layer for saving/loading document trees.
//!
//! Supports both JSON (human-readable) and bincode (efficient binary) formats.

use crate::error::{PageIndexError, Result};
use crate::tree::DocumentTree;
use std::fs;
use std::path::Path;

/// Default filename for the tree index.
pub const DEFAULT_INDEX_FILENAME: &str = "tree_index.json";

/// Save format for tree indexes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaveFormat {
    /// JSON format (human-readable, larger).
    Json,
    /// Bincode format (binary, compact).
    Bincode,
}

impl SaveFormat {
    /// Determine format from file extension.
    pub fn from_path(path: &Path) -> Self {
        match path.extension().and_then(|e| e.to_str()) {
            Some("json") => SaveFormat::Json,
            Some("bin") | Some("bincode") => SaveFormat::Bincode,
            _ => SaveFormat::Json, // Default to JSON
        }
    }
}

/// Save a DocumentTree to a file.
pub fn save_tree(tree: &DocumentTree, path: &Path) -> Result<()> {
    let format = SaveFormat::from_path(path);
    save_tree_with_format(tree, path, format)
}

/// Save a DocumentTree with specific format.
pub fn save_tree_with_format(tree: &DocumentTree, path: &Path, format: SaveFormat) -> Result<()> {
    // Ensure parent directory exists
    if let Some(parent) = path.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent).map_err(|e| PageIndexError::io(parent, e))?;
        }
    }

    let data = match format {
        SaveFormat::Json => serde_json::to_string_pretty(tree)
            .map_err(|e| PageIndexError::Serialization(e.to_string()))?
            .into_bytes(),
        SaveFormat::Bincode => {
            let config = bincode::config::standard();
            bincode::encode_to_vec(tree, config)
                .map_err(|e| PageIndexError::Serialization(e.to_string()))?
        }
    };

    fs::write(path, &data).map_err(|e| PageIndexError::io(path, e))?;

    Ok(())
}

/// Load a DocumentTree from a file.
pub fn load_tree(path: &Path) -> Result<DocumentTree> {
    if !path.exists() {
        return Err(PageIndexError::IndexNotFound(path.to_path_buf()));
    }

    let format = SaveFormat::from_path(path);
    load_tree_with_format(path, format)
}

/// Load a DocumentTree with specific format.
pub fn load_tree_with_format(path: &Path, format: SaveFormat) -> Result<DocumentTree> {
    let data = fs::read(path).map_err(|e| PageIndexError::io(path, e))?;

    let tree = match format {
        SaveFormat::Json => {
            let json_str = String::from_utf8(data)
                .map_err(|e| PageIndexError::Serialization(e.to_string()))?;
            serde_json::from_str(&json_str)
                .map_err(|e| PageIndexError::Serialization(e.to_string()))?
        }
        SaveFormat::Bincode => {
            let config = bincode::config::standard();
            let (tree, _): (DocumentTree, usize) = bincode::decode_from_slice(&data, config)
                .map_err(|e| PageIndexError::Serialization(e.to_string()))?;
            tree
        }
    };

    Ok(tree)
}

/// Check if an index file exists at the given path.
pub fn tree_exists(path: &Path) -> bool {
    path.exists() && path.is_file()
}

/// Get the size of an index file in bytes.
pub fn tree_size(path: &Path) -> Result<u64> {
    let metadata = fs::metadata(path).map_err(|e| PageIndexError::io(path, e))?;
    Ok(metadata.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::TreeNode;
    use tempfile::TempDir;

    fn create_test_tree() -> DocumentTree {
        let mut ch1 = TreeNode::new("Chapter 1: Introduction", 1, 10);
        ch1.structure = Some("1".to_string());
        ch1.add_child(TreeNode::new("Section 1.1", 1, 5).with_structure("1.1"));
        ch1.add_child(TreeNode::new("Section 1.2", 6, 10).with_structure("1.2"));

        let mut ch2 = TreeNode::new("Chapter 2: Methods", 11, 20);
        ch2.structure = Some("2".to_string());

        DocumentTree::new("Test Document", vec![ch1, ch2], 20)
    }

    #[test]
    fn test_save_and_load_json() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test_tree.json");

        let original = create_test_tree();
        save_tree(&original, &path).unwrap();

        assert!(tree_exists(&path));

        let loaded = load_tree(&path).unwrap();

        assert_eq!(loaded.name, original.name);
        assert_eq!(loaded.total_pages, original.total_pages);
        assert_eq!(loaded.node_count(), original.node_count());
    }

    #[test]
    fn test_save_and_load_bincode() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test_tree.bin");

        let original = create_test_tree();
        save_tree(&original, &path).unwrap();

        assert!(tree_exists(&path));

        let loaded = load_tree(&path).unwrap();

        assert_eq!(loaded.name, original.name);
        assert_eq!(loaded.total_pages, original.total_pages);
    }

    #[test]
    fn test_format_detection() {
        assert_eq!(
            SaveFormat::from_path(Path::new("test.json")),
            SaveFormat::Json
        );
        assert_eq!(
            SaveFormat::from_path(Path::new("test.bin")),
            SaveFormat::Bincode
        );
        assert_eq!(
            SaveFormat::from_path(Path::new("test.bincode")),
            SaveFormat::Bincode
        );
        assert_eq!(SaveFormat::from_path(Path::new("test")), SaveFormat::Json);
    }

    #[test]
    fn test_load_nonexistent() {
        let result = load_tree(Path::new("/nonexistent/tree.json"));
        assert!(result.is_err());
    }

    #[test]
    fn test_tree_size() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test_tree.json");

        let tree = create_test_tree();
        save_tree(&tree, &path).unwrap();

        let size = tree_size(&path).unwrap();
        assert!(size > 0);
    }

    #[test]
    fn test_json_is_readable() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test_tree.json");

        let tree = create_test_tree();
        save_tree(&tree, &path).unwrap();

        // Read as text and verify it's valid JSON
        let content = fs::read_to_string(&path).unwrap();
        assert!(content.contains("Test Document"));
        assert!(content.contains("Chapter 1"));
    }
}
