# Rust Page Indexer

A Rust port of [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex) - an LLM-powered hierarchical tree indexing system for document search.

## Overview

Unlike traditional RAG systems that chunk documents and use vector embeddings, PageIndex:

1. **Uses LLM reasoning** to extract the document's hierarchical structure (table of contents, sections, subsections)
2. **Maps sections to physical page indices** for precise retrieval
3. **Navigates the tree using LLM reasoning** during search to find relevant sections

This approach provides more accurate retrieval for structured documents like books, papers, and technical documentation.

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd rust_page_indexer

# Build release binary
cargo build --release

# The binary will be at ./target/release/rust_page_indexer
```

## Configuration

The indexer requires an LLM API endpoint. It supports any OpenAI-compatible API.

### Option 1: Environment Variables (Recommended)

```bash
export LLM_API_BASE="https://api.openai.com"      # Or your LLM proxy URL
export LLM_API_KEY="your-api-key"
export LLM_MODEL="gpt-4"                          # Or claude-latest, etc.

# Optional
export LLM_MAX_TOKENS="4096"
export LLM_TEMPERATURE="0.0"
```

### Option 2: Configuration File

Create `~/.config/rust-page-indexer/config.yaml`:

```yaml
llm:
  api_base: "https://api.openai.com"
  api_key: "your-api-key"
  model: "gpt-4"
  max_tokens: 4096
  temperature: 0.0
```

**Note:** Environment variables take precedence over the config file.

### Using with LLM Proxies

If you have access to an OpenAI-compatible LLM proxy, configure it as:

```bash
export LLM_API_BASE="https://your-llm-proxy.example.com/"
export LLM_API_KEY="your-proxy-key"
export LLM_MODEL="claude-3-opus"  # or gpt-4, etc.
```

## Usage

### Test LLM Connection

```bash
./target/release/rust_page_indexer test
```

This verifies your LLM configuration is working.

### Index a Document

```bash
# Index a text file
./target/release/rust_page_indexer index document.txt

# Specify output path
./target/release/rust_page_indexer index document.txt -o my_index.json
```

The indexer will:
1. Load the document
2. Call the LLM to extract hierarchical structure
3. Save the tree index as JSON (or bincode with `.bin` extension)

### Search the Index

```bash
# Basic search
./target/release/rust_page_indexer search "What is ownership in Rust?"

# Specify index file
./target/release/rust_page_indexer search "query" -i my_index.json

# Limit results
./target/release/rust_page_indexer search "query" -k 3

# Include content snippets (requires original document)
./target/release/rust_page_indexer search "query" --with-content -d document.txt
```

### View Index Structure

```bash
# Pretty-print tree structure
./target/release/rust_page_indexer show data/tree_index.json

# Output as JSON
./target/release/rust_page_indexer show data/tree_index.json --json
```

### Index Information

```bash
./target/release/rust_page_indexer info data/tree_index.json
```

## CLI Reference

```
rust_page_indexer <COMMAND>

Commands:
  index   Build a tree index for a document
  search  Search a tree index using LLM reasoning
  show    Display the tree structure of an index
  info    Show information about an index
  test    Test LLM connection
  help    Print this message or the help of the given subcommand(s)
```

### index

```
rust_page_indexer index [OPTIONS] <DOCUMENT>

Arguments:
  <DOCUMENT>  Path to the document file (text file)

Options:
  -o, --output <OUTPUT>  Output path for the tree index file [default: data/tree_index.json]
```

### search

```
rust_page_indexer search [OPTIONS] <QUERY>

Arguments:
  <QUERY>  The search query

Options:
  -i, --index <INDEX>        Path to the tree index file [default: data/tree_index.json]
  -k, --top-k <TOP_K>        Number of results to return [default: 5]
      --with-content         Include section content in results
  -d, --document <DOCUMENT>  Path to the original document (required with --with-content)
```

## Library Usage

```rust
use rust_page_indexer::{
    config::Config,
    document::Document,
    indexer::TreeIndexer,
    search::TreeSearcher,
    llm::LlmClient,
    persistence::{save_tree, load_tree},
};
use std::path::Path;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load configuration
    let config = Config::load()?;
    config.validate()?;

    // Create LLM client
    let client = LlmClient::new(config.llm.clone());

    // Load a document
    let document = Document::from_text_file(Path::new("document.txt"))?;

    // Build tree index
    let indexer = TreeIndexer::new(client.clone());
    let tree = indexer.index(&document).await?;

    // Save the tree for later use
    save_tree(&tree, Path::new("tree_index.json"))?;

    // Search the tree
    let searcher = TreeSearcher::new(client);
    let results = searcher.search(&tree, "your query here").await?;

    for result in results {
        println!("{}: {:?} (pages {}-{})", 
            result.title, 
            result.relevance,
            result.start_index,
            result.end_index
        );
        println!("  Reason: {}", result.reason);
    }

    Ok(())
}
```

## Architecture

```
src/
├── main.rs          # CLI entry point
├── lib.rs           # Library exports
├── config.rs        # Configuration (env vars + YAML)
├── document.rs      # Page-based document model
├── tree.rs          # TreeNode/DocumentTree structures
├── indexer.rs       # LLM-based tree generation
├── search.rs        # LLM reasoning search
├── persistence.rs   # JSON/bincode serialization
├── error.rs         # Error types
└── llm/
    ├── mod.rs
    ├── client.rs    # OpenAI-compatible HTTP client
    └── prompts.rs   # LLM prompts (ported from Python)
```

## How It Works

### Indexing

1. **Document Loading**: Text files are loaded as single-page documents (PDF support planned)
2. **Structure Extraction**: The LLM analyzes the document and extracts:
   - Section titles and hierarchy (e.g., "1. Introduction", "1.1 Background")
   - Physical page indices where each section starts
3. **Tree Construction**: Flat TOC items are converted into a hierarchical tree
4. **Persistence**: Tree is saved as JSON (human-readable) or bincode (compact)

### Searching

1. **Tree Analysis**: The LLM receives the tree structure and your query
2. **Reasoning**: The LLM reasons about which sections are relevant
3. **Results**: Returns ranked sections with:
   - Title and page range
   - Relevance level (high/medium/low)
   - Explanation of why it's relevant

## Differences from Python PageIndex

| Feature | Python Original | Rust Port |
|---------|----------------|-----------|
| PDF Support | Yes | Planned |
| TOC Detection | Yes (multi-mode) | Simplified (direct generation) |
| Page Verification | Yes | Planned |
| Async | asyncio | tokio |
| Config | Command args | Env vars + YAML |

## Roadmap

- [ ] PDF document support
- [ ] TOC detection and extraction
- [ ] Page index verification
- [ ] Multi-page document chunking
- [ ] Streaming LLM responses
- [ ] Batch indexing

## License

MIT

## Credits

Based on [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex).
