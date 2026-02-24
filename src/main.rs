//! Rust Page Indexer CLI
//!
//! An LLM-powered hierarchical tree indexing system for document search.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use rust_page_indexer::{
    config::Config,
    document::Document,
    indexer::TreeIndexer,
    llm::LlmClient,
    persistence::{load_tree, save_tree, tree_exists, tree_size},
    search::TreeSearcher,
};
use std::path::PathBuf;
use std::time::Instant;

/// Rust Page Indexer - An LLM-powered hierarchical tree indexing system
#[derive(Parser)]
#[command(name = "page-indexer")]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build a tree index for a document
    Index {
        /// Path to the document file (text file)
        document: PathBuf,

        /// Output path for the tree index file
        #[arg(short, long, default_value = "data/tree_index.json")]
        output: PathBuf,
    },

    /// Search a tree index using LLM reasoning
    Search {
        /// The search query
        query: String,

        /// Path to the tree index file
        #[arg(short, long, default_value = "data/tree_index.json")]
        index: PathBuf,

        /// Number of results to return
        #[arg(short = 'k', long, default_value_t = 5)]
        top_k: usize,

        /// Include section content in results
        #[arg(long)]
        with_content: bool,

        /// Path to the original document (required if --with-content is set)
        #[arg(short, long)]
        document: Option<PathBuf>,
    },

    /// Display the tree structure of an index
    Show {
        /// Path to the tree index file
        #[arg(default_value = "data/tree_index.json")]
        index: PathBuf,

        /// Output as JSON instead of formatted tree
        #[arg(long)]
        json: bool,
    },

    /// Show information about an index
    Info {
        /// Path to the tree index file
        #[arg(default_value = "data/tree_index.json")]
        index: PathBuf,
    },

    /// Test LLM connection
    Test,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Index { document, output } => cmd_index(document, output).await,
        Commands::Search {
            query,
            index,
            top_k,
            with_content,
            document,
        } => cmd_search(query, index, top_k, with_content, document).await,
        Commands::Show { index, json } => cmd_show(index, json),
        Commands::Info { index } => cmd_info(index),
        Commands::Test => cmd_test().await,
    }
}

async fn cmd_index(document_path: PathBuf, output: PathBuf) -> Result<()> {
    println!("Loading configuration...");
    let config = Config::load().context("Failed to load configuration")?;
    config.validate().context("Invalid configuration")?;

    println!("Indexing document: {}", document_path.display());
    println!("Using model: {}", config.llm.model);

    let start = Instant::now();

    // Load document
    let document = Document::from_text_file(&document_path).context("Failed to load document")?;

    println!(
        "  Document: {} ({} pages, ~{} tokens)",
        document.name,
        document.page_count(),
        document.total_tokens()
    );

    // Create client and indexer
    let client = LlmClient::new(config.llm);
    let indexer = TreeIndexer::new(client);

    // Build tree index
    println!("\nBuilding tree index via LLM...");
    let tree = indexer
        .index(&document)
        .await
        .context("Failed to build tree index")?;

    let build_duration = start.elapsed();

    // Show stats
    println!("\nTree Index Built:");
    println!("  Sections:    {}", tree.node_count());
    println!("  Max depth:   {}", tree.max_depth());
    println!("  Build time:  {:.2?}", build_duration);

    // Save tree
    save_tree(&tree, &output).context("Failed to save tree index")?;

    let size = tree_size(&output)?;
    println!("\nIndex saved to: {}", output.display());
    println!("  File size: {:.1} KB", size as f64 / 1024.0);

    Ok(())
}

async fn cmd_search(
    query: String,
    index_path: PathBuf,
    top_k: usize,
    with_content: bool,
    document_path: Option<PathBuf>,
) -> Result<()> {
    if !tree_exists(&index_path) {
        anyhow::bail!(
            "Index not found at '{}'. Run 'index' command first.",
            index_path.display()
        );
    }

    if with_content && document_path.is_none() {
        anyhow::bail!("--document is required when using --with-content");
    }

    println!("Loading configuration...");
    let config = Config::load().context("Failed to load configuration")?;
    config.validate().context("Invalid configuration")?;

    let tree = load_tree(&index_path).context("Failed to load tree index")?;

    let client = LlmClient::new(config.llm.clone());
    let searcher = TreeSearcher::new(client);

    println!("Searching for: \"{}\"", query);
    println!("Using model: {}", config.llm.model);
    println!();

    let start = Instant::now();

    let results = if with_content {
        let document = Document::from_text_file(document_path.as_ref().unwrap())
            .context("Failed to load document")?;
        searcher
            .search_with_content(&tree, &document, &query)
            .await
            .context("Search failed")?
    } else {
        searcher
            .search(&tree, &query)
            .await
            .context("Search failed")?
    };

    let search_duration = start.elapsed();

    if results.is_empty() {
        println!("No relevant sections found.");
    } else {
        println!("Results:");
        println!("{}", "─".repeat(60));

        for (i, result) in results.iter().take(top_k).enumerate() {
            println!(
                "{:>2}. {} [pages {}-{}] ({:?})",
                i + 1,
                result.title,
                result.start_index,
                result.end_index,
                result.relevance
            );
            println!("    Reason: {}", result.reason);

            if let Some(content) = &result.content {
                println!("    Content preview:");
                let preview: String = content.chars().take(200).collect();
                for line in preview.lines().take(3) {
                    println!("      {}", line);
                }
                if content.len() > 200 {
                    println!("      ...");
                }
            }
            println!();
        }

        println!("{}", "─".repeat(60));
        println!("Found {} results in {:.2?}", results.len(), search_duration);
    }

    Ok(())
}

fn cmd_show(index_path: PathBuf, json: bool) -> Result<()> {
    if !tree_exists(&index_path) {
        anyhow::bail!(
            "Index not found at '{}'. Run 'index' command first.",
            index_path.display()
        );
    }

    let tree = load_tree(&index_path).context("Failed to load tree index")?;

    if json {
        let json_str = tree.to_json().context("Failed to serialize tree")?;
        println!("{}", json_str);
    } else {
        println!("{}", tree.format());
    }

    Ok(())
}

fn cmd_info(index_path: PathBuf) -> Result<()> {
    if !tree_exists(&index_path) {
        anyhow::bail!(
            "Index not found at '{}'. Run 'index' command first.",
            index_path.display()
        );
    }

    let tree = load_tree(&index_path).context("Failed to load tree index")?;
    let size = tree_size(&index_path)?;

    println!("Tree Index Information");
    println!("{}", "─".repeat(40));
    println!("  Document:     {}", tree.name);
    println!("  Total pages:  {}", tree.total_pages);
    println!("  Sections:     {}", tree.node_count());
    println!("  Max depth:    {}", tree.max_depth());
    println!("  File size:    {:.1} KB", size as f64 / 1024.0);
    println!("  Index path:   {}", index_path.display());

    if let Some(desc) = &tree.description {
        println!("  Description:  {}", desc);
    }

    Ok(())
}

async fn cmd_test() -> Result<()> {
    println!("Testing LLM connection...\n");

    let config = Config::load().context("Failed to load configuration")?;

    println!("Configuration:");
    println!("  API Base:  {}", config.llm.api_base);
    println!("  Model:     {}", config.llm.model);
    println!(
        "  API Key:   {}...",
        &config.llm.api_key[..config.llm.api_key.len().min(8)]
    );
    println!();

    if let Err(e) = config.validate() {
        println!("Configuration error: {}", e);
        return Ok(());
    }

    let client = LlmClient::new(config.llm);

    println!("Sending test request...");
    match client.test_connection().await {
        Ok(()) => {
            println!("Connection successful!");
        }
        Err(e) => {
            println!("Connection failed: {}", e);
        }
    }

    Ok(())
}
