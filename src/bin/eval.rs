//! Evaluation CLI binary for comparing PageIndex vs Vector Search.
//!
//! Usage:
//!   eval sample              # Run on built-in sample dataset
//!   eval quality <path>      # Run on QuALITY dataset
//!   eval custom <path>       # Run on custom JSON dataset
//!
//! Options:
//!   --max-items <N>          # Limit number of items
//!   --top-k <N>              # Number of results to retrieve (default: 3)
//!   --pageindex-only         # Only run PageIndex
//!   --vector-only            # Only run vector search
//!   --verbose                # Verbose output
//!   --output <path>          # Save results to JSON file

use anyhow::Result;
use clap::{Parser, Subcommand};
use rust_page_indexer::config::Config;
use rust_page_indexer::eval::{
    Benchmark, BenchmarkConfig, BenchmarkResults, ChunkConfig, create_sample_dataset,
    load_quality_dataset, load_simple_dataset,
};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "eval")]
#[command(about = "Evaluate PageIndex vs Vector Search", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Maximum number of items to evaluate
    #[arg(long, global = true)]
    max_items: Option<usize>,

    /// Number of top results to retrieve
    #[arg(long, global = true, default_value = "3")]
    top_k: usize,

    /// Only run PageIndex (skip vector search)
    #[arg(long, global = true)]
    pageindex_only: bool,

    /// Only run vector search (skip PageIndex)
    #[arg(long, global = true)]
    vector_only: bool,

    /// Verbose output
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Save results to JSON file
    #[arg(short, long, global = true)]
    output: Option<PathBuf>,

    /// Chunk size for vector search
    #[arg(long, global = true, default_value = "512")]
    chunk_size: usize,

    /// Chunk overlap for vector search
    #[arg(long, global = true, default_value = "50")]
    chunk_overlap: usize,
}

#[derive(Subcommand)]
enum Commands {
    /// Run on built-in sample dataset (for quick testing)
    Sample,

    /// Run on QuALITY dataset (JSONL format)
    Quality {
        /// Path to QuALITY dataset file
        path: PathBuf,
    },

    /// Run on custom JSON dataset
    Custom {
        /// Path to custom dataset JSON file
        path: PathBuf,
    },

    /// Download QuALITY dataset
    Download {
        /// Output directory
        #[arg(default_value = "data")]
        output_dir: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Handle download command separately
    if let Commands::Download { output_dir } = &cli.command {
        return download_quality(output_dir).await;
    }

    // Load LLM config
    let config = Config::load()?;
    let llm_config = config.llm;

    println!("LLM API Base: {}", llm_config.api_base);
    println!("LLM Model: {}", llm_config.model);

    // Load dataset
    let dataset = match &cli.command {
        Commands::Sample => {
            println!("Using sample dataset...");
            create_sample_dataset()
        }
        Commands::Quality { path } => {
            println!("Loading QuALITY dataset from {:?}...", path);
            load_quality_dataset(path)?
        }
        Commands::Custom { path } => {
            println!("Loading custom dataset from {:?}...", path);
            load_simple_dataset(path)?
        }
        Commands::Download { .. } => unreachable!(),
    };

    println!("Dataset: {} ({} items)", dataset.name, dataset.len());

    // Build benchmark config
    let benchmark_config = BenchmarkConfig {
        top_k: cli.top_k,
        chunk_config: ChunkConfig {
            chunk_size: cli.chunk_size,
            chunk_overlap: cli.chunk_overlap,
        },
        run_pageindex: !cli.vector_only,
        run_vector: !cli.pageindex_only,
        max_items: cli.max_items,
        verbose: cli.verbose,
    };

    // Validate config
    if cli.pageindex_only && cli.vector_only {
        eprintln!("Error: Cannot use both --pageindex-only and --vector-only");
        std::process::exit(1);
    }

    if !benchmark_config.run_pageindex && !benchmark_config.run_vector {
        eprintln!("Error: At least one search method must be enabled");
        std::process::exit(1);
    }

    // Run benchmark
    let benchmark = Benchmark::new(llm_config, benchmark_config);
    let results: BenchmarkResults = benchmark.run(&dataset).await?;

    // Print summary
    results.print_summary();

    // Save results if requested
    if let Some(output_path) = cli.output {
        let json = serde_json::to_string_pretty(&results)?;
        std::fs::write(&output_path, json)?;
        println!("Results saved to {:?}", output_path);
    }

    Ok(())
}

/// Download QuALITY dataset from the official NYU source.
async fn download_quality(output_dir: &PathBuf) -> Result<()> {
    use std::fs;

    // Create output directory
    fs::create_dir_all(output_dir)?;

    // Official NYU QuALITY dataset source
    let url = "https://raw.githubusercontent.com/nyu-mll/quality/main/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.dev";
    let output_path = output_dir.join("quality_dev.jsonl");

    println!("Downloading QuALITY dataset...");
    println!("URL: {}", url);
    println!("Output: {:?}", output_path);

    let response = reqwest::get(url).await?;

    if !response.status().is_success() {
        eprintln!("Failed to download: HTTP {}", response.status());
        std::process::exit(1);
    }

    let content = response.text().await?;
    fs::write(&output_path, &content)?;

    // Count items
    let item_count = content.lines().filter(|l| !l.trim().is_empty()).count();

    println!("Downloaded {} items to {:?}", item_count, output_path);
    println!("\nTo run evaluation:");
    println!("  eval quality {:?}", output_path);

    Ok(())
}
