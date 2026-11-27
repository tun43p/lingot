use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use engine::gguf::{Gguf, Qwen3Config};

#[derive(Parser)]
#[command(name = "lingot", version, about = "LLM inference engine")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Print the metadata and tensors of a GGUF file
    Inspect { file: PathBuf },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Inspect { file } => inspect(&file),
    }
}

fn inspect(file: &PathBuf) -> Result<()> {
    let g = Gguf::open(file)?;
    println!(
        "GGUF v{}: {} metadata, {} tensors, alignment {}, data offset {}",
        g.version,
        g.metadata.len(),
        g.tensors.len(),
        g.alignment,
        g.data_offset
    );

    println!("\nmetadata:");
    for (key, value) in &g.metadata {
        println!("  {key} = {value}");
    }

    println!("\nconfig:\n  {:?}", Qwen3Config::from_gguf(&g)?);

    // One tab-separated line per tensor: name, type, dims (GGUF order), file offset, bytes.
    println!("\ntensors:");
    for t in &g.tensors {
        let dims: Vec<String> = t.dims.iter().map(u64::to_string).collect();
        println!(
            "  {}\t{}\t{}\t{}\t{}",
            t.name,
            t.ty,
            dims.join("x"),
            t.file_offset(),
            t.n_bytes()
        );
    }
    Ok(())
}
