use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use engine::backend::cpu::Cpu;
use engine::gguf::{Gguf, Qwen3Config};
use engine::model::Model;
use engine::tokenizer::Tokenizer;

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
    /// Run a prompt and write every activation to a safetensors file (for ref/compare.py)
    Dump {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long)]
        out: PathBuf,
        /// Only write the tokens and the logits
        #[arg(long)]
        logits_only: bool,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Inspect { file } => inspect(&file),
        Command::Dump {
            model,
            prompt,
            out,
            logits_only,
        } => dump(&model, &prompt, &out, logits_only),
    }
}

fn inspect(file: &Path) -> Result<()> {
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

/// Tensors are named like the Hugging Face dump (ref/dump.py), one row per prompt token:
/// `[T, n]` in F32, plus `tokens` `[T]` in I32. Q and K rows are `[n_head][head_dim]`.
fn dump(model: &Path, prompt: &str, out: &Path, logits_only: bool) -> Result<()> {
    let gguf = Gguf::open(model)?;
    let tokens = Tokenizer::from_gguf(&gguf)?.encode(prompt);
    let model = Model::load(&gguf, Cpu)?;

    let mut tensors: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    let mut state = model.new_state();
    for token in &tokens {
        let logits = model.forward(&mut state, *token, &mut |name, data| {
            if !logits_only {
                tensors
                    .entry(name.to_string())
                    .or_default()
                    .extend_from_slice(data);
            }
        });
        tensors.entry("logits".into()).or_default().extend(logits);
    }

    // safetensors: u64 header length, JSON header, raw little-endian data.
    let n = tokens.len();
    let mut entries = vec![format!(
        r#""tokens":{{"dtype":"I32","shape":[{n}],"data_offsets":[0,{}]}}"#,
        n * 4
    )];
    let mut offset = n * 4;
    for (name, data) in &tensors {
        let end = offset + data.len() * 4;
        entries.push(format!(
            r#""{name}":{{"dtype":"F32","shape":[{n},{}],"data_offsets":[{offset},{end}]}}"#,
            data.len().checked_div(n).unwrap_or(0)
        ));
        offset = end;
    }
    let mut header = format!("{{{}}}", entries.join(","));
    header.push_str(&" ".repeat(header.len().next_multiple_of(8) - header.len()));

    let file = std::fs::File::create(out).with_context(|| format!("creating {}", out.display()))?;
    let mut w = std::io::BufWriter::new(file);
    w.write_all(&(header.len() as u64).to_le_bytes())?;
    w.write_all(header.as_bytes())?;
    for token in &tokens {
        w.write_all(&(*token as i32).to_le_bytes())?;
    }
    for data in tensors.values() {
        for v in data {
            w.write_all(&v.to_le_bytes())?;
        }
    }
    w.flush()?;
    Ok(())
}
