use std::collections::BTreeMap;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use engine::backend::cpu::Cpu;
use engine::gguf::{Gguf, Qwen3Config};
use engine::model::Model;
use engine::sample::{Sampler, SamplerConfig, argmax};
use engine::session::{Message, Role, Session, render_chat};
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
    /// Chat with a model: one answer with --prompt, interactive without
    Run(RunArgs),
    /// Run a raw prompt and write every activation to a safetensors file (for ref/compare.py)
    Dump(DumpArgs),
}

#[derive(Args)]
struct RunArgs {
    #[arg(long)]
    model: PathBuf,
    /// User message. Without it, messages are read from stdin, one per line.
    #[arg(long)]
    prompt: Option<String>,
    /// Maximum number of tokens generated per answer
    #[arg(long, default_value_t = 512)]
    max_tokens: usize,
    /// Context size in tokens (prompt + answers), capped by the model's own limit
    #[arg(long, default_value_t = 4096)]
    ctx: usize,
    /// Temperature, 0 for greedy [default: the model's general.sampling.temp, else 1]
    #[arg(long)]
    temp: Option<f32>,
    /// [default: the model's general.sampling.top_k, else off]
    #[arg(long)]
    top_k: Option<usize>,
    /// [default: the model's general.sampling.top_p, else off]
    #[arg(long)]
    top_p: Option<f32>,
    /// [default: the model's general.sampling.min_p, else off]
    #[arg(long)]
    min_p: Option<f32>,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Disable Qwen3's thinking mode
    #[arg(long)]
    no_think: bool,
}

#[derive(Args)]
struct DumpArgs {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    prompt: String,
    #[arg(long)]
    out: PathBuf,
    /// Only write the tokens and the logits
    #[arg(long)]
    logits_only: bool,
    /// Also generate this many tokens greedily, without stopping on end tokens
    #[arg(long, default_value_t = 0)]
    r#gen: usize,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Inspect { file } => inspect(&file),
        Command::Run(args) => run(&args),
        Command::Dump(args) => dump(&args),
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

fn run(args: &RunArgs) -> Result<()> {
    let gguf = Gguf::open(&args.model)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let model = Model::load(&gguf, Cpu)?;

    // Sampling defaults recommended by the model file, when it has them.
    let meta = |key: &str| gguf.get(&format!("general.sampling.{key}"));
    let float = |key: &str| meta(key).and_then(|v| v.as_f32());
    let mut sampler = Sampler::new(SamplerConfig {
        temperature: args.temp.or(float("temp")).unwrap_or(1.0),
        top_k: args
            .top_k
            .or(meta("top_k").and_then(|v| v.as_u64()).map(|k| k as usize))
            .unwrap_or(0),
        top_p: args.top_p.or(float("top_p")).unwrap_or(1.0),
        min_p: args.min_p.or(float("min_p")).unwrap_or(0.0),
        seed: args.seed,
    });
    let mut session = Session::new(&model, args.ctx, tokenizer.stop_tokens());
    let mut messages: Vec<Message> = Vec::new();
    let mut stdout = std::io::stdout();

    let mut answer = |messages: &mut Vec<Message>, content: String| -> Result<()> {
        messages.push(Message {
            role: Role::User,
            content,
        });
        let prompt = tokenizer.encode(&render_chat(messages, !args.no_think));
        let mut stream = tokenizer.stream();
        let mut reply = String::new();
        let stats = session.generate(&prompt, &mut sampler, args.max_tokens, &mut |token| {
            let text = stream.push(token);
            print!("{text}");
            let _ = stdout.flush();
            reply.push_str(&text);
        })?;
        let rest = stream.finish();
        println!("{rest}");
        reply.push_str(&rest);
        messages.push(Message {
            role: Role::Assistant,
            content: reply,
        });
        eprintln!(
            "[prompt: {} tokens, {} evaluated at {:.1} tok/s | generated: {} tokens at {:.1} tok/s | first token: {:.2} s | end: {:?}]",
            stats.prompt_tokens,
            stats.prefill_tokens,
            stats.prefill_tokens as f64 / stats.prefill_secs,
            stats.generated_tokens,
            stats.generated_tokens as f64 / stats.decode_secs,
            stats.first_token_secs,
            stats.stop
        );
        Ok(())
    };

    if let Some(prompt) = &args.prompt {
        return answer(&mut messages, prompt.clone());
    }
    // Interactive: one user message per line, an empty line or end of input quits.
    let mut lines = std::io::stdin().lock().lines();
    loop {
        eprint!("> ");
        match lines.next().transpose()? {
            Some(line) if !line.trim().is_empty() => answer(&mut messages, line)?,
            _ => return Ok(()),
        }
    }
}

/// Tensors are named like the Hugging Face dump (ref/dump.py). Activations have one row per
/// prompt token (`[T, n]`, Q and K rows being `[n_head][head_dim]`), `gen_logits` one row per
/// generated token. All F32, except `tokens` and `gen_tokens` in I32.
fn dump(args: &DumpArgs) -> Result<()> {
    let gguf = Gguf::open(&args.model)?;
    let tokens = Tokenizer::from_gguf(&gguf)?.encode(&args.prompt);
    let model = Model::load(&gguf, Cpu)?;

    let mut traced: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    let mut state = model.new_state(tokens.len() + args.r#gen);
    let mut logits = Vec::new();
    for token in &tokens {
        logits = model.forward(&mut state, *token, &mut |name, data| {
            if !args.logits_only {
                traced
                    .entry(name.to_string())
                    .or_default()
                    .extend_from_slice(data);
            }
        });
        traced
            .entry("logits".into())
            .or_default()
            .extend_from_slice(&logits);
    }

    let mut gen_tokens = Vec::new();
    let mut gen_logits = Vec::new();
    for step in 0..args.r#gen {
        let token = argmax(&logits);
        gen_tokens.push(token);
        gen_logits.extend_from_slice(&logits);
        if step + 1 < args.r#gen {
            logits = model.forward(&mut state, token, &mut |_, _| {});
        }
    }

    // (name, dtype, rows, values as little-endian bytes)
    let ints = |v: &[u32]| v.iter().flat_map(|t| (*t as i32).to_le_bytes()).collect();
    let floats = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let mut entries: Vec<(&str, &str, usize, Vec<u8>)> =
        vec![("tokens", "I32", tokens.len(), ints(&tokens))];
    for (name, data) in &traced {
        entries.push((name, "F32", tokens.len(), floats(data)));
    }
    if args.r#gen > 0 {
        entries.push(("gen_tokens", "I32", args.r#gen, ints(&gen_tokens)));
        entries.push(("gen_logits", "F32", args.r#gen, floats(&gen_logits)));
    }

    // safetensors: u64 header length, JSON header, raw data.
    let mut offset = 0;
    let mut header = Vec::new();
    for (name, dtype, rows, bytes) in &entries {
        let cols = (bytes.len() / 4).checked_div(*rows).unwrap_or(0);
        let shape = if *dtype == "I32" {
            format!("[{rows}]")
        } else {
            format!("[{rows},{cols}]")
        };
        let end = offset + bytes.len();
        header.push(format!(
            r#""{name}":{{"dtype":"{dtype}","shape":{shape},"data_offsets":[{offset},{end}]}}"#
        ));
        offset = end;
    }
    let mut header = format!("{{{}}}", header.join(","));
    header.push_str(&" ".repeat(header.len().next_multiple_of(8) - header.len()));

    let file = std::fs::File::create(&args.out)
        .with_context(|| format!("creating {}", args.out.display()))?;
    let mut w = std::io::BufWriter::new(file);
    w.write_all(&(header.len() as u64).to_le_bytes())?;
    w.write_all(header.as_bytes())?;
    for (_, _, _, bytes) in &entries {
        w.write_all(bytes)?;
    }
    w.flush()?;
    Ok(())
}
