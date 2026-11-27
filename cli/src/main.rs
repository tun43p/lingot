use std::collections::BTreeMap;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use engine::backend::Backend;
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
    /// Measure prefill and decode speed, like llama-bench
    Bench(BenchArgs),
    /// Perplexity of a text file, computed like llama-perplexity
    Perplexity(PerplexityArgs),
    /// Run a raw prompt and write every activation to a safetensors file (for ref/compare.py)
    Dump(DumpArgs),
}

#[derive(Clone, Copy, ValueEnum)]
enum BackendKind {
    /// Scalar reference
    Cpu,
    /// GPU
    Metal,
}

#[cfg(target_os = "macos")]
fn metal() -> Result<engine::backend::metal::Metal> {
    engine::backend::metal::Metal::new().map_err(anyhow::Error::msg)
}

/// Call a function generic over the backend with the backend chosen on the command line.
macro_rules! with_backend {
    ($kind:expr, $function:ident, $args:expr) => {
        match $kind {
            BackendKind::Cpu => $function($args, Cpu),
            #[cfg(target_os = "macos")]
            BackendKind::Metal => $function($args, metal()?),
            #[cfg(not(target_os = "macos"))]
            BackendKind::Metal => anyhow::bail!("the Metal backend needs macOS"),
        }
    };
}

#[derive(Args)]
struct BenchArgs {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, value_enum, default_value = "metal")]
    backend: BackendKind,
    /// Prompt length for the prefill test (pp)
    #[arg(short = 'p', long, default_value_t = 512)]
    n_prompt: usize,
    /// Tokens generated for the decode test (tg)
    #[arg(short = 'n', long, default_value_t = 128)]
    n_gen: usize,
    #[arg(short, long, default_value_t = 5)]
    repetitions: usize,
}

#[derive(Args)]
struct RunArgs {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, value_enum, default_value = "metal")]
    backend: BackendKind,
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
struct PerplexityArgs {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, value_enum, default_value = "metal")]
    backend: BackendKind,
    /// Text file, e.g. WikiText-2's wiki.test.raw
    #[arg(short, long)]
    file: PathBuf,
    /// Chunk size in tokens
    #[arg(short, long, default_value_t = 512)]
    ctx: usize,
    /// Only use the first N chunks [default: all]
    #[arg(long)]
    chunks: Option<usize>,
}

#[derive(Args)]
struct DumpArgs {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, value_enum, default_value = "cpu")]
    backend: BackendKind,
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
        Command::Run(args) => with_backend!(args.backend, run, &args),
        Command::Bench(args) => with_backend!(args.backend, bench, &args),
        Command::Perplexity(args) => with_backend!(args.backend, perplexity, &args),
        Command::Dump(args) => with_backend!(args.backend, dump, &args),
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

fn run<B: Backend>(args: &RunArgs, backend: B) -> Result<()> {
    let gguf = Gguf::open(&args.model)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let model = Model::load(&gguf, backend)?;

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
            true
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

/// Same tests as `llama-bench -p N -n M`: `pp` evaluates a prompt of N tokens in one batch from
/// an empty context, `tg` generates M tokens one at a time from an empty context. Tokens are
/// random (fixed seed); one unmeasured warm-up run comes first.
fn bench<B: Backend>(args: &BenchArgs, backend: B) -> Result<()> {
    let gguf = Gguf::open(&args.model)?;
    let model = Model::load(&gguf, backend)?;
    let (n_prompt, n_gen) = (args.n_prompt, args.n_gen);
    anyhow::ensure!(n_prompt > 0 && n_gen > 0 && args.repetitions > 0);

    let mut seed = 0x2545_F491_4F6C_DD1Du64; // xorshift
    let mut random_token = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % model.config.vocab_size as u64) as u32
    };
    let prompt: Vec<u32> = (0..n_prompt).map(|_| random_token()).collect();
    let first = random_token();
    let mut state = model.new_state(n_prompt.max(n_gen));

    let prefill = |state: &mut _| {
        state_reset(state);
        let start = std::time::Instant::now();
        model.forward(state, &prompt, n_prompt - 1, None);
        n_prompt as f64 / start.elapsed().as_secs_f64()
    };
    let decode = |state: &mut _| {
        state_reset(state);
        let mut token = first;
        let start = std::time::Instant::now();
        for _ in 0..n_gen {
            token = argmax(&model.forward(state, &[token], 0, None));
        }
        n_gen as f64 / start.elapsed().as_secs_f64()
    };

    prefill(&mut state);
    decode(&mut state);
    let pp: Vec<f64> = (0..args.repetitions).map(|_| prefill(&mut state)).collect();
    let tg: Vec<f64> = (0..args.repetitions).map(|_| decode(&mut state)).collect();

    let summary = |runs: &[f64]| {
        let mean = runs.iter().sum::<f64>() / runs.len() as f64;
        let var =
            runs.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (runs.len().max(2) - 1) as f64;
        format!("{mean:.2} ± {:.2}", var.sqrt())
    };
    let name = args.model.file_name().unwrap_or_default().to_string_lossy();
    println!("| model | test | t/s |");
    println!("| --- | --- | --- |");
    println!("| {name} | pp{n_prompt} | {} |", summary(&pp));
    println!("| {name} | tg{n_gen} | {} |", summary(&tg));
    Ok(())
}

fn state_reset<B: Backend>(state: &mut engine::model::State<B>) {
    state.truncate(0);
}

/// Same method as llama-perplexity: the text is cut into chunks of `ctx` tokens, each chunk is
/// evaluated from an empty context, and only the second half of each chunk is scored (the
/// tokens at positions ctx/2 + 1 .. ctx - 1, given everything before them).
fn perplexity<B: Backend>(args: &PerplexityArgs, backend: B) -> Result<()> {
    let gguf = Gguf::open(&args.model)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let model = Model::load(&gguf, backend)?;
    let text = std::fs::read_to_string(&args.file)
        .with_context(|| format!("reading {}", args.file.display()))?;
    let tokens = tokenizer.encode(&text);

    let n_ctx = args.ctx;
    anyhow::ensure!(n_ctx >= 4, "--ctx must be at least 4");
    let available = tokens.len() / n_ctx;
    let n_chunks = args.chunks.map_or(available, |n| n.min(available));
    anyhow::ensure!(n_chunks > 0, "the file has fewer than {n_ctx} tokens");
    eprintln!(
        "{} tokens, calculating perplexity over {n_chunks} chunks, n_ctx={n_ctx}",
        tokens.len()
    );

    let first = n_ctx / 2;
    let (mut nll, mut nll2, mut count) = (0.0f64, 0.0f64, 0usize);
    let mut state = model.new_state(n_ctx);
    for (i, chunk) in tokens.chunks_exact(n_ctx).take(n_chunks).enumerate() {
        state.truncate(0);
        // The last token of a chunk is only a target: no need to evaluate it.
        let vocab = model.config.vocab_size;
        let all = model.forward(&mut state, &chunk[..n_ctx - 1], first, None);
        for (j, logits) in (first..).zip(all.chunks_exact(vocab)) {
            // -log softmax(logits)[target], subtracting the max first.
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f64 = logits.iter().map(|l| ((l - max) as f64).exp()).sum();
            let loss = sum.ln() - (logits[chunk[j + 1] as usize] - max) as f64;
            nll += loss;
            nll2 += loss * loss;
            count += 1;
        }
        print!("[{}]{:.4},", i + 1, (nll / count as f64).exp());
        std::io::stdout().flush()?;
    }
    println!();

    let mean = nll / count as f64;
    let ppl = mean.exp();
    let variance = nll2 / count as f64 - mean * mean;
    let error = if variance > 0.0 {
        (variance / (count - 1) as f64).sqrt() * ppl
    } else {
        0.0
    };
    println!("Final estimate: PPL = {ppl:.4} +/- {error:.5}");
    Ok(())
}

/// Tensors are named like the Hugging Face dump (ref/dump.py). Activations have one row per
/// prompt token (`[T, n]`, Q and K rows being `[n_head][head_dim]`), `gen_logits` one row per
/// generated token. All F32, except `tokens` and `gen_tokens` in I32.
fn dump<B: Backend>(args: &DumpArgs, backend: B) -> Result<()> {
    let gguf = Gguf::open(&args.model)?;
    let tokens = Tokenizer::from_gguf(&gguf)?.encode(&args.prompt);
    let model = Model::load(&gguf, backend)?;

    let mut traced: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    let mut state = model.new_state(tokens.len() + args.r#gen);
    anyhow::ensure!(!tokens.is_empty(), "empty prompt");
    let mut record = |name: &str, data: &[f32]| {
        traced.insert(name.to_string(), data.to_vec());
    };
    let trace: Option<engine::model::Trace> = if args.logits_only {
        None
    } else {
        Some(&mut record)
    };
    let all = model.forward(&mut state, &tokens, 0, trace);
    let mut logits = all[all.len() - model.config.vocab_size..].to_vec();
    traced.insert("logits".into(), all);

    let mut gen_tokens = Vec::new();
    let mut gen_logits = Vec::new();
    for step in 0..args.r#gen {
        let token = argmax(&logits);
        gen_tokens.push(token);
        gen_logits.extend_from_slice(&logits);
        if step + 1 < args.r#gen {
            logits = model.forward(&mut state, &[token], 0, None);
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
