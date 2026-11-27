//! lingot-server: OpenAI-compatible chat completions for a single user.
//!
//! The model lives on one worker thread, which takes requests from a queue and runs them one
//! at a time. HTTP handlers only parse, enqueue and relay what the worker sends back.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, mpsc as queue};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::{Parser, ValueEnum};
use engine::backend::Backend;
use engine::backend::cpu::Cpu;
use engine::gguf::Gguf;
use engine::model::Model;
use engine::sample::{Sampler, SamplerConfig};
use engine::session::{Message, Role, Session, StopReason, render_chat};
use engine::tokenizer::Tokenizer;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::mpsc;

#[derive(Parser)]
#[command(
    name = "lingot-server",
    version,
    about = "OpenAI-compatible server for lingot"
)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, value_enum, default_value = "metal")]
    backend: BackendKind,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8080)]
    port: u16,
    /// Context size in tokens (prompt + answer), capped by the model's own limit
    #[arg(long, default_value_t = 4096)]
    ctx: usize,
    /// Disable Qwen3's thinking mode
    #[arg(long)]
    no_think: bool,
}

#[derive(Clone, Copy, ValueEnum)]
enum BackendKind {
    Cpu,
    Metal,
}

/// One chat completion to run.
struct Job {
    messages: Vec<Message>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    seed: u64,
    max_tokens: usize,
    stop: Vec<String>,
    /// Where the worker reports; when the receiver is gone the client has left.
    events: mpsc::Sender<Event>,
}

enum Event {
    Text(String),
    Done {
        prompt_tokens: usize,
        completion_tokens: usize,
        finish_reason: &'static str,
    },
    Error(String),
}

/// Owns the model and answers jobs one at a time, in order of arrival.
fn worker<B: Backend>(
    args: &Args,
    backend: B,
    ready: queue::Sender<Result<()>>,
    jobs: queue::Receiver<Job>,
) {
    let loaded = (|| {
        let gguf = Gguf::open(&args.model)?;
        let tokenizer = Tokenizer::from_gguf(&gguf)?;
        Ok::<_, engine::gguf::Error>((gguf, tokenizer))
    })();
    let (gguf, tokenizer) = match loaded {
        Ok(loaded) => loaded,
        Err(e) => return drop(ready.send(Err(e.into()))),
    };
    let model = match Model::load(&gguf, backend) {
        Ok(model) => model,
        Err(e) => return drop(ready.send(Err(e.into()))),
    };
    // Sampling defaults recommended by the model file, when it has them.
    let meta = |key: &str| gguf.get(&format!("general.sampling.{key}"));
    let float = |key: &str| meta(key).and_then(|v| v.as_f32());
    let defaults = SamplerConfig {
        temperature: float("temp").unwrap_or(1.0),
        top_k: meta("top_k").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
        top_p: float("top_p").unwrap_or(1.0),
        min_p: float("min_p").unwrap_or(0.0),
        seed: 0,
    };
    let mut session = Session::new(&model, args.ctx, tokenizer.stop_tokens());
    let _ = ready.send(Ok(()));

    for job in jobs {
        if job.events.is_closed() {
            continue; // the client left while waiting in the queue
        }
        let prompt = tokenizer.encode(&render_chat(&job.messages, !args.no_think));
        let mut sampler = Sampler::new(SamplerConfig {
            temperature: job.temperature.unwrap_or(defaults.temperature),
            top_p: job.top_p.unwrap_or(defaults.top_p),
            seed: job.seed,
            ..defaults
        });

        let mut stream = tokenizer.stream();
        let mut held = String::new(); // text not sent yet: it may be the start of a stop string
        let mut hit_stop_string = false;
        let mut client_gone = false;
        let result = session.generate(&prompt, &mut sampler, job.max_tokens, &mut |token| {
            held.push_str(&stream.push(token));
            let (ready_len, stopped) = split_at_stop(&held, &job.stop);
            hit_stop_string = stopped;
            let text: String = held.drain(..ready_len).collect();
            if !text.is_empty() && job.events.blocking_send(Event::Text(text)).is_err() {
                client_gone = true;
            }
            !stopped && !client_gone
        });

        let event = match result {
            Err(e) => Event::Error(e.to_string()),
            Ok(_) if client_gone => continue,
            Ok(stats) => {
                if !hit_stop_string {
                    held.push_str(&stream.finish());
                    if !held.is_empty() {
                        let _ = job.events.blocking_send(Event::Text(held));
                    }
                }
                Event::Done {
                    prompt_tokens: stats.prompt_tokens,
                    completion_tokens: stats.generated_tokens,
                    finish_reason: match stats.stop {
                        StopReason::Stop | StopReason::Cancelled => "stop",
                        StopReason::MaxTokens | StopReason::ContextFull => "length",
                    },
                }
            }
        };
        let _ = job.events.blocking_send(event);
    }
}

/// How many bytes of `text` can be sent now, and whether a stop string was reached.
/// Text that could still turn into a stop string is held back.
fn split_at_stop(text: &str, stops: &[String]) -> (usize, bool) {
    if let Some(at) = stops.iter().filter_map(|s| text.find(s.as_str())).min() {
        return (at, true);
    }
    let held = stops
        .iter()
        .flat_map(|s| {
            (1..s.len())
                .filter(|n| s.is_char_boundary(*n))
                .map(move |n| &s[..n])
        })
        .filter(|prefix| text.ends_with(prefix))
        .map(str::len)
        .max()
        .unwrap_or(0);
    (text.len() - held, false)
}

struct App {
    jobs: queue::Sender<Job>,
    model: String,
    started: u64,
    requests: AtomicU64,
}

#[derive(Deserialize)]
struct ChatRequest {
    messages: Vec<RequestMessage>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<usize>,
    max_completion_tokens: Option<usize>,
    stop: Option<OneOrMany>,
    seed: Option<u64>,
    #[serde(default)]
    stream: bool,
}

#[derive(Deserialize)]
struct RequestMessage {
    role: String,
    content: Content,
}

/// Message content is a string, or a list of parts of which only the text ones are used.
#[derive(Deserialize)]
#[serde(untagged)]
enum Content {
    Text(String),
    Parts(Vec<Part>),
}

#[derive(Deserialize)]
struct Part {
    #[serde(rename = "type")]
    kind: String,
    text: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum OneOrMany {
    One(String),
    Many(Vec<String>),
}

fn bad_request(message: impl Into<String>) -> Response {
    let body = json!({"error": {"message": message.into(), "type": "invalid_request_error"}});
    (StatusCode::BAD_REQUEST, Json(body)).into_response()
}

fn server_error(message: impl Into<String>) -> Response {
    let body = json!({"error": {"message": message.into(), "type": "server_error"}});
    (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

async fn models(State(app): State<Arc<App>>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{"id": app.model, "object": "model", "created": app.started, "owned_by": "lingot"}],
    }))
}

fn to_messages(request: &[RequestMessage]) -> Result<Vec<Message>, String> {
    let mut messages = Vec::new();
    for m in request {
        let role = match m.role.as_str() {
            "system" | "developer" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            other => return Err(format!("unsupported message role `{other}`")),
        };
        let content = match &m.content {
            Content::Text(text) => text.clone(),
            Content::Parts(parts) => {
                if let Some(p) = parts.iter().find(|p| p.kind != "text") {
                    return Err(format!("unsupported content part type `{}`", p.kind));
                }
                parts.iter().filter_map(|p| p.text.as_deref()).collect()
            }
        };
        messages.push(Message { role, content });
    }
    match messages.last() {
        Some(last) if last.role == Role::User => Ok(messages),
        _ => Err("the last message must be a user message".into()),
    }
}

async fn chat_completions(
    State(app): State<Arc<App>>,
    Json(request): Json<ChatRequest>,
) -> Response {
    let messages = match to_messages(&request.messages) {
        Ok(messages) => messages,
        Err(e) => return bad_request(e),
    };
    let stop = match request.stop {
        None => Vec::new(),
        Some(OneOrMany::One(s)) => vec![s],
        Some(OneOrMany::Many(v)) => v,
    };
    let number = app.requests.fetch_add(1, Ordering::Relaxed);
    // Without a seed, every request samples differently.
    let seed = request.seed.unwrap_or_else(|| {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        nanos as u64 ^ number
    });
    let (events, mut received) = mpsc::channel(64);
    let job = Job {
        messages,
        temperature: request.temperature,
        top_p: request.top_p,
        seed,
        max_tokens: request
            .max_completion_tokens
            .or(request.max_tokens)
            .unwrap_or(usize::MAX),
        stop: stop.into_iter().filter(|s| !s.is_empty()).collect(),
        events,
    };
    if app.jobs.send(job).is_err() {
        return server_error("the model worker has stopped");
    }
    let id = format!("chatcmpl-{number}");

    if request.stream {
        return bad_request("streaming is not available in this build");
    }

    // If the client disconnects, this future is dropped, `received` with it, and the worker
    // stops at its next token.
    let mut content = String::new();
    while let Some(event) = received.recv().await {
        match event {
            Event::Text(text) => content.push_str(&text),
            Event::Error(e) => return bad_request(e),
            Event::Done {
                prompt_tokens,
                completion_tokens,
                finish_reason,
            } => {
                return Json(json!({
                    "id": id,
                    "object": "chat.completion",
                    "created": now(),
                    "model": app.model,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": finish_reason,
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }))
                .into_response();
            }
        }
    }
    server_error("the model worker has stopped")
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let args = Args::parse();
    let model = args
        .model
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    let address = format!("{}:{}", args.host, args.port);

    let (jobs, job_queue) = queue::channel();
    let (ready, is_ready) = queue::channel();
    std::thread::spawn(move || match args.backend {
        BackendKind::Cpu => worker(&args, Cpu, ready, job_queue),
        #[cfg(target_os = "macos")]
        BackendKind::Metal => match engine::backend::metal::Metal::new() {
            Ok(metal) => worker(&args, metal, ready, job_queue),
            Err(e) => drop(ready.send(Err(anyhow::Error::msg(e)))),
        },
        #[cfg(not(target_os = "macos"))]
        BackendKind::Metal => {
            drop(ready.send(Err(anyhow::anyhow!("the Metal backend needs macOS"))))
        }
    });
    is_ready
        .recv()
        .context("the model worker died while loading")??;

    let app = Arc::new(App {
        jobs,
        model,
        started: now(),
        requests: AtomicU64::new(0),
    });
    let router = Router::new()
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(app);
    let listener = tokio::net::TcpListener::bind(&address)
        .await
        .with_context(|| format!("cannot listen on {address}"))?;
    eprintln!("listening on http://{address}");
    axum::serve(listener, router).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::split_at_stop;

    #[test]
    fn stop_strings_are_cut_and_their_beginnings_held_back() {
        let stops = ["END".to_string(), "\n\n".to_string()];
        assert_eq!(split_at_stop("hello", &stops), (5, false));
        assert_eq!(split_at_stop("hello E", &stops), (6, false)); // "E" may start "END"
        assert_eq!(split_at_stop("hello EN", &stops), (6, false));
        assert_eq!(split_at_stop("hello END more", &stops), (6, true));
        assert_eq!(split_at_stop("a\n", &stops), (1, false));
        assert_eq!(split_at_stop("a\n\nb END", &stops), (1, true)); // earliest stop wins
        assert_eq!(split_at_stop("hello", &[]), (5, false));
    }
}
