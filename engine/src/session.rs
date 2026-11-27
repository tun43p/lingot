//! Chat formatting and a generation session: prefill, decode loop, stop tokens, KV cache reuse.

use std::time::Instant;

use crate::backend::Backend;
use crate::gguf::{Error, Result};
use crate::model::{Model, State};
use crate::sample::Sampler;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

/// Qwen3's ChatML template for a conversation ending with a user message, with the generation
/// prompt. Like the original template, reasoning (`<think>...</think>`) is dropped from past
/// assistant turns, and `think = false` pre-fills an empty reasoning block.
pub fn render_chat(messages: &[Message], think: bool) -> String {
    let mut out = String::new();
    for m in messages {
        let (role, content) = match m.role {
            Role::System => ("system", m.content.as_str()),
            Role::User => ("user", m.content.as_str()),
            Role::Assistant => {
                let answer = match m.content.rsplit_once("</think>") {
                    Some((_, after)) => after.trim_start_matches('\n'),
                    None => &m.content,
                };
                ("assistant", answer)
            }
        };
        out.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
    }
    out.push_str("<|im_start|>assistant\n");
    if !think {
        out.push_str("<think>\n\n</think>\n\n");
    }
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The model produced an end-of-generation token.
    Stop,
    MaxTokens,
    ContextFull,
}

#[derive(Debug, Clone, Copy)]
pub struct Stats {
    pub prompt_tokens: usize,
    /// Prompt tokens actually evaluated (the rest came from the KV cache).
    pub prefill_tokens: usize,
    pub prefill_secs: f64,
    pub generated_tokens: usize,
    pub decode_secs: f64,
    /// From the start of the call to the first generated token.
    pub first_token_secs: f64,
    pub stop: StopReason,
}

pub struct Session<'m, 'g, B: Backend> {
    model: &'m Model<'g, B>,
    state: State<B>,
    /// Tokens currently in the KV cache.
    cached: Vec<u32>,
    stop_tokens: Vec<u32>,
}

impl<'m, 'g, B: Backend> Session<'m, 'g, B> {
    /// `n_ctx` is the maximum number of tokens (prompt + generation) the session can hold.
    pub fn new(model: &'m Model<'g, B>, n_ctx: usize, stop_tokens: &[u32]) -> Self {
        Session {
            model,
            state: model.new_state(n_ctx),
            cached: Vec::new(),
            stop_tokens: stop_tokens.to_vec(),
        }
    }

    /// Evaluate `prompt` (reusing the cache for the prefix it shares with the previous call),
    /// then generate until a stop token, `max_tokens` or a full context. `on_token` receives
    /// each generated token; the stop token is not passed on.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        sampler: &mut Sampler,
        max_tokens: usize,
        on_token: &mut dyn FnMut(u32),
    ) -> Result<Stats> {
        if prompt.is_empty() {
            return Err(Error::Invalid("empty prompt".into()));
        }
        if prompt.len() > self.state.n_ctx() {
            return Err(Error::Invalid(format!(
                "prompt has {} tokens, context holds {}",
                prompt.len(),
                self.state.n_ctx()
            )));
        }
        let start = Instant::now();

        // Keep the shared prefix, but always evaluate the last prompt token: we need its logits.
        let shared = self.cached.iter().zip(prompt).take_while(|(a, b)| a == b);
        let keep = shared.count().min(prompt.len() - 1);
        self.state.truncate(keep);
        self.cached.truncate(keep);

        // Prefill: the rest of the prompt in one batch; only the last token's logits matter.
        let rest = &prompt[keep..];
        let mut logits = self
            .model
            .forward(&mut self.state, rest, rest.len() - 1, None);
        self.cached.extend_from_slice(rest);
        let prefill_secs = start.elapsed().as_secs_f64();

        let mut generated_tokens = 0;
        let mut first_token_secs = 0.0;
        let stop = loop {
            if generated_tokens == max_tokens {
                break StopReason::MaxTokens;
            }
            let token = sampler.sample(&logits);
            if generated_tokens == 0 {
                first_token_secs = start.elapsed().as_secs_f64();
            }
            if self.stop_tokens.contains(&token) {
                break StopReason::Stop;
            }
            on_token(token);
            generated_tokens += 1;
            if self.state.pos() == self.state.n_ctx() {
                break StopReason::ContextFull;
            }
            logits = self.model.forward(&mut self.state, &[token], 0, None);
            self.cached.push(token);
        };

        Ok(Stats {
            prompt_tokens: prompt.len(),
            prefill_tokens: prompt.len() - keep,
            prefill_secs,
            generated_tokens,
            decode_secs: start.elapsed().as_secs_f64() - prefill_secs,
            first_token_secs,
            stop,
        })
    }
}
