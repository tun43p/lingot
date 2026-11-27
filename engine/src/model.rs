//! Qwen3 forward pass on top of a `Backend`, for a batch of consecutive tokens.

use crate::backend::{AttentionShape, Backend, Weight};
use crate::gguf::{Error, Gguf, Qwen3Config, Result, TensorType};

struct Layer<'a, B: Backend> {
    attn_norm: B::Buf,
    attn_q: B::Weights<'a>,
    attn_k: B::Weights<'a>,
    attn_v: B::Weights<'a>,
    attn_q_norm: B::Buf,
    attn_k_norm: B::Buf,
    attn_output: B::Weights<'a>,
    ffn_norm: B::Buf,
    ffn_gate: B::Weights<'a>,
    ffn_up: B::Weights<'a>,
    ffn_down: B::Weights<'a>,
}

pub struct Model<'a, B: Backend> {
    pub config: Qwen3Config,
    backend: B,
    token_embd: B::Weights<'a>,
    layers: Vec<Layer<'a, B>>,
    output_norm: B::Buf,
    output: B::Weights<'a>,
}

/// KV cache: per layer, K and V buffers preallocated for `n_ctx` positions and indexed by
/// position, `[position][n_head_kv][head_dim]`. Only the first `pos` positions are valid.
pub struct State<B: Backend> {
    k: Vec<B::Buf>,
    v: Vec<B::Buf>,
    pos: usize,
    n_ctx: usize,
}

impl<B: Backend> State<B> {
    /// Number of positions in the cache.
    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn n_ctx(&self) -> usize {
        self.n_ctx
    }

    /// Forget everything from position `pos` on.
    pub fn truncate(&mut self, pos: usize) {
        self.pos = self.pos.min(pos);
    }
}

/// Receives every intermediate activation of a forward pass, named like the Hugging Face
/// dump (`hidden.NN`, `lNN.q_norm`, ...), one row per token of the batch.
pub type Trace<'t> = &'t mut dyn FnMut(&str, &[f32]);

impl<'a, B: Backend> Model<'a, B> {
    pub fn load(gguf: &'a Gguf, backend: B) -> Result<Self> {
        let config = Qwen3Config::from_gguf(gguf)?;
        let c = &config;
        if c.key_length != c.value_length
            || !c.key_length.is_multiple_of(2)
            || c.head_count_kv == 0
            || !c.head_count.is_multiple_of(c.head_count_kv)
        {
            return Err(Error::Invalid(format!(
                "unsupported attention shape: {c:?}"
            )));
        }
        let (d, ff) = (c.embedding_length, c.feed_forward_length);
        let (q_dim, kv_dim) = (c.head_count * c.key_length, c.head_count_kv * c.key_length);

        let tensor = |name: &str, dims: &[usize]| {
            let t = gguf
                .tensor(name)
                .ok_or_else(|| Error::Invalid(format!("missing tensor `{name}`")))?;
            if t.dims.iter().map(|d| *d as usize).ne(dims.iter().copied()) {
                return Err(Error::Invalid(format!(
                    "tensor `{name}` has dims {:?}, expected {dims:?}",
                    t.dims
                )));
            }
            Ok(t)
        };
        // GGUF dims are innermost first: [cols, rows].
        let weight = |name: &str, rows: usize, cols: usize| -> Result<B::Weights<'a>> {
            let t = tensor(name, &[cols, rows])?;
            let quantized = matches!(t.ty, TensorType::Q8_0 | TensorType::Q4_0);
            // Quantized rows are made of whole 32-weight blocks.
            if t.ty == TensorType::F16 || (quantized && !cols.is_multiple_of(32)) {
                return Err(Error::Invalid(format!(
                    "tensor `{name}` ({}, {cols} columns) is not supported",
                    t.ty
                )));
            }
            Ok(backend.weights(Weight {
                ty: t.ty,
                rows,
                cols,
                data: gguf.data(t),
            }))
        };
        let norm = |name: &str, len: usize| -> Result<B::Buf> {
            let t = tensor(name, &[len])?;
            if t.ty != TensorType::F32 {
                return Err(Error::Invalid(format!("tensor `{name}` must be F32")));
            }
            let (chunks, _) = gguf.data(t).as_chunks::<4>();
            let values: Vec<f32> = chunks.iter().map(|b| f32::from_le_bytes(*b)).collect();
            Ok(backend.upload(&values))
        };

        let token_embd = weight("token_embd.weight", c.vocab_size, d)?;
        let layers = (0..c.block_count)
            .map(|i| {
                let n = |s: &str| format!("blk.{i}.{s}.weight");
                Ok(Layer {
                    attn_norm: norm(&n("attn_norm"), d)?,
                    attn_q: weight(&n("attn_q"), q_dim, d)?,
                    attn_k: weight(&n("attn_k"), kv_dim, d)?,
                    attn_v: weight(&n("attn_v"), kv_dim, d)?,
                    attn_q_norm: norm(&n("attn_q_norm"), c.key_length)?,
                    attn_k_norm: norm(&n("attn_k_norm"), c.key_length)?,
                    attn_output: weight(&n("attn_output"), d, q_dim)?,
                    ffn_norm: norm(&n("ffn_norm"), d)?,
                    ffn_gate: weight(&n("ffn_gate"), ff, d)?,
                    ffn_up: weight(&n("ffn_up"), ff, d)?,
                    ffn_down: weight(&n("ffn_down"), d, ff)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        // Tied embeddings: some files have no output.weight.
        let output = match gguf.tensor("output.weight") {
            Some(_) => weight("output.weight", c.vocab_size, d)?,
            None => weight("token_embd.weight", c.vocab_size, d)?,
        };
        let output_norm = norm("output_norm.weight", d)?;

        Ok(Model {
            config,
            backend,
            token_embd,
            layers,
            output_norm,
            output,
        })
    }

    /// A KV cache for at most `n_ctx` positions (capped by the model's context length).
    pub fn new_state(&self, n_ctx: usize) -> State<B> {
        let c = &self.config;
        let n_ctx = n_ctx.min(c.context_length);
        let size = n_ctx * c.head_count_kv * c.key_length;
        let caches = || {
            (0..self.layers.len())
                .map(|_| self.backend.alloc(size))
                .collect()
        };
        State {
            k: caches(),
            v: caches(),
            pos: 0,
            n_ctx,
        }
    }

    /// Feed `tokens` at the next positions. Returns the logits of the tokens from index
    /// `logits_from` on, row by row (`[tokens.len() - logits_from][vocab]`): row `i` predicts
    /// the token that follows `tokens[logits_from + i]`.
    pub fn forward(
        &self,
        state: &mut State<B>,
        tokens: &[u32],
        logits_from: usize,
        mut trace: Option<Trace>,
    ) -> Vec<f32> {
        let (c, b) = (&self.config, &self.backend);
        let (d, hd, eps) = (c.embedding_length, c.key_length, c.rms_norm_eps);
        let n = tokens.len();
        assert!(n > 0 && logits_from <= n);
        assert!(state.pos + n <= state.n_ctx, "context length exceeded");
        assert!(
            tokens.iter().all(|t| (*t as usize) < c.vocab_size),
            "token out of vocabulary"
        );
        let shape = AttentionShape {
            pos: state.pos,
            n_head: c.head_count,
            n_head_kv: c.head_count_kv,
            head_dim: hd,
        };
        // Reading a buffer back is only worth it when someone is listening.
        let mut emit = |name: &str, buf: &B::Buf| {
            if let Some(trace) = trace.as_mut() {
                trace(name, &b.read(buf));
            }
        };

        let mut x = b.alloc(n * d);
        let mut xn = b.alloc(n * d);
        let mut q = b.alloc(n * c.head_count * hd);
        let mut k = b.alloc(n * c.head_count_kv * hd);
        let mut v = b.alloc(n * c.head_count_kv * hd);
        let mut qn = b.alloc(n * c.head_count * hd);
        let mut kn = b.alloc(n * c.head_count_kv * hd);
        let mut attn = b.alloc(n * c.head_count * hd);
        let mut out = b.alloc(n * d);
        let mut gate = b.alloc(n * c.feed_forward_length);
        let mut up = b.alloc(n * c.feed_forward_length);
        let mut hidden = b.alloc(n * c.feed_forward_length);

        b.embed(&self.token_embd, tokens, &mut x);
        emit("hidden.00", &x);

        for (i, l) in self.layers.iter().enumerate() {
            b.rms_norm(&x, &l.attn_norm, eps, &mut xn);
            b.matmul(&l.attn_q, &xn, &mut q);
            b.matmul(&l.attn_k, &xn, &mut k);
            b.matmul(&l.attn_v, &xn, &mut v);

            // QK-Norm: one RMSNorm per head, before RoPE.
            b.rms_norm(&q, &l.attn_q_norm, eps, &mut qn);
            b.rms_norm(&k, &l.attn_k_norm, eps, &mut kn);
            emit(&format!("l{i:02}.q_norm"), &qn);
            emit(&format!("l{i:02}.k_norm"), &kn);

            b.rope(&mut qn, n, hd, state.pos, c.rope_freq_base);
            b.rope(&mut kn, n, hd, state.pos, c.rope_freq_base);
            emit(&format!("l{i:02}.q_rope"), &qn);
            emit(&format!("l{i:02}.k_rope"), &kn);

            b.attention(
                shape,
                &qn,
                &kn,
                &v,
                &mut state.k[i],
                &mut state.v[i],
                &mut attn,
            );
            b.matmul(&l.attn_output, &attn, &mut out);
            emit(&format!("l{i:02}.attn_out"), &out);
            b.add(&mut x, &out);

            b.rms_norm(&x, &l.ffn_norm, eps, &mut xn);
            b.matmul(&l.ffn_gate, &xn, &mut gate);
            b.matmul(&l.ffn_up, &xn, &mut up);
            b.silu_mul(&gate, &up, &mut hidden);
            b.matmul(&l.ffn_down, &hidden, &mut out);
            emit(&format!("l{i:02}.mlp_out"), &out);
            b.add(&mut x, &out);
            if i + 1 < self.layers.len() {
                emit(&format!("hidden.{:02}", i + 1), &x);
            }
        }
        state.pos += n;

        // Like Hugging Face, the last hidden state is taken after the final norm.
        b.rms_norm(&x, &self.output_norm, eps, &mut xn);
        emit(&format!("hidden.{:02}", self.layers.len()), &xn);

        let rows = n - logits_from;
        if rows == 0 {
            // Nothing to return, but make sure the work (KV cache writes) is done.
            b.read(&b.slice(&xn, 0, 1));
            return Vec::new();
        }
        let last = b.slice(&xn, logits_from * d, rows * d);
        let mut logits = b.alloc(rows * c.vocab_size);
        b.matmul(&self.output, &last, &mut logits);
        b.read(&logits)
    }
}
