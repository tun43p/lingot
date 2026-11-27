//! Qwen3 forward pass, one token at a time, on top of a `Backend`.

use crate::backend::{Backend, Weight};
use crate::gguf::{Error, Gguf, Qwen3Config, Result, TensorType};

struct Layer<'a> {
    attn_norm: Vec<f32>,
    attn_q: Weight<'a>,
    attn_k: Weight<'a>,
    attn_v: Weight<'a>,
    attn_q_norm: Vec<f32>,
    attn_k_norm: Vec<f32>,
    attn_output: Weight<'a>,
    ffn_norm: Vec<f32>,
    ffn_gate: Weight<'a>,
    ffn_up: Weight<'a>,
    ffn_down: Weight<'a>,
}

pub struct Model<'a, B: Backend> {
    pub config: Qwen3Config,
    backend: B,
    token_embd: Weight<'a>,
    layers: Vec<Layer<'a>>,
    output_norm: Vec<f32>,
    output: Weight<'a>,
}

/// KV cache: per layer, K and V of every position seen so far, `[position][n_head_kv][head_dim]`.
#[derive(Default)]
pub struct State {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    pos: usize,
}

impl<'a, B: Backend> Model<'a, B> {
    pub fn load(gguf: &'a Gguf, backend: B) -> Result<Self> {
        let config = Qwen3Config::from_gguf(gguf)?;
        let c = &config;
        if c.key_length != c.value_length
            || c.key_length % 2 != 0
            || c.head_count_kv == 0
            || c.head_count % c.head_count_kv != 0
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
        let weight = |name: &str, rows: usize, cols: usize| -> Result<Weight<'a>> {
            let t = tensor(name, &[cols, rows])?;
            if !matches!(t.ty, TensorType::BF16 | TensorType::F32) {
                return Err(Error::Invalid(format!(
                    "tensor `{name}` is {}, only BF16 and F32 weights are supported for now",
                    t.ty
                )));
            }
            Ok(Weight {
                ty: t.ty,
                rows,
                cols,
                data: gguf.data(t),
            })
        };
        let norm = |name: &str, len: usize| -> Result<Vec<f32>> {
            let t = tensor(name, &[len])?;
            if t.ty != TensorType::F32 {
                return Err(Error::Invalid(format!("tensor `{name}` must be F32")));
            }
            let (chunks, _) = gguf.data(t).as_chunks::<4>();
            Ok(chunks.iter().map(|b| f32::from_le_bytes(*b)).collect())
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
            None => token_embd,
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

    pub fn new_state(&self) -> State {
        State {
            k: vec![Vec::new(); self.layers.len()],
            v: vec![Vec::new(); self.layers.len()],
            pos: 0,
        }
    }

    /// Feed one token at the next position and return the logits for the token after it.
    /// `trace` receives every intermediate activation, named like the Hugging Face dump
    /// (`hidden.NN`, `lNN.q_norm`, ...); pass `&mut |_, _| {}` to ignore them.
    pub fn forward(
        &self,
        state: &mut State,
        token: u32,
        trace: &mut dyn FnMut(&str, &[f32]),
    ) -> Vec<f32> {
        let (c, b) = (&self.config, &self.backend);
        let (d, hd, eps) = (c.embedding_length, c.key_length, c.rms_norm_eps);
        assert!(state.pos < c.context_length, "context length exceeded");
        assert!(
            (token as usize) < c.vocab_size,
            "token {token} out of vocabulary"
        );

        let mut x = vec![0.0; d];
        let mut xn = vec![0.0; d];
        let mut q = vec![0.0; c.head_count * hd];
        let mut k = vec![0.0; c.head_count_kv * hd];
        let mut v = vec![0.0; c.head_count_kv * hd];
        let mut normed = vec![0.0; hd];
        let mut attn = vec![0.0; c.head_count * hd];
        let mut out = vec![0.0; d];
        let mut gate = vec![0.0; c.feed_forward_length];
        let mut up = vec![0.0; c.feed_forward_length];
        let mut hidden = vec![0.0; c.feed_forward_length];

        b.embed(&self.token_embd, token, &mut x);
        trace("hidden.00", &x);

        for (i, l) in self.layers.iter().enumerate() {
            b.rms_norm(&x, &l.attn_norm, eps, &mut xn);
            b.matvec(&l.attn_q, &xn, &mut q);
            b.matvec(&l.attn_k, &xn, &mut k);
            b.matvec(&l.attn_v, &xn, &mut v);

            // QK-Norm: one RMSNorm per head, before RoPE.
            for (heads, weight) in [(&mut q, &l.attn_q_norm), (&mut k, &l.attn_k_norm)] {
                for head in heads.chunks_exact_mut(hd) {
                    b.rms_norm(head, weight, eps, &mut normed);
                    head.copy_from_slice(&normed);
                }
            }
            trace(&format!("l{i:02}.q_norm"), &q);
            trace(&format!("l{i:02}.k_norm"), &k);

            b.rope(&mut q, hd, state.pos, c.rope_freq_base);
            b.rope(&mut k, hd, state.pos, c.rope_freq_base);
            trace(&format!("l{i:02}.q_rope"), &q);
            trace(&format!("l{i:02}.k_rope"), &k);

            state.k[i].extend_from_slice(&k);
            state.v[i].extend_from_slice(&v);
            b.attention(
                &q,
                &state.k[i],
                &state.v[i],
                c.head_count,
                c.head_count_kv,
                hd,
                &mut attn,
            );
            b.matvec(&l.attn_output, &attn, &mut out);
            trace(&format!("l{i:02}.attn_out"), &out);
            b.add(&mut x, &out);

            b.rms_norm(&x, &l.ffn_norm, eps, &mut xn);
            b.matvec(&l.ffn_gate, &xn, &mut gate);
            b.matvec(&l.ffn_up, &xn, &mut up);
            b.silu_mul(&gate, &up, &mut hidden);
            b.matvec(&l.ffn_down, &hidden, &mut out);
            trace(&format!("l{i:02}.mlp_out"), &out);
            b.add(&mut x, &out);
            if i + 1 < self.layers.len() {
                trace(&format!("hidden.{:02}", i + 1), &x);
            }
        }

        // Like Hugging Face, the last hidden state is taken after the final norm.
        b.rms_norm(&x, &self.output_norm, eps, &mut xn);
        trace(&format!("hidden.{:02}", self.layers.len()), &xn);
        let mut logits = vec![0.0; c.vocab_size];
        b.matvec(&self.output, &xn, &mut logits);
        state.pos += 1;
        logits
    }
}
