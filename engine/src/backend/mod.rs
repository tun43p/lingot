//! The operations the model needs. The model only knows this trait.

pub mod cpu;

use crate::gguf::TensorType;

/// A weight matrix as stored in the GGUF file: `rows` rows of `cols` values, row-major.
#[derive(Debug, Clone, Copy)]
pub struct Weight<'a> {
    pub ty: TensorType,
    pub rows: usize,
    pub cols: usize,
    pub data: &'a [u8],
}

pub trait Backend {
    /// `out` = row `token` of `w`.
    fn embed(&self, w: &Weight, token: u32, out: &mut [f32]);

    /// `out` = `w` · `x`, with `x.len() == w.cols` and `out.len() == w.rows`.
    fn matvec(&self, w: &Weight, x: &[f32], out: &mut [f32]);

    /// `out` = `x` / sqrt(mean(`x`²) + `eps`) * `weight`.
    fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]);

    /// Rotary position embedding, NEOX layout (element `i` pairs with `i + head_dim / 2`),
    /// applied in place to each consecutive head of `x`.
    fn rope(&self, x: &mut [f32], head_dim: usize, pos: usize, freq_base: f32);

    /// Attention of one query token over every cached position (causal by construction).
    /// `q` is `[n_head][head_dim]`, `k` and `v` are `[position][n_head_kv][head_dim]`,
    /// `out` is `[n_head][head_dim]`. Each K/V head serves `n_head / n_head_kv` query heads.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        out: &mut [f32],
    );

    /// `out` = SiLU(`gate`) * `up`.
    fn silu_mul(&self, gate: &[f32], up: &[f32], out: &mut [f32]);

    /// `x` += `y`.
    fn add(&self, x: &mut [f32], y: &[f32]);
}
