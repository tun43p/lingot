//! The operations the model needs. The model only knows this trait.
//!
//! Activations live in backend-owned buffers (`Buf`), so a GPU backend can chain every
//! operation of a forward pass without coming back to the CPU; `read` is the only way out.
//! Every operation works on a batch of `n` tokens laid out row by row (`[n][dim]`): `n` is 1
//! when decoding and the prompt length when prefilling.

pub mod cpu;
// metal backend added below

use crate::gguf::TensorType;

/// A weight matrix as stored in the GGUF file: `rows` rows of `cols` values, row-major.
#[derive(Debug, Clone, Copy)]
pub struct Weight<'a> {
    pub ty: TensorType,
    pub rows: usize,
    pub cols: usize,
    pub data: &'a [u8],
}

#[derive(Debug, Clone, Copy)]
pub struct AttentionShape {
    /// Position of the first token of the batch.
    pub pos: usize,
    pub n_head: usize,
    /// Each K/V head serves `n_head / n_head_kv` query heads.
    pub n_head_kv: usize,
    pub head_dim: usize,
}

pub trait Backend {
    /// A vector of f32 owned by the backend.
    type Buf;
    /// A weight matrix prepared by the backend, borrowing the GGUF data.
    type Weights<'a>;

    fn weights<'a>(&self, w: Weight<'a>) -> Self::Weights<'a>;
    /// A zeroed buffer of `len` values.
    fn alloc(&self, len: usize) -> Self::Buf;
    fn upload(&self, data: &[f32]) -> Self::Buf;
    /// Wait for pending work and copy the buffer back.
    fn read(&self, buf: &Self::Buf) -> Vec<f32>;
    /// Read-only view (or copy) of `len` values starting at `start`.
    fn slice(&self, buf: &Self::Buf, start: usize, len: usize) -> Self::Buf;

    /// `out[t]` = row `tokens[t]` of `w`.
    fn embed(&self, w: &Self::Weights<'_>, tokens: &[u32], out: &mut Self::Buf);

    /// `out[t]` = `w` · `x[t]`, with rows of `w.cols` values in `x` and `w.rows` in `out`.
    fn matmul(&self, w: &Self::Weights<'_>, x: &Self::Buf, out: &mut Self::Buf);

    /// RMSNorm of each consecutive chunk of `weight.len()` values:
    /// `out` = `x` / sqrt(mean(`x`²) + `eps`) * `weight`.
    fn rms_norm(&self, x: &Self::Buf, weight: &Self::Buf, eps: f32, out: &mut Self::Buf);

    /// Rotary position embedding, NEOX layout (element `i` pairs with `i + head_dim / 2`),
    /// in place. `x` holds `n` tokens of consecutive heads; token `t` is at position `pos + t`.
    fn rope(&self, x: &mut Self::Buf, n: usize, head_dim: usize, pos: usize, freq_base: f32);

    /// Store the batch's `k` and `v` (`[n][n_head_kv][head_dim]`) in the caches at positions
    /// `pos..pos + n`, then attend: token `t` sees positions `0..=pos + t` (causal).
    /// `q` and `out` are `[n][n_head][head_dim]`, caches are `[position][n_head_kv][head_dim]`.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        shape: AttentionShape,
        q: &Self::Buf,
        k: &Self::Buf,
        v: &Self::Buf,
        k_cache: &mut Self::Buf,
        v_cache: &mut Self::Buf,
        out: &mut Self::Buf,
    );

    /// `out` = SiLU(`gate`) * `up`.
    fn silu_mul(&self, gate: &Self::Buf, up: &Self::Buf, out: &mut Self::Buf);

    /// `x` += `y`.
    fn add(&self, x: &mut Self::Buf, y: &Self::Buf);
}
