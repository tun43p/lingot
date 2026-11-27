//! Scalar f32 CPU backend: the correctness reference. Readable first; only `matmul` is
//! parallel (rayon over output values) so the tests do not take minutes.

use rayon::prelude::*;

use super::{AttentionShape, Backend, Weight};
use crate::gguf::TensorType;

pub struct Cpu;

/// BF16 is the top 16 bits of an f32.
fn bf16(b: &[u8]) -> f32 {
    f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)
}

fn f32le(b: &[u8]) -> f32 {
    f32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

fn f16(b: &[u8]) -> f32 {
    half::f16::from_le_bytes([b[0], b[1]]).to_f32()
}

// Quantized blocks hold 32 weights: a scale `d` in f16, then the quantized values.
const BLOCK: usize = 32;
/// Q8_0: `d`, then 32 int8. Weight = q * d.
const Q8_0_BYTES: usize = 2 + 32;
/// Q4_0: `d`, then 16 bytes of nibbles. The low nibble of byte `j` is weight `j`, the high
/// nibble is weight `j + 16`. Weight = (q - 8) * d.
const Q4_0_BYTES: usize = 2 + 16;

fn nibbles(byte: u8) -> (f32, f32) {
    ((byte & 0x0F) as f32 - 8.0, (byte >> 4) as f32 - 8.0)
}

/// Bytes taken by one row of `cols` weights.
fn row_bytes(ty: TensorType, cols: usize) -> usize {
    match ty {
        TensorType::F32 => cols * 4,
        TensorType::BF16 => cols * 2,
        TensorType::Q8_0 | TensorType::Q4_0 => {
            assert!(
                cols.is_multiple_of(BLOCK),
                "quantized rows must be whole blocks"
            );
            cols / BLOCK
                * if ty == TensorType::Q8_0 {
                    Q8_0_BYTES
                } else {
                    Q4_0_BYTES
                }
        }
        TensorType::F16 => panic!("CPU backend does not support F16 weights"),
    }
}

/// Decode one row of weights to f32.
fn decode_row(ty: TensorType, row: &[u8], out: &mut [f32]) {
    match ty {
        TensorType::F32 => {
            for (o, b) in out.iter_mut().zip(row.chunks_exact(4)) {
                *o = f32le(b);
            }
        }
        TensorType::BF16 => {
            for (o, b) in out.iter_mut().zip(row.chunks_exact(2)) {
                *o = bf16(b);
            }
        }
        TensorType::Q8_0 => {
            for (out, block) in out
                .chunks_exact_mut(BLOCK)
                .zip(row.chunks_exact(Q8_0_BYTES))
            {
                let d = f16(block);
                for (o, q) in out.iter_mut().zip(&block[2..]) {
                    *o = (*q as i8) as f32 * d;
                }
            }
        }
        TensorType::Q4_0 => {
            for (out, block) in out
                .chunks_exact_mut(BLOCK)
                .zip(row.chunks_exact(Q4_0_BYTES))
            {
                let d = f16(block);
                for (j, byte) in block[2..].iter().enumerate() {
                    let (lo, hi) = nibbles(*byte);
                    out[j] = lo * d;
                    out[j + BLOCK / 2] = hi * d;
                }
            }
        }
        TensorType::F16 => panic!("CPU backend does not support F16 weights"),
    }
}

/// Dot product of one row of weights with `x`, dequantizing on the fly.
fn dot(ty: TensorType, row: &[u8], x: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    match ty {
        TensorType::F32 => {
            for (b, x) in row.chunks_exact(4).zip(x) {
                sum += f32le(b) * x;
            }
        }
        TensorType::BF16 => {
            for (b, x) in row.chunks_exact(2).zip(x) {
                sum += bf16(b) * x;
            }
        }
        TensorType::Q8_0 => {
            for (block, x) in row.chunks_exact(Q8_0_BYTES).zip(x.chunks_exact(BLOCK)) {
                let d = f16(block);
                for (q, x) in block[2..].iter().zip(x) {
                    sum += (*q as i8) as f32 * d * x;
                }
            }
        }
        TensorType::Q4_0 => {
            for (block, x) in row.chunks_exact(Q4_0_BYTES).zip(x.chunks_exact(BLOCK)) {
                let d = f16(block);
                let (x_lo, x_hi) = x.split_at(BLOCK / 2);
                for (j, byte) in block[2..].iter().enumerate() {
                    let (lo, hi) = nibbles(*byte);
                    sum += lo * d * x_lo[j];
                    sum += hi * d * x_hi[j];
                }
            }
        }
        TensorType::F16 => panic!("CPU backend does not support F16 weights"),
    }
    sum
}

impl Backend for Cpu {
    type Buf = Vec<f32>;
    type Weights<'a> = Weight<'a>;

    fn weights<'a>(&self, w: Weight<'a>) -> Weight<'a> {
        w
    }

    fn alloc(&self, len: usize) -> Vec<f32> {
        vec![0.0; len]
    }

    fn upload(&self, data: &[f32]) -> Vec<f32> {
        data.to_vec()
    }

    fn read(&self, buf: &Vec<f32>) -> Vec<f32> {
        buf.clone()
    }

    fn slice(&self, buf: &Vec<f32>, start: usize, len: usize) -> Vec<f32> {
        buf[start..start + len].to_vec()
    }

    fn embed(&self, w: &Weight, tokens: &[u32], out: &mut Vec<f32>) {
        let size = row_bytes(w.ty, w.cols);
        assert!(out.len() == tokens.len() * w.cols);
        for (token, out) in tokens.iter().zip(out.chunks_exact_mut(w.cols)) {
            let row = *token as usize;
            assert!(row < w.rows);
            decode_row(w.ty, &w.data[row * size..(row + 1) * size], out);
        }
    }

    fn matmul(&self, w: &Weight, x: &Vec<f32>, out: &mut Vec<f32>) {
        let size = row_bytes(w.ty, w.cols);
        assert!(x.len().is_multiple_of(w.cols) && out.len() == x.len() / w.cols * w.rows);
        out.par_iter_mut().enumerate().for_each(|(i, o)| {
            let (t, r) = (i / w.rows, i % w.rows);
            let row = &w.data[r * size..(r + 1) * size];
            *o = dot(w.ty, row, &x[t * w.cols..(t + 1) * w.cols]);
        });
    }

    fn rms_norm(&self, x: &Vec<f32>, weight: &Vec<f32>, eps: f32, out: &mut Vec<f32>) {
        let n = weight.len();
        assert!(x.len().is_multiple_of(n) && x.len() == out.len());
        for (x, out) in x.chunks_exact(n).zip(out.chunks_exact_mut(n)) {
            let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
            let scale = 1.0 / (mean_sq + eps).sqrt();
            for ((o, x), w) in out.iter_mut().zip(x).zip(weight) {
                *o = x * scale * w;
            }
        }
    }

    fn rope(&self, x: &mut Vec<f32>, n: usize, head_dim: usize, pos: usize, freq_base: f32) {
        let half = head_dim / 2;
        let per_token = x.len() / n;
        for (t, token) in x.chunks_exact_mut(per_token).enumerate() {
            for head in token.chunks_exact_mut(head_dim) {
                let (lo, hi) = head.split_at_mut(half);
                for i in 0..half {
                    let inv_freq = 1.0 / freq_base.powf(2.0 * i as f32 / head_dim as f32);
                    let (sin, cos) = ((pos + t) as f32 * inv_freq).sin_cos();
                    let (a, b) = (lo[i], hi[i]);
                    lo[i] = a * cos - b * sin;
                    hi[i] = b * cos + a * sin;
                }
            }
        }
    }

    fn attention(
        &self,
        shape: AttentionShape,
        q: &Vec<f32>,
        k: &Vec<f32>,
        v: &Vec<f32>,
        k_cache: &mut Vec<f32>,
        v_cache: &mut Vec<f32>,
        out: &mut Vec<f32>,
    ) {
        let AttentionShape {
            pos,
            n_head,
            n_head_kv,
            head_dim,
        } = shape;
        let step = n_head_kv * head_dim; // one cached position
        let n = k.len() / step;
        assert!(q.len() == n * n_head * head_dim && out.len() == q.len() && k.len() == v.len());
        k_cache[pos * step..(pos + n) * step].copy_from_slice(k);
        v_cache[pos * step..(pos + n) * step].copy_from_slice(v);

        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut scores = vec![0.0f32; pos + n];
        for t in 0..n {
            let scores = &mut scores[..pos + t + 1]; // causal: up to the token's own position
            for h in 0..n_head {
                let at = (t * n_head + h) * head_dim;
                let q = &q[at..at + head_dim];
                let kv = h / (n_head / n_head_kv) * head_dim; // offset of the shared K/V head
                for (p, score) in scores.iter_mut().enumerate() {
                    let k = &k_cache[p * step + kv..p * step + kv + head_dim];
                    *score = q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * scale;
                }

                // Softmax, subtracting the max first so the exponential cannot overflow.
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut total = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - max).exp();
                    total += *s;
                }

                let out = &mut out[at..at + head_dim];
                out.fill(0.0);
                for (p, s) in scores.iter().enumerate() {
                    let weight = s / total;
                    let v = &v_cache[p * step + kv..p * step + kv + head_dim];
                    for (o, v) in out.iter_mut().zip(v) {
                        *o += weight * v;
                    }
                }
            }
        }
    }

    fn silu_mul(&self, gate: &Vec<f32>, up: &Vec<f32>, out: &mut Vec<f32>) {
        assert!(gate.len() == up.len() && gate.len() == out.len());
        for ((o, g), u) in out.iter_mut().zip(gate).zip(up) {
            *o = g / (1.0 + (-g).exp()) * u;
        }
    }

    fn add(&self, x: &mut Vec<f32>, y: &Vec<f32>) {
        assert!(x.len() == y.len());
        for (x, y) in x.iter_mut().zip(y) {
            *x += y;
        }
    }
}
