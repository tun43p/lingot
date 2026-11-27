//! Scalar f32 CPU backend: the correctness reference. Readable first; only `matvec` is
//! parallel (rayon over rows) so the tests do not take minutes.

use rayon::prelude::*;

use super::{Backend, Weight};
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
    fn embed(&self, w: &Weight, token: u32, out: &mut [f32]) {
        let size = row_bytes(w.ty, w.cols);
        let row = token as usize;
        assert!(row < w.rows && out.len() == w.cols);
        decode_row(w.ty, &w.data[row * size..(row + 1) * size], out);
    }

    fn matvec(&self, w: &Weight, x: &[f32], out: &mut [f32]) {
        assert!(x.len() == w.cols && out.len() == w.rows);
        out.par_iter_mut()
            .zip(w.data.par_chunks_exact(row_bytes(w.ty, w.cols)))
            .for_each(|(o, row)| *o = dot(w.ty, row, x));
    }

    fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
        assert!(x.len() == weight.len() && x.len() == out.len());
        let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let scale = 1.0 / (mean_sq + eps).sqrt();
        for ((o, x), w) in out.iter_mut().zip(x).zip(weight) {
            *o = x * scale * w;
        }
    }

    fn rope(&self, x: &mut [f32], head_dim: usize, pos: usize, freq_base: f32) {
        let half = head_dim / 2;
        for head in x.chunks_exact_mut(head_dim) {
            let (lo, hi) = head.split_at_mut(half);
            for i in 0..half {
                let inv_freq = 1.0 / freq_base.powf(2.0 * i as f32 / head_dim as f32);
                let (sin, cos) = (pos as f32 * inv_freq).sin_cos();
                let (a, b) = (lo[i], hi[i]);
                lo[i] = a * cos - b * sin;
                hi[i] = b * cos + a * sin;
            }
        }
    }

    fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        out: &mut [f32],
    ) {
        let step = n_head_kv * head_dim; // one cached position
        let n_pos = k.len() / step;
        assert!(q.len() == n_head * head_dim && out.len() == q.len() && k.len() == v.len());
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut scores = vec![0.0f32; n_pos];

        for h in 0..n_head {
            let q = &q[h * head_dim..(h + 1) * head_dim];
            let kv = h / (n_head / n_head_kv) * head_dim; // offset of the shared K/V head
            for (p, score) in scores.iter_mut().enumerate() {
                let k = &k[p * step + kv..p * step + kv + head_dim];
                *score = q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * scale;
            }

            // Softmax, subtracting the max first so the exponential cannot overflow.
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut total = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max).exp();
                total += *s;
            }

            let out = &mut out[h * head_dim..(h + 1) * head_dim];
            out.fill(0.0);
            for (p, s) in scores.iter().enumerate() {
                let weight = s / total;
                let v = &v[p * step + kv..p * step + kv + head_dim];
                for (o, v) in out.iter_mut().zip(v) {
                    *o += weight * v;
                }
            }
        }
    }

    fn silu_mul(&self, gate: &[f32], up: &[f32], out: &mut [f32]) {
        assert!(gate.len() == up.len() && gate.len() == out.len());
        for ((o, g), u) in out.iter_mut().zip(gate).zip(up) {
            *o = g / (1.0 + (-g).exp()) * u;
        }
    }

    fn add(&self, x: &mut [f32], y: &[f32]) {
        assert!(x.len() == y.len());
        for (x, y) in x.iter_mut().zip(y) {
            *x += y;
        }
    }
}
