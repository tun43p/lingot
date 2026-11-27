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

/// (bytes per value, decoder) for the weight types this backend can read.
fn decoder(ty: TensorType) -> (usize, fn(&[u8]) -> f32) {
    match ty {
        TensorType::BF16 => (2, bf16),
        TensorType::F32 => (4, f32le),
        other => panic!("CPU backend does not support {other} weights yet"),
    }
}

impl Backend for Cpu {
    fn embed(&self, w: &Weight, token: u32, out: &mut [f32]) {
        let (size, decode) = decoder(w.ty);
        let row = token as usize;
        assert!(row < w.rows && out.len() == w.cols);
        let bytes = &w.data[row * w.cols * size..(row + 1) * w.cols * size];
        for (o, b) in out.iter_mut().zip(bytes.chunks_exact(size)) {
            *o = decode(b);
        }
    }

    fn matvec(&self, w: &Weight, x: &[f32], out: &mut [f32]) {
        let (size, decode) = decoder(w.ty);
        assert!(x.len() == w.cols && out.len() == w.rows);
        out.par_iter_mut()
            .zip(w.data.par_chunks_exact(w.cols * size))
            .for_each(|(o, row)| {
                let mut sum = 0.0f32;
                for (b, x) in row.chunks_exact(size).zip(x) {
                    sum += decode(b) * x;
                }
                *o = sum;
            });
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
