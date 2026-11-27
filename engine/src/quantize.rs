//! f32 -> Q8_0 / Q4_0 quantization, byte-identical to llama.cpp's reference quantizers
//! (`quantize_row_q8_0_ref` and `quantize_row_q4_0_ref` in ggml-quants.c).

const BLOCK: usize = 32;

fn scale_bytes(d: f32) -> [u8; 2] {
    half::f16::from_f32(d).to_le_bytes()
}

/// Blocks of 34 bytes: scale `d` in f16, then 32 int8 with weight ≈ q * d.
/// `values.len()` must be a multiple of 32.
pub fn quantize_q8_0(values: &[f32]) -> Vec<u8> {
    assert!(values.len().is_multiple_of(BLOCK));
    let mut out = Vec::with_capacity(values.len() / BLOCK * 34);
    for block in values.chunks_exact(BLOCK) {
        let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let inv = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend(scale_bytes(d));
        // `round` is half away from zero, like C's roundf.
        out.extend(block.iter().map(|v| (v * inv).round() as i8 as u8));
    }
    out
}

/// Blocks of 18 bytes: scale `d` in f16, then 16 bytes of nibbles with weight ≈ (q - 8) * d.
/// The low nibble of byte `j` is weight `j`, the high nibble is weight `j + 16`.
/// `values.len()` must be a multiple of 32.
pub fn quantize_q4_0(values: &[f32]) -> Vec<u8> {
    assert!(values.len().is_multiple_of(BLOCK));
    let mut out = Vec::with_capacity(values.len() / BLOCK * 18);
    for block in values.chunks_exact(BLOCK) {
        // The value with the largest magnitude (first one on ties), sign included: it maps to -8.
        let mut max = 0.0f32;
        for v in block {
            if max.abs() < v.abs() {
                max = *v;
            }
        }
        let d = max / -8.0;
        let inv = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend(scale_bytes(d));
        // Truncation after + 8.5, like the C cast to int8_t; +8 would give 16, hence the min.
        let q = |v: f32| ((v * inv + 8.5) as i8).min(15) as u8;
        for j in 0..BLOCK / 2 {
            out.push(q(block[j]) | (q(block[j + BLOCK / 2]) << 4));
        }
    }
    out
}
