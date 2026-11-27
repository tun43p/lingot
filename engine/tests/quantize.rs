//! Our quantizers must reproduce `llama-quantize --pure` byte for byte: every BF16 tensor of
//! Qwen3-0.6B is quantized and compared with the same tensor in the Q8_0 and Q4_0 files.
//! Needs models/qwen3-0.6B-{bf16,q8_0,q4_0}.gguf.

use engine::gguf::{Gguf, TensorType};
use engine::quantize::{quantize_q4_0, quantize_q8_0};

fn open(kind: &str) -> Gguf {
    let path = format!(
        "{}/../models/qwen3-0.6B-{kind}.gguf",
        env!("CARGO_MANIFEST_DIR")
    );
    Gguf::open(&path).unwrap_or_else(|e| panic!("missing {path}: {e}"))
}

#[test]
fn matches_llama_quantize_byte_for_byte() {
    let source = open("bf16");
    let targets: [(Gguf, TensorType, fn(&[f32]) -> Vec<u8>); 2] = [
        (open("q8_0"), TensorType::Q8_0, quantize_q8_0),
        (open("q4_0"), TensorType::Q4_0, quantize_q4_0),
    ];

    let mut compared = 0;
    for t in source.tensors.iter().filter(|t| t.ty == TensorType::BF16) {
        let (chunks, _) = source.data(t).as_chunks::<2>();
        let values: Vec<f32> = chunks
            .iter()
            .map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16))
            .collect();
        for (file, ty, quantize) in &targets {
            let want = file.tensor(&t.name).unwrap();
            assert_eq!(want.ty, *ty, "{}", t.name);
            let got = quantize(&values);
            let want = file.data(want);
            assert_eq!(got.len(), want.len(), "{} {ty}", t.name);
            if let Some(i) = got.iter().zip(want).position(|(a, b)| a != b) {
                panic!(
                    "{} {ty}: first difference at byte {i}: got {}, want {}",
                    t.name, got[i], want[i]
                );
            }
        }
        compared += 1;
    }
    assert_eq!(compared, 198);
}
