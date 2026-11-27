//! Needs models/qwen3-0.6B-bf16.gguf (see bench/baseline.md for how to produce it).
//! Tensor-by-tensor comparison with llama.cpp's reader is in ref/check_inspect.py.

use engine::gguf::{Error, Gguf, Qwen3Config, TensorType};

const MODEL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../models/qwen3-0.6B-bf16.gguf"
);

#[test]
fn reads_qwen3_0_6b() {
    let g = Gguf::open(MODEL).expect("missing models/qwen3-0.6B-bf16.gguf");
    assert_eq!(g.alignment, 32);
    assert_eq!(g.data_offset % 32, 0);
    assert_eq!(g.tensors.len(), 311);

    // Values from the model's Hugging Face config.json.
    assert_eq!(
        Qwen3Config::from_gguf(&g).unwrap(),
        Qwen3Config {
            vocab_size: 151936,
            block_count: 28,
            context_length: 40960,
            embedding_length: 1024,
            feed_forward_length: 3072,
            head_count: 16,
            head_count_kv: 8,
            key_length: 128,
            value_length: 128,
            rope_freq_base: 1_000_000.0,
            rms_norm_eps: 1e-6,
        }
    );

    // dims[0] is the innermost dimension: PyTorch's [3072, 1024] is stored as [1024, 3072].
    let t = g.tensor("blk.0.ffn_gate.weight").unwrap();
    assert_eq!(
        (t.ty, t.dims.as_slice()),
        (TensorType::BF16, &[1024, 3072][..])
    );
    assert_eq!(g.data(t).len(), 1024 * 3072 * 2);

    // RMSNorm weights are F32 and all finite.
    let t = g.tensor("output_norm.weight").unwrap();
    assert_eq!(t.ty, TensorType::F32);
    let data = g.data(t);
    assert_eq!(data.len(), 1024 * 4);
    let (values, _) = data.as_chunks::<4>();
    assert!(values.iter().all(|b| f32::from_le_bytes(*b).is_finite()));

    // Tensors are aligned, in bounds and do not overlap.
    let mut spans: Vec<_> = g
        .tensors
        .iter()
        .map(|t| (t.file_offset(), t.n_bytes()))
        .collect();
    spans.sort();
    assert!(spans.iter().all(|(start, _)| start % 32 == 0));
    assert!(spans.windows(2).all(|w| w[0].0 + w[0].1 <= w[1].0));
}

#[test]
fn rejects_bad_files() {
    let dir = std::env::temp_dir();
    let open = |name: &str, bytes: &[u8]| {
        let path = dir.join(format!("lingot-test-{}-{name}", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        let result = Gguf::open(&path).map(|_| ());
        std::fs::remove_file(&path).unwrap();
        result
    };

    assert!(matches!(
        open("magic", b"GGML\x03\0\0\0"),
        Err(Error::BadMagic)
    ));
    assert!(matches!(
        open("version", b"GGUF\x02\0\0\0"),
        Err(Error::Version(2))
    ));

    // A real header cut in the middle of the metadata must fail cleanly, not panic.
    let head = std::fs::read(MODEL).expect("missing models/qwen3-0.6B-bf16.gguf");
    assert!(matches!(
        open("truncated", &head[..10_000]),
        Err(Error::Truncated(_))
    ));
}
