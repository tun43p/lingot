//! Forward pass regression test: top-1 next token at every position of one prompt must match
//! Hugging Face (engine/tests/fixtures/top1.json, from ref/gen_top1_fixture.py).
//! The full activation-by-activation comparison is ref/compare.py.
//! Needs models/qwen3-0.6B-bf16.gguf.

use engine::backend::cpu::Cpu;
use engine::gguf::Gguf;
use engine::model::Model;

#[test]
fn top1_matches_hugging_face() {
    let model_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../models/qwen3-0.6B-bf16.gguf"
    );
    let fixture_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/top1.json");
    let gguf = Gguf::open(model_path).expect("missing models/qwen3-0.6B-bf16.gguf");
    let model = Model::load(&gguf, Cpu).unwrap();

    let fixture: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixture_path).unwrap()).unwrap();
    // One prompt keeps the test short; it has accents, so multi-byte tokens are covered.
    let prompt = &fixture[0];
    assert_eq!(prompt["name"], "french");
    let ints = |key: &str| -> Vec<u32> {
        prompt[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect()
    };

    let mut state = model.new_state(64);
    for (pos, (token, want)) in ints("tokens").iter().zip(ints("top1")).enumerate() {
        let logits = model.forward(&mut state, *token, &mut |_, _| {});
        let got = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32;
        assert_eq!(got, want, "position {pos}");
    }
}
