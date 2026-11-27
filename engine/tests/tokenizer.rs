//! Compares the tokenizer with Hugging Face on ref/tokenizer_corpus.json
//! (the 5 prompts + 500 varied texts, built by ref/tokenizer_corpus.py).
//! Needs models/qwen3-0.6B-bf16.gguf.

use engine::gguf::Gguf;
use engine::tokenizer::Tokenizer;

const MODEL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../models/qwen3-0.6B-bf16.gguf"
);
const CORPUS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../ref/tokenizer_corpus.json");

#[test]
fn matches_hugging_face() {
    let gguf = Gguf::open(MODEL).expect("missing models/qwen3-0.6B-bf16.gguf");
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    assert_eq!(tok.vocab_size(), 151936);

    let corpus: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(CORPUS).unwrap()).unwrap();
    let corpus = corpus.as_array().unwrap();
    assert_eq!(corpus.len(), 505);

    for entry in corpus {
        let text = entry["text"].as_str().unwrap();
        let decoded = entry["decoded"].as_str().unwrap();
        let ids: Vec<u32> = entry["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();

        assert_eq!(tok.encode(text), ids, "encode {text:?}");
        // `decoded` is the NFC form of `text`, so this is also the round trip.
        assert_eq!(tok.decode(&ids), decoded, "decode {text:?}");

        // Streaming: same text, and no chunk ever holds a split character.
        let mut stream = tok.stream();
        let mut streamed = String::new();
        for id in &ids {
            let chunk = stream.push(*id);
            assert!(
                decoded.contains('\u{FFFD}') || !chunk.contains('\u{FFFD}'),
                "{text:?}"
            );
            streamed.push_str(&chunk);
        }
        streamed.push_str(&stream.finish());
        assert_eq!(streamed, decoded, "stream {text:?}");
    }
}

#[test]
fn stream_holds_back_incomplete_characters() {
    let gguf = Gguf::open(MODEL).expect("missing models/qwen3-0.6B-bf16.gguf");
    let tok = Tokenizer::from_gguf(&gguf).unwrap();

    // Feed "é🎉" one byte-token at a time: nothing comes out until a character is complete.
    let byte_id = |b: u8| (0..256u32).find(|id| tok.token_bytes(*id) == [b]).unwrap();
    let mut stream = tok.stream();
    let chunks: Vec<String> = "é🎉".bytes().map(|b| stream.push(byte_id(b))).collect();
    assert_eq!(chunks, ["", "é", "", "", "", "🎉"]);
    assert_eq!(stream.finish(), "");

    // A stream that ends mid-character flushes a replacement character.
    let mut stream = tok.stream();
    assert_eq!(stream.push(byte_id("é".as_bytes()[0])), "");
    assert_eq!(stream.finish(), "\u{FFFD}");
}
