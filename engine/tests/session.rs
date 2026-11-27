//! Chat template against Hugging Face (engine/tests/fixtures/chat.json, from
//! ref/gen_chat_fixture.py) and a two-turn generation on Qwen3-0.6B.
//! Needs models/qwen3-0.6B-bf16.gguf.

use engine::backend::cpu::Cpu;
use engine::gguf::Gguf;
use engine::model::Model;
use engine::sample::{Sampler, SamplerConfig};
use engine::session::{Message, Role, Session, StopReason, render_chat};
use engine::tokenizer::Tokenizer;

const MODEL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../models/qwen3-0.6B-bf16.gguf"
);

#[test]
fn chat_template_matches_hugging_face() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/chat.json");
    let cases: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let cases = cases.as_array().unwrap();
    assert_eq!(cases.len(), 10);
    for case in cases {
        let messages: Vec<Message> = case["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| Message {
                role: match m["role"].as_str().unwrap() {
                    "system" => Role::System,
                    "user" => Role::User,
                    "assistant" => Role::Assistant,
                    other => panic!("unknown role {other}"),
                },
                content: m["content"].as_str().unwrap().to_string(),
            })
            .collect();
        let think = case["think"].as_bool().unwrap();
        assert_eq!(
            render_chat(&messages, think),
            case["text"].as_str().unwrap()
        );
    }
}

#[test]
fn stop_tokens_come_from_the_file() {
    let gguf = Gguf::open(MODEL).expect("missing models/qwen3-0.6B-bf16.gguf");
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let ids: Vec<u32> = ["<|endoftext|>", "<|im_end|>"]
        .iter()
        .map(|s| tok.encode(s)[0])
        .collect();
    assert_eq!(tok.stop_tokens(), ids);
}

#[test]
fn two_turns_reuse_the_cache() {
    let gguf = Gguf::open(MODEL).expect("missing models/qwen3-0.6B-bf16.gguf");
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let model = Model::load(&gguf, Cpu).unwrap();
    let greedy = SamplerConfig {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        seed: 0,
    };
    let user = |content: &str| Message {
        role: Role::User,
        content: content.into(),
    };
    let run = |session: &mut Session<Cpu>, messages: &[Message]| {
        let prompt = tok.encode(&render_chat(messages, false));
        let mut out = Vec::new();
        let stats = session
            .generate(&prompt, &mut Sampler::new(greedy), 24, &mut |t| out.push(t))
            .unwrap();
        (out, stats)
    };

    // Turn 1: a short answer that ends on a stop token, which is not passed on.
    let mut messages = vec![user("Reply with the single word: yes")];
    let mut session = Session::new(&model, 256, tok.stop_tokens());
    let (reply, stats) = run(&mut session, &messages);
    assert_eq!(stats.stop, StopReason::Stop);
    assert_eq!(stats.prefill_tokens, stats.prompt_tokens);
    assert!(!reply.is_empty() && reply.iter().all(|t| !tok.stop_tokens().contains(t)));

    // Turn 2 on the same session only evaluates what the cache does not already hold...
    messages.push(Message {
        role: Role::Assistant,
        content: tok.decode(&reply),
    });
    messages.push(user("Now reply with the single word: no"));
    let (second, stats) = run(&mut session, &messages);
    assert!(stats.prefill_tokens < stats.prompt_tokens);

    // ...and gives exactly what a fresh session gives on the whole conversation.
    let mut fresh = Session::new(&model, 256, tok.stop_tokens());
    let (expected, fresh_stats) = run(&mut fresh, &messages);
    assert_eq!(fresh_stats.prefill_tokens, fresh_stats.prompt_tokens);
    assert_eq!(second, expected);

    // A context too small for the generation stops cleanly.
    let prompt = tok.encode(&render_chat(&messages[..1], false));
    let mut tiny = Session::new(&model, prompt.len(), tok.stop_tokens());
    let (out, stats) = run(&mut tiny, &messages[..1]);
    assert_eq!((out.len(), stats.stop), (1, StopReason::ContextFull));
}
