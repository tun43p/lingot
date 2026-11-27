"""Exercise a running lingot-server with the openai Python SDK.

Usage: python ref/test_server.py [--base-url http://127.0.0.1:8080/v1]
Start the server first, e.g.:
  cargo run --release --bin lingot-server -- --model models/qwen3-0.6B-q8_0.gguf --no-think

Covers: model listing, a multi-turn conversation without and with streaming, usage, stop
strings, max_tokens, an invalid request, and a client that disconnects mid-stream.
"""

import argparse
import time

import openai

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
args = ap.parse_args()
client = openai.OpenAI(base_url=args.base_url, api_key="unused")

model = client.models.list().data[0].id
print(f"model: {model}")

SYSTEM = {"role": "system", "content": "You are a concise assistant. Answer in one short sentence."}
TURNS = [
    "My name is Alex and my favourite colour is green. Just say hello.",
    "What is my name?",
    "And what is my favourite colour?",
]


def conversation(stream):
    messages = [SYSTEM]
    for turn in TURNS:
        messages.append({"role": "user", "content": turn})
        if stream:
            chunks = list(
                client.chat.completions.create(
                    model=model, messages=messages, temperature=0, max_tokens=60, stream=True
                )
            )
            assert chunks[0].choices[0].delta.role == "assistant"
            reply = "".join(c.choices[0].delta.content or "" for c in chunks)
            finish, usage = chunks[-1].choices[0].finish_reason, chunks[-1].usage
            assert all(c.object == "chat.completion.chunk" for c in chunks)
            assert all(c.choices[0].finish_reason is None for c in chunks[:-1])
            assert len(chunks) > 3, "the answer did not arrive in several chunks"
        else:
            r = client.chat.completions.create(
                model=model, messages=messages, temperature=0, max_tokens=60
            )
            assert r.object == "chat.completion" and r.choices[0].message.role == "assistant"
            reply, finish, usage = r.choices[0].message.content, r.choices[0].finish_reason, r.usage
        assert finish == "stop", finish
        assert usage.prompt_tokens > 0 and usage.completion_tokens > 0
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
        messages.append({"role": "assistant", "content": reply})
        print(f"  user: {turn}\n  assistant: {reply.strip()}  [{usage.total_tokens} tokens]")
    # The model must have used the earlier turns.
    assert "alex" in messages[4]["content"].lower(), messages[4]
    assert "green" in messages[6]["content"].lower(), messages[6]
    return [m["content"] for m in messages if m["role"] == "assistant"]


print("multi-turn, no streaming:")
plain = conversation(stream=False)
print("multi-turn, streaming:")
streamed = conversation(stream=True)
assert plain == streamed, "greedy answers differ between streaming and non-streaming"

count = [{"role": "user", "content": "Count from 1 to 10, separated by commas."}]
r = client.chat.completions.create(model=model, messages=count, temperature=0, stop=["5"])
assert "4" in r.choices[0].message.content and "5" not in r.choices[0].message.content
assert r.choices[0].finish_reason == "stop"
print(f"stop string: {r.choices[0].message.content!r}")

r = client.chat.completions.create(model=model, messages=count, temperature=0, max_tokens=4)
assert r.choices[0].finish_reason == "length" and r.usage.completion_tokens == 4
print(f"max_tokens: {r.choices[0].message.content!r}")

try:
    client.chat.completions.create(model=model, messages=[{"role": "assistant", "content": "hi"}])
    raise SystemExit("an invalid request was accepted")
except openai.BadRequestError as e:
    print(f"invalid request: {e.status_code}")

# Disconnect mid-stream: the server must stop generating and serve the next request at once.
story = [{"role": "user", "content": "Write a very long story about a dragon, at least 2000 words."}]
start = time.time()
stream = client.chat.completions.create(model=model, messages=story, max_tokens=3000, stream=True)
for i, _ in enumerate(stream):
    if i == 5:
        break
stream.close()
r = client.chat.completions.create(model=model, messages=count, temperature=0, max_tokens=4)
elapsed = time.time() - start
assert elapsed < 10, f"the server kept generating after the disconnect ({elapsed:.1f} s)"
print(f"disconnect: next answer {elapsed:.1f} s after the abandoned stream started")

print("PASS")
