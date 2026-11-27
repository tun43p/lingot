"""Generate engine/tests/fixtures/chat.json: conversations rendered by the Hugging Face
chat template of Qwen3, with and without thinking.

Usage: python ref/gen_chat_fixture.py   (needs models/hf/Qwen3-0.6B)
"""

import json
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).parent.parent
tok = AutoTokenizer.from_pretrained(ROOT / "models/hf/Qwen3-0.6B")

U, A, S = "user", "assistant", "system"
CONVERSATIONS = [
    [(U, "Hello!")],
    [(S, "You are a pirate."), (U, "Who are you?")],
    [(U, "2+2?"), (A, "4"), (U, "And 3+3?")],
    [
        (U, "2+2?"),
        (A, "<think>\nSimple addition.\n</think>\n\n4"),
        (U, "And 3+3?"),
    ],
    [
        (S, "Be brief.\nNo emojis."),
        (U, "Bonjour, ça va ?"),
        (A, "<think>\n\n</think>\n\nOui, et toi ?"),
        (U, "Très bien 🎉"),
        (A, "Parfait."),
        (U, "  spaces and\n\nnewlines kept  "),
    ],
]

cases = []
for conv in CONVERSATIONS:
    messages = [{"role": r, "content": c} for r, c in conv]
    for think in (True, False):
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=think
        )
        cases.append({"messages": messages, "think": think, "text": text})

out = ROOT / "engine/tests/fixtures/chat.json"
out.write_text(json.dumps(cases, indent=1, ensure_ascii=False) + "\n")
print(f"wrote {out} ({len(cases)} cases)")
