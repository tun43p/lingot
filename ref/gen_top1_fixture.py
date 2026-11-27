"""Generate engine/tests/fixtures/top1.json from the Hugging Face f32 dump of Qwen3-0.6B:
for each prompt, the token ids and the top-1 next token at every position.

Usage: python ref/gen_top1_fixture.py   (needs ref/dumps/qwen3-0.6B-f32)
"""

import json
from pathlib import Path

from safetensors.numpy import load_file

ROOT = Path(__file__).parent.parent
DUMP = ROOT / "ref/dumps/qwen3-0.6B-f32"

index = json.loads((DUMP / "index.json").read_text())
fixture = [
    {
        "name": p["name"],
        "tokens": p["tokens"],
        "top1": load_file(DUMP / p["file"])["logits"].argmax(axis=1).tolist(),
    }
    for p in index["prompts"]
]
out = ROOT / "engine/tests/fixtures/top1.json"
out.write_text(json.dumps(fixture) + "\n")
print(f"wrote {out}")
