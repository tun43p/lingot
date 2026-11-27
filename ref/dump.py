"""Dump Hugging Face reference activations for Qwen3.

f32 (default): per-layer activations, final logits and a 100-token greedy
generation, for each prompt in prompts.json.
bf16: final logits only (sanity check for models too large for f32).

Output: <out>/prompt_<i>.safetensors (all tensors f32, tokens i32) + index.json.
No timestamps anywhere, so two runs can be compared byte for byte.

Tensor names (NN = zero-padded layer index, T = prompt length):
  tokens               [T]        prompt token ids
  hidden.NN            [T, d]     hidden.00 = embeddings, hidden.NN = output of layer NN-1;
                                  the last one is taken after the final norm
  lNN.attn_out         [T, d]     attention output after o_proj, before the residual add
  lNN.mlp_out          [T, d]     MLP output, before the residual add
  lNN.q_norm/.k_norm   [H, T, hd] Q/K after QK-Norm, before RoPE
  lNN.q_rope/.k_rope   [H, T, hd] Q/K after RoPE
  logits               [T, vocab]
  gen_tokens           [100]      greedy continuation (no EOS stop)
  gen_logits           [100, vocab] gen_tokens[i] == argmax(gen_logits[i])
"""

import argparse
import json
from pathlib import Path

import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3 import modeling_qwen3

GEN_STEPS = 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local HF model directory")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--dtype", choices=["f32", "bf16"], default="f32")
    args = ap.parse_args()

    full = args.dtype == "f32"
    torch.use_deterministic_algorithms(True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32 if full else torch.bfloat16
    ).eval()
    layers = model.model.layers

    rec = None  # dict while capturing the prompt pass, None otherwise

    def save(name):
        def hook(_mod, _inp, out):
            if rec is not None:
                rec[name] = out[0] if isinstance(out, tuple) else out

        return hook

    if full:
        for i, layer in enumerate(layers):
            layer.self_attn.register_forward_hook(save(f"l{i:02d}.attn_out"))
            layer.mlp.register_forward_hook(save(f"l{i:02d}.mlp_out"))
            # [1, T, H, hd] here; stored as [H, T, hd] like the RoPE outputs
            layer.self_attn.q_norm.register_forward_hook(save(f"l{i:02d}.q_norm"))
            layer.self_attn.k_norm.register_forward_hook(save(f"l{i:02d}.k_norm"))

        # RoPE is a plain function, not a module: wrap it. Called once per layer, in order.
        orig_rope = modeling_qwen3.apply_rotary_pos_emb
        rope_calls = []

        def rope(q, k, cos, sin, **kw):
            q, k = orig_rope(q, k, cos, sin, **kw)
            if rec is not None:
                rope_calls.append((q, k))
            return q, k

        modeling_qwen3.apply_rotary_pos_emb = rope

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = json.loads((Path(__file__).parent / "prompts.json").read_text())
    index = {
        "model": Path(args.model).name,
        "dtype": args.dtype,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "prompts": [],
    }

    for pi, prompt in enumerate(prompts):
        ids = tok(prompt["text"], return_tensors="pt").input_ids
        tensors = {"tokens": ids[0].to(torch.int32)}
        with torch.no_grad():
            if full:
                rec = {}
                rope_calls.clear()
            out = model(ids, output_hidden_states=full, use_cache=full)
            tensors["logits"] = out.logits[0]

            if full:
                assert len(rope_calls) == len(layers), "RoPE wrapper missed a layer"
                for i, (q, k) in enumerate(rope_calls):
                    rec[f"l{i:02d}.q_rope"], rec[f"l{i:02d}.k_rope"] = q[0], k[0]
                for name, t in rec.items():
                    if name.endswith("_norm"):
                        t = t[0].transpose(0, 1)
                    elif t.dim() == 3 and t.shape[0] == 1:
                        t = t[0]
                    tensors[name] = t
                for i, h in enumerate(out.hidden_states):
                    tensors[f"hidden.{i:02d}"] = h[0]
                rec = None

                gen_tokens, gen_logits = [], []
                logits, past = out.logits[0, -1], out.past_key_values
                for _ in range(GEN_STEPS):
                    nxt = logits.argmax()
                    gen_tokens.append(nxt)
                    gen_logits.append(logits)
                    step = model(nxt.view(1, 1), past_key_values=past, use_cache=True)
                    logits, past = step.logits[0, -1], step.past_key_values
                tensors["gen_tokens"] = torch.stack(gen_tokens).to(torch.int32)
                tensors["gen_logits"] = torch.stack(gen_logits)

        tensors = {
            k: (v if v.dtype == torch.int32 else v.float()).contiguous()
            for k, v in tensors.items()
        }
        for k, v in tensors.items():
            assert v.dtype == torch.int32 or torch.isfinite(v).all(), f"non-finite {k}"
        file = f"prompt_{pi}.safetensors"
        save_file(tensors, out_dir / file)
        entry = {
            "name": prompt["name"],
            "text": prompt["text"],
            "file": file,
            "tokens": tensors["tokens"].tolist(),
            "tensors": {k: list(v.shape) for k, v in tensors.items()},
        }
        if full:
            entry["gen_tokens"] = tensors["gen_tokens"].tolist()
            entry["gen_text"] = tok.decode(tensors["gen_tokens"])
        index["prompts"].append(entry)
        print(f"{prompt['name']}: {ids.shape[1]} tokens, {len(tensors)} tensors")

    (out_dir / "index.json").write_text(
        json.dumps(index, indent=1, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
