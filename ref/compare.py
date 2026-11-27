"""Compare lingot with a Hugging Face reference dump, in forward order, and report
the first layer and tensor that diverge.

Usage (after `cargo build --release`):
  python ref/compare.py --gguf models/qwen3-0.6B-bf16.gguf --ref ref/dumps/qwen3-0.6B-f32
  python ref/compare.py --gguf models/qwen3-8B-bf16.gguf   --ref ref/dumps/qwen3-8B-bf16

Runs `lingot dump` on each prompt of the reference, then compares every tensor.
Error of a tensor = max over positions of  max|ours - ref| / max|ref|  on that position.

Full reference (f32): every tensor must be within --rtol and top-1 must match everywhere.
Logits-only reference (bf16): top-1 must match on at least --min-top1 of the positions.
Exit code 1 on failure.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

ROOT = Path(__file__).parent.parent
LINGOT = ROOT / "target/release/lingot"
KINDS = ["q_norm", "k_norm", "q_rope", "k_rope", "attn_out", "mlp_out"]


def forward_order(names):
    n_layers = sum(1 for n in names if n.startswith("hidden.")) - 1
    order = []
    for i in range(max(n_layers, 0)):
        order.append(f"hidden.{i:02d}")
        order += [f"l{i:02d}.{k}" for k in KINDS]
    if n_layers >= 0:
        order.append(f"hidden.{n_layers:02d}")
    return order + ["logits"]


def rel_err(ours, ref):
    """(error, position of the worst row)"""
    diff = np.abs(ours - ref).max(axis=1)
    scale = np.maximum(np.abs(ref).max(axis=1), np.finfo(np.float32).tiny)
    err = diff / scale
    return float(err.max()), int(err.argmax())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--ref", required=True, help="directory written by ref/dump.py")
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--min-top1", type=float, default=0.99)
    args = ap.parse_args()

    ref_dir = Path(args.ref)
    index = json.loads((ref_dir / "index.json").read_text())
    out_dir = ROOT / "ref/dumps" / f"lingot-{Path(args.gguf).stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    full = index["dtype"] == "f32"
    ok = True
    worst = {}  # tensor kind -> worst error over all prompts
    top1_same = top1_total = 0

    for p in index["prompts"]:
        ref = load_file(ref_dir / p["file"])
        out = out_dir / p["file"]
        cmd = [LINGOT, "dump", "--model", args.gguf, "--prompt", p["text"], "--out", out]
        subprocess.run(cmd + ([] if full else ["--logits-only"]), check=True)
        ours = load_file(out)

        if not np.array_equal(ours["tokens"], ref["tokens"]):
            print(f"{p['name']}: TOKENS DIFFER")
            ok = False
            continue
        t = len(ref["tokens"])

        first = None
        for name in forward_order(ref.keys()):
            r = ref[name]
            if r.ndim == 3:  # Q/K are [heads, T, head_dim] in the reference
                r = r.transpose(1, 0, 2)
            err, pos = rel_err(ours[name], r.reshape(t, -1))
            kind = name.split(".")[0] if name[0] != "l" or name == "logits" else name.split(".")[1]
            worst[kind] = max(worst.get(kind, 0.0), err)
            if full and err > args.rtol and first is None:
                first = (name, err, pos)

        same = int((ours["logits"].argmax(axis=1) == ref["logits"].argmax(axis=1)).sum())
        top1_same += same
        top1_total += t
        status = "ok"
        if first:
            ok = False
            status = f"FIRST DIVERGENCE: {first[0]} at position {first[2]}, error {first[1]:.3e}"
        print(f"{p['name']}: {t} tokens, top-1 {same}/{t}, {status}")

    print("\nworst error per tensor kind:")
    for kind, err in worst.items():
        print(f"  {kind:<9} {err:.3e}")
    ratio = top1_same / max(top1_total, 1)
    need = 1.0 if full else args.min_top1
    print(f"\ntop-1 agreement: {top1_same}/{top1_total} = {ratio:.4f} (required {need})")
    if full:
        print(f"tolerance: {args.rtol:g}")
    if ratio < need:
        ok = False
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
