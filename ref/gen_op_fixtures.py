"""Generate engine/tests/fixtures/ops.json: NumPy reference values for each Backend op,
on small batches of tokens.

Usage: python ref/gen_op_fixtures.py
Inputs are float32, expected outputs are computed in float64.
"""

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent.parent / "engine/tests/fixtures/ops.json"
rng = np.random.default_rng(0)


def rand(*shape, scale=1.0):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def flat(a):
    return np.asarray(a, dtype=np.float64).ravel().tolist()


def weights(kind, rows, cols):
    """Random weight matrix of the given GGUF type: (raw bytes, values as float64)."""
    if kind == "f32":
        w = rand(rows, cols)
        return w.tobytes(), w.astype(np.float64)
    if kind == "bf16":  # top 16 bits of an f32
        bits = (rand(rows, cols).view(np.uint32) >> 16).astype(np.uint16)
        return bits.tobytes(), (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    raw, values = b"", []
    for _ in range(rows * cols // 32):
        d = np.float16(rng.uniform(0.001, 0.05) * rng.choice([-1, 1]))
        if kind == "q8_0":  # d, then 32 int8: w = q * d
            q = rng.integers(-128, 128, 32, dtype=np.int8)
            values.append(q.astype(np.float64) * float(d))
        else:  # q4_0: d, then 16 bytes; low nibble = weight j, high nibble = weight j + 16
            q = rng.integers(0, 256, 16, dtype=np.uint8)
            both = np.concatenate([q & 0x0F, q >> 4]).astype(np.float64)
            values.append((both - 8) * float(d))
        raw += d.tobytes() + q.tobytes()
    return raw, np.concatenate(values).reshape(rows, cols)


fx = {}

# matmul / embed for every weight type: 5 rows x 64 cols, batch of 3 tokens
x = rand(3, 64)
tokens = [4, 0, 4]
for kind in ("f32", "bf16", "q8_0", "q4_0"):
    raw, w = weights(kind, 5, 64)
    fx[f"matmul_{kind}"] = {
        "rows": 5, "cols": 64, "weight_bytes": list(raw), "x": flat(x),
        "out": flat(x.astype(np.float64) @ w.T),
    }
    fx[f"embed_{kind}"] = {
        "rows": 5, "cols": 64, "weight_bytes": list(raw), "tokens": tokens, "out": flat(w[tokens]),
    }

# rms_norm: 3 chunks of 16 sharing one weight vector
x, weight, eps = rand(3, 16, scale=3.0), rand(16), 1e-6
x64 = x.astype(np.float64)
fx["rms_norm"] = {
    "x": flat(x), "weight": flat(weight), "eps": eps,
    "out": flat(x64 / np.sqrt(np.mean(x64**2, axis=1, keepdims=True) + eps) * weight),
}

# rope, NEOX layout: 2 tokens at positions 5 and 6, 3 heads of dim 8
n, n_head, head_dim, pos, base = 2, 3, 8, 5, 1_000_000.0
x = rand(n, n_head, head_dim).astype(np.float64)
theta = (pos + np.arange(n))[:, None, None] / base ** (np.arange(0, head_dim, 2) / head_dim)
a, b = x[..., : head_dim // 2], x[..., head_dim // 2 :]
fx["rope"] = {
    "x": flat(x), "n": n, "head_dim": head_dim, "pos": pos, "freq_base": base,
    "out": flat(np.concatenate([a * np.cos(theta) - b * np.sin(theta), b * np.cos(theta) + a * np.sin(theta)], axis=-1)),
}

# attention: 4 query heads sharing 2 K/V heads. The cache (room for 6 positions) already
# holds 2 positions; a batch of 3 tokens is added at positions 2..4. Token t sees 0..=2+t.
# Scores are scaled up so a softmax without max-subtraction would overflow, and the unused
# cache slot is filled with huge values that must never be read.
n_head, n_kv, head_dim, pos, n, n_ctx = 4, 2, 8, 2, 3, 6
q = rand(n, n_head, head_dim, scale=40.0).astype(np.float64)
k_cache = rand(n_ctx, n_kv, head_dim, scale=40.0)
v_cache = rand(n_ctx, n_kv, head_dim)
k_cache[pos:], v_cache[pos:] = 1e30, 1e30
k, v = rand(n, n_kv, head_dim, scale=40.0), rand(n, n_kv, head_dim)
k_after, v_after = k_cache.copy(), v_cache.copy()
k_after[pos : pos + n], v_after[pos : pos + n] = k, v
out = np.zeros((n, n_head, head_dim))
for t in range(n):
    seen = pos + t + 1
    for h in range(n_head):
        kv = h // (n_head // n_kv)
        s = k_after[:seen, kv].astype(np.float64) @ q[t, h] / np.sqrt(head_dim)
        p = np.exp(s - s.max())
        out[t, h] = (p / p.sum()) @ v_after[:seen, kv].astype(np.float64)
fx["attention"] = {
    "pos": pos, "n_head": n_head, "n_head_kv": n_kv, "head_dim": head_dim,
    "q": flat(q), "k": flat(k), "v": flat(v),
    "k_cache": flat(k_cache), "v_cache": flat(v_cache),
    "k_cache_after": flat(k_after), "v_cache_after": flat(v_after), "out": flat(out),
}

# silu_mul, add
gate, up = rand(12, scale=4.0).astype(np.float64), rand(12).astype(np.float64)
fx["silu_mul"] = {"gate": flat(gate), "up": flat(up), "out": flat(gate / (1 + np.exp(-gate)) * up)}
x, y = rand(12), rand(12)
fx["add"] = {"x": flat(x), "y": flat(y), "out": flat(x.astype(np.float64) + y)}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(fx, indent=1) + "\n")
print(f"wrote {OUT} ({len(fx)} fixtures)")
