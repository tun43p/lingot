"""Generate engine/tests/fixtures/ops.json: NumPy reference values for each Backend op.

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


def as_bf16(w):
    """Truncate f32 to bf16: (u16 bit patterns, the values they represent)."""
    bits = (w.view(np.uint32) >> 16).astype(np.uint16)
    return bits, (bits.astype(np.uint32) << 16).view(np.float32)


def flat(a):
    return np.asarray(a, dtype=np.float64).ravel().tolist()


fx = {}

# matvec / embed, F32 and BF16 weights (5 rows x 7 cols)
w, x = rand(5, 7), rand(7)
bits, wb = as_bf16(rand(5, 7))
fx["matvec_f32"] = {
    "rows": 5, "cols": 7, "weight_bytes": list(w.tobytes()), "x": flat(x),
    "out": flat(w.astype(np.float64) @ x),
}
fx["matvec_bf16"] = {
    "rows": 5, "cols": 7, "weight_bytes": list(bits.tobytes()), "x": flat(x),
    "out": flat(wb.astype(np.float64) @ x),
}
fx["embed_bf16"] = {
    "rows": 5, "cols": 7, "weight_bytes": list(bits.tobytes()), "token": 3, "out": flat(wb[3]),
}

# matvec / embed on quantized weights: 3 rows of 2 blocks of 32 weights
def blocks(n, kind):
    """n random blocks: (raw bytes, dequantized weights)."""
    raw, values = b"", []
    for _ in range(n):
        d = np.float16(rng.uniform(0.001, 0.05) * rng.choice([-1, 1]))
        if kind == "q8_0":  # d, then 32 int8: w = q * d
            q = rng.integers(-128, 128, 32, dtype=np.int8)
            raw += d.tobytes() + q.tobytes()
            values.append(q.astype(np.float64) * float(d))
        else:  # q4_0: d, then 16 bytes; low nibble = weight j, high nibble = weight j + 16
            q = rng.integers(0, 256, 16, dtype=np.uint8)
            raw += d.tobytes() + q.tobytes()
            both = np.concatenate([q & 0x0F, q >> 4]).astype(np.float64)
            values.append((both - 8) * float(d))
    return raw, np.concatenate(values)


x = rand(64)
for kind in ("q8_0", "q4_0"):
    raw, w = blocks(6, kind)
    w = w.reshape(3, 64)
    fx[f"matvec_{kind}"] = {
        "rows": 3, "cols": 64, "weight_bytes": list(raw), "x": flat(x), "out": flat(w @ x),
    }
    fx[f"embed_{kind}"] = {
        "rows": 3, "cols": 64, "weight_bytes": list(raw), "token": 1, "out": flat(w[1]),
    }

# rms_norm
x, weight, eps = rand(16, scale=3.0), rand(16), 1e-6
x64 = x.astype(np.float64)
fx["rms_norm"] = {
    "x": flat(x), "weight": flat(weight), "eps": eps,
    "out": flat(x64 / np.sqrt(np.mean(x64**2) + eps) * weight),
}

# rope, NEOX layout: 3 heads of dim 8 at position 5
head_dim, pos, base = 8, 5, 1_000_000.0
x = rand(3, head_dim).astype(np.float64)
theta = pos / base ** (np.arange(0, head_dim, 2) / head_dim)
a, b = x[:, : head_dim // 2], x[:, head_dim // 2 :]
fx["rope"] = {
    "x": flat(x), "head_dim": head_dim, "pos": pos, "freq_base": base,
    "out": flat(np.concatenate([a * np.cos(theta) - b * np.sin(theta), b * np.cos(theta) + a * np.sin(theta)], axis=1)),
}

# attention: 4 query heads sharing 2 K/V heads, 3 cached positions.
# Scores are scaled up so a softmax without max-subtraction would overflow.
n_head, n_kv, head_dim, n_pos = 4, 2, 8, 3
q = rand(n_head, head_dim, scale=40.0).astype(np.float64)
k = rand(n_pos, n_kv, head_dim, scale=40.0).astype(np.float64)
v = rand(n_pos, n_kv, head_dim).astype(np.float64)
out = np.zeros((n_head, head_dim))
for h in range(n_head):
    kv = h // (n_head // n_kv)
    s = k[:, kv] @ q[h] / np.sqrt(head_dim)
    p = np.exp(s - s.max())
    out[h] = (p / p.sum()) @ v[:, kv]
fx["attention"] = {
    "q": flat(q), "k": flat(k), "v": flat(v),
    "n_head": n_head, "n_head_kv": n_kv, "head_dim": head_dim, "out": flat(out),
}

# silu_mul, add
gate, up = rand(12, scale=4.0).astype(np.float64), rand(12).astype(np.float64)
fx["silu_mul"] = {"gate": flat(gate), "up": flat(up), "out": flat(gate / (1 + np.exp(-gate)) * up)}
x, y = rand(12), rand(12)
fx["add"] = {"x": flat(x), "y": flat(y), "out": flat(x.astype(np.float64) + y)}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(fx, indent=1) + "\n")
print(f"wrote {OUT} ({len(fx)} fixtures)")
