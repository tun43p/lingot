// Compute kernels of the Metal backend. One kernel per Backend operation (and per weight type
// where weights are read). Layouts and semantics are those documented on the Backend trait.

#include <metal_stdlib>
using namespace metal;

// Shared by every kernel; each one uses the fields it needs. Must match `Params` in metal.rs.
struct Params {
    uint rows;      // weight rows
    uint cols;      // weight columns
    uint n;         // tokens in the batch
    uint head_dim;
    uint n_head;    // heads per token in the buffer at hand
    uint n_head_kv;
    uint pos;       // position of the first token of the batch
    uint len;       // element count / chunk length / destination offset, depending on the kernel
    float eps;
    float scale;
};

// ---- Weight decoding: value `i` of a row-major weight matrix ----

struct block_q8_0 {
    half d;
    char qs[32];
};

struct block_q4_0 {
    half d;
    uchar qs[16];
};

static inline float at_f32(device const float *w, uint i) {
    return w[i];
}

// BF16 is the top 16 bits of an f32.
static inline float at_bf16(device const ushort *w, uint i) {
    return as_type<float>(uint(w[i]) << 16);
}

static inline float at_q8_0(device const block_q8_0 *w, uint i) {
    return float(w[i / 32].qs[i % 32]) * float(w[i / 32].d);
}

// Low nibble of byte j = weight j, high nibble = weight j + 16.
static inline float at_q4_0(device const block_q4_0 *w, uint i) {
    uint j = i % 32;
    uchar byte = w[i / 32].qs[j % 16];
    uint q = j < 16 ? (byte & 0x0F) : (byte >> 4);
    return (float(q) - 8.0f) * float(w[i / 32].d);
}

// embed: out[t][c] = w[tokens[t]][c].                      grid (cols, n)
// matmul: out[t][r] = sum_c w[r][c] * x[t][c].             grid (rows, n)
#define WEIGHT_KERNELS(NAME, TYPE)                                                        \
    kernel void embed_##NAME(constant Params &p [[buffer(0)]],                            \
                             device const TYPE *w [[buffer(1)]],                          \
                             device const uint *tokens [[buffer(2)]],                     \
                             device float *out [[buffer(3)]],                             \
                             uint2 gid [[thread_position_in_grid]]) {                     \
        out[gid.y * p.cols + gid.x] = at_##NAME(w, tokens[gid.y] * p.cols + gid.x);       \
    }                                                                                     \
    kernel void matmul_##NAME(constant Params &p [[buffer(0)]],                           \
                              device const TYPE *w [[buffer(1)]],                         \
                              device const float *x [[buffer(2)]],                        \
                              device float *out [[buffer(3)]],                            \
                              uint2 gid [[thread_position_in_grid]]) {                    \
        float sum = 0.0f;                                                                 \
        for (uint c = 0; c < p.cols; c++) {                                               \
            sum += at_##NAME(w, gid.x * p.cols + c) * x[gid.y * p.cols + c];              \
        }                                                                                 \
        out[gid.y * p.rows + gid.x] = sum;                                                \
    }

WEIGHT_KERNELS(f32, float)
WEIGHT_KERNELS(bf16, ushort)
WEIGHT_KERNELS(q8_0, block_q8_0)
WEIGHT_KERNELS(q4_0, block_q4_0)

// RMSNorm of chunk gid, `len` values long.                 grid (chunks)
kernel void rms_norm(constant Params &p [[buffer(0)]],
                     device const float *x [[buffer(1)]],
                     device const float *weight [[buffer(2)]],
                     device float *out [[buffer(3)]],
                     uint gid [[thread_position_in_grid]]) {
    uint base = gid * p.len;
    float sum = 0.0f;
    for (uint i = 0; i < p.len; i++) {
        sum += x[base + i] * x[base + i];
    }
    float scale = 1.0f / sqrt(sum / float(p.len) + p.eps);
    for (uint i = 0; i < p.len; i++) {
        out[base + i] = x[base + i] * scale * weight[i];
    }
}

// RoPE, NEOX layout, in place. `inv_freq` holds head_dim / 2 frequencies.
// grid (head_dim / 2, heads in the buffer), n_head = heads per token
kernel void rope(constant Params &p [[buffer(0)]],
                 device float *x [[buffer(1)]],
                 device const float *inv_freq [[buffer(2)]],
                 uint2 gid [[thread_position_in_grid]]) {
    uint half_dim = p.head_dim / 2;
    uint lo = gid.y * p.head_dim + gid.x;
    uint hi = lo + half_dim;
    float angle = float(p.pos + gid.y / p.n_head) * inv_freq[gid.x];
    float c = cos(angle);
    float s = sin(angle);
    float a = x[lo];
    float b = x[hi];
    x[lo] = a * c - b * s;
    x[hi] = b * c + a * s;
}

// dst[len + i] = src[i].                                   grid (source length)
kernel void copy_at(constant Params &p [[buffer(0)]],
                    device const float *src [[buffer(1)]],
                    device float *dst [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
    dst[p.len + gid] = src[gid];
}

// Attention, step 1: scores[t][h][position] = q[t][h] . k_cache[position][kv head] * scale.
// `len` = pos + n, the row length of `scores`.             grid (len, n_head, n)
kernel void attention_scores(constant Params &p [[buffer(0)]],
                             device const float *q [[buffer(1)]],
                             device const float *k_cache [[buffer(2)]],
                             device float *scores [[buffer(3)]],
                             uint3 gid [[thread_position_in_grid]]) {
    if (gid.x > p.pos + gid.z) {
        return; // causal: token t only sees positions 0..=pos + t
    }
    uint qi = (gid.z * p.n_head + gid.y) * p.head_dim;
    uint kv = gid.y / (p.n_head / p.n_head_kv);
    uint ki = (gid.x * p.n_head_kv + kv) * p.head_dim;
    float sum = 0.0f;
    for (uint i = 0; i < p.head_dim; i++) {
        sum += q[qi + i] * k_cache[ki + i];
    }
    scores[(gid.z * p.n_head + gid.y) * p.len + gid.x] = sum * p.scale;
}

// Attention, step 2: softmax of each row of scores, in place, over the visible positions.
// The max is subtracted first so the exponential cannot overflow.   grid (n_head, n)
kernel void attention_softmax(constant Params &p [[buffer(0)]],
                              device float *scores [[buffer(1)]],
                              uint2 gid [[thread_position_in_grid]]) {
    uint base = (gid.y * p.n_head + gid.x) * p.len;
    uint seen = p.pos + gid.y + 1;
    float top = scores[base];
    for (uint i = 1; i < seen; i++) {
        top = max(top, scores[base + i]);
    }
    float total = 0.0f;
    for (uint i = 0; i < seen; i++) {
        float e = exp(scores[base + i] - top);
        scores[base + i] = e;
        total += e;
    }
    for (uint i = 0; i < seen; i++) {
        scores[base + i] /= total;
    }
}

// Attention, step 3: out[t][h][i] = sum over visible positions of prob * v_cache[position][kv][i].
// grid (head_dim, n_head, n)
kernel void attention_mix(constant Params &p [[buffer(0)]],
                          device const float *probs [[buffer(1)]],
                          device const float *v_cache [[buffer(2)]],
                          device float *out [[buffer(3)]],
                          uint3 gid [[thread_position_in_grid]]) {
    uint row = gid.z * p.n_head + gid.y;
    uint kv = gid.y / (p.n_head / p.n_head_kv);
    uint seen = p.pos + gid.z + 1;
    float sum = 0.0f;
    for (uint i = 0; i < seen; i++) {
        sum += probs[row * p.len + i] * v_cache[(i * p.n_head_kv + kv) * p.head_dim + gid.x];
    }
    out[row * p.head_dim + gid.x] = sum;
}

// out = SiLU(gate) * up.                                   grid (len)
kernel void silu_mul(constant Params &p [[buffer(0)]],
                     device const float *gate [[buffer(1)]],
                     device const float *up [[buffer(2)]],
                     device float *out [[buffer(3)]],
                     uint gid [[thread_position_in_grid]]) {
    float g = gate[gid];
    out[gid] = g / (1.0f + exp(-g)) * up[gid];
}

// x += y.                                                  grid (len)
kernel void add(constant Params &p [[buffer(0)]],
                device float *x [[buffer(1)]],
                device const float *y [[buffer(2)]],
                uint gid [[thread_position_in_grid]]) {
    x[gid] += y[gid];
}
