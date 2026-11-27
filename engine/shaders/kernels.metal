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
#define EMBED_KERNEL(NAME, TYPE)                                                          \
    kernel void embed_##NAME(constant Params &p [[buffer(0)]],                            \
                             device const TYPE *w [[buffer(1)]],                          \
                             device const uint *tokens [[buffer(2)]],                     \
                             device float *out [[buffer(3)]],                             \
                             uint2 gid [[thread_position_in_grid]]) {                     \
        out[gid.y * p.cols + gid.x] = at_##NAME(w, tokens[gid.y] * p.cols + gid.x);       \
    }

EMBED_KERNEL(f32, float)
EMBED_KERNEL(bf16, ushort)
EMBED_KERNEL(q8_0, block_q8_0)
EMBED_KERNEL(q4_0, block_q4_0)

// ---- matmul: out[t][r] = sum_c w[r][c] * x[t][c] ----
//
// One simdgroup (32 threads) computes R rows for one token. The threads split the columns so
// that, at each step, the simdgroup reads one contiguous run of weight bytes per row; each
// thread keeps a partial sum per row and `simd_sum` adds them up.
// Dispatched in threadgroups of 32 threads on a grid of (ceil(rows / R), n).
// R and the per-thread step were picked by measurement on an M2 Max (see bench/history.md);
// R must match `matmul_rows` in metal.rs.

#define MATMUL_ARGS(TYPE)                                                                 \
    constant Params &p [[buffer(0)]], device const TYPE *w [[buffer(1)]],                 \
    device const float *x [[buffer(2)]], device float *out [[buffer(3)]],                 \
    uint2 group [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]

#define MATMUL_BEGIN(R)                                                                   \
    uint row0 = group.x * R;                                                              \
    uint nr = min(uint(R), p.rows - row0);                                                \
    device const float *xt = x + group.y * p.cols;                                        \
    float sum[R] = {0.0f};

#define MATMUL_END                                                                        \
    for (uint r = 0; r < nr; r++) {                                                       \
        float total = simd_sum(sum[r]);                                                   \
        if (lane == 0) {                                                                  \
            out[group.y * p.rows + row0 + r] = total;                                     \
        }                                                                                 \
    }

static inline float4 load4(device const float *v) {
    return *(device const packed_float4 *)v;
}

static inline float4 load4_bf16(device const ushort *v) {
    return as_type<float4>(uint4(*(device const packed_ushort4 *)v) << 16);
}

// 4 rows per simdgroup; each thread takes 4 consecutive values per step, the simdgroup
// covers 128. cols % 4 == 0.
kernel void matmul_f32(MATMUL_ARGS(float)) {
    MATMUL_BEGIN(4)
    for (uint c = lane * 4; c < p.cols; c += 128) {
        float4 xv = load4(xt + c);
        for (uint r = 0; r < nr; r++) {
            sum[r] += dot(load4(w + (row0 + r) * p.cols + c), xv);
        }
    }
    MATMUL_END
}

kernel void matmul_bf16(MATMUL_ARGS(ushort)) {
    MATMUL_BEGIN(4)
    for (uint c = lane * 4; c < p.cols; c += 128) {
        float4 xv = load4(xt + c);
        for (uint r = 0; r < nr; r++) {
            sum[r] += dot(load4_bf16(w + (row0 + r) * p.cols + c), xv);
        }
    }
    MATMUL_END
}

// 8 rows per simdgroup; each thread takes one whole block per step, the simdgroup covers 32.
kernel void matmul_q8_0(MATMUL_ARGS(block_q8_0)) {
    MATMUL_BEGIN(8)
    uint nb = p.cols / 32;
    for (uint b = lane; b < nb; b += 32) {
        device const float *xb = xt + b * 32;
        float4 x0 = load4(xb), x1 = load4(xb + 4), x2 = load4(xb + 8), x3 = load4(xb + 12);
        float4 x4 = load4(xb + 16), x5 = load4(xb + 20), x6 = load4(xb + 24), x7 = load4(xb + 28);
        for (uint r = 0; r < nr; r++) {
            device const block_q8_0 *block = w + (row0 + r) * nb + b;
            device const packed_char4 *q = (device const packed_char4 *)(block->qs);
            float acc = dot(float4(char4(q[0])), x0) + dot(float4(char4(q[1])), x1)
                      + dot(float4(char4(q[2])), x2) + dot(float4(char4(q[3])), x3)
                      + dot(float4(char4(q[4])), x4) + dot(float4(char4(q[5])), x5)
                      + dot(float4(char4(q[6])), x6) + dot(float4(char4(q[7])), x7);
            sum[r] += float(block->d) * acc;
        }
    }
    MATMUL_END
}

// 8 rows per simdgroup; each thread takes one whole block per step (16 bytes: low nibbles are
// weights 0..16, high nibbles weights 16..32), the simdgroup covers 32.
// sum (q - 8) * d * x  =  d * (sum q * x  -  8 * sum x)
kernel void matmul_q4_0(MATMUL_ARGS(block_q4_0)) {
    MATMUL_BEGIN(8)
    uint nb = p.cols / 32;
    for (uint b = lane; b < nb; b += 32) {
        device const float *xb = xt + b * 32;
        float4 lo0 = load4(xb), lo1 = load4(xb + 4), lo2 = load4(xb + 8), lo3 = load4(xb + 12);
        float4 hi0 = load4(xb + 16), hi1 = load4(xb + 20), hi2 = load4(xb + 24), hi3 = load4(xb + 28);
        float xsum = dot(lo0 + lo1 + lo2 + lo3 + hi0 + hi1 + hi2 + hi3, float4(1.0f));
        for (uint r = 0; r < nr; r++) {
            device const block_q4_0 *block = w + (row0 + r) * nb + b;
            device const packed_uchar4 *q = (device const packed_uchar4 *)(block->qs);
            uint4 b0 = uint4(uchar4(q[0])), b1 = uint4(uchar4(q[1]));
            uint4 b2 = uint4(uchar4(q[2])), b3 = uint4(uchar4(q[3]));
            float acc = dot(float4(b0 & 0x0F), lo0) + dot(float4(b1 & 0x0F), lo1)
                      + dot(float4(b2 & 0x0F), lo2) + dot(float4(b3 & 0x0F), lo3)
                      + dot(float4(b0 >> 4), hi0) + dot(float4(b1 >> 4), hi1)
                      + dot(float4(b2 >> 4), hi2) + dot(float4(b3 >> 4), hi3);
            sum[r] += float(block->d) * (acc - 8.0f * xsum);
        }
    }
    MATMUL_END
}

// 16 consecutive weights of a row, starting at column `c` (a multiple of 16), as 4 float4.

static inline float4x4 load16_f32(device const float *w, uint row, uint cols, uint c) {
    device const float *v = w + row * cols + c;
    return float4x4(load4(v), load4(v + 4), load4(v + 8), load4(v + 12));
}

static inline float4x4 load16_bf16(device const ushort *w, uint row, uint cols, uint c) {
    device const ushort *v = w + row * cols + c;
    return float4x4(load4_bf16(v), load4_bf16(v + 4), load4_bf16(v + 8), load4_bf16(v + 12));
}

static inline float4x4 load16_q8_0(device const block_q8_0 *w, uint row, uint cols, uint c) {
    device const block_q8_0 *block = w + row * (cols / 32) + c / 32;
    device const packed_char4 *q = (device const packed_char4 *)(block->qs + c % 32);
    float d = float(block->d);
    return float4x4(float4(char4(q[0])) * d, float4(char4(q[1])) * d,
                    float4(char4(q[2])) * d, float4(char4(q[3])) * d);
}

// The first 16 weights of a block are the low nibbles of its 16 bytes, the last 16 the high ones.
static inline float4x4 load16_q4_0(device const block_q4_0 *w, uint row, uint cols, uint c) {
    device const block_q4_0 *block = w + row * (cols / 32) + c / 32;
    device const packed_uchar4 *q = (device const packed_uchar4 *)(block->qs);
    uint shift = (c % 32) / 4; // 0 for the low nibbles, 4 for the high ones
    float d = float(block->d);
    float4x4 m;
    for (uint i = 0; i < 4; i++) {
        m[i] = (float4((uint4(uchar4(q[i])) >> shift) & 0x0F) - 8.0f) * d;
    }
    return m;
}

// ---- Tiled matmul for batches (prefill) ----
//
// One threadgroup of 128 threads (4 simdgroups) computes a tile of 64 rows x 32 tokens, so
// each weight is read and decoded once for 32 tokens. Columns are walked 32 at a time: the
// threads first copy a 64 x 32 slice of weights (`sa`) and a 32 x 32 slice of activations
// (`sb`) to threadgroup memory, then each simdgroup accumulates its quarter of the tile
// (32 rows x 16 tokens) with 8 x 8 simdgroup matrices. Tiles on the edges are padded with zeros.
// Dispatched in threadgroups of 128 threads on a grid of (ceil(rows / 64), ceil(n / 32)).
#define TILED_KERNEL(NAME, TYPE)                                                          \
    kernel void matmul_tiled_##NAME(constant Params &p [[buffer(0)]],                     \
                                    device const TYPE *w [[buffer(1)]],                   \
                                    device const float *x [[buffer(2)]],                  \
                                    device float *out [[buffer(3)]],                      \
                                    uint2 group [[threadgroup_position_in_grid]],         \
                                    ushort tid [[thread_index_in_threadgroup]],           \
                                    ushort sg [[simdgroup_index_in_threadgroup]]) {       \
        threadgroup float sa[64 * 32]; /* [row][column], later the result [row][token] */ \
        threadgroup float sb[32 * 32]; /* [token][column] */                              \
        /* Loading duties: 2 threads per row (16 weights each), 4 per token (8 values). */\
        uint a_row = tid / 2, a_col = (tid % 2) * 16;                                     \
        uint b_row = tid / 4, b_col = (tid % 4) * 8;                                      \
        uint row = group.x * 64 + a_row;                                                  \
        uint token = group.y * 32 + b_row;                                                \
        /* Computing duties: this simdgroup's 4 x 2 blocks of 8 rows x 8 tokens. */       \
        uint row_block = (sg % 2) * 4, token_block = (sg / 2) * 2;                        \
        simdgroup_float8x8 mc[8];                                                         \
        for (uint i = 0; i < 8; i++) {                                                    \
            mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);                         \
        }                                                                                 \
        for (uint c = 0; c < p.cols; c += 32) {                                           \
            float4x4 a = float4x4(0.0f);                                                  \
            if (row < p.rows) {                                                           \
                a = load16_##NAME(w, row, p.cols, c + a_col);                             \
            }                                                                             \
            float4 b0 = 0.0f, b1 = 0.0f;                                                  \
            if (token < p.n) {                                                            \
                b0 = load4(x + token * p.cols + c + b_col);                               \
                b1 = load4(x + token * p.cols + c + b_col + 4);                           \
            }                                                                             \
            threadgroup_barrier(mem_flags::mem_threadgroup);                              \
            for (uint i = 0; i < 4; i++) {                                                \
                *(threadgroup packed_float4 *)(sa + a_row * 32 + a_col + i * 4) = a[i];   \
            }                                                                             \
            *(threadgroup packed_float4 *)(sb + b_row * 32 + b_col) = b0;                 \
            *(threadgroup packed_float4 *)(sb + b_row * 32 + b_col + 4) = b1;             \
            threadgroup_barrier(mem_flags::mem_threadgroup);                              \
            for (uint k = 0; k < 32; k += 8) {                                            \
                simdgroup_float8x8 mb[2];                                                 \
                for (uint t = 0; t < 2; t++) {                                            \
                    /* 8 columns x 8 tokens: the transpose of a block of sb */           \
                    simdgroup_load(mb[t], sb, 32, ulong2(k, (token_block + t) * 8), true);\
                }                                                                         \
                for (uint r = 0; r < 4; r++) {                                            \
                    simdgroup_float8x8 ma;                                                \
                    simdgroup_load(ma, sa, 32, ulong2(k, (row_block + r) * 8));           \
                    for (uint t = 0; t < 2; t++) {                                        \
                        simdgroup_multiply_accumulate(mc[r * 2 + t], ma, mb[t], mc[r * 2 + t]); \
                    }                                                                     \
                }                                                                         \
            }                                                                             \
        }                                                                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                  \
        for (uint r = 0; r < 4; r++) {                                                    \
            for (uint t = 0; t < 2; t++) {                                                \
                simdgroup_store(mc[r * 2 + t], sa, 32,                                    \
                                ulong2((token_block + t) * 8, (row_block + r) * 8));      \
            }                                                                             \
        }                                                                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                  \
        /* Each thread writes 16 results of its row. */                                   \
        for (uint i = 0; i < 16; i++) {                                                   \
            uint t = group.y * 32 + a_col + i;                                            \
            if (row < p.rows && t < p.n) {                                                \
                out[t * p.rows + row] = sa[a_row * 32 + a_col + i];                       \
            }                                                                             \
        }                                                                                 \
    }

TILED_KERNEL(f32, float)
TILED_KERNEL(bf16, ushort)
TILED_KERNEL(q8_0, block_q8_0)
TILED_KERNEL(q4_0, block_q4_0)

// RMSNorm of one chunk of `len` values per threadgroup, 4 values per thread: each simdgroup
// sums its squares (`simd_sum`), the partial sums meet in threadgroup memory, then every
// thread scales its 4 values. Needs len % 128 == 0 and len <= 4096.
// Dispatched in threadgroups of len / 4 threads on a grid of (chunks).
kernel void rms_norm(constant Params &p [[buffer(0)]],
                     device const float *x [[buffer(1)]],
                     device const float *weight [[buffer(2)]],
                     device float *out [[buffer(3)]],
                     uint group [[threadgroup_position_in_grid]],
                     ushort tid [[thread_index_in_threadgroup]],
                     ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[32];
    uint at = group * p.len + tid * 4;
    float4 v = load4(x + at);
    float sum = simd_sum(dot(v, v));
    if (lane == 0) {
        partial[tid / 32] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < p.len / 128; i++) {
        total += partial[i];
    }
    float scale = 1.0f / sqrt(total / float(p.len) + p.eps);
    *(device packed_float4 *)(out + at) = v * scale * load4(weight + tid * 4);
}

// Same for any chunk length: one simdgroup per chunk, each thread taking every 32nd value.
// Dispatched in threadgroups of 32 threads on a grid of (chunks).
kernel void rms_norm_any(constant Params &p [[buffer(0)]],
                         device const float *x [[buffer(1)]],
                         device const float *weight [[buffer(2)]],
                         device float *out [[buffer(3)]],
                         uint group [[threadgroup_position_in_grid]],
                         ushort lane [[thread_index_in_simdgroup]]) {
    uint base = group * p.len;
    float sum = 0.0f;
    for (uint i = lane; i < p.len; i += 32) {
        sum += x[base + i] * x[base + i];
    }
    float scale = 1.0f / sqrt(simd_sum(sum) / float(p.len) + p.eps);
    for (uint i = lane; i < p.len; i += 32) {
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
// `len` = pos + n, the row length of `scores`. head_dim % 4 == 0.   grid (len, n_head, n)
kernel void attention_scores(constant Params &p [[buffer(0)]],
                             device const float *q [[buffer(1)]],
                             device const float *k_cache [[buffer(2)]],
                             device float *scores [[buffer(3)]],
                             uint3 gid [[thread_position_in_grid]]) {
    if (gid.x > p.pos + gid.z) {
        return; // causal: token t only sees positions 0..=pos + t
    }
    uint kv = gid.y / (p.n_head / p.n_head_kv);
    device const float *qh = q + (gid.z * p.n_head + gid.y) * p.head_dim;
    device const float *kh = k_cache + (gid.x * p.n_head_kv + kv) * p.head_dim;
    float4 sum = 0.0f;
    for (uint i = 0; i < p.head_dim; i += 4) {
        sum += load4(qh + i) * load4(kh + i);
    }
    scores[(gid.z * p.n_head + gid.y) * p.len + gid.x] = dot(sum, float4(1.0f)) * p.scale;
}

// Attention, step 2: softmax of each row of scores, in place, over the visible positions.
// One simdgroup per row; each thread takes every 32nd position. The max is subtracted first
// so the exponential cannot overflow.
// Dispatched in threadgroups of 32 threads on a grid of (n_head, n).
kernel void attention_softmax(constant Params &p [[buffer(0)]],
                              device float *scores [[buffer(1)]],
                              uint2 group [[threadgroup_position_in_grid]],
                              ushort lane [[thread_index_in_simdgroup]]) {
    uint base = (group.y * p.n_head + group.x) * p.len;
    uint seen = p.pos + group.y + 1;
    float top = -INFINITY;
    for (uint i = lane; i < seen; i += 32) {
        top = max(top, scores[base + i]);
    }
    top = simd_max(top);
    float total = 0.0f;
    for (uint i = lane; i < seen; i += 32) {
        float e = exp(scores[base + i] - top);
        scores[base + i] = e;
        total += e;
    }
    total = simd_sum(total);
    for (uint i = lane; i < seen; i += 32) {
        scores[base + i] /= total;
    }
}

// Attention, step 3: out[t][h] = sum over visible positions of prob * v_cache[position][kv],
// 4 values per thread. head_dim % 4 == 0.                  grid (head_dim / 4, n_head, n)
kernel void attention_mix(constant Params &p [[buffer(0)]],
                          device const float *probs [[buffer(1)]],
                          device const float *v_cache [[buffer(2)]],
                          device float *out [[buffer(3)]],
                          uint3 gid [[thread_position_in_grid]]) {
    uint row = gid.z * p.n_head + gid.y;
    uint kv = gid.y / (p.n_head / p.n_head_kv);
    uint seen = p.pos + gid.z + 1;
    device const float *v = v_cache + kv * p.head_dim + gid.x * 4;
    float4 sum = 0.0f;
    for (uint i = 0; i < seen; i++) {
        sum += probs[row * p.len + i] * load4(v + i * p.n_head_kv * p.head_dim);
    }
    *(device packed_float4 *)(out + row * p.head_dim + gid.x * 4) = sum;
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
