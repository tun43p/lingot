# Performance history

Metal backend on the M2 Max, `lingot bench --model models/qwen3-8B-q4_0.gguf` unless noted.
Tokens per second. Each change was picked from the per-kernel GPU times printed with
`LINGOT_PROFILE=1`, and re-validated with the op tests and `ref/compare.py`.

Intermediate rows were measured with shorter runs (the test is given in the cell) and 1 to 3
repetitions; they show the direction, not a precise figure. llama.cpp on the same file:
pp512 478.83, tg128 43.59 (bench/baseline.md).

| Date       | Change                                                                 | pp            | tg            |
| ---------- | ---------------------------------------------------------------------- | ------------- | ------------- |
| 2026-11-27 | Naive kernels: one thread per output value                             | 3.43 (pp64)   | 2.11 (tg16)   |
| 2026-11-27 | matmul: one simdgroup per 4 rows, contiguous reads, `simd_sum`         | 83.78 (pp128) | 11.77 (tg64)  |
| 2026-11-27 | rms_norm: one simdgroup per chunk instead of one thread (was 75 % of GPU time) | 87.49 (pp128) | 36.54 (tg64)  |
| 2026-11-27 | rms_norm: whole threadgroup, 4 values per thread, shared partial sums  | 84.02 (pp128) | 40.06 (tg64)  |
| 2026-11-27 | Q4_0/Q8_0 matmul: one whole block per thread, 8 rows per simdgroup     | 89.41 (pp128) | 46.17 (tg64), 40.89 (tg128) |
| 2026-11-27 | Tiled matmul for batches (64 rows x 32 tokens, `simdgroup_matrix`); attention softmax on a simdgroup | 135.59 (pp512) | 44.17 (tg128) |
| 2026-11-27 | Tile loading and attention dot products with 4-wide vector loads       | 286.35 (pp512) | 43.07 (tg128) |

Rows per simdgroup tried for the Q4_0 matmul (GPU µs per call, decode): half a block per
thread: 2 rows 113.8, 4 rows 80.0, 8 rows 91.5, 16 rows 132.2; one block per thread: 2 rows
107.3, 4 rows 81.8, 8 rows 76.1, 16 rows 97.1. For BF16, 16 values per thread with 8 rows was
slower (tg128 13.81) than 4 values per thread with 4 rows (15.95).

## State at the end of phase 5 (2026-11-27)

`lingot bench --model <file>` with the defaults (pp512, tg128, 5 repetitions, mean ± stddev),
llama.cpp figures from bench/baseline.md.

| File               | pp512          | % of llama.cpp | tg128        | % of llama.cpp | % of ceiling |
| ------------------ | -------------- | -------------- | ------------ | -------------- | ------------ |
| qwen3-8B-bf16.gguf | 280.84 ± 12.33 | 62 %           | 17.64 ± 0.18 | 94 %           | 74 %         |
| qwen3-8B-q8_0.gguf | 301.17 ± 8.20  | 62 %           | 27.92 ± 1.03 | 110 %          | 61 %         |
| qwen3-8B-q4_0.gguf | 290.26 ± 10.38 | 61 %           | 43.25 ± 0.71 | 99 %           | 50 %         |

tg128 in Q4_0 is at 43.25 tokens/s: the 52 tokens/s target (60 % of the ceiling) is not met.

Metal logits against the CPU backend on the 5 prompts (`ref/compare.py --ref cpu --backend
metal`, worst error, top-1 228/228 everywhere): 0.6B BF16 7.3e-6, Q8_0 1.1e-5, Q4_0 2.1e-5;
8B BF16 1.2e-5, Q8_0 1.3e-5, Q4_0 1.4e-5.

Perplexity on the full WikiText-2 test set (584 chunks), `lingot perplexity` on Metal against
llama-perplexity: 8B Q8_0 10.1332 vs 10.1334, 8B Q4_0 10.5275 vs 10.5275.
