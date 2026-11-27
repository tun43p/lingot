# llama.cpp baseline

Numbers to beat. Measured on 2026-11-27.

## Setup

- Machine: MacBook Pro, Apple M2 Max, 32 GB, macOS 27.2.1, on AC power.
- llama.cpp: 1.0.0, build 12359, commit `b1b770d90` (Homebrew bottle, with ggml 1.1.0).
- Conversion scripts: `vendor/llama.cpp` checked out at the same commit.
- Sources: `Qwen/Qwen3-0.6B` @ `c1899de289a04d12100db370d81485cdf75e47ca`,
  `Qwen/Qwen3-8B` @ `b968826d9c46dd6066d109eabc6255188de91218`.
- WikiText-2: `wiki.test.raw` from
  `https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip`
  (sha256 `173c87a5…9e7dd08`), 584 chunks at `n_ctx=512`.

## Files

For `M` in `0.6B`, `8B`:

```bash
python vendor/llama.cpp/convert_hf_to_gguf.py models/hf/Qwen3-$M --outtype bf16 --outfile models/qwen3-$M-bf16.gguf
llama-quantize --pure models/qwen3-$M-bf16.gguf models/qwen3-$M-q8_0.gguf Q8_0
llama-quantize --pure models/qwen3-$M-bf16.gguf models/qwen3-$M-q4_0.gguf Q4_0
gguf-dump models/qwen3-$M-<type>.gguf
```

| File                  | Size      | Tensor types (from `gguf-dump`) |
| --------------------- | --------- | ------------------------------- |
| qwen3-0.6B-bf16.gguf  | 1.5 GB    | 198 BF16, 113 F32               |
| qwen3-0.6B-q8_0.gguf  | 805 MB    | 198 Q8_0, 113 F32               |
| qwen3-0.6B-q4_0.gguf  | 429 MB    | 198 Q4_0, 113 F32               |
| qwen3-8B-bf16.gguf    | 15.26 GiB | 254 BF16, 145 F32               |
| qwen3-8B-q8_0.gguf    | 8.11 GiB  | 254 Q8_0, 145 F32               |
| qwen3-8B-q4_0.gguf    | 4.29 GiB  | 254 Q4_0, 145 F32               |

F32 tensors are the 1-D norm weights only. Both models have an explicit
`output.weight` tensor, including the 0.6B (tied embeddings are written twice
by this llama.cpp version).

## Speed: Qwen3-8B, pp512 / tg128

```bash
M=models/qwen3-8B-bf16.gguf,models/qwen3-8B-q8_0.gguf,models/qwen3-8B-q4_0.gguf
llama-bench -m $M -p 512 -n 128 -ngl 99 -o md   # Metal
llama-bench -m $M -p 512 -n 128 -ngl 0 -o md    # CPU
```

Defaults otherwise: 5 repetitions, 8 threads, batch 2048, ubatch 512, flash-attn auto.
Tokens per second, mean ± stddev.

| Format | Metal pp512    | Metal tg128  | CPU pp512     | CPU tg128    |
| ------ | -------------- | ------------ | ------------- | ------------ |
| BF16   | 454.00 ± 55.57 | 18.84 ± 0.35 | 73.44 ± 1.83  | 1.09 ± 0.02  |
| Q8_0   | 485.30 ± 89.10 | 25.43 ± 0.89 | 101.59 ± 1.23 | 14.65 ± 0.07 |
| Q4_0   | 478.83 ± 63.33 | 43.59 ± 2.15 | 100.93 ± 6.79 | 24.43 ± 0.68 |

Metal pp512 is noisy (stddev 12-18 %); rerun before using it as a tight target.

### Against the theoretical ceiling (docs/plan.md, 400 GB/s)

| Format | Ceiling tg | llama.cpp Metal tg128 | % of ceiling | lingot target (60 %) | 50 % of llama.cpp |
| ------ | ---------- | --------------------- | ------------ | -------------------- | ----------------- |
| BF16   | ~24        | 18.84                 | 78 %         | ~15                  | 9.4               |
| Q8_0   | ~46        | 25.43                 | 55 %         | ~28                  | 12.7              |
| Q4_0   | ~87        | 43.59                 | 50 %         | ~52                  | 21.8              |

The 60 % roofline target is above llama.cpp itself for Q8_0 (28 vs 25.4) and
Q4_0 (52 vs 43.6).

## Perplexity: WikiText-2 test

```bash
W=models/wikitext/wikitext-2-raw/wiki.test.raw
llama-perplexity -m models/qwen3-8B-<q>.gguf   -f $W -c 512 -ngl 99
llama-perplexity -m models/qwen3-0.6B-<q>.gguf -f $W -c 512 -ngl 99 --chunks 20
```

Defaults otherwise: batch 2048, `n_seq=4`.

| File                 | Chunks     | PPL               |
| -------------------- | ---------- | ----------------- |
| qwen3-8B-q8_0.gguf   | 584 (full) | 10.1334 ± 0.08212 |
| qwen3-8B-q4_0.gguf   | 584 (full) | 10.5275 ± 0.08486 |
| qwen3-0.6B-q8_0.gguf | 20         | 23.6391 ± 1.12559 |
| qwen3-0.6B-q4_0.gguf | 20         | 26.9929 ± 1.27023 |

## Reference dumps

```bash
python ref/dump.py --model models/hf/Qwen3-0.6B --out ref/dumps/qwen3-0.6B-f32
python ref/dump.py --model models/hf/Qwen3-8B --out ref/dumps/qwen3-8B-bf16 --dtype bf16
```

torch 2.11.0 (CPU), transformers 5.23.0; full versions in `ref/requirements.txt`.
The 0.6B f32 dump was run twice: all 6 files have identical SHA-256.

| File                  | SHA-256 (qwen3-0.6B-f32)                                           |
| --------------------- | ------------------------------------------------------------------ |
| index.json            | `d75f6b8a1f0b3b18552a6a9d6949b912b9e4088c4dbe0dcdc7d5116510b94022` |
| prompt_0.safetensors  | `5b72f288382a93b71d0e11940c484cf501a21c3c2549afdf74ae61dde8d510a5` |
| prompt_1.safetensors  | `516bd113038d53ffe07fe11d14922ee9d1bc4b5c4f99ee17c45e806e30554d58` |
| prompt_2.safetensors  | `4b0b8cfbe1d7689e612293da16d350d49bb423061d223125d0d4d833401fae30` |
| prompt_3.safetensors  | `63105adafc37f29500c53cbb917a8d9420f6f799a9f6ff34770cd11e9c2ab3f7` |
| prompt_4.safetensors  | `ae77d14f17aa77af6d1a00b32fccab896691516f7d97cbc2e2cb8cf90d10a1c9` |
