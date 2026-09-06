# Qwen3.8-Flash-Next (llama.cpp, RTX 5090)

The same 180B model on a 32 GB RTX 5090: llama.cpp with every routed expert
streaming from system RAM, tail layers on the card. Full 262,144-token
window, vision included.

**42-43 tok/s decode on short prompts, 40.9 at 6K, 38.2 at 24K; prefill
850-940 tok/s**, measured 2026-08-31, speculation off.

| | |
|---|---|
| **Engine** | llama.cpp b10705 (`2578138397d7`) + [`maxspeed.patch`](../flash-next-5090/maxspeed.patch), built by [`build-engine.sh`](../flash-next-5090/build-engine.sh) into `llamacpp-qwen4exp:2578138397d7-p50825275` |
| **Weights** | [`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) UD-Q4_K_XL, 103.7 GiB in 4 shards, revision `c8b5954a88c2775c546b92593eda40ea041d3176` (the 2026-08-28 imatrix requant; earlier uploads differ) |
| **Vision** | `mmproj-BF16.gguf`, 865 MiB, same repo |
| **Card** | RTX 5090, 30.0 GiB of 31.8 at `--n-cpu-moe 43` |
| **Host** | ~100 GiB free RAM for the mmap page cache, 48 physical cores, NVMe |

llama.cpp is the only engine with per-tensor placement (`--n-cpu-moe`); vLLM
and SGLang need the ~78 GiB core GPU-resident. 10 of 512 experts fire per
token, so DDR4 traffic is ~1.4 GiB/token, not the 71.7 GiB the experts
occupy.

## Memory

```
GPU   5.2 GiB  non-expert tensors
   +  0.9 GiB  vision ViT
   +  6.0 GiB  KV at 262,144 tokens, f16 (12 of 48 layers)
   +  6.0 GiB  compute buffers + CUDA context (ubatch 2048)
   = 18.1 GiB  at --n-cpu-moe 48 (no experts on GPU)
HOST 71.7 GiB  routed experts (mmap)
   + 26.8 GiB  PLE table, IQ4_NL
```

`--n-cpu-moe N`: layers 0..N-1 experts on host, N..47 on the card. Expert
layers are 1.46 GiB each except 2, 4, 30, 46, 47 at 1.71-1.90 GiB.

| `--n-cpu-moe` | VRAM | decode | |
|---|---|---|---|
| 48 | 18.1 GiB | ~39.8 | floor |
| **43** | **30.0 GiB** | **43.4** | shipped |
| 42 | 31.7 GiB peak at 32K | 43.8 | too thin |
| 38 | 33.2 GiB | | does not fit |

Judge peak VRAM through a deep-context run, not the load line.

## Quant

| | UD-Q4_K_XL | UD-IQ4_XS |
|---|---|---|
| routed experts | 71.7 GiB: Q4_K x94, Q5_1 x43, Q8_0 x5, Q5_K x2 | 55.5 GiB: IQ3_S x94, IQ4_NL x43, Q8_0 x5, IQ4_XS x2 |
| PLE table | 26.8 GiB, IQ4_NL | 26.8 GiB, IQ4_NL |
| dense + shared + embed | 5.1 GiB | 4.9 GiB |

UD-IQ4_XS is a 3.4-bit expert build; the bytes it saves are host RAM, and
decode is not bandwidth-bound here (~51 of ~205 GB/s), so it is the wrong
trade. Open question: the PLE table at 4.5 bits is the one tensor materially
below the PRO 6000 build's FP8. A Q8_0 requant
(`llama-quantize --tensor-type per_layer_token_embd=Q8_0`, +24 GiB host RAM)
is untested.

## Engine

Base: mainline b10705, which carries [#27742](https://github.com/ggml-org/llama.cpp/pull/27742)
(`qwen4exp`) and [#28011](https://github.com/ggml-org/llama.cpp/pull/28011)
(kv-cells scan early exit). `maxspeed.patch` merges the still-open PRs:

| PR | Head | Effect |
|---|---|---|
| [#27941](https://github.com/ggml-org/llama.cpp/pull/27941) | `868e2f52` | QSA blocks keyed per sequence; `gridDim.y` overflow fix at full depth; kv-unified NaN fix. Required: fixes silent logit drift on long thinking chains |
| [#27879](https://github.com/ggml-org/llama.cpp/pull/27879) | `a7fc7e40` only | GDN QK norm as `rsqrt(sum+eps)`. Its rollback flag (`edb6dec`) is excluded: corrupts multi-sequence recurrent state ([#28019](https://github.com/ggml-org/llama.cpp/pull/28019)) |
| [#28023](https://github.com/ggml-org/llama.cpp/pull/28023) | `ead00aed` | QSA indexer head-sum by slices (prefill) |
| [#27977](https://github.com/ggml-org/llama.cpp/pull/27977) | `db40b22d` | QSA gather-window decode split, bitmap used-cells (depth decode). Draft PR |
| [#27836](https://github.com/ggml-org/llama.cpp/pull/27836) | `1d8de7c1` | MTP draft head. Compiled in, off |
| [#27861](https://github.com/ggml-org/llama.cpp/pull/27861) | `bccbacdb` | GPU LRU cache for host experts. Compiled in, off |

The image tag is keyed on base commit + patch sha256. As PRs merge, drop
their hunks; when the patch is empty, use `ghcr.io/ggml-org/llama.cpp:server-cuda`.

## Build and download

```bash
./flash-next-5090/build-engine.sh   # → llamacpp-qwen4exp:2578138397d7-p50825275
```

Fetches the pinned tarball, `git apply --check` then applies the patch,
compiles for `sm_120` only inside `nvidia/cuda:13.1.2-devel`, bakes into
`13.1.2-runtime`. ~15 minutes. `BUILD_CPUSET=48-55` pins the compile. If CUDA
13 fails on the pinned commit, drop both images to 12.8.1 (`sm_120` needs
>= 12.8).

```bash
hf download unsloth/Qwen3.8-Flash-Next-GGUF \
    --revision c8b5954a88c2775c546b92593eda40ea041d3176 \
    --include 'UD-Q4_K_XL/*' 'mmproj-BF16.gguf' \
    --local-dir /path/to/flash-next-gguf
```

## Launch

```bash
docker run -d --name qwen38-flash-next-5090 \
    --restart unless-stopped \
    --gpus '"device=0"' \
    --ulimit memlock=-1 \
    --cpuset-cpus 0-47 \
    -p 8001:8080 \
    -v /path/to/flash-next-gguf:/models:ro \
    -e LLAMA_ATTN_ROT_DISABLE=1 \
    llamacpp-qwen4exp:2578138397d7-p50825275 \
    --model /models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
    --mmproj /models/mmproj-BF16.gguf \
    --alias Qwen3.8-Flash-Next \
    --host 0.0.0.0 --port 8080 \
    -ngl 999 --n-cpu-moe 43 \
    --ctx-size 262144 --parallel 1 \
    --flash-attn on --cache-type-k f16 --cache-type-v f16 \
    --threads 48 --threads-batch 48 \
    --batch-size 2048 --ubatch-size 2048 \
    --load-mode mmap \
    --cont-batching --metrics --slots --jinja
```

`/health` in under a minute. Experts and PLE fault in from disk on demand;
the first requests are slow.

- `--cpuset-cpus 0-47 --threads 48`: physical cores, no SMT (a measured
  loss). llama.cpp's own `--cpu-mask` did not place threads; the docker
  cpuset does. Anything else streaming from DDR4 roughly halves decode.
- `--n-cpu-moe 43`: CUDA OOM at load means raise it. At 48 the budget is over
  from the dense side: lower `--ctx-size` (~0.75 GiB per 32K), `--ubatch-size`
  to 1024, or drop `--mmproj`.
- `--load-mode mmap`: mlock would pin 100 GiB from the loading thread; direct
  I/O bypasses the page cache the expert faulting needs.
- `--ubatch-size 2048`: prefill copies host experts to the GPU per batch;
  the big ubatch amortises PCIe and is most of the 6.0 GiB of buffers.
- `--parallel 1`: two slots cost each stream ~29% (24.1 + 24.2 vs 33.5
  single). `--ctx-size` is the total across slots.
- `LLAMA_ATTN_ROT_DISABLE=1`: the `qwen4exp` attention path aborts at load
  with quantised-KV rotation active. KV is f16; belt and braces.
- The checkpoint's chat template raises on `reasoning_effort: high|max`;
  pass `--chat-template-file` with a fork if clients send those.

## Speculation: off

- MTP self-draft (`--spec-type draft-mtp` on a GGUF with `blk.48` grafted)
  corrupts output on every patch composition tried: token salad or mid-answer
  degradation, acceptance falling from 63-66% to 3-11%, decode 15-24 tok/s,
  VRAM 32.1 GiB. Cause per #28019: PLE convolution and QSA indexer state have
  no rollback snapshot.
- n-gram speculation: ~0% gain on real prompts; serialises multi-slot
  decode. Synthetic filler accepts ~100% and reads 86 tok/s. Bench on real
  prompts.
- `--moe-expert-cache 64`: 41.7 tok/s warm, 1.7-5.6 cold, 31.3 GiB. The
  same VRAM in `--n-cpu-moe 43` gives 43.4 with no cold phase.

## Measured

| context | decode tok/s | prefill tok/s |
|---|---|---|
| ~500 | 42.3-43.0 | |
| ~2K | | 936 |
| 6K | 40.9 | |
| 8K | | 873 |
| 24K | 38.2 | |
| 32K | | 848 |

Measurement rules:

- Bench until consecutive runs agree. Cold NVMe is pass one; deep-context
  rows are still cold on pass two (16 tok/s at 32K on pass two, 82 on pass
  three).
- Synthetic filler can read decode 0: the model EOSes on gibberish at token
  1. Use [`probe-long.sh`](../flash-next-5090/probe-long.sh), which prefills
  varied prose and asks a question: `PORT=8001 ./flash-next-5090/probe-long.sh 24000 200`.
- Change the served alias on every output-affecting engine change. A grader
  that caches answers by model identity re-derives a broken build's score
  without generating a token.

Quality: with #27941, mid-difficulty thinking scores at parity with the
NVFP4 build. The hardest tier scores about half (19 of 55 vs 39), with
genuine wrong answers, pointing at the 4.5-bit PLE table.

## Sources

- [#27742](https://github.com/ggml-org/llama.cpp/pull/27742), [#27941](https://github.com/ggml-org/llama.cpp/pull/27941), [#27879](https://github.com/ggml-org/llama.cpp/pull/27879), [#28023](https://github.com/ggml-org/llama.cpp/pull/28023), [#27977](https://github.com/ggml-org/llama.cpp/pull/27977), [#27836](https://github.com/ggml-org/llama.cpp/pull/27836), [#27861](https://github.com/ggml-org/llama.cpp/pull/27861), [#28019](https://github.com/ggml-org/llama.cpp/pull/28019)
- [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
