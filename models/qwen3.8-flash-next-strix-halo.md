# Qwen3.8-Flash-Next (llama.cpp Vulkan, AMD Strix Halo)

The 180B model on a 128 GB Ryzen AI MAX+ 395 mini-PC with no discrete GPU.
The whole 87 GiB ROCmFP4 quant sits in GTT and the Radeon 8060S runs it
through Vulkan, from a fork that reads the quant's tensor types and its
per-head PLE layout. Text only.

| | |
|---|---|
| **Engine** | [LaurentZuijdwijk/llama.cpp](https://github.com/LaurentZuijdwijk/llama.cpp) branch `vulkan/qwen4exp-rocmfpx` at `5e085d1` (2026-08-31), Vulkan, image `llamacpp-qwen4exp-vulkan:5e085d123` |
| **Weights** | [`agentionai/Qwen3.8-Flash-Next-ROCmFP4-FAST-imatrix-GGUF`](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-ROCmFP4-FAST-imatrix-GGUF), `Qwen3.8-Flash-Next-ROCmFP4-FAST-v2-ple16.gguf`, 93,484,237,760 bytes, 4.23 bpw, sha256 `552a7a162f6a620c3aa0850d070086bc2b95094e0a4e8b860694c7f212cb59d8` |
| **Box** | Ryzen AI MAX+ 395, Radeon 8060S (gfx1151), 128 GB LPDDR5X; Fedora 44, kernel 7.1 |
| **Vision** | the quant repo ships an f16 mmproj; not configured |
| **Memory** | 92 GiB of GTT at 131,072 context with q8_0 KV: 87.1 weights, 1.5 KV, 3–4 compute |
| **Decode** | 23.9 tok/s at 1.3K, 23.1 at 6K, 21.6 at 23K |
| **Prefill** | 321–353 tok/s at 1–6K, 243 at 23K, 187 at 59K |

Why this fork and quant: the `Q4_0_ROCMFP4` / `Q3_0_ROCMFPX` types exist only
in the ROCmFPX fork family, and the `ple16` file splits the 51B n-gram table
per head so every piece is under Vulkan's 4 GiB buffer limit and GPU
resident. A table left on the CPU drops prefill under 20 tok/s. The quant
measures +2.48% perplexity over BF16 (4.106 vs 4.007, wikitext-2). Vulkan,
not HIP: the ROCmFP4 kernels are Vulkan-only, the image is 730 MB, and the
passthrough is one `/dev/dri` render node.

## Host

The APU has a 4 GiB VRAM carve-out; everything else the GPU maps comes from
GTT, which the default limit makes far too small. Kernel boot parameters
(124 GiB of GTT on a 128 GB box), or the load fails:

```
amdgpu.gttsize=126976 ttm.pages_limit=32505856 ttm.page_pool_size=32505856
```

## Build and download

```bash
./flash-next-strix/build-engine.sh   # → llamacpp-qwen4exp-vulkan:5e085d123, 10–20 min
```

[`flash-next-strix/Dockerfile`](../flash-next-strix/Dockerfile): clone the
fork at the pin, build `llama-server` against the LunarG Vulkan SDK on
Ubuntu 24.04 (noble's headers are too old for ggml's cooperative-matrix
shaders), bake the binary and its versioned `.so` files onto a runtime with
Mesa from the kisak PPA (noble's RADV predates Strix Halo; the fork gates its
LDS-stride prefill path on RADV 25.3+). The container's Vulkan driver is the
one that runs the model.

```bash
hf download agentionai/Qwen3.8-Flash-Next-ROCmFP4-FAST-imatrix-GGUF \
    Qwen3.8-Flash-Next-ROCmFP4-FAST-v2-ple16.gguf \
    --local-dir /path/to/flash-next-strix
sha256sum /path/to/flash-next-strix/Qwen3.8-Flash-Next-ROCmFP4-FAST-v2-ple16.gguf
```

Verify it. A partial file loads and produces nonsense.

## Launch

```bash
docker run -d --name qwen38-flash-next-strix \
    --restart unless-stopped \
    --device /dev/dri/renderD128 \
    --group-add "$(getent group render | cut -d: -f3)" \
    --network host \
    -v /path/to/flash-next-strix:/models:ro \
    llamacpp-qwen4exp-vulkan:5e085d123 \
    --model /models/Qwen3.8-Flash-Next-ROCmFP4-FAST-v2-ple16.gguf \
    --alias Qwen3.8-Flash-Next \
    --host 127.0.0.1 --port 8094 \
    --device Vulkan0 --n-gpu-layers 999 \
    --ctx-size 131072 --parallel 1 \
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
    --batch-size 2048 --ubatch-size 512 \
    --threads 2 --threads-batch 2 --poll 0 \
    --jinja
```

`/health` after 30–40 seconds. The whole file is mapped at load, so the
first request runs at full speed.

## Rules

- Do not set `GGML_VK_DENSE_WAVE32=1`. The fork README suggests it for MoE;
  on this model every reply becomes a run of `/`. It also reads 485–565
  tok/s of prefill, so a speed-only benchmark would ship it.
- `--ubatch-size 512` at a 131,072 window. 2048 (+14% prefill) times out the
  GPU compute ring at 65,536 tokens of depth or more (`amdgpu: ring
  comp_1.2.0 timeout`). 512 and 1024 are safe at 131,072; take 2048 only
  with `--ctx-size` at or below 32,768.
- `--network host` and `--host 127.0.0.1`: a single-user box. Bind
  `0.0.0.0` and add `--api-key` for a network.
- `--threads 2 --poll 0`. Every layer is on the GPU; the default 16 threads
  busy-wait at 1424% CPU for the same tok/s. Two threads read 148%.
- `--flash-attn on` with q8_0 KV is the quant card's run line and what its
  throughput table was measured with.
- The native 262,144 window costs another 1.5 GiB of KV and fits.
- Pin the commit, not the branch. A pin three days older ran 35% slower at
  prompt processing (predates the Vulkan large-k TOP_K radix select and the
  QSA pooled-key cache).
- Speculation is off. The published 2.28 GiB MTP drafter is precision
  mismatched with the imatrix build; no matched drafter exists.
- Send `cache_prompt: false` when measuring, or read `timings.prompt_n`. A
  re-sent prompt reports the few uncached tokens as "32 tok/s prefill".
- Read the answer as well as the timings:
  `PORT=8094 ./flash-next-5090/probe-chat.sh doc.txt`.

## Measured

Idle box, natural prompts, temperature 0, cache cold on every row:

| context | prefill tok/s | decode tok/s |
|---|---|---|
| 1.3K | 321–331 | 23.9 |
| 6K | 351–353 | 23.1–23.2 |
| 23K | 243 | 21.6 |
| 59K | 187 | |

The quant card on the same GPU with a 96 GiB carve-out: 423 / 357 / 245 /
138 tok/s prefill at 512 / 8K / 32K / 128K; 27.8 / 24.7 / 19.7 tok/s decode
at 512 / 32K / 131K.

## Sources

- [agentionai/Qwen3.8-Flash-Next-ROCmFP4-FAST-imatrix-GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-ROCmFP4-FAST-imatrix-GGUF): the quant, its run line and throughput table
- [LaurentZuijdwijk/llama.cpp](https://github.com/LaurentZuijdwijk/llama.cpp): the fork; its README for the ubatch warning, the RADV 25.3 gate and the wave32 variable
- [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX): origin of the ROCmFP4 tensor types
- [ggml-org/llama.cpp#27742](https://github.com/ggml-org/llama.cpp/pull/27742), [#21948](https://github.com/ggml-org/llama.cpp/issues/21948)
- [kisak-mesa PPA](https://launchpad.net/~kisak/+archive/ubuntu/kisak-mesa)
