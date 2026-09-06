# Qwen3.8-Flash-Next (llama.cpp Vulkan, AMD Strix Halo)

The same 180B model on a 128 GB Ryzen AI MAX+ 395 mini-PC, no discrete GPU.
Every tensor GPU-resident in unified memory, run by the Radeon 8060S iGPU
through Vulkan, from a pinned llama.cpp fork and one public GGUF.

**23-24 tok/s decode on short prompts, 21.6 at 23K; prefill 320-350 tok/s at
1-6K tokens, 243 at 23K, 187 at 59K**, measured 2026-09-06, speculation off.

| | |
|---|---|
| **Engine** | [LaurentZuijdwijk/llama.cpp](https://github.com/LaurentZuijdwijk/llama.cpp) branch `vulkan/qwen4exp-rocmfpx` at `5e085d123` (build b10809, 2026-08-31), Vulkan only, built by [`build-engine.sh`](../flash-next-strix/build-engine.sh) into `llamacpp-qwen4exp-vulkan:5e085d123` |
| **Weights** | [`agentionai/Qwen3.8-Flash-Next-ROCmFP4-FAST-imatrix-GGUF`](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-ROCmFP4-FAST-imatrix-GGUF), `Qwen3.8-Flash-Next-ROCmFP4-FAST-v2-ple16.gguf`, 93,484,237,760 bytes (87.06 GiB), 4.23 bpw, sha256 `552a7a162f6a620c3aa0850d070086bc2b95094e0a4e8b860694c7f212cb59d8` |
| **Box** | Ryzen AI MAX+ 395, Radeon 8060S (gfx1151), 128 GB LPDDR5X, Fedora 44, kernel 7.1 |
| **Memory** | 92 GiB GTT in use at 131,072 context with q8_0 KV; ~30 GB left for the host |
| **Vision** | the quant repo ships an f16 mmproj; not configured |

Mainline llama.cpp cannot serve this file: the `Q4_0_ROCMFP4` /
`Q3_0_ROCMFPX` tensor types (from
[charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX)) and the
per-head PLE layout exist only in the fork. Quant card: +2.48% perplexity vs
unquantised (4.106 vs 4.007 on wikitext-2), imatrix on 1540 chunks.

`ple16`: the 51.2B n-gram table split per attention head so every piece is
under Vulkan's 4 GiB single-buffer limit and GPU-resident. The plain v2 file
has it as one 28.8 GiB tensor and needs `--ngram-on-disk` or host RAM; a
CPU-side table collapses prefill to under 20 tok/s.

Vulkan, not HIP: the ROCmFP4 kernels are Vulkan-only, the image is 730 MB,
and passthrough is one `/dev/dri` render node.

## Memory

The APU has a 4 GiB VRAM carve-out; the rest comes from GTT, which the kernel
caps too low by default. Boot with:

```
amdgpu.gttsize=126976 ttm.pages_limit=32505856 ttm.page_pool_size=32505856
```

124 GiB of GTT on a 128 GB box. In use at the shipped settings:

```
87.1 GiB  weights, PLE table included
 1.5 GiB  KV at 131,072 tokens, q8_0 (~12 KiB/token; f16 doubles it)
 3-4 GiB  compute buffers at ubatch 512
92 GiB    reported by the driver
```

The native 262,144 window costs another 1.5 GiB and fits. A GTT limit below
what the model needs fails the load; there is no fallback.

## Build and download

```bash
./flash-next-strix/build-engine.sh   # → llamacpp-qwen4exp-vulkan:5e085d123
```

[`Dockerfile`](../flash-next-strix/Dockerfile): clone at the pinned commit,
compile against the LunarG Vulkan SDK on Ubuntu 24.04 (stock noble headers
are too old for ggml's cooperative-matrix shaders), runtime image with Mesa
from the kisak PPA. Noble's stock RADV predates Strix Halo and does not list
the GPU; the fork also gates its LDS-stride prefill path on RADV >= 25.3. The
container's Mesa is the one that runs the model. 10-20 minutes.

The commit is pinned, not the branch. `5e085d123` carries the Vulkan large-k
TOP_K radix path and the QSA pooled-key cache; a pin three days older ran
prefill ~35% slower.

```bash
hf download agentionai/Qwen3.8-Flash-Next-ROCmFP4-FAST-imatrix-GGUF \
    Qwen3.8-Flash-Next-ROCmFP4-FAST-v2-ple16.gguf \
    --local-dir /path/to/flash-next-strix
sha256sum /path/to/flash-next-strix/Qwen3.8-Flash-Next-ROCmFP4-FAST-v2-ple16.gguf
```

Verify the sha: a partial file loads and produces garbage.

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

`/health` after 30-40 s; the whole file is mapped at load, no cold phase.

- `--threads 2 --threads-batch 2 --poll 0`: with every layer on the GPU the
  CPU pool only spin-waits. Default (16 threads) burns 14 cores at 100%
  during decode for nothing; 2 threads gives the same throughput. `--poll 0`
  alone changes nothing.

  | threads | CPU during decode | prefill | decode |
  |---|---|---|---|
  | 16, poll 50 | 1424% | 228 | 21.2 |
  | 4, poll 0 | 347% | 229 | 21.5 |
  | 2, poll 0 | 148% | 229 | 21.5 |

- `--flash-attn on`, q8_0 KV: the quant card's run line. No prefill change at
  1-6K; halves KV; required for the card's 32K and 128K numbers.
- `--ubatch-size 512`: the fork recommends 2048 for MoE (+14% prefill at 6K
  here), but ubatch 2048 at context depth >= 65,536 times out the GPU
  compute ring (`amdgpu: ring comp_1.2.0 timeout`), on stock llama.cpp too.
  512 and 1024 are safe at 131,072. Use 2048 only with `--ctx-size` <= 32,768.
- `--network host`, bind 127.0.0.1: single-user box. Bind `0.0.0.0` and add
  `--api-key` for a network.
- `--parallel 1`: not measured with more.

**Never set `GGML_VK_DENSE_WAVE32=1`.** The fork README recommends it for MoE
models. On this model it corrupts the computation: every reply is a run of
`/`, thinking on or off. Bisected on this build: flash attention, q8_0 KV and
batch sizes are each clean; only this variable breaks it. The broken
configuration reads 485-565 tok/s prefill.

Speculation: the quant repo publishes a 2.28 GiB MTP drafter
(`Qwen3.8-Flash-Next-MTP-ROCmFP4-FAST-GGUF`, card reports up to 40 tok/s).
Not enabled; not validated for coherence here. See the
[5090 page](qwen3.8-flash-next-5090.md#speculation-off) for why MTP
self-draft on this architecture needs checking on real prompts first.

## Measured

Prompt cache cold on every row, shipped build and flags:

| context | prefill tok/s | decode tok/s |
|---|---|---|
| 1.3K | 321-331 | 23.9 |
| 6K | 351-353 chat, 284 raw `/completion` | 23.1 |
| 23K | 243 | 21.6 |
| 59K | 187 | |

Quant card, same file and commit, 96 GiB carve-out: prefill 423 / 357 / 245 /
138 at 512 / 8K / 32K / 128K; decode 27.8 / 24.7 / 19.7 at 512 / 32K / 131K.

Measurement rules:

- A re-sent prompt is a cached prompt. `timings.prompt_per_second` covers
  only the tokens processed; a warm-up plus repeat reports 4 tokens in 120 ms
  as "32 tok/s". Check `timings.prompt_n` against `prompt_tokens`, or send
  `cache_prompt: false`.
- A fast engine can be wrong. Read the output after every engine or flag
  change. [`probe-long.sh`](../flash-next-5090/probe-long.sh) works here
  unchanged: `PORT=8094 ./flash-next-5090/probe-long.sh 24000 200`.

## Sources

- [agentionai/Qwen3.8-Flash-Next-ROCmFP4-FAST-imatrix-GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-ROCmFP4-FAST-imatrix-GGUF)
- [LaurentZuijdwijk/llama.cpp](https://github.com/LaurentZuijdwijk/llama.cpp) README: ubatch/compute-ring warning, RADV 25.3 gate, wave32
- [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX)
- [ggml-org/llama.cpp#21948](https://github.com/ggml-org/llama.cpp/issues/21948), `MUL_MAT_ID` as the Vulkan MoE prefill bottleneck on gfx1151
- [kisak-mesa PPA](https://launchpad.net/~kisak/+archive/ubuntu/kisak-mesa)
