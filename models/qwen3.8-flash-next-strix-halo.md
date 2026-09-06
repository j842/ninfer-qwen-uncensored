# Qwen3.8-Flash-Next (llama.cpp Vulkan, AMD Strix Halo)

The same 180B model as the [RTX PRO 6000](qwen3.8-flash-next.md) and
[RTX 5090](qwen3.8-flash-next-5090.md) pages, on a 128 GB Ryzen AI MAX+ 395
mini-PC with no discrete GPU at all. Every tensor lives in the APU's unified
memory and the Radeon 8060S iGPU runs it through Vulkan, so this is the
cheapest box on which the whole model sits GPU-resident. Not NInfer, and
there is no artifact to download: you build a small engine image from a
pinned llama.cpp fork and pull one public GGUF.

**23-24 tok/s single-stream decode on short prompts, 21.6 at 23K tokens of
context; prefill 320-350 tok/s at 1-6K tokens, 243 at 23K, 187 at 59K**,
measured 2026-09-06 with speculation off. Prefill is where the 5090's
system-RAM build beats this box by 2.5x and the PRO 6000 by 30x; decode is
about half the 5090 and a tenth of the PRO 6000.

| | |
|---|---|
| **Engine** | [LaurentZuijdwijk/llama.cpp](https://github.com/LaurentZuijdwijk/llama.cpp) branch `vulkan/qwen4exp-rocmfpx` at `5e085d123` (build `b10809`), Vulkan backend, built by [`build-engine.sh`](../flash-next-strix/build-engine.sh) |
| **Weights** | [`agentionai/Qwen3.8-Flash-Next-ROCmFP4-FAST-imatrix-GGUF`](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-ROCmFP4-FAST-imatrix-GGUF), file `Qwen3.8-Flash-Next-ROCmFP4-FAST-v2-ple16.gguf`, 87.06 GiB in one file, 4.23 bpw |
| **Box** | Ryzen AI MAX+ 395, Radeon 8060S (gfx1151), 128 GB LPDDR5X; Fedora 44, kernel 7.1, Mesa 26.1 RADV on the host |
| **Memory** | 92 GiB of GTT in use at a 131,072-token context with q8_0 KV; about 30 GB left for everything else |
| **Vision** | the quant repo ships an f16 mmproj; not configured here |

## Why this fork, this quant, and Vulkan

The model is 180B total: a 125B MoE core with 6B active, a 51B n-gram (PLE)
embedding table and a 4B MTP head. On the 5090 the experts stream from
system RAM because 32 GB of VRAM cannot hold them. On Strix Halo there is no
such split to make: the iGPU addresses system memory through GTT, so the
question is only whether the whole thing fits in the carve-out the kernel
allows, and at 4.23 bpw it does with room to spare.

Three things make mainline llama.cpp unusable for this file and this box:

- **The tensor types.** ROCmFP4-FAST is a 4-bit block-FP4 format hand-ported
  from [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX). The
  `Q4_0_ROCMFP4` / `Q3_0_ROCMFPX` types exist only in that fork family. The
  quant card measures it at +2.48% perplexity over the unquantised reference
  (4.106 against 4.007 on wikitext-2), imatrix-calibrated on 1540 chunks, and
  30 GB smaller than an IQ4_XS of comparable quality.
- **The PLE layout.** The 51.2B n-gram table is a single 28.8 GiB tensor in
  the plain v2 file, which is over the 4 GiB single-buffer limit Vulkan
  imposes, so it has to be paged from disk (`--ngram-on-disk`) or held on the
  host. The `ple16` file splits the table per attention head so every piece
  is GPU-resident. It is a byte-for-byte restructuring, not a requant, and
  only this fork reads it. Any layout that leaves the table on the CPU
  collapses prefill to under 20 tok/s.
- **The backend.** ROCm/HIP wins prompt processing on other large MoEs on
  this silicon by 20-30%, but the ROCmFP4 kernels are Vulkan-only, the image
  is 730 MB instead of many GB of ROCm SDK, and the GPU passthrough is one
  `/dev/dri` render node with no `/dev/kfd` and no `HSA_OVERRIDE` games.

## What fits where

A Strix Halo APU has a small fixed VRAM carve-out (4 GiB here) and takes the
rest of what the GPU uses from GTT, which is ordinary system RAM the kernel
lets the GPU map. The default GTT limit is far too small for this model, so
the box boots with:

```
amdgpu.gttsize=126976 ttm.pages_limit=32505856 ttm.page_pool_size=32505856
```

That is 124 GiB of GTT on a 128 GB machine. Once loaded at the shipped
settings the driver reports 92 GiB of GTT in use:

```
87.1 GiB  weights, every tensor including the per-head PLE table
 1.5 GiB  KV at 131,072 tokens, q8_0 (about 12 KiB/token; f16 is double)
 3-4 GiB  compute buffers at ubatch 512
92 GiB    measured
```

The remaining ~30 GB is what the rest of the machine gets, so this coexists
with a desktop session and small services but not with a second large
model. The native 262,144-token window costs another 1.5 GiB of KV at q8_0
and fits; the default here is 131,072 because that is what was measured.
There is no swap to fall back on: if the GTT limit is below what the model
needs, the load fails rather than degrading.

## Building it

```bash
./flash-next-strix/build-engine.sh   # → llamacpp-qwen4exp-vulkan:5e085d123
```

A plain `docker build` of [`flash-next-strix/Dockerfile`](../flash-next-strix/Dockerfile):
clones the fork at the pinned commit, compiles `llama-server` against the
LunarG Vulkan SDK inside an Ubuntu 24.04 stage (the stock noble Vulkan
headers are too old for ggml's cooperative-matrix shaders), and bakes the
binary plus its versioned `.so` files into a runtime image whose Mesa comes
from the kisak PPA. That last part is not optional: noble's own RADV predates
Strix Halo and never lists the GPU at all. The fork also gates its
LDS-stride prefill path on RADV 25.3 or newer. The Vulkan driver that runs
the model is the one inside the container, so the host's Mesa version does
not matter beyond the kernel driver.

The commit is pinned, not the branch: the fork is a fast-moving research
tree. `5e085d123` is the head of `vulkan/qwen4exp-rocmfpx` on 2026-08-31 and
the commit the quant card's throughput table was measured on. A pin from
three days earlier ran 35% slower at prompt processing because it predated
the Vulkan large-k TOP_K radix selection and the QSA pooled-key cache, both
on this model's prefill path.

Download the single GGUF into a directory you will mount at `/models`:

```bash
hf download agentionai/Qwen3.8-Flash-Next-ROCmFP4-FAST-imatrix-GGUF \
    Qwen3.8-Flash-Next-ROCmFP4-FAST-v2-ple16.gguf \
    --local-dir /path/to/flash-next-strix
sha256sum /path/to/flash-next-strix/Qwen3.8-Flash-Next-ROCmFP4-FAST-v2-ple16.gguf
# 552a7a162f6a620c3aa0850d070086bc2b95094e0a4e8b860694c7f212cb59d8
```

93,484,237,760 bytes. Verify it: a partial or corrupt file loads, runs, and
produces garbage rather than failing.

## Launching it

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

`/health` answers after 30-40 seconds; the whole 87 GiB is mapped into GTT
at load, so there is no cold-page phase and the first request runs at full
speed. Notes on the flags that are not self-explanatory:

- `--threads 2 --threads-batch 2 --poll 0`: **the single biggest change to
  how the box feels.** With every layer on the GPU the CPU thread pool has
  nothing to compute, but llama-server still sizes it to the 16 physical
  cores and those threads busy-wait on the GPU between kernels: 14 cores at
  100% for the whole of every generation, all of it heat. Two threads give
  identical throughput at a twentieth of the CPU.

  | threads | CPU during decode | prefill | decode |
  |---|---|---|---|
  | 16 (default), poll 50 | 1424% | 228 tok/s | 21.2 |
  | 16, `--poll 0` | 1438% | 229 | 21.5 |
  | 4, `--poll 0` | 347% | 229 | 21.5 |
  | **2, `--poll 0`** | **148%** | 229 | 21.5 |

  Polling alone changes nothing; thread count is the lever. (Measured on the
  earlier pin, hence the lower prefill; the shipped build reads ~55% CPU.)
- `--flash-attn on --cache-type-k q8_0 --cache-type-v q8_0`: the quant
  card's run line, and what its throughput table was measured with. At 1-6K
  tokens neither changed prefill here; q8_0 halves KV memory and the card's
  numbers at 32K and 128K depend on both.
- `--batch-size 2048 --ubatch-size 512`: the fork recommends ubatch 2048 for
  MoE models (+14% prefill measured here on a 6K prompt), **but ubatch 2048
  at a context depth of 65,536 or more reproducibly times out the GPU
  compute ring** (`amdgpu: ring comp_1.2.0 timeout`, recovered by a ring
  reset). The fork README documents this and it reproduces on stock
  llama.cpp. At a 131,072 window that means 512, or 1024 which the README
  also lists as safe. Take 2048 only if you also cap `--ctx-size` at
  32,768.
- `--device Vulkan0 --n-gpu-layers 999`: the only GPU on the box. Vulkan
  needs the render node and membership of its group; no `/dev/kfd`.
- `--network host`: a single-user box serving localhost. Bind `0.0.0.0` and
  add `--api-key` if it faces a network.
- `--parallel 1`: one slot, a big single window. Not measured with more.

**Do not set `GGML_VK_DENSE_WAVE32=1`.** The fork README suggests it for
MoE models (a 32-wide retile of the quantised matmul, +5-6% prefill on the
two MoEs it was measured on). On this model it silently corrupts the
computation: every reply becomes a run of `/` characters, with thinking on
or off, and with thinking off the model reports that its own prompt
"appears to be corrupted". Bisected on the shipped build: flash attention,
q8_0 KV and the batch sizes are each clean; only this environment variable
breaks it. The broken configuration also read 485-565 tok/s of prefill,
which is how a speed-only benchmark would have shipped it.

## Speculation is off

The quant repo publishes a matching 2.28 GiB MTP drafter
(`Qwen3.8-Flash-Next-MTP-ROCmFP4-FAST-GGUF`) and the card reports up to
40 tok/s of decode with it on content that accepts well. It is not enabled
here: the numbers on this page are spec-off, and the
[5090 page](qwen3.8-flash-next-5090.md#speculation-is-off-and-must-stay-off-on-this-build)
records why MTP self-drafting on this architecture needs a coherence check
on real prompts before it is trusted. Wire it in with `--spec-type
draft-mtp --spec-draft-model /models/<drafter>.gguf` if you want to try.

## Measured performance

Idle box, natural prompts, temperature 0, prompt cache cold for every row
(see the first trap below), shipped build and flags:

| context | prefill tok/s | decode tok/s |
|---|---|---|
| 1.3K | 321-331 | 23.9 |
| 6K | 351-353 (chat endpoint), 284 (raw `/completion`) | 23.1-23.2 |
| 23K | 243 | 21.6 |
| 59K | 187 | |

The quant card, same file, same fork commit, q8_0 KV, on the same GPU with
a 96 GiB carve-out: 423 tok/s at 512, 357 at 8K, 245 at 32K, 138 at 128K
prefill; 27.8, 24.7 and 19.7 tok/s decode at 512, 32K and 131K. The prefill
rows above land on the card's curve; decode is 2-4 tok/s under it, which is
within the difference between a bench harness and a server answering chat
completions with a thinking model.

Two traps in measuring this, both of which produced a wrong number on the
way to this page:

- **A re-sent prompt is a cached prompt.** llama-server caches the prompt
  prefix, and `timings.prompt_per_second` in the response is computed over
  the tokens it actually processed. Send a benchmark prompt twice, as a
  warm-up and a measurement, and the second run reports 4 tokens in 120 ms
  as "32 tok/s prefill". Read `timings.prompt_n` next to `prompt_tokens`, or
  pass `cache_prompt: false`, before believing a prefill figure. The real
  rate was seven times higher than the number that triggered an afternoon
  of investigation.
- **A fast engine can be wrong.** The wave32 build above was the fastest
  configuration measured and produced nothing but slashes. Every engine or
  flag change needs a coherence read, not a health check.
  [`flash-next-5090/probe-long.sh`](../flash-next-5090/probe-long.sh) works
  unchanged against this server and sends with the cache off:

```bash
PORT=8094 ./flash-next-5090/probe-long.sh 24000 200
```

## Sources

- [agentionai/Qwen3.8-Flash-Next-ROCmFP4-FAST-imatrix-GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-ROCmFP4-FAST-imatrix-GGUF), the quant, its run line and throughput table
- [LaurentZuijdwijk/llama.cpp](https://github.com/LaurentZuijdwijk/llama.cpp), the fork; its README for the ubatch/compute-ring warning, the RADV 25.3 gate and the wave32 variable
- [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX), origin of the ROCmFP4 tensor types
- [ggml-org/llama.cpp#27742](https://github.com/ggml-org/llama.cpp/pull/27742), the `qwen4exp` architecture in mainline
- [ggml-org/llama.cpp#21948](https://github.com/ggml-org/llama.cpp/issues/21948), `MUL_MAT_ID` as the Vulkan MoE prefill bottleneck on gfx1151
- [kisak-mesa PPA](https://launchpad.net/~kisak/+archive/ubuntu/kisak-mesa), the RADV that knows about Strix Halo
- [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), the model card
