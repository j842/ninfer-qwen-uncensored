# Ling-3.0-tiny (llama.cpp Vulkan, Intel Arc A750)

inclusionAI's Ling-3.0-tiny (7.9B total, 1.3B active) on an 8 GB Intel Arc
A750, stock llama.cpp Vulkan, one public GGUF. Nothing to build.

**40.5 tok/s decode, 812-1096 tok/s prefill up to 18K tokens, 73,728-token
window across 2 slots in 6.79 GiB**, measured 2026-08-20.

| | |
|---|---|
| **Engine** | `ghcr.io/ggml-org/llama.cpp:server-vulkan`, build 10460 or newer (`bailingmoe3` merged 2026-08-17, [#26608](https://github.com/ggml-org/llama.cpp/pull/26608)) |
| **Weights** | [`bloomer010/Ling-3.0-tiny-GGUF`](https://huggingface.co/bloomer010/Ling-3.0-tiny-GGUF), `Ling-3.0-tiny-UD-Q4_K_XL.gguf`, 5.34 GB |
| **Model** | [`inclusionAI/Ling-3.0-tiny`](https://huggingface.co/inclusionAI/Ling-3.0-tiny), MIT. 128 routed experts + 1 shared, 8 per token, 24 layers in a 3:1 KDA/MLA stack, 131,072 native context, no MTP block |
| **Card** | Arc A750 8 GB (Alchemist), 6.79 GiB used at `--ctx-size 73728` |
| **Host** | Docker, DRM render node, Mesa Vulkan. No SYCL |

Why this model: on a bandwidth-bound card, bytes per token matter. 1.3B
active reads a sixth of a dense 8B. Quality on an in-house 0-100 scale, all
on this card:

| | quant | quality | decode | context/slot |
|---|---|---|---|---|
| Granite 4.1 8B | Q4_K_M | 52 | 35.5 | 16K |
| Ling-3.0-tiny | Q4_K_M | 61 | 56 | 49K possible |
| Ling-3.0-tiny | UD-Q4_K_XL | 69 | 40.5 | 36K, shipped |

Q4_K_M at `--ctx-size 98304` is the throughput option.

## Quant

- K-quant, not I-quant: I-quant dequant is slow on Arc.
- UD-Q4_K_XL keeps attention, embeddings, first and last layers at higher
  bits. KDA projections are precision-sensitive. Q6_K (6.50 GB) leaves no
  compute buffer on 8 GB.
- Use bloomer010's repo: it carries the 2026-08-07 SwiGLU-clamp metadata fix
  and was re-uploaded after the llama.cpp merge. GGUFs without the clamp
  values are silently wrong.

## Engine

Check the image is at or past the merge before blaming config:

```bash
docker run --rm --entrypoint /app/llama-server \
    ghcr.io/ggml-org/llama.cpp:server-vulkan --version
# version: 0.1.2-dev (build 10499, commit 6d0549831)   need build >= 10460
```

Two version-line formats exist (semver from v0.1.2, 2026-08-18); a parser
for the old shape reads `0` from `0.1.2-dev`.

## Required settings

- `--flash-attn off`. With FA on, one KDA/MLA Vulkan shader hangs the GPU on
  prompts above ~550 tokens (`i915: GPU HANG ecode 12:1:85def5fb`, then
  `ErrorDeviceLost`). Nondeterministic; 3161 tokens never survives. Not a
  VRAM issue. Smaller batch sizes and f16 KV do not help. SYCL is stable at
  17.7 tok/s. FA off costs ~5% decode. The flag is `on|off|auto`, default
  `auto`; pass `off` explicitly.
- `--cache-type-k f16 --cache-type-v f16`. Quantised V needs flash
  attention; quantised K alone fails to load.
- `--cache-ram 0`. Slot-state save to the prompt cache
  (`server_slot::prompt_save` -> `vk::Queue::submit`) loses the device;
  KDA recurrent state cannot be serialised on Vulkan. Default is 8192 MiB
  (on). Cost: no prefix reuse across requests.
- `--parallel 2`, explicit. Default -1 auto-picked four slots on this card.
- `--reasoning auto --reasoning-budget 1024`. `--reasoning off` only sets a
  default that callers override with `enable_thinking`; `--reasoning-budget
  0` pins it off but the model then reasons in the answer channel and runs
  past 13K tokens. 1024 thinking tokens is ~18 s. -1 for a full reasoning
  worker.
- `--n-predict 8192` is the default when a caller sends no `max_tokens`, not
  a ceiling; an explicit larger `max_tokens` is not clamped. The reasoning
  budget is the runaway defence.

## Context

f16 KV costs 24.03 KiB/token; only 6 MLA layers keep a per-token cache
(latent 512 + rope 64), the 18 KDA layers hold constant-size state. Base
4.62 GiB (Q4_K_M weights + compute) + 0.48 GiB for UD-Q4_K_XL.

| `--ctx-size` | VRAM | free of 8 GiB |
|---|---|---|
| 65536 | 6.61 GiB | 1.39 |
| **73728** | **6.79 GiB** | **1.21, shipped** |
| 81920 | 6.98 GiB | 1.02 |
| 98304 | 7.36 GiB | 0.64, too tight |
| 131072 | 7.62 GiB (Q4_K_M) | `failed to fit params` |

`--ctx-size` is the total across slots: 73728 / 2 = 36,864 per slot. Read the
real figures from the log (`KV self size`, `Vulkan0 compute buffer size`) and
`drm-total-local0` in the process fdinfo.

## Launch

The A750 is PCI ID `8086:56a1`; with an Intel iGPU present, the first Intel
render node is the iGPU.

```bash
lspci -D -d 8086:56a1                      # 0000:03:00.0
ls /sys/bus/pci/devices/0000:03:00.0/drm/  # card1  renderD129
```

```bash
hf download bloomer010/Ling-3.0-tiny-GGUF \
    Ling-3.0-tiny-UD-Q4_K_XL.gguf --local-dir /path/to/models
```

```bash
docker run -d --name ling-30-tiny \
    --restart unless-stopped \
    --device /dev/dri/renderD129 \
    --group-add "$(stat -c %g /dev/dri/renderD129)" \
    -p 8082:8082 \
    -v /path/to/models:/models:ro \
    ghcr.io/ggml-org/llama.cpp:server-vulkan \
    -m /models/Ling-3.0-tiny-UD-Q4_K_XL.gguf \
    --host 0.0.0.0 --port 8082 \
    --n-gpu-layers 999 \
    --ctx-size 73728 --parallel 2 \
    --flash-attn off \
    --cache-type-k f16 --cache-type-v f16 \
    --cache-ram 0 \
    --reasoning auto --reasoning-budget 1024 \
    --n-predict 8192 \
    --jinja
```

`/health` within seconds. If it dies at load, step down one row of the
context table.

## Measurement notes

- A throughput sample taken while requests ramp onto 2 slots (6.6 tok/s
  under 16 concurrent) is not the single-stream rate (48.3 on real traffic).
- A prefill probe above ~550 tokens on a FA-on server reads "connection
  refused", not a crash. Blank metric = check the container.

## Sources

- [inclusionAI/Ling-3.0-tiny](https://huggingface.co/inclusionAI/Ling-3.0-tiny)
- [bloomer010/Ling-3.0-tiny-GGUF](https://huggingface.co/bloomer010/Ling-3.0-tiny-GGUF)
- [ggml-org/llama.cpp#26608](https://github.com/ggml-org/llama.cpp/pull/26608)
