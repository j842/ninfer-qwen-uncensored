# Ling-3.0-tiny (llama.cpp Vulkan, Intel Arc A750)

inclusionAI's Ling-3.0-tiny (7.9B total, 1.3B active) on an 8 GB Intel Arc
A750, stock llama.cpp Vulkan, one public GGUF. Nothing to build.

| | |
|---|---|
| **Engine** | `ghcr.io/ggml-org/llama.cpp:server-vulkan-b10795`, digest `sha256:f862c901a0089bc3dc326bd24f208bfcd147839783328d252e911a004c94ff3c` (2026-09-04) |
| **Weights** | [`bloomer010/Ling-3.0-tiny-GGUF`](https://huggingface.co/bloomer010/Ling-3.0-tiny-GGUF), `Ling-3.0-tiny-UD-Q4_K_XL.gguf`, 5.34 GB |
| **Model** | [`inclusionAI/Ling-3.0-tiny`](https://huggingface.co/inclusionAI/Ling-3.0-tiny), MIT. 128 routed experts + 1 shared, 8 per token, 24 layers in a 3:1 KDA/MLA stack, 131,072 native context |
| **Card** | Arc A750 8 GB (Alchemist), 6.79 GiB used at `--ctx-size 73728` |
| **Context** | 73,728 across 2 slots, 36,864 each, f16 KV |
| **Decode** | 40.5 tok/s single stream |
| **Prefill** | 812–1,217 tok/s on prompts to 18K |

The quant: a K-quant, since I-quant dequantisation is slow on Arc, from the
repo that carries the 2026-08-07 SwiGLU-clamp metadata fix and was
re-uploaded after `bailingmoe3` merged (llama.cpp #26608, 2026-08-17). GGUFs
without the clamp values are silently wrong. UD-Q4_K_XL scored 69 on our
0–100 grader against Q4_K_M's 61, for 15 tok/s of decode; Q4_K_M at
`--ctx-size 98304` is the throughput option, and Q6_K (6.50 GB) leaves no
room for the compute buffer. No MTP block, so no self-speculation.

## Launch

The A750 is PCI `8086:56a1`; on a host with an Intel iGPU the first render
node is the wrong card:

```bash
lspci -D -d 8086:56a1                      # 0000:03:00.0
ls /sys/bus/pci/devices/0000:03:00.0/drm/  # card1  renderD129
hf download bloomer010/Ling-3.0-tiny-GGUF Ling-3.0-tiny-UD-Q4_K_XL.gguf --local-dir /path/to/models
```

```bash
docker run -d --name ling-30-tiny \
    --restart unless-stopped \
    --device /dev/dri/renderD129 \
    --group-add "$(stat -c %g /dev/dri/renderD129)" \
    --network host \
    --health-cmd "curl -sf http://localhost:8082/health || exit 1" \
    --health-interval 30s --health-timeout 5s \
    --health-start-period 120s --health-retries 3 \
    -v /path/to/models:/models:ro \
    ghcr.io/ggml-org/llama.cpp@sha256:f862c901a0089bc3dc326bd24f208bfcd147839783328d252e911a004c94ff3c \
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

Loads in about two seconds.

## Rules

- `--flash-attn off`. With it on, one shader in the KDA/MLA Vulkan path
  hangs the GPU on prompts over ~550 tokens (`i915 GPU HANG ... ecode
  12:1:85def5fb`, then `ErrorDeviceLost`). Costs ~5% decode. Pass the value:
  the flag defaults to `auto`.
- `--cache-ram 0`. Saving slot state to the prompt cache loses the device
  (`vk::Queue::submit: ErrorDeviceLost` from `prompt_save`). The default is
  8192 MiB, on.
- f16 KV both ways. Quantised V needs flash attention; quantised K alone
  fails to load.
- Add `--health-cmd` on the real port. The image's baked-in check curls
  8080, so a server on any other port reads unhealthy forever.
- Keep `--ubatch-size` at or below 2048 (defaults: batch 2048, ubatch 512).
  llama.cpp [#27638](https://github.com/ggml-org/llama.cpp/issues/27638): on
  Intel ANV, KDA prompt processing degrades and the device is lost above it.
- Pass `--parallel` even for 1. The default (-1) picked four slots.
- `--reasoning auto --reasoning-budget 1024`, not `--reasoning off`. Denied a
  thinking channel, this hybrid model reasons in the answer channel and runs
  past 13,000 tokens. Off alone is also overridden by any caller sending
  `enable_thinking`.
- `--n-predict` is a default, not a ceiling: an explicit `max_tokens` above
  it is honoured. The reasoning budget is the defence against runaways.
- Context budget at f16, 24 KiB/token: 65536 → 6.61 GiB, 73728 → 6.79,
  81920 → 6.98, 98304 → 7.36 (too tight). Read `KV self size` and
  `Vulkan0 compute buffer size` from the log rather than estimating.
- Pin by digest. Dated `server-vulkan-bNNNNN` tags exist; the bare tag moves.

## Sources

- [inclusionAI/Ling-3.0-tiny](https://huggingface.co/inclusionAI/Ling-3.0-tiny), [bloomer010/Ling-3.0-tiny-GGUF](https://huggingface.co/bloomer010/Ling-3.0-tiny-GGUF)
- [ggml-org/llama.cpp#26608](https://github.com/ggml-org/llama.cpp/pull/26608), [#27638](https://github.com/ggml-org/llama.cpp/issues/27638)
