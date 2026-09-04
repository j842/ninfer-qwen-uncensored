# Ling-3.0-tiny (llama.cpp Vulkan, Intel Arc A750)

inclusionAI's Ling-3.0-tiny on an 8 GB Intel Arc A750 (Alchemist, Xe1), served
by stock llama.cpp Vulkan off a public GGUF. 7.9B total parameters, 1.3B active
per token. Nothing to build and no artifact to download: `bailingmoe3` support
went upstream on 2026-08-17, so this is the rolling `server-vulkan` image plus
one file. The cheap end of what we run, and the page exists for two
Vulkan-on-Arc crashes that both look like broken hardware.

**40.5 tok/s single-stream decode, 812-1096 tok/s prefill on prompts up to
18,029 tokens, a 73,728-token window across 2 slots in 6.79 GiB of 8**,
measured 2026-08-20.

| | |
|---|---|
| **Engine** | stock `ghcr.io/ggml-org/llama.cpp:server-vulkan`, build 10460 or newer |
| **Weights** | [`bloomer010/Ling-3.0-tiny-GGUF`](https://huggingface.co/bloomer010/Ling-3.0-tiny-GGUF), `Ling-3.0-tiny-UD-Q4_K_XL.gguf`, 5.34 GB |
| **Card** | one Intel Arc A750 8 GB, 6.79 GiB used at `--ctx-size 73728` |
| **Model** | [`inclusionAI/Ling-3.0-tiny`](https://huggingface.co/inclusionAI/Ling-3.0-tiny), MIT. 128 routed experts + 1 shared, 8 routed per token, 24 layers in a 3:1 KDA/MLA stack, 131,072 native context |
| **Host** | Docker, a DRM render node, Mesa's Vulkan driver. No SYCL, no oneAPI |

For scale, inclusionAI measure the FP8 weights at 100-105 tok/s on a DGX Spark
and 86-90 tok/s on an M4 Pro MacBook. This is a four-bit quant on a 2022 gaming
card that cost less than either.

## Why this model on this card

Decode on an A750 is memory-bandwidth-bound, so bytes read per token is the
number that matters, not parameter count. Ling-3.0-tiny reads 1.3B of its 7.9B
per token; the dense Granite 4.1 8B it replaced read all of its 8B, so roughly
a sixth of the traffic. Temper the arithmetic: MoE expert-gather on Vulkan is
less efficient than a dense GEMM, so expect a large multiple, not exactly 6x.

The real reason for the swap was quality. Artificial Analysis rate
Ling-3.0-tiny 25 on Intelligence Index v4.1.1 against Granite 4.1 8B's 12, and
our own graded benchmark agreed. All three of these were measured on this same
card, quality on an in-house 0-100 scale:

| | quant | quality | decode | context/slot |
|---|---|---|---|---|
| Granite 4.1 8B | Q4_K_M | 52 | 35.5 tok/s | 16K |
| Ling-3.0-tiny | Q4_K_M | 61 | 56 tok/s | 49K possible |
| Ling-3.0-tiny | **UD-Q4_K_XL** | **69** | **40.5 tok/s** | **36K, shipped** |

69 puts an 8 GB card above a 24 GB Arc Pro B60 running a 26B MoE (63) on the
same scale; the best worker we run, the RTX PRO 6000
[Flash-Next](qwen3.8-flash-next.md) box, scores 96. The dynamic quant costs
about 15 tok/s of decode and buys 8 points of quality. That trade was taken
deliberately, because this worker's job is to be the good cheap one rather than
the fast one. Q4_K_M at `--ctx-size 98304` is the alternative if throughput is
what you want instead.

## The quant

A K-quant, not an I-quant: I-quant dequantisation is slow on Arc, which is the
same rule the other cards in this class follow. UD-Q4_K_XL keeps the
precision-sensitive tensors (attention, embeddings, first and last layers) at
higher bits and compresses the rest harder, so it is both smaller than Q5_K_M
(5.64 GB) and more faithful than the plain Q4_K_M (4.82 GB). That matters
doubly here, because Ling's KDA projections are precision-sensitive, which is
why the published quants keep them high. Q6_K (6.50 GB) leaves nothing for the
compute buffer on this card.

The `UD-` is bloomer010's label rather than an unsloth-built file, so the recipe
was unverified going in. It paid off: 61 to 69 for +0.52 GB, which comes out of
the context budget rather than out of spare headroom.

Use that repo specifically. It carries the 2026-08-07 metadata fix that added
Ling 3.0's trained per-layer SwiGLU clamp values, and it was re-uploaded on
2026-08-17 after the llama.cpp merge, so it matches what the stock image
expects. GGUFs built before the clamp fix are silently wrong, and several of the
Ling-3.0-tiny repos that appeared on 2026-08-11 either predate it or omit it.

The file has no bundled MTP block (`num_nextn_predict_layers = 0`), so there is
no self-speculation available and no exposure to the "MTP can't be disabled"
crash the pull request describes.

## The engine

`bailingmoe3` support ([PR #26608](https://github.com/ggml-org/llama.cpp/pull/26608))
merged on 2026-08-17 07:49 UTC as `373336672029`, carrying the Q-LoRA attention
path this model requires (`q_lora_rank = 256`), the trained-SwiGLU-clamp
metadata fix, and the Bailing V3 multi-argument tool-call delimiter fix. So the
stock rolling tag works and there is no source build.

Mind the tag lag if you hit an unknown-architecture failure at load: the tag has
to have been rebuilt since the merge, and your host has to have re-pulled it.
The merge is build 10460, bracketed by release b10456 (one commit behind it) and
b10470 (ten ahead). Check the image before blaming the config:

```bash
docker run --rm --entrypoint /app/llama-server \
    ghcr.io/ggml-org/llama.cpp:server-vulkan --version
```

Two version-line formats are in the wild, since llama.cpp moved to semver around
v0.1.2 on 2026-08-18, and anything that parses only the older shape reads the
`0` out of `0.1.2-dev`:

```
version: 0.1.2-dev (build 10499, commit 6d0549831)
version: 10335 (74ce15741)
```

## Flash attention hangs the GPU

Reproducible, and it presents as a hardware fault:

```
i915 0000:03:00.0: [drm] GPU HANG: ecode 12:1:85def5fb, in llama-server
i915 0000:03:00.0: [drm] llama-server[...] context reset due to GPU hang
-> vk::Device::waitForFences: ErrorDeviceLost -> the server dies
```

Same ecode every time. One shader in the new KDA/MLA Vulkan path wedges the
card. It fires on prompts somewhere above ~550 tokens, nondeterministically:
1072 tokens sometimes survived, 3161 never did. So it is not a VRAM problem and
not fixable by trimming context. What was ruled out by measurement on
2026-08-19, all on Q4_K_M:

| | decode | prefill at 3161 | prompts over 1K |
|---|---|---|---|
| Vulkan, FA on | 59 tok/s | crashes | GPU hang |
| Vulkan, FA off | 56 tok/s | 1248 tok/s | stable to 18,029 tokens |
| SYCL (`server-intel`), FA on | 17.7 tok/s | 316 tok/s | stable |

`--batch-size 512 --ubatch-size 256`, the sizes the pull request thread
suggested, was worse and died at 550. f16 KV with flash attention still on died
at 3161 as usual. SYCL is stable and gives up two thirds of decode, which
matches the Granite A/B on the same card, so Vulkan is still the right backend
here; it is flash attention that has to go.

`--flash-attn off` costs about 5% of decode and improves prefill. Note that the
flag takes `on|off|auto` and defaults to `auto`, so anything that gates on the
string `on` and otherwise passes nothing leaves flash attention enabled. That is
the shape of bug that makes an A/B silently compare on against auto.

The consequence lands on the KV cache. Without flash attention the V cache
cannot be quantised (`quantized V cache requires flash_attn`), and quantising
only K fails to load, so both are f16 and the cache costs about twice what it
otherwise would. The context budget below is built around that.

## The prompt cache kills the device

A second crash, a different code path, also not a tuning choice:

```
ggml_vulkan: device lost on Vulkan0
terminate called after throwing an instance of 'vk::DeviceLostError'
  what():  vk::Queue::submit: ErrorDeviceLost
... server_slot::prompt_save -> llama_state_seq_get_data_ext -> vk::Queue::submit
```

The server dies whenever llama.cpp saves slot state to the prompt cache.
Serialising KDA's recurrent state is evidently not something the Vulkan backend
can do. It crashed twice inside the first handful of requests. `--cache-ram 0`
is the fix and it has to be set explicitly, because `-cram` defaults to 8192 MiB
(enabled). The cost is prefix reuse across requests.

This was measured against the fork commit that later merged, so assume it is
still there until a retest on a current image says otherwise. It is worth
reporting upstream as well: the Vulkan reports on the pull request are all AMD
gfx1151, and this is a precise non-AMD signature. Those AMD reports of freezes
and 30+ minute load timeouts did not reproduce here, for what it is worth. This
model loads in about two seconds.

## Context, budgeted against measured VRAM

The KV cache is cheap on this architecture, which is what makes a real window
possible on 8 GB. Of 24 layers only six are MLA layers that keep a per-token
cache at all, and each of those stores a compressed latent
(`kv_lora_rank = 512`) plus the rope part (`qk_rope_head_dim = 64`) rather than
16 heads by 128 dims of real K and V. The other 18 are KDA: linear attention
with a constant-size recurrent state that does not grow with context.

Cheap is not free. Measured on this card at f16, 24.03 KiB per token across the
whole cache, on a fixed base of 4.62 GiB of weights and compute buffer for the
4.82 GB Q4_K_M. UD-Q4_K_XL adds about 0.48 GiB of weights on top of that.

| `--ctx-size` | VRAM of 8 GB | free |
|---|---|---|
| 65536 | 6.61 GiB | 1.39 |
| **73728** | **6.79 GiB** | **1.21, shipped** |
| 81920 | 6.98 GiB | 1.02 |
| 98304 | 7.36 GiB | 0.64, too tight |

Do not chase the GGUF's declared 131072. Q4_K_M at that size measured 7.62 GiB
of 8 and logged `failed to fit params to free device memory`; it loads, and
there is nothing left for a bad day. (The model's real window is 131,072 native,
and inclusionAI's own SGLang recipe reaches 262,144 with YaRN at factor 2.
Neither is reachable on this card with f16 KV.)

`--ctx-size` is the total across slots, so 73728 with `--parallel 2` is 36,864
per slot, which is the usable window a client reads from `/props`. Still 2.3x
what the dense 8B advertised. Two slots rather than one because a single slot
would advertise more context and halve concurrency.

Pass `--parallel` even when the value is 1. The llama.cpp default is -1, meaning
auto, and here auto picked four slots, so a config that only passes the flag when
it is greater than one quietly asks an 8 GB card for four slots at whatever
context you set. Harmless on a roomy card, not on this one.

## Thinking needs a budget, not a switch

Ling-3.0-tiny is a hybrid reasoning model and ships with thinking on. On a small
card that is a trap, and it took two wrong answers to find the right one. All
measured 2026-08-19.

First attempt, `--reasoning off`. That only sets the server default. The routing
layer in front of this worker probes thinking support, saw that it worked,
recorded the dialect, and thereafter sent `{"enable_thinking": true}` on any
prompt it scored as hard, overriding the default. Both slots sat 7,000 to 10,000
tokens deep in a single answer and the cold-start profile never finished.

Second attempt, pin it off properly: `--reasoning off` together with
`--reasoning-budget 0`. That fixed detection, since nothing injects thinking at a
model that reports it cannot think, but it did not fix the length. Answers still
ran past 13,000 tokens. Deny a hybrid reasoning model its thinking channel and on
a hard question it will reason in the answer channel instead, and loop there. Its
published quality is measured with thinking on; this is the model working as
designed.

What works is a budget. `--reasoning auto --reasoning-budget 1024` leaves the
chat template and the caller to decide whether to think and caps how long: 1024
thinking tokens is about 18 seconds here. Set the budget to -1 for a full
reasoning worker, or 0 to pin it off, having read the paragraph above.

`--n-predict 8192` sets the default length when a caller sends no `max_tokens`,
and despite the flag name it is not a ceiling. On build 10499 a request with an
explicit `"max_tokens": 20000` was not clamped down to it and produced no
"exceeds server configuration" warning. The reasoning budget is the real defence
against a runaway generation.

## Running it

Pick the right render node first. If the host has an integrated Intel GPU as
well, the first Intel card by PCI order is the iGPU, not the A750, and a
container that binds it starts happily and is very slow. The A750 is PCI ID
`8086:56a1`:

```bash
lspci -D -d 8086:56a1                      # 0000:03:00.0 VGA ... DG2 [Arc A750]
ls /sys/bus/pci/devices/0000:03:00.0/drm/  # card1  renderD129
```

Fetch the weights, then start the server:

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

`--n-gpu-layers 999` puts every layer on the card; 5.34 GB of weights fits with
room for the cache. `/health` answers within a few seconds of the container
starting. If it dies at load, step down one row of the context table before
touching anything else, and read the real figures out of the log rather than
estimating them: the `KV self size` and `Vulkan0 compute buffer size` lines at
startup, and `drm-total-local0` in the process fdinfo for what the card actually
holds.

## Two ways the numbers mislead

- **A throughput sample taken during a concurrency ramp is not the worker's
  speed.** Our monitoring records about 6.6 tok/s here, because the closing speed
  sample of its profile lands while 16 requests are being pushed at 2 slots. The
  single-stream rate observed on real traffic is 48.3. Nothing is wrong with the
  card; the sample is of a saturated per-stream rate, and anything that ranks or
  alerts on it will be wrong about this worker.
- **A crashing server reads as a missing metric rather than as a crash.**
  Prefill went unmeasured on this worker for its first five days. The probe used
  a 1024-token prompt, which was tripping the flash attention hang, so all three
  attempts logged "connection refused" and the metric stayed blank. Fixing flash
  attention fixed the metric as a side effect: 1217 tok/s, against the dense 8B's
  643.

One more thing that reads as a fault and is not. With an idle fleet and a front
end that ranks by expected completion time, this worker gets nothing: 8 of 8
trivial prompts went to a faster card, which is the correct answer, since that
card is faster end to end (58.9 tok/s decode and 31,664 tok/s prefill against
48.3 and 1,217 here). Under load it spills as intended. Of 14 concurrent
requests this worker took 2, matching its 2 slots.

## Sources

- [inclusionAI/Ling-3.0-tiny](https://huggingface.co/inclusionAI/Ling-3.0-tiny), the model card
- [bloomer010/Ling-3.0-tiny-GGUF](https://huggingface.co/bloomer010/Ling-3.0-tiny-GGUF), the quant
- [ggml-org/llama.cpp#26608](https://github.com/ggml-org/llama.cpp/pull/26608), `bailingmoe3` support, merged 2026-08-17
