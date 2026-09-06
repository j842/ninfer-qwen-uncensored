# Ornith-1.5-35B-A3B (NInfer, RTX 5090)

shisa-ai's 35B MoE with a distilled MTP head and the z-lab DFlash draft
model, as one [NInfer](https://github.com/Neroued/ninfer) artifact. Not
abliterated. The checkpoint matches NInfer's `qwen3_6_35b_a3b` target tensor
for tensor (1045 names, shapes, dtypes, MTP head included), so the 27B build
pattern applies.

| | |
|---|---|
| **Artifact** | `ornith_1_5_35b_a3b.ninfer`, 22,783,246,080 bytes, sha256 `bc58fa4900d99560904bb94987704e712091a8e72a1a91d07242313631a919a3` |
| **Download** | [huggingJDE/Ornith-1.5-35B-A3B-NInfer](https://huggingface.co/huggingJDE/Ornith-1.5-35B-A3B-NInfer) (every input is Apache-2.0 or MIT) |
| **Base** | [`shisa-ai/Ornith-1.5-35B-A3B-MTP`](https://huggingface.co/shisa-ai/Ornith-1.5-35B-A3B-MTP), BF16, 16 shards + `model-mtp.safetensors`, ~72 GB |
| **DFlash** | [`z-lab/Qwen3.6-35B-A3B-DFlash`](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash), ~1.7 GB, trained on base Qwen3.6-35B-A3B |
| **Frontend** | official `Qwen/Qwen3.6-35B-A3B`, pinned in [`frontend-qwen3_6_35b_a3b.sha256`](../frontend-qwen3_6_35b_a3b.sha256) |
| **Architecture** | 35B MoE, 256 experts, 8 active, 40 hybrid linear/full-attention layers, vision tower, 1-layer MTP, 262,144 context |
| **Recipe** | groupwise-int `qwen3_6_35b_a3b-v2` |
| **Engine** | NInfer `ad0f3d38` (2026-09-04), CUDA 13.1, `sm_120a` |
| **Decode** | 415–429 tok/s single stream through a router (200-token replies); the engine logs 650–675 tok/s at 69–73% MTP acceptance |

## Build

```bash
./build-ornith.sh   # → work-ornith/out/ornith_1_5_35b_a3b.ninfer
```

Five containerised steps: converter source at the pinned commit, base
weights, DFlash weights, graft the official frontend (Ornith's own
`chat_template.jinja` keeps prior-turn `<think>` blocks; the converter pins
the official one), convert with `--dflash-model`. Requirements and overrides
are the [27B page](qwen3.8-27b-uncensored.md)'s. The published sha256 was
produced by converter `b2b96bae`; the recipe id is unchanged at `ad0f3d38`
but byte-identity from the newer converter has not been checked.

## Serve

```bash
ninfer-serve ornith_1_5_35b_a3b.ninfer \
    --model-id default \
    --host 127.0.0.1 --port 6107 \
    --max-context 262144 \
    --max-concurrency 8 \
    --kv-capacity auto --kv-dtype int8 \
    --temperature 0.6 --top-p 0.95 \
    --spec mtp --draft-tokens 3 --lm-head-draft \
    --vision
```

## Rules

- Use MTP, not DFlash. NInfer pads each request's draft KV reservation by
  the draft window under MTP but not under DFlash, so a long DFlash
  generation can abort with `KV materialization exceeds active entitlement`.
- `--kv-dtype int8` is what fits full context plus the vision tower in
  32 GB. `fp8`, `nvfp4` and `k8v4` also exist now; not measured here.
- `--default-thinking-budget N` force-closes `<think>` at N tokens. Use it
  instead of a proxy-side rescue if replies burn their budget thinking.
- The API rejects `response_format` and any `chat_template_kwargs` other
  than `enable_thinking`/`preserve_thinking`. Translate in a proxy.
- The MTP head was distilled for Ornith; the DFlash head was not. Outputs
  are verified by the target either way.

## Publish

```bash
HF_TOKEN=hf_...  ./upload-ornith.sh   # → <you>/Ornith-1.5-35B-A3B-NInfer
```

Uploads the artifact, its conversion report and [`hf-ornith/`](../hf-ornith/)
(card, LICENSE, NOTICE). `HF_REPO` overrides the name; `PRIVATE=true` creates
it private.
