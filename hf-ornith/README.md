---
license: apache-2.0
base_model: shisa-ai/Ornith-1.5-35B-A3B-MTP
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - ninfer
  - moe
  - mtp
  - dflash
  - speculative-decoding
  - quantized
---

# Ornith-1.5-35B-A3B-NInfer

[Ornith-1.5-35B-A3B](https://huggingface.co/shisa-ai/Ornith-1.5-35B-A3B-MTP) —
shisa-ai's 35B MoE model with an MTP speculation head distilled from
Qwen3.6-35B-A3B — as a single-file [NInfer](https://github.com/Neroued/ninfer)
artifact: groupwise-int quantised, DFlash draft model embedded, official Qwen
frontend included, ready to serve on an RTX 5090.

| | |
|---|---|
| **File** | `ornith_1_5_35b_a3b.ninfer` (~21.22 GiB) |
| **Architecture** | 35B MoE (256 experts, 8 active per token), 40 hybrid linear/full-attention layers, vision tower, 1-layer MTP, 262144 context |
| **Recipe** | groupwise-int `qwen3_6_35b_a3b-v2` |
| **Converter** | NInfer commit `b2b96bae4dd88f95b9ea8126d68fae3b88caa374` (2026-08-18) |
| **Runtime** | `ninfer-serve` on a single RTX 5090 (32 GB) — NInfer targets `sm_120a` only |

## What was changed from the base model

- BF16 weights quantised to NInfer's groupwise-int `qwen3_6_35b_a3b-v2` recipe
  and packed into one `.ninfer` file together with the distilled MTP head, the
  [z-lab DFlash](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash) draft
  model, the vision tower, and the tokenizer/chat-template frontend.
- The frontend resources are the official, sha256-pinned
  [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) copies.
  Ornith's own `chat_template.jinja` (which keeps prior-turn `<think>` blocks)
  is replaced by the official template, which strips them — the standard Qwen
  convention.

The build is fully reproducible from pinned public inputs:
[j842/ninfer-qwen-uncensored](https://github.com/j842/ninfer-qwen-uncensored)
(`./build-ornith.sh`). The converter's `*.conversion.json` report is included
in this repo.

## Serving

With MTP speculation (supports vision):

```bash
ninfer-serve ornith_1_5_35b_a3b.ninfer \
    --model-id default \
    --host 127.0.0.1 --port 6107 \
    --max-context 262144 \
    --max-concurrency 4 \
    --kv-capacity auto --kv-dtype int8 \
    --spec mtp --draft-tokens 3 --lm-head-draft \
    --vision
```

For text-only serving, try `--spec dflash --draft-tokens 7` instead (MTP and
DFlash are mutually exclusive; DFlash cannot combine with `--vision`).
`--kv-dtype int8` is what lets full context and the vision tower coexist in
32 GB. NInfer exposes an OpenAI-compatible `/v1/chat/completions`.

**Speculation caveats:** the MTP head was distilled for Ornith specifically, so
acceptance should be near the official Qwen3.6-35B-A3B model's (~80%, ~3.4
tokens/round at window 3). The DFlash head was trained against *base*
Qwen3.6-35B-A3B — outputs are unaffected (speculation is always verified by the
target model) but acceptance, and therefore speed, may dip on a finetune.
Benchmark both on your workload.

## Licence

Apache-2.0 for this distribution — see [LICENSE](LICENSE) and
[NOTICE](NOTICE). The artifact combines:

- **Ornith target weights** — MIT, via
  [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B),
  as merged in shisa-ai's Apache-2.0 distribution
- **Distilled MTP head** — Apache-2.0, derived from Qwen3.6-35B-A3B's MTP head
  by [shisa-ai](https://huggingface.co/shisa-ai/Ornith-1.5-35B-A3B-MTP)
- **DFlash draft model** — Apache-2.0,
  [z-lab/Qwen3.6-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash)
- **Frontend resources** — Apache-2.0,
  [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- **Converter and recipe** — Apache-2.0,
  [NInfer](https://github.com/Neroued/ninfer)

This is not an official Qwen, shisa-ai, ornith-ai, or NInfer release.
