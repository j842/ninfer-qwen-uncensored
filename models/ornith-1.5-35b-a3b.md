# Ornith-1.5-35B-A3B (NInfer, RTX 5090)

shisa-ai's 35B MoE with a distilled MTP head and the z-lab DFlash draft
model, as a single-file [NInfer](https://github.com/Neroued/ninfer) artifact.
Not abliterated. The checkpoint is a tensor-for-tensor drop-in for NInfer's
`qwen3_6_35b_a3b` target (all 1045 names, shapes and dtypes, including the
19-tensor MTP head), so the 27B build pattern applies.

Prebuilt and published, every input Apache-2.0 or MIT:
[huggingJDE/Ornith-1.5-35B-A3B-NInfer](https://huggingface.co/huggingJDE/Ornith-1.5-35B-A3B-NInfer).

```
sha256  bc58fa4900d99560904bb94987704e712091a8e72a1a91d07242313631a919a3
bytes   22783246080
file    ornith_1_5_35b_a3b.ninfer
```

Or build it:

```bash
./build-ornith.sh   # → work-ornith/out/ornith_1_5_35b_a3b.ninfer
```

| | |
|---|---|
| **Output** | `ornith_1_5_35b_a3b.ninfer`, 21.22 GiB |
| **Base** | [`shisa-ai/Ornith-1.5-35B-A3B-MTP`](https://huggingface.co/shisa-ai/Ornith-1.5-35B-A3B-MTP), BF16, 16 shards + `model-mtp.safetensors`, ~72 GB. MTP head distilled from Qwen3.6-35B-A3B |
| **DFlash** | [`z-lab/Qwen3.6-35B-A3B-DFlash`](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash), ~1.7 GB, trained against base Qwen3.6-35B-A3B |
| **Frontend** | official `Qwen/Qwen3.6-35B-A3B`, pinned in [`frontend-qwen3_6_35b_a3b.sha256`](../frontend-qwen3_6_35b_a3b.sha256) |
| **Architecture** | 35B MoE, 256 experts, 8 active, 40 hybrid linear/full-attention layers, vision tower, 1-layer MTP, 262144 context |
| **Recipe** | groupwise-int `qwen3_6_35b_a3b-v2` |
| **Requirements** | [README](../README.md#building-the-ninfer-artifacts): Docker + NVIDIA runtime, ~11 GB VRAM, ~110 GB disk |

## Build

Same converter commit and step pattern as the
[27B build](qwen3.8-27b-uncensored.md#build) (including the `git`-in-container
requirement and the overrides), plus a DFlash download, in `work-ornith/`.
Differences:

- The converter requires `--dflash-model`; the artifact embeds the DFlash
  head alongside MTP.
- Ornith's `chat_template.jinja` keeps prior-turn `<think>` blocks. The
  converter pins the official template, so the artifact has standard Qwen
  behaviour.
- The MTP head was distilled for Ornith (acceptance ~80%, ~3.4 tokens/round
  at window 3). The DFlash head was trained on base Qwen3.6, so its
  acceptance may be lower on this finetune. Outputs are unaffected either way;
  speculation is verified by the target.

## Serve

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

Use MTP, not DFlash. `--spec dflash --draft-tokens 7` is faster on paper
(text-only; DFlash excludes `--vision`) but an engine bug aborts long
generations with `KV materialization exceeds active entitlement`: NInfer pads
each request's draft KV reservation by the draft window under MTP but not
under DFlash (`request_plan_impl.h`, `plan_request`), so a DFlash round near
the end of the output budget can cross a 64-token KV page past its
reservation. Larger `--draft-tokens` makes it likelier; `--kv-capacity` and
`--kv-dtype` do not help.

Measured on the official `qwen3_6_35b_a3b` artifact on a 5090: ~593 tok/s
single-stream with MTP=3, ~1,314 tok/s aggregate at concurrency 8, ~271 tok/s
with speculation off. Prefill ~13,700 tok/s into the 262,144-token window.

## Publish

[`upload-ornith.sh`](../upload-ornith.sh) uploads the artifact, its conversion
report, and the card / LICENSE / NOTICE from [`hf-ornith/`](../hf-ornith/)
(which keep the upstream shisa-ai notices). Runs in a container because
`work-ornith/out` is root-owned.

```bash
HF_TOKEN=hf_...  ./upload-ornith.sh   # → <you>/Ornith-1.5-35B-A3B-NInfer
```

`HF_REPO` overrides the name; `PRIVATE=true` creates it private.
