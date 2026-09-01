# Ornith-1.5-35B-A3B (NInfer)

shisa-ai's 35B MoE with a distilled MTP head and the z-lab DFlash draft model,
converted into a single-file [NInfer](https://github.com/Neroued/ninfer)
artifact for the RTX 5090. Not abliterated. It is in this repo because the
checkpoint is an exact drop-in for NInfer's registered `qwen3_6_35b_a3b`
target, so the same build pattern applies.

**Prebuilt.** This exact artifact is published at
[huggingJDE/Ornith-1.5-35B-A3B-NInfer](https://huggingface.co/huggingJDE/Ornith-1.5-35B-A3B-NInfer).
Download it instead of building, then verify:

```
sha256  bc58fa4900d99560904bb94987704e712091a8e72a1a91d07242313631a919a3
bytes   22783246080
file    ornith_1_5_35b_a3b.ninfer
```

Or build it yourself:

```bash
./build-ornith.sh   # → work-ornith/out/ornith_1_5_35b_a3b.ninfer
```

| | |
|---|---|
| **Output** | `ornith_1_5_35b_a3b.ninfer` (~21.22 GiB) |
| **Base model** | [`shisa-ai/Ornith-1.5-35B-A3B-MTP`](https://huggingface.co/shisa-ai/Ornith-1.5-35B-A3B-MTP), BF16, 16 shards + `model-mtp.safetensors`, ~72 GB. MTP head distilled from Qwen3.6-35B-A3B |
| **DFlash companion** | [`z-lab/Qwen3.6-35B-A3B-DFlash`](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash) (~1.7 GB) |
| **Official frontend** | `Qwen/Qwen3.6-35B-A3B`, pinned in [`frontend-qwen3_6_35b_a3b.sha256`](../frontend-qwen3_6_35b_a3b.sha256) |
| **Architecture** | 35B MoE (256 experts, 8 active), 40 hybrid linear/full-attention layers, vision tower, 1-layer MTP, 262144 context |
| **Recipe** | groupwise-int `qwen3_6_35b_a3b-v2` |

Build requirements are in the [README](../README.md#building-the-ninfer-artifacts):
Docker with the NVIDIA runtime, ~11 GB of free VRAM, ~110 GB of disk.

## Why an Ornith checkpoint converts on a Qwen target

The converter's preflight demands an exact tensor contract (names, shapes,
dtypes, no extras) plus its pinned config values, but it never hashes the
weights themselves. Ornith's merged checkpoint matches the official
Qwen3.6-35B-A3B inventory tensor for tensor: all 1045 names, shapes and dtypes,
including the 19-tensor MTP head, verified against both repos' safetensors
headers before this script was written.

## How the build differs from the 27B build

Same converter commit, same step pattern plus a DFlash download step, and a
`work-ornith/` working directory. Three real differences:

- **Two weight inputs.** The converter requires `--dflash-model`; the artifact
  embeds the DFlash draft model alongside the MTP head.
- **Chat template.** Ornith ships five of the six frontend files byte-identical
  to the official ones. Its `chat_template.jinja` keeps prior-turn `<think>`
  blocks where the official template strips them. The converter pins the
  official template, so the artifact gets official (standard Qwen) behaviour.
  Expect no practical difference.
- **Two speculation heads.** The MTP head was distilled for Ornith
  specifically, so acceptance should be near the official model's (~80%, ~3.4
  tokens/round at window 3). The DFlash head was trained against *base*
  Qwen3.6-35B-A3B, so outputs are unaffected (speculation is always verified by
  the target model) but acceptance, and therefore speed, may dip on a finetune.
  Benchmark both.

The full four-step mechanics, the missing-`git` gotcha and the environment
overrides are the same as the
[Qwen3.8-27B build](qwen3.8-27b-uncensored.md#how-the-build-works); read that
page first if you are adapting the pipeline.

## Serving

On the 5090, with MTP (supports vision):

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

For text-only serving, try `--spec dflash --draft-tokens 7` instead. MTP and
DFlash are mutually exclusive, and DFlash cannot combine with `--vision`.

For reference, the official `qwen3_6_35b_a3b` artifact measures ~593 tok/s
single-stream decode with MTP=3 on a 5090, and ~1,314 aggregate tok/s at
concurrency 8.

## Publishing to Hugging Face

The reference artifact is live at
[huggingJDE/Ornith-1.5-35B-A3B-NInfer](https://huggingface.co/huggingJDE/Ornith-1.5-35B-A3B-NInfer).
Every input is Apache-2.0 or MIT, so redistributing the artifact is fine with
attribution. [`upload-ornith.sh`](../upload-ornith.sh) uploads the artifact,
its conversion report, and the model card / LICENSE / NOTICE from
[`hf-ornith/`](../hf-ornith/), which preserve the upstream shisa-ai notices:

```bash
HF_TOKEN=hf_...  ./upload-ornith.sh   # → <you>/Ornith-1.5-35B-A3B-NInfer
```

Override `HF_REPO` for a different repo name, `PRIVATE=true` to create it
private. Like the build, it runs in a container, because rootful Docker leaves
`work-ornith/out` root-owned.
