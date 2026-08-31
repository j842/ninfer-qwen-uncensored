# ninfer-qwen-uncensored

Reproducible builds of two models as single-file [NInfer](https://github.com/Neroued/ninfer)
artifacts, for serving on an RTX 5090:

- **[Build 1: Qwen3.8-27B Uncensored](#build-1-qwen38-27b-uncensored)** —
  abliterated Qwen3.8-27B (`./build.sh`)
- **[Build 2: Ornith-1.5-35B-A3B](#build-2-ornith-15-35b-a3b)** — shisa-ai's
  MoE model with a distilled MTP head (`./build-ornith.sh`)

NInfer is a C++/CUDA inference engine built exclusively for the 5090 (Blackwell,
`sm_120a`). It does not load Hugging Face checkpoints — it serves a `.ninfer`
artifact: one file carrying the quantised weights, the MTP speculation head, the
vision tower, and the tokenizer/chat-template frontend. Both builds run NInfer's
own converter over public weights, deterministically and from scratch. There is
no prebuilt artifact to download — the point is that **you rebuild it yourself
from pinned inputs**. One command each.

## Requirements

- **Docker** with the **NVIDIA container runtime** (`--gpus` support). Every
  heavy step runs in a pinned container; the host itself only needs `curl` and
  `python3`.
- A CUDA GPU with **~11 GB of free VRAM** for the quantise step — any card, not
  necessarily the 5090 you serve on (the reference build used a 5060 Ti). No
  GPU? `DEVICE=cpu` works: slower, but it cannot OOM and needs no NVIDIA
  runtime.
- **Disk:** ~90 GB free for build 1, ~110 GB for build 2.
- To *serve* the result you need an actual **RTX 5090** — NInfer targets
  `sm_120a` only.

---

## Build 1: Qwen3.8-27B Uncensored

```bash
./build.sh        # → work/out/qwen3_8_27b_uncensored.ninfer
```

| | |
|---|---|
| **Output** | `qwen3_8_27b_uncensored.ninfer` (~16.96 GiB / 18,210,531,328 bytes) |
| **Base model** | [`JonathanColetti/Qwen3.8-27B-Uncensored`](https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored) — abliterated BF16, 12 shards + `model-mtp.safetensors`, ~55 GB |
| **Architecture** | Qwen3.5-family multimodal: 27B, hidden 5120, 64 layers, 24 heads, vocab 248320, vision tower + 1-layer MTP |
| **Quantisation** | Groupwise-int recipe `qwen3_8_27b-v1` (Q4/Q5/Q6 body; token embedding + output head `W8G32_F16S`) |
| **Capabilities** | Text + vision, thinking mode, MTP speculation |

"Abliterated" means the base model's refusal direction has been removed from its
weights, so it does not decline requests the way the instruct model does. That
work is already done and public — this repo only re-packages those weights into
NInfer's format. See [Responsible use](#responsible-use).

### How the build works

Everything is pinned so two clean runs consume the same bytes: NInfer converter
commit `b2b96bae4dd88f95b9ea8126d68fae3b88caa374` (2026-08-18), the base repo
above, the official `Qwen/Qwen3.8-27B` frontend (sha256-pinned in
[`frontend.sha256`](frontend.sha256)), and the
`nvidia/cuda:13.1.2-runtime-ubuntu24.04` convert image. Four steps:

1. **Fetch the converter** — the NInfer source tarball at the pinned commit
   (`curl | tar` on the host). Only the `tools/convert/...` Python is used;
   nothing is compiled.
2. **Download the base weights** into `work/ckpt` (resumable, `hf_transfer`, in
   a `python:3.12-slim` container). Includes the MTP shard the converter uses to
   derive the speculation head.
3. **Graft the official frontend.** The abliterated repo ships
   trivially-different tokenizer/config copies, and the converter's preflight
   sha-pins six frontend files — so the build overwrites them with the official
   `Qwen/Qwen3.8-27B` copies, then verifies via
   [`verify_frontend.py`](verify_frontend.py). The graft runs in a container
   because rootful Docker leaves `work/ckpt` root-owned after step 2.
4. **Convert.** `tools.convert.qwen3_8_27b.convert` quantises the body, builds
   the proposal head from the MTP tensors plus the converter's ranking fixture,
   keeps the vision tower, and writes the `.ninfer` file plus a
   `*.conversion.json` report. Roughly ten minutes on a mid-range CUDA card.

**The one gotcha:** the converter stamps its git revision into the report by
shelling out to `git rev-parse HEAD`. The stock CUDA image has no `git`, so the
last step raises `FileNotFoundError: 'git'` — *after* the artifact is fully
written, making a successful build look failed. `build.sh` installs `git` in the
convert container; in a tarball checkout the stamp is simply empty, which is
fine. Keep `git` in the container if you adapt this pipeline.

Common overrides (all optional, shown with defaults):

```bash
W="$(pwd)/work"                 # working directory
DEVICE=cuda                     # or: cpu
GPU_SELECT=all                  # or: device=GPU-<uuid>  (nvidia-smi -L)
CPUSET=                         # e.g. 0-7 to pin the heavy steps
BASE_REPO=JonathanColetti/Qwen3.8-27B-Uncensored
NINFER_COMMIT=b2b96bae4dd88f95b9ea8126d68fae3b88caa374
HF_TOKEN=                       # optional; raises HF rate limits on the big pulls
```

`NINFER_REPO`, `FRONTEND_REPO`, `OUT_FILE` and `CONVERT_IMAGE` can also be
overridden; see the top of `build.sh`.

### Reproducibility

The reference build produces:

```
sha256  714565ed29db4415322e9bc13a3464dc1fd8fcc911234740a79af67934e49969
bytes   18210531328
file    qwen3_8_27b_uncensored.ninfer
```

Byte-exact output additionally depends on the quantisation device and the
PyTorch/CUDA build (low-bit rounding differs between GPU/CPU and torch
versions). To match the reference sha, quantise with `--device cuda` on the
pinned image. Functionally the artifacts are equivalent either way.

### Serving

The `.ninfer` file is self-contained — point `ninfer-serve` at it (build the
engine at the pinned commit in a `nvidia/cuda:13.1.2-devel-ubuntu24.04`
container; see [the NInfer repo](https://github.com/Neroued/ninfer)):

```bash
ninfer-serve qwen3_8_27b_uncensored.ninfer \
    --model-id default \
    --host 127.0.0.1 --port 6106 \
    --max-context 262144 \
    --max-concurrency 4 \
    --kv-capacity auto --kv-dtype int8 \
    --spec mtp --draft-tokens 3 --lm-head-draft \
    --vision
```

`--kv-dtype int8` is what lets full context and the vision tower coexist in
32 GB — bf16 KV runs out of memory at high context. NInfer exposes an
OpenAI-compatible `/v1/chat/completions`, but rejects unknown
`chat_template_kwargs` and has no constrained-JSON mode — a thin translating
proxy helps if your clients speak those dialects.

To revert to the official (censored) model, serve the official
[`neroued/Qwen3.8-27B-NInfer`](https://huggingface.co/neroued/Qwen3.8-27B-NInfer)
artifact instead — same engine, same flags, same recipe; only the base weights
differ.

---

## Build 2: Ornith-1.5-35B-A3B

MoE, MTP + DFlash. Not abliterated — it's here because the checkpoint is an
exact drop-in for NInfer's registered `qwen3_6_35b_a3b` target and the same
build pattern applies.

```bash
./build-ornith.sh   # → work-ornith/out/ornith_1_5_35b_a3b.ninfer
```

| | |
|---|---|
| **Output** | `ornith_1_5_35b_a3b.ninfer` (~21.22 GiB) |
| **Base model** | [`shisa-ai/Ornith-1.5-35B-A3B-MTP`](https://huggingface.co/shisa-ai/Ornith-1.5-35B-A3B-MTP) — BF16, 16 shards + `model-mtp.safetensors`, ~72 GB. MTP head distilled from Qwen3.6-35B-A3B |
| **DFlash companion** | [`z-lab/Qwen3.6-35B-A3B-DFlash`](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash) (~1.7 GB) |
| **Official frontend** | `Qwen/Qwen3.6-35B-A3B` — pinned in [`frontend-qwen3_6_35b_a3b.sha256`](frontend-qwen3_6_35b_a3b.sha256) |
| **Architecture** | 35B MoE (256 experts, 8 active), 40 hybrid linear/full-attention layers, vision tower, 1-layer MTP, 262144 context |
| **Recipe** | groupwise-int `qwen3_6_35b_a3b-v2` |

Why this works: the converter's preflight demands an exact tensor contract
(names, shapes, dtypes — no extras) plus its pinned config values, but never
hashes the weights themselves. Ornith's merged checkpoint matches the official
Qwen3.6-35B-A3B inventory tensor-for-tensor — all 1045 names, shapes and dtypes,
including the 19-tensor MTP head — verified against both repos' safetensors
headers before this script was written.

Differences from build 1 (same converter commit, same step pattern plus a
DFlash download step, `work-ornith/` working dir):

- **Two weight inputs.** The converter requires `--dflash-model`; the artifact
  embeds the DFlash draft model alongside the MTP head.
- **Chat template.** Ornith ships five of the six frontend files byte-identical
  to the official ones; its `chat_template.jinja` keeps prior-turn `<think>`
  blocks where the official template strips them. The converter pins the
  official template, so the artifact gets official (standard Qwen) behaviour —
  expect no practical difference.
- **Speculation heads.** The MTP head was distilled for Ornith specifically, so
  acceptance should be near the official model's (~80%, ~3.4 tokens/round at
  window 3). The DFlash head was trained against *base* Qwen3.6-35B-A3B —
  outputs are unaffected (speculation is always verified by the target model)
  but acceptance, and therefore speed, may dip on a finetune. Benchmark both.

Serving on the 5090, with MTP (supports vision):

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
DFlash are mutually exclusive; DFlash cannot combine with `--vision`). For
reference, the official `qwen3_6_35b_a3b` artifact measures ~593 tok/s
single-stream decode with MTP=3 on a 5090, ~1,314 aggregate tok/s at
concurrency 8.

---

## Responsible use

The base model of build 1 is already abliterated and already public; this repo
does not create that capability, it only converts existing public weights into
a different serving format for self-hosting. An uncensored model will follow
instructions a safety-trained one refuses — that makes **you** responsible for
what you ask it to do and for what you expose it to. Don't put an unfiltered
model in front of untrusted users or the public without your own guardrails,
and comply with the base model's and Qwen's licences and with the law where you
operate. Provided as-is, for research and self-hosting.

## Layout

```
build.sh                          Qwen3.8-27B uncensored build (four steps)
build-ornith.sh                   Ornith-1.5-35B-A3B MoE build (five steps)
verify_frontend.py                sha256 pin check for the grafted frontend resources
frontend.sha256                   the six official Qwen3.8-27B frontend pins
frontend-qwen3_6_35b_a3b.sha256   the six official Qwen3.6-35B-A3B frontend pins
README.md                         this file
```

## Credits

- [NInfer](https://github.com/Neroued/ninfer) — the 5090 engine, converter, and groupwise-int recipes.
- [Qwen](https://huggingface.co/Qwen) — the Qwen3.8-27B and Qwen3.6-35B-A3B base models and frontend resources.
- [`JonathanColetti/Qwen3.8-27B-Uncensored`](https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored) — the abliterated base weights.
- [`shisa-ai/Ornith-1.5-35B-A3B-MTP`](https://huggingface.co/shisa-ai/Ornith-1.5-35B-A3B-MTP) — Ornith with the distilled MTP head, merged into the Qwen3.6-35B-A3B checkpoint layout.
- [`z-lab/Qwen3.6-35B-A3B-DFlash`](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash) — the DFlash draft model.
