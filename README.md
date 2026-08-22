# ninfer-qwen-uncensored

Reproducible build of an **abliterated ("uncensored") Qwen3.8-27B** as a
single-file [NInfer](https://github.com/Neroued/ninfer) artifact, for serving on
an RTX 5090.

NInfer is a C++/CUDA inference engine built exclusively for the 5090 (Blackwell,
`sm_120a`). It does not load Hugging Face checkpoints directly — it serves a
`.ninfer` artifact: one file carrying the quantised weights, the in-checkpoint
MTP speculation head, the vision tower, and the tokenizer/chat-template
frontend. This repo runs NInfer's own converter over public abliterated weights
to produce that artifact, deterministically and from scratch.

There is no prebuilt artifact to download here — the weights are large and the
point is that **you rebuild it yourself from pinned inputs**. One command does it.

---

## What you get

| | |
|---|---|
| **Output** | `qwen3_8_27b_uncensored.ninfer` (~16.96 GiB / 18,210,531,328 bytes) |
| **Base model** | [`JonathanColetti/Qwen3.8-27B-Uncensored`](https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored) — abliterated BF16 |
| **Architecture** | Qwen3.5-family multimodal: 27B, hidden 5120, 64 layers, 24 heads, vocab 248320, vision tower + 1-layer MTP |
| **Quantisation** | Groupwise-int (Q4/Q5/Q6 body; token embedding + output head `W8G32_F16S`) — recipe `qwen3_8_27b-v1` |
| **Capabilities** | Text + vision, thinking mode, MTP speculation with the optimised proposal head |
| **Runtime** | `ninfer-serve` on a single RTX 5090 (32 GB) |

"Abliterated" means the base model's refusal direction has been removed from its
weights, so it does not decline requests the way the instruct model does. That
work is already done and public — this repo only re-packages those weights into
NInfer's format. See [Responsible use](#responsible-use).

---

## Pinned inputs

Everything the build depends on is pinned, so two clean runs consume the same
bytes:

| Input | Pin |
|---|---|
| NInfer converter | commit `b2b96bae4dd88f95b9ea8126d68fae3b88caa374` (2026-08-18) |
| Base weights | `JonathanColetti/Qwen3.8-27B-Uncensored` (12 safetensor shards + `model-mtp.safetensors`, ~55 GB) |
| Official frontend | `Qwen/Qwen3.8-27B` — six resources, sha256-pinned in [`frontend.sha256`](frontend.sha256) |
| Convert container | `nvidia/cuda:13.1.2-runtime-ubuntu24.04` |

The converter defines the recipe *and* embeds a sha256 table for the six
frontend files. The abliterated repo ships trivially-different tokenizer/config
copies, so the build overwrites them with the official ones and verifies against
the pins before quantising — otherwise the converter's own preflight rejects the
checkpoint.

---

## Requirements

- **Docker** with the **NVIDIA container runtime** (`--gpus` support). Every
  heavy step runs in a pinned container; nothing is installed on the host.
- A CUDA GPU with **~11 GB of free VRAM** for the quantise step. This does *not*
  have to be the 5090 you serve on — the reference build quantised on a 5060 Ti.
  No GPU? Set `DEVICE=cpu` (slower, but it cannot OOM and needs no NVIDIA
  runtime).
- **~90 GB free disk**: ~55 GB base weights + the ~17 GB artifact + headroom.
- To *serve* the result you need an actual **RTX 5090** — NInfer targets
  `sm_120a` only.

---

## Quick start

```bash
git clone https://github.com/j842/ninfer-qwen-uncensored
cd ninfer-qwen-uncensored
./build.sh
```

The artifact lands at `work/out/qwen3_8_27b_uncensored.ninfer`. On a warm cache
(weights already downloaded) the quantise step is roughly ten minutes on a
mid-range CUDA card.

Common overrides (all optional, shown with defaults):

```bash
W="$(pwd)/work"                 # working directory
DEVICE=cuda                     # or: cpu
GPU_SELECT=all                  # or: device=GPU-<uuid>  (nvidia-smi -L)
CPUSET=                         # e.g. 48-55,104-111 to pin the heavy steps
BASE_REPO=JonathanColetti/Qwen3.8-27B-Uncensored
NINFER_COMMIT=b2b96bae4dd88f95b9ea8126d68fae3b88caa374
```

Example — quantise on one specific card, pinned to a CPU block:

```bash
GPU_SELECT="device=GPU-f0a12b84-81c2-d26e-9522-80e0bb8f11a8" \
CPUSET="48-55,104-111" ./build.sh
```

---

## What the build does

[`build.sh`](build.sh) runs four steps, each in a container:

1. **Fetch the converter.** Downloads the NInfer source tarball at the pinned
   commit. Only the `tools/convert/...` Python is used — nothing is compiled at
   build time.
2. **Download the base weights.** Pulls `JonathanColetti/Qwen3.8-27B-Uncensored`
   into `work/ckpt` (resumable, via `hf_transfer`). This includes the MTP shard,
   which the converter uses to derive the speculation head.
3. **Graft the official frontend.** Overwrites the six tokenizer/chat-template/
   preprocessor files with the official `Qwen/Qwen3.8-27B` copies, then runs
   [`verify_frontend.py`](verify_frontend.py) against [`frontend.sha256`](frontend.sha256).
4. **Convert.** Runs `tools.convert.qwen3_8_27b.convert`, which quantises the
   body to the groupwise-int recipe, builds the proposal head from the
   checkpoint's MTP tensors plus the converter's ranking fixture, keeps the
   vision tower, and writes the single `.ninfer` file plus a
   `*.conversion.json` report.

### The one gotcha worth knowing

The converter stamps its own git revision into the conversion report by shelling
out to `git rev-parse HEAD`. The stock CUDA runtime image has **no `git`**, so
the very last step raises `FileNotFoundError: 'git'` — *after* the artifact is
already fully written (all 1124 tensors), which makes it look like a failed build
that actually succeeded.

`build.sh` avoids this by installing `git` in the convert container. Because the
source is a tarball checkout with no `.git` directory, `git rev-parse` simply
returns nothing and the revision stamp is left empty — which is correct and
expected. If you ever adapt this pipeline, keep `git` in the container.

---

## Reproducibility

The inputs are fully pinned, and the reference build produces:

```
sha256  714565ed29db4415322e9bc13a3464dc1fd8fcc911234740a79af67934e49969
bytes   18210531328
file    qwen3_8_27b_uncensored.ninfer
```

Byte-for-byte identical output additionally depends on the quantisation
**device** and the **PyTorch/CUDA build**: GPU and CPU quantisation can differ in
low-bit rounding, and different torch versions can too. If you need to match the
reference sha exactly, quantise with `--device cuda` on a CUDA GPU using the
`nvidia/cuda:13.1.2-runtime-ubuntu24.04` image as pinned. Functionally the
artifacts are equivalent regardless; the sha is a bit-exactness check, not a
correctness one.

---

## Serving the artifact

The `.ninfer` file is self-contained — point `ninfer-serve` at it. You need the
NInfer engine built for your 5090 (see [its repo](https://github.com/Neroued/ninfer);
build the pinned commit in a `nvidia/cuda:13.1.2-devel-ubuntu24.04` container).
The reference serving flags:

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

`--kv-dtype int8` is what lets the full context and the vision tower coexist in
32 GB — bf16 KV runs the card out of memory at high context. `--spec mtp
--draft-tokens 3 --lm-head-draft` turns on the in-checkpoint MTP speculation.
NInfer exposes an OpenAI-compatible `/v1/chat/completions`. Note that it rejects
unknown `chat_template_kwargs` and has no constrained-JSON mode, so a thin
translating proxy is useful if your clients speak those dialects.

---

## Reverting to the official (censored) model

This artifact is a drop-in swap for the official
[`neroued/Qwen3.8-27B-NInfer`](https://huggingface.co/neroued/Qwen3.8-27B-NInfer)
groupwise-int artifact — same engine, same flags, same recipe. To serve the
official refusal-trained model instead, just serve that file. The only
difference is the base weights this repo feeds the converter.

---

## Responsible use

The base model is already abliterated and already public; this repo does not
create that capability, it only converts existing public weights into a
different serving format for self-hosting. An uncensored model will follow
instructions a safety-trained one refuses — that makes **you** responsible for
what you ask it to do and for what you expose it to. Don't put an unfiltered
model in front of untrusted users or the public without your own guardrails, and
comply with the base model's and Qwen's licences and with the law where you
operate. Provided as-is, for research and self-hosting.

---

## Layout

```
build.sh             end-to-end reproducible build (four containerised steps)
verify_frontend.py   sha256 pin check for the grafted frontend resources
frontend.sha256      the six official-frontend pins (mirror the converter's)
README.md            this file
```

## Credits

- [NInfer](https://github.com/Neroued/ninfer) — the 5090 engine, converter, and groupwise-int recipe.
- [Qwen](https://huggingface.co/Qwen) — the Qwen3.8-27B base model and frontend resources.
- [`JonathanColetti/Qwen3.8-27B-Uncensored`](https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored) — the abliterated base weights.
