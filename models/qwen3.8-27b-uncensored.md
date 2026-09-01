# Qwen3.8-27B Uncensored (NInfer)

An abliterated Qwen3.8-27B, converted into a single-file
[NInfer](https://github.com/Neroued/ninfer) artifact for the RTX 5090.

```bash
./build.sh        # → work/out/qwen3_8_27b_uncensored.ninfer
```

| | |
|---|---|
| **Output** | `qwen3_8_27b_uncensored.ninfer` (~16.96 GiB / 18,210,531,328 bytes) |
| **Base model** | [`JonathanColetti/Qwen3.8-27B-Uncensored`](https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored), abliterated BF16, 12 shards + `model-mtp.safetensors`, ~55 GB |
| **Architecture** | Qwen3.5-family multimodal: 27B, hidden 5120, 64 layers, 24 heads, vocab 248320, vision tower + 1-layer MTP |
| **Quantisation** | Groupwise-int recipe `qwen3_8_27b-v1` (Q4/Q5/Q6 body; token embedding and output head `W8G32_F16S`) |
| **Capabilities** | Text + vision, thinking mode, MTP speculation |

Build requirements are in the [README](../README.md#building-the-ninfer-artifacts):
Docker with the NVIDIA runtime, ~11 GB of free VRAM, ~90 GB of disk.

"Abliterated" means the base model's refusal direction has been removed from
its weights, so it does not decline requests the way the instruct model does.
That work is already done and public. This repo only re-packages those weights
into NInfer's format. See [Responsible use](../README.md#responsible-use).

There is no prebuilt artifact for this model, deliberately. Build it yourself.

## How the build works

Everything is pinned so that two clean runs consume the same bytes: NInfer
converter commit `b2b96bae4dd88f95b9ea8126d68fae3b88caa374` (2026-08-18), the
base repo above, the official `Qwen/Qwen3.8-27B` frontend (sha256-pinned in
[`frontend.sha256`](../frontend.sha256)), and the
`nvidia/cuda:13.1.2-runtime-ubuntu24.04` convert image. Four steps:

1. **Fetch the converter.** The NInfer source tarball at the pinned commit
   (`curl | tar` on the host). Only the `tools/convert/...` Python is used;
   nothing is compiled.
2. **Download the base weights** into `work/ckpt` (resumable, `hf_transfer`, in
   a `python:3.12-slim` container). This includes the MTP shard the converter
   uses to derive the speculation head.
3. **Graft the official frontend.** The abliterated repo ships
   trivially-different tokenizer/config copies, and the converter's preflight
   sha-pins six frontend files, so the build overwrites them with the official
   `Qwen/Qwen3.8-27B` copies and then verifies via
   [`verify_frontend.py`](../verify_frontend.py). The graft runs in a container
   because rootful Docker leaves `work/ckpt` root-owned after step 2.
4. **Convert.** `tools.convert.qwen3_8_27b.convert` quantises the body, builds
   the proposal head from the MTP tensors plus the converter's ranking fixture,
   keeps the vision tower, and writes the `.ninfer` file plus a
   `*.conversion.json` report. Roughly ten minutes on a mid-range CUDA card.

### The one gotcha

The converter stamps its git revision into the report by shelling out to `git
rev-parse HEAD`. The stock CUDA image has no `git`, so the last step raises
`FileNotFoundError: 'git'` *after* the artifact is fully written, which makes a
successful build look failed. `build.sh` installs `git` in the convert
container; in a tarball checkout the stamp is simply empty, which is fine. Keep
`git` in the container if you adapt this pipeline.

### Overrides

All optional, shown with their defaults:

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
overridden; see the top of [`build.sh`](../build.sh).

## Reproducibility

The reference build produces:

```
sha256  714565ed29db4415322e9bc13a3464dc1fd8fcc911234740a79af67934e49969
bytes   18210531328
file    qwen3_8_27b_uncensored.ninfer
```

Byte-exact output additionally depends on the quantisation device and the
PyTorch/CUDA build, because low-bit rounding differs between GPU and CPU and
between torch versions. To match the reference sha, quantise with `--device
cuda` on the pinned image. Functionally the artifacts are equivalent either
way.

## Serving

The `.ninfer` file is self-contained. Point `ninfer-serve` at it, having built
the engine at the pinned commit in a `nvidia/cuda:13.1.2-devel-ubuntu24.04`
container (see [the NInfer repo](https://github.com/Neroued/ninfer)):

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
32 GB; bf16 KV runs out of memory at high context. NInfer exposes an
OpenAI-compatible `/v1/chat/completions`, but it rejects unknown
`chat_template_kwargs` and has no constrained-JSON mode, so a thin translating
proxy helps if your clients speak those dialects.

To revert to the official (censored) model, serve the official
[`neroued/Qwen3.8-27B-NInfer`](https://huggingface.co/neroued/Qwen3.8-27B-NInfer)
artifact instead: same engine, same flags, same recipe, only the base weights
differ.
