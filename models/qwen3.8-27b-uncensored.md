# Qwen3.8-27B Uncensored (NInfer, RTX 5090)

Abliterated Qwen3.8-27B as a single-file [NInfer](https://github.com/Neroued/ninfer)
artifact. Not redistributed: build it.

```bash
./build.sh        # → work/out/qwen3_8_27b_uncensored.ninfer
```

| | |
|---|---|
| **Output** | `qwen3_8_27b_uncensored.ninfer`, 18,210,531,328 bytes (16.96 GiB) |
| **Base** | [`JonathanColetti/Qwen3.8-27B-Uncensored`](https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored), BF16, 12 shards + `model-mtp.safetensors`, ~55 GB |
| **Architecture** | Qwen3.5-family multimodal: 27B, hidden 5120, 64 layers, 24 heads, vocab 248320, vision tower, 1-layer MTP |
| **Recipe** | groupwise-int `qwen3_8_27b-v1`: Q4/Q5/Q6 body, embedding and head `W8G32_F16S` |
| **Converter** | NInfer commit `b2b96bae4dd88f95b9ea8126d68fae3b88caa374`, run in `nvidia/cuda:13.1.2-runtime-ubuntu24.04` |
| **Frontend** | official `Qwen/Qwen3.8-27B`, pinned in [`frontend.sha256`](../frontend.sha256) |
| **Requirements** | [README](../README.md#building-the-ninfer-artifacts): Docker + NVIDIA runtime, ~11 GB VRAM, ~90 GB disk |

"Abliterated": the refusal direction is removed from the weights. Done and
published upstream; this repo only repackages. See
[Responsible use](../README.md#responsible-use).

## Build

Four containerised steps, ~10 minutes of GPU time after the download:

1. Fetch the converter source tarball at the pinned commit. Only
   `tools/convert/...` is used; nothing is compiled.
2. Download the base weights into `work/ckpt` (resumable, `hf_transfer`,
   `python:3.12-slim`). Includes the MTP shard.
3. Graft the official frontend. The base repo ships altered tokenizer/config
   files; the converter sha-pins six of them, so the build overwrites them
   with the official copies and checks via
   [`verify_frontend.py`](../verify_frontend.py).
4. Convert: `tools.convert.qwen3_8_27b.convert` quantises the body, builds the
   MTP proposal head, keeps the vision tower, writes the `.ninfer` and a
   `*.conversion.json` report.

Facts that matter if you adapt this:

- The convert container needs `git`. The converter stamps `git rev-parse HEAD`
  into the report after writing the artifact; without `git` it raises
  `FileNotFoundError` on a finished build. `build.sh` installs it.
- Steps 3 and 4 run in containers because rootful Docker leaves `work/ckpt`
  root-owned.
- The reference sha below needs `--device cuda` on the pinned image. CPU or
  another torch build gives a functionally equivalent but not byte-identical
  file.

```
sha256  714565ed29db4415322e9bc13a3464dc1fd8fcc911234740a79af67934e49969
bytes   18210531328
```

Overrides, shown with defaults:

```bash
W="$(pwd)/work"
DEVICE=cuda                     # or cpu
GPU_SELECT=all                  # or device=GPU-<uuid>
CPUSET=                         # e.g. 0-7
BASE_REPO=JonathanColetti/Qwen3.8-27B-Uncensored
NINFER_COMMIT=b2b96bae4dd88f95b9ea8126d68fae3b88caa374
HF_TOKEN=
```

`NINFER_REPO`, `FRONTEND_REPO`, `OUT_FILE`, `CONVERT_IMAGE`: see the top of
[`build.sh`](../build.sh).

## Serve

Engine built at the pinned commit in `nvidia/cuda:13.1.2-devel-ubuntu24.04`
(see the NInfer repo).

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

- `--kv-dtype int8` is required for full context plus vision in 32 GB; bf16 KV
  runs out of memory at high context.
- NInfer's `/v1/chat/completions` rejects unknown `chat_template_kwargs` and
  has no constrained-JSON mode. Put a translating proxy in front if clients
  need either.
- The censored equivalent is
  [`neroued/Qwen3.8-27B-NInfer`](https://huggingface.co/neroued/Qwen3.8-27B-NInfer):
  same engine, flags and recipe.
