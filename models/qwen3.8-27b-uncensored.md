# Qwen3.8-27B Uncensored (NInfer, RTX 5090)

Abliterated Qwen3.8-27B as one [NInfer](https://github.com/Neroued/ninfer)
artifact. Not redistributed: build it.

```bash
./build.sh        # → work/out/qwen3_8_27b_uncensored.ninfer
```

| | |
|---|---|
| **Output** | `qwen3_8_27b_uncensored.ninfer`, 18,210,531,328 bytes (16.96 GiB) |
| **Base** | [`JonathanColetti/Qwen3.8-27B-Uncensored`](https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored), BF16, 12 shards + `model-mtp.safetensors`, ~55 GB |
| **Architecture** | 27B dense, hidden 5120, 64 layers, vocab 248320, vision tower, 1-layer MTP |
| **Recipe** | groupwise-int `qwen3_8_27b-v1`: Q4/Q5/Q6 body, embedding and head `W8G32_F16S` |
| **Converter** | NInfer `ad0f3d384b5cbcec4a48a3951c287b4e9831443e` (2026-09-04) in `nvidia/cuda:13.1.2-runtime-ubuntu24.04` |
| **Frontend** | official `Qwen/Qwen3.8-27B`, pinned in [`frontend.sha256`](../frontend.sha256) |
| **Requirements** | Docker + NVIDIA runtime, ~11 GB VRAM, ~90 GB disk ([README](../README.md#building-the-ninfer-artifacts)) |

"Abliterated": the refusal direction is removed from the weights. Done and
published upstream; this repo only repackages. See
[Responsible use](../README.md#responsible-use).

## Build

Four containerised steps, ~10 minutes of GPU time after the download:

1. Converter source tarball at the pinned commit. Only `tools/convert/...`
   is used.
2. Base weights into `work/ckpt` (resumable, `hf_transfer`). Includes the
   MTP shard.
3. Graft the official frontend: the base repo ships altered tokenizer and
   config files, and the converter sha-pins six of them.
   [`verify_frontend.py`](../verify_frontend.py) checks before the quantise.
4. Convert: `tools.convert.qwen3_8_27b.convert` quantises the body, builds
   the MTP proposal head, keeps the vision tower, writes the `.ninfer` and a
   `*.conversion.json` report.

Overrides, with defaults:

```bash
W="$(pwd)/work"
DEVICE=cuda                     # or cpu
GPU_SELECT=all                  # or device=GPU-<uuid>
CPUSET=                         # e.g. 0-7
BASE_REPO=JonathanColetti/Qwen3.8-27B-Uncensored
NINFER_COMMIT=ad0f3d384b5cbcec4a48a3951c287b4e9831443e
HF_TOKEN=
```

Reference sha256 from converter `b2b96bae` on `--device cuda`:
`714565ed29db4415322e9bc13a3464dc1fd8fcc911234740a79af67934e49969`. CPU or a
different converter commit gives an equivalent file, not a byte-identical
one.

## Serve

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

## Rules

- The convert container needs `git`: the converter runs `git rev-parse HEAD`
  after writing the artifact and raises without it. `build.sh` installs it.
- Steps 3 and 4 run in containers because rootful Docker leaves `work/ckpt`
  root-owned.
- `--kv-dtype int8` for full context plus vision in 32 GB; bf16 KV runs out
  of memory at high context.
- The API rejects `response_format` and unknown `chat_template_kwargs`.
  Translate in a proxy.
- The censored equivalent is
  [`neroued/Qwen3.8-27B-NInfer`](https://huggingface.co/neroued/Qwen3.8-27B-NInfer):
  same engine, flags and recipe.
