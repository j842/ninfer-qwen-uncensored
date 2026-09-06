# Qwen3.8-Flash-Next (SGLang, RTX PRO 6000)

The Qwen4 architecture preview on one RTX PRO 6000 Blackwell (96 GB): the
stock SGLang image with a seven-piece sm_120 patch stack applied at
container start. Full 262,144-token window, text and image.

| | |
|---|---|
| **Model** | 180B total: 125B core (6B active, 512 experts, 10 per token), 51B PLE n-gram table, 4B MTP head. 48 layers: 36 gated-DeltaNet + 12 QSA. KV only on the 12 QSA layers: 12 KiB/token at fp8 |
| **Checkpoint** | [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4), 135.3 GB. Routed experts NVFP4; everything else BF16 |
| **Image** | `lmsysorg/sglang@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1` (tag `qwen38flashnext` as pushed 2026-08-26). Pin the digest: the tag moves |
| **Patches** | six from [gabrielolympie/sglang-flashnext-sm120](https://github.com/gabrielolympie/sglang-flashnext-sm120) at `67d2f92`, sha256-pinned and fetched (no licence upstream), plus SGLang PR #36556 applied by [`sm120-patch.py`](../flash-next/sm120-patch.py) |
| **Card** | 92.5 GB used, 3.14 GB free. Host: 51.2 GB pinned for the FP8 PLE table |
| **Decode** | 257 tok/s prose, 324 code at shallow depth; 236 / 288 at 250K |
| **Prefill** | 11,000–12,600 tok/s across the whole window |

Not usable: `Qwen/Qwen3.8-Flash-Next-FP8` (125.2 GiB resident, does not
fit); vLLM (loads only the first-party FP8 build).

## Patch stack

Pure Python under `python/sglang/**` plus one Triton kernel. No source build.

| Patch | Effect |
|---|---|
| PR [#36556](https://github.com/sgl-project/sglang/pull/36556) | sm_120 QSA decode: TRTLLM sparse-decode gate accepts SM120, fallback routed to the FA4 dispatcher. Also fixes the `qwen3_coder` tool parser token-0 loop ([#36537](https://github.com/sgl-project/sglang/issues/36537)) |
| `0001b` | RecoverSSM + WY output-only MTP verify on sm_120 (stock gate is `sm_major in (9, 10)`) |
| `0002` | fp8 tile dequant for QSA sparse prefill: makes `--kv-cache-dtype fp8_e4m3` work ([#36545](https://github.com/sgl-project/sglang/issues/36545)). Halves KV; the reason 262K fits |
| `0003` | fp32 prefill state for the sm_120 FlashInfer GDN kernel |
| `0004` | low-M bf16 dense GEMMs on a Triton split-K kernel (~90% of DRAM bandwidth vs cuBLAS's 20–75% under graph capture) |
| `0005` | W8A16 fp8 weight-only serving for low-M dense GEMMs. Bandwidth, not memory |
| `0006` | fp8 weight-only for the HyperConnection mix and decode lm_head |

`sm120-patch.py` applies them and refuses to start the server unless ten
postconditions (one symbol per patch) hold. `0002`, `0004`, `0005` apply
with `git apply`; `0001b`, `0003`, `0006` need `patch -F3` (one hunk each).
A restart re-runs it on the patched layer as a no-op.

## Launch

```bash
./flash-next/fetch-patches.sh    # → flash-next/patches/, sha256-verified
```

```bash
docker run -d --name qwen38-flash-next \
    --restart unless-stopped \
    --gpus '"device=0"' \
    --network host --ipc host \
    --shm-size 16g --memory 112g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --cap-add SYS_PTRACE \
    -v /path/to/Qwen3.8-Flash-Next-NVFP4:/models:ro \
    -v "$PWD/flash-next/patches:/patches:ro" \
    -v "$PWD/flash-next/patches/hot_tokens_64k.pt:/hot_tokens_64k.pt:ro" \
    -v "$PWD/flash-next/sm120-patch.py:/sm120-patch.py:ro" \
    -v "$PWD/flash-next/chat_template.jinja:/chat_template.jinja:ro" \
    -v qwen38fn_jit_cache:/root/.cache \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
    -e SGLANG_SM120_LOWM_FP8_WEIGHT=1 \
    -e SGLANG_SM120_LM_HEAD_FP8=1 \
    -e XDG_CACHE_HOME=/root/.cache \
    -e MAX_JOBS=4 -e FLASHINFER_NVCC_THREADS=2 \
    -e OMP_NUM_THREADS=8 \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    --entrypoint bash \
    lmsysorg/sglang@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1 \
    -c 'python3 /sm120-patch.py && exec python3 -m sglang.launch_server \
        --model-path /models --served-model-name default --tp 1 \
        --trust-remote-code \
        --quantization modelopt_fp4 --fp4-gemm-backend flashinfer_cutlass \
        --kv-cache-dtype fp8_e4m3 \
        --context-length 262144 --max-total-tokens 262144 \
        --mem-fraction-static 0.95 \
        --page-size 64 --chunked-prefill-size 4096 \
        --ple-offload-embedding \
        --linear-attn-prefill-backend flashinfer \
        --linear-attn-decode-backend flashinfer \
        --mamba-ssm-dtype bfloat16 --mamba-radix-cache-strategy extra_buffer \
        --mamba-track-interval 64 --max-mamba-cache-size 36 \
        --gdn-mtp-cache-mode none \
        --max-running-requests 6 --cuda-graph-max-bs-decode 6 \
        --speculative-algorithm NEXTN --speculative-num-steps 3 \
        --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
        --speculative-accept-threshold-single 0.3 \
        --speculative-accept-threshold-acc 0.3 \
        --speculative-token-map /hot_tokens_64k.pt \
        --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
        --chat-template /chat_template.jinja \
        --sampling-defaults model \
        --enable-metrics --enable-request-time-stats-logging \
        --watchdog-timeout 1800 \
        --host 0.0.0.0 --port 8001'
```

First start: ~20 minutes of autotune on a cold JIT cache, paid once into the
named volume. `--disable-flashinfer-autotune` skips it.

## Rules

- After any image change, generate past the first KV page and grep for
  `!!!!`. Stock QSA decode on sm_120 emits token 0 past ~65 tokens at page
  size 64; short prompts pass. #36556 is the fix.
- `--max-total-tokens` = context length, below what the memory fraction
  allows. That is where the fp8-KV saving becomes free VRAM. Lowering
  `--mem-fraction-static` shrinks context instead.
- `--max-mamba-cache-size` = 6 x `--max-running-requests`. Below that SGLang
  silently clamps `max_running_requests`; re-read it from the log.
- `--speculative-num-steps 3` is the ceiling: draft tokens (steps + 1) must
  not exceed the QSA compress ratio of 4. 4 loads 135 GB then dies; 2 loses
  ~9% on code.
- Accept thresholds 0.3 are exact at temperature 0 and lossy above it, and
  the model card wants 1.0 thinking / 0.7 non-thinking, so production is
  lossy. A temperature-0 grader cannot see the difference. Set both to 1.0
  if answers go flat. `--speculative-token-map` (FR-Spec, 64K hot tokens) is
  lossless: verification stays against the full distribution.
- `--mamba-ssm-dtype bfloat16`. fp8 SSM state corrupts output.
- `--tp 1`: `moe_intermediate_size` 640 shards below kernel tile size.
- `--ulimit memlock=-1` is mandatory for `--ple-offload-embedding`.
  `OMP_NUM_THREADS=8` because the PLE gather runs host-side on decode.
- `--memory 112g` and `MAX_JOBS=4` cap the JIT compile storm (~3 GB per
  `cicc`) on a host holding a 51 GB pinned table.
- The vendored chat template accepts `reasoning_effort: high|max`; the
  checkpoint's own raises on them.
- 3.14 GB free is thin. Levers in cost order: `--cuda-graph-max-bs-decode`
  6 to 4 (~0.8 GB); `--gdn-mtp-cache-mode full` (1.33 GB, loses `0001b`);
  context and pool to 196608 (0.75 GB). Watch the log for
  `available_gpu_mem`, `SIGQUIT`, `watchdog`.

## Measured

Single stream, idle card (acceptance counters are worker-global):

| depth | prose tok/s | prose accept | code tok/s | code accept | prefill |
|---|---|---|---|---|---|
| ~0 | 257.1 | | 323.6 | | TTFT 94 ms |
| 4,996 | 248.7 | 2.69 | 309.4 | 3.36 | ~12,100 |
| 59,941 | 249.3 | 2.73 | 303.1 | 3.32 | ~12,600 |
| 189,896 | 243.9 | 2.74 | 297.6 | 3.33 | ~11,600 |
| 249,857 | 236.2 | 2.68 | 287.8 | 3.25 | ~11,000 |

Acceptance out of 4. Unpatched image: 139.6 prose, 186.5 code at ~0 depth.

Memory, from the server log: 81.35 GB weights, 0.51 MTP head, 2.06 GDN state
(36 slots), 3.00 KV at 262K fp8, 0.26 draft KV, 1.44 workspace, 2.23 CUDA
graphs; 3.14 GB free.

## Sources

- [SGLang cookbook: Qwen3.8-Flash-Next](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next)
- [gabrielolympie/sglang-flashnext-sm120](https://github.com/gabrielolympie/sglang-flashnext-sm120), [jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000)
- SGLang [#36531](https://github.com/sgl-project/sglang/issues/36531), [#36537](https://github.com/sgl-project/sglang/issues/36537), [#36545](https://github.com/sgl-project/sglang/issues/36545), [#36556](https://github.com/sgl-project/sglang/pull/36556)
