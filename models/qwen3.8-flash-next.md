# Qwen3.8-Flash-Next (SGLang, RTX PRO 6000)

The Qwen4 architecture preview on one RTX PRO 6000 Blackwell (96 GB), served
by the stock SGLang image with a seven-piece sm_120 patch stack applied at
container start. Full 262,144-token window.

**257 tok/s prose / 324 tok/s code at shallow depth, 236 / 288 at 250K
tokens, prefill 11,000-12,600 tok/s throughout**, single stream, measured
2026-08-31.

| | |
|---|---|
| **Model** | 180B total: 125B core (6B active), 51B PLE n-gram table, 4B MTP head. 512 experts, 10 per token, `moe_intermediate_size` 640. 48 layers: 36 gated-DeltaNet + 12 QSA. KV only on the 12 QSA layers, 2 KV heads x 256: 12 KiB/token at fp8 |
| **Checkpoint** | [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4), 135.3 GB. Routed experts NVFP4 (W4A4, group 16, FP8 block scales); everything else BF16 and byte-identical to first-party. GSM8K 97.27% vs 97.12-97.50 BF16; AIME26 98.75 pass@1 |
| **Image** | `lmsysorg/sglang@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1` (tag `qwen38flashnext`, pushed 2026-08-26). Pinned by digest |
| **Patches** | six from [gabrielolympie/sglang-flashnext-sm120](https://github.com/gabrielolympie/sglang-flashnext-sm120) at `67d2f92` (sha256-pinned, fetched not committed: no licence upstream) + SGLang PR #36556 |
| **Modality** | text + image. Video encoder disabled |

Alternatives that do not apply: `Qwen/Qwen3.8-Flash-Next-FP8` (125.2 GiB
resident, does not fit), `lovedheart/...-NVFP4-FP8` (less validated), vLLM
(loads only the first-party FP8 build).

## Memory, from the server log

```
GPU   81.35 GB  weights (68.0 NVFP4 experts + BF16 dense)
    +  0.51 GB  MTP draft head (NVFP4)
    +  2.06 GB  GDN state, 36 slots, bf16
    +  3.00 GB  KV at 262,144 tokens, fp8
    +  0.26 GB  draft KV
    +  1.44 GB  workspace
    +  2.23 GB  CUDA graphs (verify 1.33 + draft 0.66 + extend 0.24)
    =  3.14 GB  free on a 95.6 GiB card
HOST 51.2 GB    FP8 PLE table, pinned (--ple-offload-embedding)
```

## Patch stack

All pure Python under `python/sglang/**` plus one Triton kernel; no source
build, no CUDA toolchain.

| Patch | Effect |
|---|---|
| PR [#36556](https://github.com/sgl-project/sglang/pull/36556) | sm_120 QSA decode: accepts SM120 in the TRTLLM sparse-decode gate and routes the fallback to the FA4 dispatcher. Also fixes the `qwen3_coder` tool parser token-0 loop ([#36537](https://github.com/sgl-project/sglang/issues/36537)) |
| `0001b` | RecoverSSM + WY output-only MTP verify on sm_120 (stock gate is `sm_major in (9, 10)`). One launch per T draft tokens instead of the Triton per-token kernel |
| `0002` | fp8 tile dequant for QSA sparse prefill: makes `--kv-cache-dtype fp8_e4m3` work ([#36545](https://github.com/sgl-project/sglang/issues/36545)) |
| `0003` | fp32 prefill state for the sm_120 FlashInfer GDN kernel (upstream allows bf16 only at CC == 10) |
| `0004` | low-M bf16 dense GEMMs to a Triton split-K kernel (~90% of DRAM bandwidth vs cuBLAS's 20-75% under graph capture) |
| `0005` | W8A16 fp8 weight-only serving for low-M dense GEMMs. Not a memory saving |
| `0006` | fp8 weight-only for the HyperConnection mix and decode lm_head |

[`flash-next/sm120-patch.py`](../flash-next/sm120-patch.py) applies them to
the installed tree and refuses to start the server unless ten postconditions
(one symbol per patch) hold. Mechanics it depends on:

- `0002`, `0004`, `0005` apply with `git apply`; `0001b`, `0003`, `0006`
  need `patch -F3` (context drift, one hunk each).
- `git apply -3` is unavailable: the image's `.git` index does not contain
  the Qwen4 files.
- GNU `patch` runs with `--batch --forward` and closed stdin; otherwise it
  prompts on an already-applied hunk and the entrypoint hangs.
- The postcondition check runs in a fresh interpreter. The same process that
  imported modules earlier answers `hasattr` from stale `sys.modules`.
- Restart re-runs the script on the already-patched layer; it is a no-op when
  all postconditions hold.

## Corruption check after any image change

Stock QSA decode on sm_120 emits token 0 (`!`) past the first KV page, at an
absolute position tracking `--page-size` (~65 at 64, ~193 at 128). Short
prompts pass. PR #36556 is the fix. After every install, generate past the
page boundary and grep for the `!!!!` signature.

## Launch

```bash
./flash-next/fetch-patches.sh    # → flash-next/patches/, sha256-verified
```

```bash
docker run -d --name qwen38-flash-next \
    --gpus '"device=0"' \
    --network host --ipc host \
    --shm-size 16g --memory 112g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --cap-add SYS_PTRACE \
    -v /path/to/Qwen3.8-Flash-Next-NVFP4:/models:ro \
    -v "$PWD/flash-next/patches:/patches:ro" \
    -v "$PWD/flash-next/patches/hot_tokens_64k.pt:/hot_tokens_64k.pt:ro" \
    -v "$PWD/flash-next/sm120-patch.py:/sm120-patch.py:ro" \
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
        --sampling-defaults model \
        --enable-metrics --enable-request-time-stats-logging \
        --watchdog-timeout 1800 \
        --host 0.0.0.0 --port 8001'
```

Flags:

- `--ple-offload-embedding`: the 51.2 GB PLE table in pinned host RAM.
  `--ulimit memlock=-1` is mandatory for it; `--shm-size` and
  `OMP_NUM_THREADS=8` because the PLE gather runs host-side on the decode
  path.
- `--tp 1`: `moe_intermediate_size` 640 shards below kernel tile size at
  TP>1.
- `--memory 112g` and `MAX_JOBS=4`: cap a JIT compile storm (~3 GB per
  `cicc`) on a 125 GB host holding a 51 GB pinned table.
- Autotune costs ~20 minutes on a cold JIT cache, hence the named
  `/root/.cache` volume. `--disable-flashinfer-autotune` skips it.
- `--max-total-tokens` = context length, deliberately below what the memory
  fraction allows: that is where the fp8-KV saving becomes free VRAM.
  Lowering `--mem-fraction-static` shrinks context instead.
- `--max-mamba-cache-size 36` = 6 x `--max-running-requests`. Below that
  ratio the speculative CUDA graphs cap at bs=4, and SGLang silently clamps
  `max_running_requests`; re-read it from the log after changing either.
- `--mamba-ssm-dtype bfloat16`: GDN state is ~113 MB/slot at fp32, ~57 MB at
  bf16, and scales with concurrency, not context. fp8 SSM state corrupts
  output. Issues #36611/#36627 (fp32 assert) apply to SM90 only; #36701's
  "flashinfer unreachable on sm_120" applies to an unpatched tree.
- `--speculative-num-steps 3` is the ceiling: draft tokens (steps + 1) must
  not exceed the QSA compress ratio of 4; 4 loads all 135 GB, then dies at
  scheduler start. 2 loses ~9% on code.
- Accept thresholds 0.3 force-accept drafts the target gives >=30%. Exact at
  temperature 0, lossy above it; the model card wants 1.0 thinking / 0.7
  non-thinking, so production is lossy. Upstream C1 at temp 0.6: 179 tok/s at
  1.0, 203 at 0.5, 231 at 0.3. Set both to 1.0 if answers go flat.
- `--speculative-token-map`: FR-Spec 64K hot-token draft vocab. Lossless;
  verification is against the full distribution.
- The checkpoint's chat template raises on `reasoning_effort: high|max`;
  carry a fork with `--chat-template` if clients send those.

3.14 GB free is thin. Levers in cost order: `--cuda-graph-max-bs-decode 6`
to 4 (~0.7-0.9 GB); `--gdn-mtp-cache-mode full` (1.33 GB, loses patch
`0001b`'s verifier); context and pool to 196608 (0.75 GB). Watch the log for
`available_gpu_mem`, `SIGQUIT`, `watchdog`.

## Measured

Shallow depth, same card and checkpoint:

| | unpatched | patched |
|---|---|---|
| prose decode | 139.6 | 257.1 |
| code decode | 186.5 | 323.6 |
| TTFT | ~106 ms | 94 ms |

At depth, idle worker (acceptance counters are worker-global; measure with
nothing else running):

| depth | prose tok/s | prose accept | code tok/s | code accept | prefill |
|---|---|---|---|---|---|
| 4,996 | 248.7 | 2.69 | 309.4 | 3.36 | ~12,100 |
| 59,941 | 249.3 | 2.73 | 303.1 | 3.32 | ~12,600 |
| 189,896 | 243.9 | 2.74 | 297.6 | 3.33 | ~11,600 |
| 249,857 | 236.2 | 2.68 | 287.8 | 3.25 | ~11,000 |

Acceptance out of 4. Unpatched: prose 156/164/153, code 213/208/211 at the
first three depths; could not reach the fourth.

## Sources

- [SGLang cookbook: Qwen3.8-Flash-Next](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next)
- [gabrielolympie/sglang-flashnext-sm120](https://github.com/gabrielolympie/sglang-flashnext-sm120), [jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000)
- SGLang issues [#36531](https://github.com/sgl-project/sglang/issues/36531), [#36537](https://github.com/sgl-project/sglang/issues/36537), [#36545](https://github.com/sgl-project/sglang/issues/36545)
