# Qwen3.8-Flash-Next (llama.cpp, RTX 5090)

The 180B model on a 32 GB RTX 5090: llama.cpp with the routed experts in
system RAM, the last three expert layers on the card, MTP self-drafting on.
Full 262,144-token window, vision. llama.cpp is the only engine with
per-tensor placement (`--n-cpu-moe`); vLLM and SGLang need the ~78 GiB core
GPU-resident. 10 of 512 experts fire per token, so DDR4 traffic is ~1.4
GiB/token.

| | |
|---|---|
| **Engine** | llama.cpp b10819 (`74a7c897f049`) + [`maxspeed.patch`](../flash-next-5090/maxspeed.patch), image `llamacpp-qwen4exp:74a7c897f049-pff941add` |
| **Weights** | `Qwen3.8-Flash-Next-UD-Q4_K_XL-imx-MTP-PLE8-v1.gguf`, ~130 GiB, built below from [`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) |
| **Vision** | `mmproj-BF16.gguf`, 865 MiB, same repo |
| **Card** | RTX 5090, 30.2 GiB peak at 74K context, `--n-cpu-moe 45` |
| **Host** | ~130 GiB free RAM for the mmap page cache, 48 physical cores, NVMe |
| **Decode** | 52 tok/s short, 57 at 6K, 49 at 27K, 46 at 74K (spec off: 42 / 40 / 37 / 34) |
| **Prefill** | 953 / 899 / 827 tok/s at 2K / 8K / 32K |

## The weights

Three steps on top of the unsloth UD-Q4_K_XL imatrix requant
(revision `c8b5954a`, 4 shards, 103.7 GiB):

1. Export the model's own MTP head. [`fetch_mtp.py`](../flash-next-5090/fetch_mtp.py)
   range-reads the `mtp.*` tensors (~10 GB) out of `Qwen/Qwen3.8-Flash-Next`
   into a stub checkpoint; the patched tree's converter writes it as a GGUF.
2. Graft it as `blk.48` with [`graft_mtp.py`](../flash-next-5090/graft_mtp.py)
   (block_count 49, `nextn_predict_layers` 1). Needs `gguf-py` from the same
   tree: symlink the checkout as `tree/` beside the script.
3. Swap the PLE table from IQ4_NL (26.8 GiB) to Q8_0 (50.7 GiB) with
   [`graft_ple.py`](../flash-next-5090/graft_ple.py). The donor is shard 3 of
   unsloth's Q8_0 quant, which holds only that tensor, quantised once from
   BF16. The PLE is a per-token gather, so this costs host RAM and no decode.

```bash
hf download unsloth/Qwen3.8-Flash-Next-GGUF --revision c8b5954a88c2775c546b92593eda40ea041d3176 \
    --include 'UD-Q4_K_XL/*' 'mmproj-BF16.gguf' 'Q8_0/Qwen3.8-Flash-Next-Q8_0-00003-of-00006.gguf' \
    --local-dir /path/to/flash-next-gguf
cd /path/to/flash-next-gguf
python3 fetch_mtp.py --out mtp-ckpt
python3 /path/to/llama.cpp/convert_hf_to_gguf.py mtp-ckpt --mtp --outtype q8_0 --outfile mtp-q8_0.gguf
python3 graft_mtp.py UD-Q4_K_XL/*-0000{1,2,3,4}-of-00004.gguf mtp-q8_0.gguf merged-mtp.gguf
python3 graft_ple.py merged-mtp.gguf Q8_0/Qwen3.8-Flash-Next-Q8_0-00003-of-00006.gguf \
    Qwen3.8-Flash-Next-UD-Q4_K_XL-imx-MTP-PLE8-v1.gguf
```

The plain 4-shard UD-Q4_K_XL serves unchanged with `--spec-type` left off.
UD-IQ4_XS is the wrong trade: 94 of 144 expert tensors drop to IQ3_S, and
decode here is not bandwidth-bound (~51 of ~205 GB/s).
PR #28243 can also draft from unsloth's `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`
sidecar via `-md`; not used here.

## The engine

Base b10819 carries `qwen4exp` (#27742) and the merged follow-ups (#27941,
#28023, #28123, #28040, #27970). `maxspeed.patch` adds:

| PR | Effect |
|---|---|
| [#28243](https://github.com/ggml-org/llama.cpp/pull/28243) | MTP draft head, `--spec-type draft-mtp`, self-draft from `blk.48` |
| [#28068](https://github.com/ggml-org/llama.cpp/pull/28068) | GDN QK norm as `x * rsqrt(sum + eps)` |
| [#28213](https://github.com/ggml-org/llama.cpp/pull/28213) | QSA decode attends only the indexer-selected cells. `QWEN4EXP_QSA_GATHER=0` restores the masked path |
| unsloth [#144](https://github.com/unslothai/llama.cpp/pull/144) `5a08a717` | CUDA graph cache keyed by shape, so MTP verify batches of 2/3/4 tokens stop resetting warm-up |

```bash
./flash-next-5090/build-engine.sh   # → llamacpp-qwen4exp:74a7c897f049-pff941add, ~15 min
```

The tag is base commit + patch sha256. `BUILD_CPUSET=48-55` pins the
compile. If CUDA 13 fails on the pinned commit, drop both images to 12.8.1
(`sm_120` needs 12.8 or newer). Drop hunks as PRs merge; with an empty patch
use `ghcr.io/ggml-org/llama.cpp:server-cuda`.

## Launch

```bash
docker run -d --name qwen38-flash-next-5090 \
    --restart unless-stopped \
    --gpus '"device=0"' \
    --ulimit memlock=-1 \
    --cpuset-cpus 0-47 \
    -p 8001:8080 \
    -v /path/to/flash-next-gguf:/models:ro \
    -v "$PWD/flash-next/chat_template.jinja:/chat_template.jinja:ro" \
    -e LLAMA_ATTN_ROT_DISABLE=1 \
    llamacpp-qwen4exp:74a7c897f049-pff941add \
    --model /models/Qwen3.8-Flash-Next-UD-Q4_K_XL-imx-MTP-PLE8-v1.gguf \
    --mmproj /models/mmproj-BF16.gguf \
    --chat-template-file /chat_template.jinja \
    --alias Qwen3.8-Flash-Next \
    --host 0.0.0.0 --port 8080 \
    -ngl 999 --n-cpu-moe 45 \
    --ctx-size 262144 --parallel 1 \
    --flash-attn on --cache-type-k f16 --cache-type-v f16 \
    --threads 48 --threads-batch 48 \
    --batch-size 2048 --ubatch-size 2048 \
    --load-mode mmap \
    --spec-type draft-mtp --spec-draft-n-max 2 \
    --override-tensor 'blk\.48\.ffn_(up|down|gate|gate_up)_(ch|)exps=CPU' \
    --cont-batching --metrics --slots --jinja
```

`/health` in under a minute; experts and PLE fault in from disk on demand.

## Rules

- Do not apply llama.cpp #28118 (on-device speculative checkpoints): it
  aborts the third request, spec on or off.
- Keep `--spec-draft-n-max 2`. 3 wins at depth and loses to spec-off on
  low-acceptance prose (48% on 768-token prose).
- `--n-cpu-moe 45` is the floor with MTP on: the draft context needs ~2.5 GiB
  and 43 crash-loops on an 865 MiB `cudaMalloc` at load. CUDA OOM at load
  means raise it; at 48 the budget is over from the dense side, so lower
  `--ctx-size` (~0.75 GiB per 32K), `--ubatch-size` to 1024, or drop
  `--mmproj`.
- Physical cores only, no SMT. `--cpuset-cpus` places the threads;
  `--cpu-mask` did not.
- `--parallel 1`. A second slot costs each stream ~29%. `--ctx-size` is the
  total across slots.
- `--load-mode mmap`. mlock pins 100 GiB from the loading thread; direct I/O
  bypasses the page cache the expert faulting needs.
- n-gram speculation and `--moe-expert-cache` measured no gain here; both
  off.
- `--ubatch-size 2048`: prefill copies host experts per batch; the large
  ubatch amortises PCIe and is most of the compute buffer.
- `LLAMA_ATTN_ROT_DISABLE=1`: the `qwen4exp` attention path aborts at load
  with quantised-KV rotation active.
- The vendored chat template accepts `reasoning_effort: high|max`; the
  checkpoint's own raises on them.
- Judge depth with a real document through `/v1/chat/completions`
  ([`probe-chat.sh`](../flash-next-5090/probe-chat.sh)). Synthetic filler
  EOSes at token 1 on this build and reads as decode 0.
- Greedy output is not bit-identical spec on versus off (MUL_MAT batch
  invariance, per the #28243 thread). Not a defect.
- The first request after a restart can report 1–9 tok/s in the client's
  `timings` while the server log shows the normal rate. Read the log.
- Change the served alias on every output-affecting engine change. A grader
  that caches answers by model identity re-uses a broken build's scores.
- Bench until consecutive runs agree; the page cache is cold on pass one and
  deep-context rows are still cold on pass two.

## Measured

Idle box, temperature 0, single stream, `--n-cpu-moe 45`:

| workload | spec off | draft-mtp n=2 (accept) |
|---|---|---|
| code / fix / prose, 128 tokens | 41.5 / 41.5 / 42.1 | 51.7 (86%) / 51.9 (83%) / 50.5 (79%) |
| code / fix / prose, 768 tokens | 42.2 / 42.2 / 41.8 | 55.8 (92%) / 54.5 (84%) / 47.7 (65%) |
| chat, 6K of real documents | 40.4 | 56.7 (90%) |
| chat, 27K | 37.4 | 48.7 (86%) |
| chat, 74K | 34.1 | 46.3 (89%) |
| maths, thinking off | 40.9 | 53.5 (88%) |

Prefill 953 / 899 / 827 tok/s at 2K / 8K / 32K, unchanged by MTP.

## Memory

At `--n-cpu-moe 48` (nothing on the card) the GPU holds 18.1 GiB: 5.2 dense,
0.9 ViT, 6.0 KV at 262K f16, 6.0 compute buffers. Each expert layer moved to
the card is 1.46 GiB (1.71–1.90 for layers 2, 4, 30, 46, 47). Host page
cache: 71.7 GiB experts + 50.7 GiB PLE.

## Sources

- [#27742](https://github.com/ggml-org/llama.cpp/pull/27742), [#28243](https://github.com/ggml-org/llama.cpp/pull/28243), [#28068](https://github.com/ggml-org/llama.cpp/pull/28068), [#28213](https://github.com/ggml-org/llama.cpp/pull/28213), [#28118](https://github.com/ggml-org/llama.cpp/pull/28118)
- [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
